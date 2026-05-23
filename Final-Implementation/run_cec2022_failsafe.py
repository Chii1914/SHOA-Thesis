"""Fail-safe orchestrator for CEC2022 experiments with PSO, SHOA and SHOA-COMBINED.

This script executes a reproducible protocol over CEC2022, then builds
statistics and figures suitable for thesis/paper reporting.

Protocol highlights:
- 12 CEC2022 functions
- Dimensions: 10 and 20 (configurable)
- Independent runs per function: default 30
- Error metric: abs(f(x_best) - f*)
- Statistical comparison requested: SHO vs PSO only (Wilcoxon + rank summaries)
- Contribution plots: SHOA-COMBINED only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from opfunu.cec_based import cec2022
from scipy.stats import wilcoxon


DEFAULT_MAX_FES = {10: 200_000, 20: 1_000_000}
REPRESENTATIVE_FUNCTIONS = (1, 5, 8, 12)


def setup_orchestrator_logger(logs_dir: Path, level_name: str) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger = logging.getLogger("cec2022_failsafe")
    logger.setLevel(level)
    logger.propagate = False

    # Reconfigure handlers on each run to avoid duplicate entries.
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logs_dir / "orchestrator.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


@dataclass(frozen=True)
class JobSpec:
    algorithm: str
    dim: int
    runs: int
    functions: str
    seed: int
    max_fes: int
    max_iter: int
    output_dir: Path
    cwd: Path
    command: list[str]

    @property
    def job_id(self) -> str:
        return f"{self.algorithm}_D{self.dim}"


def parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        values.append(int(tok))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_function_ids(raw: str) -> list[int]:
    value = raw.strip().lower()
    if value == "all":
        return list(range(1, 13))

    result: set[int] = set()
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", maxsplit=1)
            start = int(left)
            end = int(right)
            if start > end:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(token))

    filtered = sorted(fid for fid in result if 1 <= fid <= 12)
    if not filtered:
        raise ValueError("No valid CEC2022 function IDs in --functions")
    return filtered


def infer_function_id(function_name: str | None, function_id: str | None) -> int:
    if function_id is not None and str(function_id).strip() != "":
        return int(float(function_id))

    if function_name is None:
        raise ValueError("Cannot infer function id from empty function_name")

    match = re.search(r"F\s*(\d+)", function_name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer function id from function_name={function_name!r}")
    return int(match.group(1))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.glob("run-*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def discover_new_or_latest_run_dir(base_dir: Path, previous: set[str]) -> Path | None:
    if not base_dir.exists():
        return None
    now = [p for p in base_dir.glob("run-*") if p.is_dir()]
    new = [p for p in now if str(p.resolve()) not in previous]
    if new:
        return sorted(new, key=lambda p: p.stat().st_mtime)[-1]
    if now:
        return sorted(now, key=lambda p: p.stat().st_mtime)[-1]
    return None


def run_job_fail_safe(
    job: JobSpec,
    *,
    logs_dir: Path,
    retries: int,
    state: dict,
    continue_on_failure: bool,
    logger: logging.Logger,
) -> dict:
    jobs_state = state.setdefault("jobs", {})
    job_state = jobs_state.setdefault(job.job_id, {})

    if job_state.get("status") == "completed":
        run_dir_str = job_state.get("run_dir", "")
        if run_dir_str and Path(run_dir_str).exists():
            logger.info("Skipping %s: already completed at %s", job.job_id, run_dir_str)
            return job_state

    attempts_done = int(job_state.get("attempts", 0))
    max_attempts = max(1, retries + 1)

    before = set(str(p.resolve()) for p in job.output_dir.glob("run-*") if p.is_dir()) if job.output_dir.exists() else set()

    for attempt in range(attempts_done + 1, max_attempts + 1):
        job_state["status"] = "running"
        job_state["attempts"] = attempt
        job_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        log_file = logs_dir / f"{job.job_id}.attempt{attempt}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting %s (attempt %d/%d) | dim=%d runs=%d max_fes=%d max_iter=%d",
            job.job_id,
            attempt,
            max_attempts,
            job.dim,
            job.runs,
            job.max_fes,
            job.max_iter,
        )
        logger.info("Command for %s: %s", job.job_id, " ".join(job.command))
        logger.info("Job log file: %s", str(log_file.resolve()))

        with log_file.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND:\n")
            handle.write(" ".join(job.command) + "\n\n")
            handle.write(f"CWD: {job.cwd}\n\n")
            handle.flush()

            completed = subprocess.run(
                job.command,
                cwd=str(job.cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        job_state["return_code"] = int(completed.returncode)
        job_state["log_file"] = str(log_file.resolve())

        if completed.returncode == 0:
            run_dir = discover_new_or_latest_run_dir(job.output_dir, before)
            if run_dir is None:
                run_dir = discover_latest_run_dir(job.output_dir)

            job_state["status"] = "completed"
            job_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            job_state["run_dir"] = str(run_dir.resolve()) if run_dir is not None else ""
            logger.info(
                "Completed %s successfully | run_dir=%s",
                job.job_id,
                job_state["run_dir"] or "<not-found>",
            )
            return job_state

        job_state["status"] = "failed"
        job_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.warning(
            "Failed %s on attempt %d/%d with return_code=%d",
            job.job_id,
            attempt,
            max_attempts,
            int(completed.returncode),
        )

        if attempt < max_attempts:
            logger.info("Retrying %s after backoff", job.job_id)
            time.sleep(min(10, 2 * attempt))

    if not continue_on_failure:
        raise RuntimeError(f"Job {job.job_id} failed after {max_attempts} attempts")

    logger.error("Job %s exhausted retries and remains failed", job.job_id)

    return job_state


def compute_pso_max_iter(max_fes: int, particles: int) -> int:
    # nfev ~ particles (initial) + particles * iter
    if particles <= 0:
        raise ValueError("particles must be > 0")
    return max(1, (max_fes - particles) // particles)


def compute_shoa_max_iter(max_fes: int, pop: int) -> int:
    # nfev ~ pop (initial) + iter * (pop + pop//2)
    if pop <= 0:
        raise ValueError("pop must be > 0")
    per_iter = pop + pop // 2
    return max(1, (max_fes - pop) // max(1, per_iter))


def compute_combined_max_iter(max_fes: int, pop: int) -> int:
    # Keep max_iter high enough so --max-fes becomes the real stop criterion.
    base = compute_shoa_max_iter(max_fes=max_fes, pop=pop)
    return int(base + 50)


def build_jobs(args: argparse.Namespace, repo_root: Path, output_root: Path) -> list[JobSpec]:
    dims = parse_int_csv(args.dims)
    function_ids = parse_function_ids(args.functions)
    functions_raw = args.functions if args.functions.strip().lower() == "all" else ",".join(str(fid) for fid in function_ids)

    py_exec = args.python_executable or sys.executable

    jobs: list[JobSpec] = []

    for dim in dims:
        max_fes = int(args.max_fes_10 if dim == 10 else args.max_fes_20)
        if max_fes <= 0:
            raise ValueError(f"Invalid MaxFEs for D={dim}: {max_fes}")

        combined_max_fes = int(args.combined_max_fes)
        if combined_max_fes <= 0:
            raise ValueError(f"Invalid --combined-max-fes: {combined_max_fes}")

        pso_iter = compute_pso_max_iter(max_fes=max_fes, particles=args.pso_particles)
        shoa_iter = compute_shoa_max_iter(max_fes=max_fes, pop=args.shoa_pop)
        combined_iter = compute_combined_max_iter(max_fes=combined_max_fes, pop=args.combined_pop)

        pso_cwd = repo_root / "Final-Implementation" / "PSO" / "cec2022"
        shoa_cwd = repo_root / "Final-Implementation" / "SHOA" / "cec2022"
        combined_cwd = repo_root / "Final-Implementation" / "SHOA-COMBINED" / "cec2022"

        pso_out = output_root / "raw" / "PSO" / f"D{dim}"
        shoa_out = output_root / "raw" / "SHOA" / f"D{dim}"
        combined_out = output_root / "raw" / "SHOA-COMBINED" / f"D{dim}"

        pso_cmd = [
            py_exec,
            "run_pso_cec2022.py",
            "--functions",
            functions_raw,
            "--dim",
            str(dim),
            "--particles",
            str(args.pso_particles),
            "--max-iter",
            str(pso_iter),
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--w",
            str(args.pso_w),
            "--c1",
            str(args.pso_c1),
            "--c2",
            str(args.pso_c2),
            "--output-dir",
            str(pso_out.resolve()),
            "--log-level",
            args.log_level,
        ]

        shoa_cmd = [
            py_exec,
            "run_shoa_cec2022.py",
            "--functions",
            functions_raw,
            "--dim",
            str(dim),
            "--pop",
            str(args.shoa_pop),
            "--max-iter",
            str(shoa_iter),
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(shoa_out.resolve()),
            "--log-level",
            args.log_level,
        ]

        combined_cmd = [
            py_exec,
            "run_cec2022_combined.py",
            "--functions",
            functions_raw,
            "--dim",
            str(dim),
            "--pop",
            str(args.combined_pop),
            "--max-iter",
            str(combined_iter),
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--max-fes",
            str(combined_max_fes),
            "--restart-enabled",
            "1" if args.combined_rescue_enabled else "0",
            "--restart-percent",
            str(args.combined_restart_percent),
            "--restart-cooldown-fes-ratio",
            str(args.combined_restart_cooldown_ratio),
            "--restart-dominance-threshold",
            str(args.combined_restart_dominance_threshold),
            "--lime-min-samples",
            str(args.combined_lime_min_samples),
            "--stagnation-lime-selection-mode",
            args.combined_lime_selection_mode,
            "--progress-every",
            str(args.combined_progress_every),
            "--output-dir",
            str(combined_out.resolve()),
            "--log-level",
            args.log_level,
        ]

        jobs.append(
            JobSpec(
                algorithm="PSO",
                dim=dim,
                runs=args.runs,
                functions=functions_raw,
                seed=args.seed,
                max_fes=max_fes,
                max_iter=pso_iter,
                output_dir=pso_out,
                cwd=pso_cwd,
                command=pso_cmd,
            )
        )
        jobs.append(
            JobSpec(
                algorithm="SHOA",
                dim=dim,
                runs=args.runs,
                functions=functions_raw,
                seed=args.seed,
                max_fes=max_fes,
                max_iter=shoa_iter,
                output_dir=shoa_out,
                cwd=shoa_cwd,
                command=shoa_cmd,
            )
        )
        jobs.append(
            JobSpec(
                algorithm="SHOA-COMBINED",
                dim=dim,
                runs=args.runs,
                functions=functions_raw,
                seed=args.seed,
                max_fes=combined_max_fes,
                max_iter=combined_iter,
                output_dir=combined_out,
                cwd=combined_cwd,
                command=combined_cmd,
            )
        )

    return jobs


def get_fstar(function_id: int, dim: int, cache: dict[tuple[int, int], float]) -> float:
    key = (function_id, dim)
    if key in cache:
        return cache[key]

    class_name = f"F{function_id}2022"
    func_class = getattr(cec2022, class_name)
    func = func_class(ndim=dim)
    fstar = float(getattr(func, "f_global", getattr(func, "f_bias", 0.0)))
    cache[key] = fstar
    return fstar


def _to_int(value: str | int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raw = str(value).strip()
    if not raw:
        return default
    return int(float(raw))


def _to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return default
    return float(raw)


def collect_experiment_data(
    completed_jobs: list[dict],
    *,
    fstar_cache: dict[tuple[int, int], float],
) -> tuple[list[dict], dict[tuple[str, int, int], list[dict]], dict[tuple[str, int, int, int], list[tuple[float, float]]]]:
    per_run_rows: list[dict] = []
    grouped_errors: dict[tuple[str, int, int], list[dict]] = {}
    convergence_points: dict[tuple[str, int, int, int], list[tuple[float, float]]] = {}

    for job in completed_jobs:
        algorithm = str(job["algorithm"])
        dim = int(job["dim"])
        run_dir = Path(job["run_dir"])

        runs_raw = read_csv_rows(run_dir / "runs_raw.csv")
        for row in runs_raw:
            fid = infer_function_id(row.get("function_name"), row.get("function_id"))
            row_dim = _to_int(row.get("dimension"), dim)
            run_number = _to_int(row.get("run_number"), 0)
            best_fitness = _to_float(row.get("best_fitness"), np.nan)
            fstar = get_fstar(fid, row_dim, fstar_cache)
            error = abs(best_fitness - fstar)

            out_row = {
                "algorithm": algorithm,
                "dimension": row_dim,
                "function_id": fid,
                "function_name": f"F{fid}",
                "run_number": run_number,
                "best_fitness": best_fitness,
                "f_star": fstar,
                "error": error,
                "run_dir": str(run_dir.resolve()),
            }
            per_run_rows.append(out_row)

            grouped_errors.setdefault((algorithm, row_dim, fid), []).append(out_row)

        full_output = read_csv_rows(run_dir / "full_output.csv")
        for row in full_output:
            fid = infer_function_id(row.get("function_name"), row.get("function_id"))
            row_dim = _to_int(row.get("dimension"), dim)
            run_number = _to_int(row.get("run_number"), 0)

            fe = row.get("fe")
            if fe is None or str(fe).strip() == "":
                fe = row.get("fe_estimate")
            fe_val = _to_float(fe, np.nan)

            best_so_far = _to_float(row.get("best_fitness_so_far"), np.nan)
            fstar = get_fstar(fid, row_dim, fstar_cache)
            err = abs(best_so_far - fstar)

            key = (algorithm, row_dim, fid, run_number)
            convergence_points.setdefault(key, []).append((fe_val, err))

    for key, values in list(convergence_points.items()):
        # Keep monotonic FE order and monotonic best-error envelope.
        ordered = sorted(values, key=lambda item: item[0])
        cleaned: list[tuple[float, float]] = []
        best_err = float("inf")
        for fe, err in ordered:
            if not np.isfinite(fe) or not np.isfinite(err):
                continue
            best_err = min(best_err, float(err))
            cleaned.append((float(fe), float(best_err)))
        convergence_points[key] = cleaned

    return per_run_rows, grouped_errors, convergence_points


def summarize_grouped_errors(grouped_errors: dict[tuple[str, int, int], list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for (algorithm, dim, fid), items in sorted(grouped_errors.items()):
        values = np.array([float(row["error"]) for row in items], dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "algorithm": algorithm,
                "dimension": dim,
                "function_id": fid,
                "function_name": f"F{fid}",
                "runs_completed": int(values.size),
                "best": float(np.min(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "worst": float(np.max(values)),
                "mean_pm_std": f"{np.mean(values):.6e} +/- {np.std(values, ddof=1) if values.size > 1 else 0.0:.6e}",
            }
        )
    return rows


def build_mean_std_table(summary_rows: list[dict], dim: int, algorithms: list[str]) -> list[dict]:
    by_key = {(int(r["dimension"]), int(r["function_id"]), str(r["algorithm"])): r for r in summary_rows}
    rows: list[dict] = []
    for fid in range(1, 13):
        row = {"function_id": fid, "function_name": f"F{fid}"}
        for algo in algorithms:
            item = by_key.get((dim, fid, algo))
            row[algo] = item["mean_pm_std"] if item is not None else "NA"
        rows.append(row)
    return rows


def rank_shoa_vs_pso(grouped_errors: dict[tuple[str, int, int], list[dict]]) -> tuple[list[dict], list[dict]]:
    ranking_rows: list[dict] = []
    avg_rows: list[dict] = []

    rank_acc: dict[tuple[int, str], list[float]] = {}

    dims = sorted({dim for (_, dim, _) in grouped_errors.keys()})
    for dim in dims:
        for fid in range(1, 13):
            shoa = grouped_errors.get(("SHOA", dim, fid), [])
            pso = grouped_errors.get(("PSO", dim, fid), [])
            if not shoa or not pso:
                continue

            shoa_mean = float(np.mean([r["error"] for r in shoa]))
            pso_mean = float(np.mean([r["error"] for r in pso]))

            if abs(shoa_mean - pso_mean) <= 1e-15:
                shoa_rank, pso_rank = 1.5, 1.5
            elif shoa_mean < pso_mean:
                shoa_rank, pso_rank = 1.0, 2.0
            else:
                shoa_rank, pso_rank = 2.0, 1.0

            ranking_rows.append(
                {
                    "dimension": dim,
                    "function_id": fid,
                    "function_name": f"F{fid}",
                    "SHOA_mean_error": shoa_mean,
                    "PSO_mean_error": pso_mean,
                    "SHOA_rank": shoa_rank,
                    "PSO_rank": pso_rank,
                }
            )

            rank_acc.setdefault((dim, "SHOA"), []).append(shoa_rank)
            rank_acc.setdefault((dim, "PSO"), []).append(pso_rank)
            rank_acc.setdefault((0, "SHOA"), []).append(shoa_rank)
            rank_acc.setdefault((0, "PSO"), []).append(pso_rank)

    for (dim, algo), values in sorted(rank_acc.items()):
        avg_rows.append(
            {
                "dimension": dim if dim != 0 else "global",
                "algorithm": algo,
                "average_rank": float(np.mean(values)),
                "functions_count": len(values),
            }
        )

    return ranking_rows, avg_rows


def paired_errors_for_wilcoxon(shoa_rows: list[dict], pso_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    shoa_by_run = {int(r["run_number"]): float(r["error"]) for r in shoa_rows}
    pso_by_run = {int(r["run_number"]): float(r["error"]) for r in pso_rows}

    common = sorted(set(shoa_by_run).intersection(pso_by_run))
    if common:
        shoa = np.array([shoa_by_run[k] for k in common], dtype=float)
        pso = np.array([pso_by_run[k] for k in common], dtype=float)
        return shoa, pso

    shoa_vals = np.array(sorted(float(r["error"]) for r in shoa_rows), dtype=float)
    pso_vals = np.array(sorted(float(r["error"]) for r in pso_rows), dtype=float)
    n = min(shoa_vals.size, pso_vals.size)
    return shoa_vals[:n], pso_vals[:n]


def wilcoxon_shoa_vs_pso(grouped_errors: dict[tuple[str, int, int], list[dict]], alpha: float = 0.05) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summary_counts: list[dict] = []

    dims = sorted({dim for (_, dim, _) in grouped_errors.keys()})
    for dim in dims:
        wins = 0
        ties = 0
        losses = 0

        for fid in range(1, 13):
            shoa = grouped_errors.get(("SHOA", dim, fid), [])
            pso = grouped_errors.get(("PSO", dim, fid), [])
            if not shoa or not pso:
                continue

            x, y = paired_errors_for_wilcoxon(shoa, pso)
            if x.size == 0 or y.size == 0:
                continue

            p_value = 1.0
            stat_value = 0.0
            try:
                result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
                p_value = float(result.pvalue)
                stat_value = float(result.statistic)
            except ValueError:
                # Typical edge case: all paired differences are zero.
                p_value = 1.0
                stat_value = 0.0

            shoa_mean = float(np.mean(x))
            pso_mean = float(np.mean(y))

            if p_value < alpha:
                if shoa_mean < pso_mean:
                    outcome = "+"
                    wins += 1
                elif shoa_mean > pso_mean:
                    outcome = "-"
                    losses += 1
                else:
                    outcome = "≈"
                    ties += 1
            else:
                outcome = "≈"
                ties += 1

            rows.append(
                {
                    "dimension": dim,
                    "function_id": fid,
                    "function_name": f"F{fid}",
                    "n_pairs": int(min(x.size, y.size)),
                    "SHOA_mean_error": shoa_mean,
                    "PSO_mean_error": pso_mean,
                    "wilcoxon_statistic": stat_value,
                    "p_value": p_value,
                    "alpha": alpha,
                    "outcome_SHOA_vs_PSO": outcome,
                }
            )

        summary_counts.append(
            {
                "dimension": dim,
                "opponent": "PSO",
                "wins_plus": wins,
                "ties_equal": ties,
                "losses_minus": losses,
            }
        )

    if summary_counts:
        total_wins = sum(int(r["wins_plus"]) for r in summary_counts)
        total_ties = sum(int(r["ties_equal"]) for r in summary_counts)
        total_losses = sum(int(r["losses_minus"]) for r in summary_counts)
        summary_counts.append(
            {
                "dimension": "global",
                "opponent": "PSO",
                "wins_plus": total_wins,
                "ties_equal": total_ties,
                "losses_minus": total_losses,
            }
        )

    return rows, summary_counts


def plot_convergence_curves(
    *,
    convergence_points: dict[tuple[str, int, int, int], list[tuple[float, float]]],
    output_dir: Path,
    dims: list[int],
    representative_functions: list[int],
    max_fes_by_dim: dict[int, int],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    algorithms = sorted({key[0] for key in convergence_points.keys()})

    for dim in dims:
        max_fes = max_fes_by_dim[dim]
        grid = np.linspace(1, max_fes, 500)

        for fid in representative_functions:
            plt.figure(figsize=(10, 6))
            plotted_any = False

            for algo in algorithms:
                run_curves: list[np.ndarray] = []
                for (k_algo, k_dim, k_fid, _run_number), series in convergence_points.items():
                    if k_algo != algo or k_dim != dim or k_fid != fid:
                        continue
                    if not series:
                        continue

                    fe = np.array([s[0] for s in series], dtype=float)
                    err = np.array([s[1] for s in series], dtype=float)
                    if fe.size == 0 or err.size == 0:
                        continue

                    curve = np.interp(grid, fe, err, left=err[0], right=err[-1])
                    run_curves.append(curve)

                if not run_curves:
                    continue

                mat = np.vstack(run_curves)
                mean_curve = np.mean(mat, axis=0)
                std_curve = np.std(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean_curve)

                plt.plot(grid, mean_curve, linewidth=2, label=f"{algo} mean")
                lower = np.maximum(mean_curve - std_curve, 1e-300)
                upper = np.maximum(mean_curve + std_curve, 1e-300)
                plt.fill_between(grid, lower, upper, alpha=0.15)
                plotted_any = True

            if not plotted_any:
                plt.close()
                continue

            plt.yscale("log")
            plt.xlabel("Function Evaluations (FEs)")
            plt.ylabel("Error to optimum |f(x)-f*|")
            plt.title(f"Convergence CEC2022 D={dim} F{fid}")
            plt.grid(alpha=0.25)
            plt.legend(loc="best")
            plt.tight_layout()

            out_path = output_dir / f"convergence_D{dim}_F{fid}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()

            manifest_rows.append(
                {
                    "plot_type": "convergence",
                    "dimension": dim,
                    "function_id": fid,
                    "path": str(out_path.resolve()),
                }
            )

    return manifest_rows


def plot_boxplots_shoa_vs_pso(
    *,
    grouped_errors: dict[tuple[str, int, int], list[dict]],
    output_dir: Path,
    dims: list[int],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for dim in dims:
        for fid in range(1, 13):
            shoa = grouped_errors.get(("SHOA", dim, fid), [])
            pso = grouped_errors.get(("PSO", dim, fid), [])
            if not shoa or not pso:
                continue

            shoa_vals = [float(r["error"]) for r in shoa]
            pso_vals = [float(r["error"]) for r in pso]

            plt.figure(figsize=(8, 6))
            plt.boxplot([shoa_vals, pso_vals], tick_labels=["SHOA", "PSO"], showfliers=True)
            plt.yscale("log")
            plt.ylabel("Error to optimum |f(x)-f*|")
            plt.title(f"Boxplot CEC2022 D={dim} F{fid} (SHOA vs PSO)")
            plt.grid(alpha=0.25)
            plt.tight_layout()

            out_path = output_dir / f"boxplot_D{dim}_F{fid}_SHOA_vs_PSO.png"
            plt.savefig(out_path, dpi=150)
            plt.close()

            manifest_rows.append(
                {
                    "plot_type": "boxplot_shoa_vs_pso",
                    "dimension": dim,
                    "function_id": fid,
                    "path": str(out_path.resolve()),
                }
            )

    return manifest_rows


def generate_combined_plots(
    *,
    completed_jobs: list[dict],
    repo_root: Path,
    python_executable: str,
    logs_dir: Path,
    logger: logging.Logger,
) -> list[dict]:
    rows: list[dict] = []

    plot_script = repo_root / "Final-Implementation" / "SHOA-COMBINED" / "cec2022" / "plot_combined_run.py"
    plot_cwd = plot_script.parent

    for job in completed_jobs:
        if str(job.get("algorithm")) != "SHOA-COMBINED":
            continue
        run_dir = Path(job.get("run_dir", ""))
        if not run_dir.exists():
            continue

        dim = int(job.get("dim", 0))
        log_file = logs_dir / f"plot_SHOA-COMBINED_D{dim}.log"

        cmd = [
            python_executable,
            str(plot_script.name),
            "--run-dir",
            str(run_dir.resolve()),
            "--log-y",
            "--target-fitness",
            "0",
        ]

        logger.info("Generating SHOA-COMBINED plots for D%d from %s", dim, str(run_dir.resolve()))
        logger.info("Combined plot command: %s", " ".join(cmd))

        with log_file.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND:\n")
            handle.write(" ".join(cmd) + "\n\n")
            handle.flush()

            completed = subprocess.run(
                cmd,
                cwd=str(plot_cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        rows.append(
            {
                "algorithm": "SHOA-COMBINED",
                "dimension": dim,
                "run_dir": str(run_dir.resolve()),
                "return_code": int(completed.returncode),
                "log_file": str(log_file.resolve()),
                "plots_dir": str((run_dir / "plots").resolve()),
            }
        )
        logger.info(
            "Combined plots finished for D%d with return_code=%d",
            dim,
            int(completed.returncode),
        )

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-safe CEC2022 orchestrator for PSO, SHOA, SHOA-COMBINED")

    parser.add_argument("--dims", type=str, default="10,20", help="Comma-separated dimensions, e.g. 10,20")
    parser.add_argument("--functions", type=str, default="1-12", help="Function selector: all | 1-12 | 1,2,3")
    parser.add_argument("--runs", type=int, default=30, help="Independent runs per function")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    parser.add_argument("--max-fes-10", type=int, default=DEFAULT_MAX_FES[10], help="MaxFEs for D=10")
    parser.add_argument("--max-fes-20", type=int, default=DEFAULT_MAX_FES[20], help="MaxFEs for D=20")

    parser.add_argument("--pso-particles", type=int, default=40)
    parser.add_argument("--pso-w", type=float, default=0.7)
    parser.add_argument("--pso-c1", type=float, default=1.7)
    parser.add_argument("--pso-c2", type=float, default=1.7)

    parser.add_argument("--shoa-pop", type=int, default=30)

    parser.add_argument("--combined-pop", type=int, default=30)
    parser.add_argument(
        "--combined-max-fes",
        type=int,
        default=500_000,
        help="MaxFEs used only by SHOA-COMBINED (does not affect PSO/SHOA).",
    )
    parser.add_argument("--combined-rescue-enabled", action="store_true", default=True)
    parser.add_argument("--combined-restart-percent", type=float, default=10.0)
    parser.add_argument("--combined-restart-cooldown-ratio", type=float, default=0.04)
    parser.add_argument("--combined-restart-dominance-threshold", type=float, default=0.90)
    parser.add_argument("--combined-lime-min-samples", type=int, default=1000)
    parser.add_argument("--combined-lime-selection-mode", type=str, default="medoid", choices=["medoid", "selected_agents"])
    parser.add_argument("--combined-progress-every", type=int, default=10)

    parser.add_argument("--retry", type=int, default=2, help="Retries per job when failures occur")
    parser.add_argument("--continue-on-failure", action="store_true", default=True)
    parser.add_argument("--skip-execution", action="store_true", help="Skip runner execution and only build reports from completed jobs in state.json")

    parser.add_argument("--output-root", type=str, default="Final-Implementation/experiments/cec2022_failsafe")
    parser.add_argument("--python-executable", type=str, default="", help="Python executable for sub-runners. Default: current interpreter")
    parser.add_argument("--log-level", type=str, default="INFO")

    parser.add_argument("--representative-functions", type=str, default="1,5,8,12")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logs_dir = output_root / "logs"
    reports_dir = output_root / "reports"
    plots_dir = reports_dir / "plots"
    tables_dir = reports_dir / "tables"

    logger = setup_orchestrator_logger(logs_dir=logs_dir, level_name=args.log_level)
    logger.info("Starting CEC2022 fail-safe orchestrator")
    logger.info("Output root: %s", str(output_root))
    logger.info(
        "Config summary | dims=%s functions=%s runs=%d seed=%d skip_execution=%s retries=%d",
        args.dims,
        args.functions,
        int(args.runs),
        int(args.seed),
        bool(args.skip_execution),
        int(args.retry),
    )
    logger.info(
        "SHOA-COMBINED params | pop=%d max_fes=%d restart_enabled=%s restart_percent=%.3f cooldown_ratio=%.4f dominance_threshold=%.4f lime_min_samples=%d lime_selection_mode=%s progress_every=%d",
        int(args.combined_pop),
        int(args.combined_max_fes),
        bool(args.combined_rescue_enabled),
        float(args.combined_restart_percent),
        float(args.combined_restart_cooldown_ratio),
        float(args.combined_restart_dominance_threshold),
        int(args.combined_lime_min_samples),
        str(args.combined_lime_selection_mode),
        int(args.combined_progress_every),
    )

    state_path = output_root / "state.json"
    state = load_json(state_path)
    state.setdefault("metadata", {})
    state["metadata"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    jobs = build_jobs(args=args, repo_root=repo_root, output_root=output_root)

    jobs_state = state.setdefault("jobs", {})
    for job in jobs:
        spec = jobs_state.setdefault(job.job_id, {})
        spec["algorithm"] = job.algorithm
        spec["dim"] = job.dim
        spec["max_fes"] = job.max_fes
        spec["max_iter"] = job.max_iter
        spec["runs"] = job.runs
        spec["functions"] = job.functions
        spec["seed"] = job.seed
        spec["cwd"] = str(job.cwd.resolve())
        spec["output_dir"] = str(job.output_dir.resolve())

    save_json(state_path, state)
    logger.info("State initialized at %s", str(state_path.resolve()))

    if not args.skip_execution:
        logger.info("Executing %d jobs", len(jobs))
        for job in jobs:
            state = load_json(state_path)
            _ = run_job_fail_safe(
                job,
                logs_dir=logs_dir,
                retries=args.retry,
                state=state,
                continue_on_failure=bool(args.continue_on_failure),
                logger=logger,
            )
            save_json(state_path, state)
            logger.info("State saved after %s", job.job_id)
    else:
        logger.info("Execution skipped (--skip-execution); generating reports from existing state")

    state = load_json(state_path)
    completed_jobs: list[dict] = []
    failed_jobs: list[dict] = []

    for job in jobs:
        record = state.get("jobs", {}).get(job.job_id, {})
        row = {
            "job_id": job.job_id,
            "algorithm": job.algorithm,
            "dim": job.dim,
            "status": record.get("status", "unknown"),
            "attempts": record.get("attempts", 0),
            "return_code": record.get("return_code", ""),
            "run_dir": record.get("run_dir", ""),
            "log_file": record.get("log_file", ""),
        }
        if row["status"] == "completed" and row["run_dir"]:
            completed_jobs.append(row)
        else:
            failed_jobs.append(row)

    write_csv(tables_dir / "job_status.csv", completed_jobs + failed_jobs)
    logger.info("Wrote job status table to %s", str((tables_dir / "job_status.csv").resolve()))

    if not completed_jobs:
        note = {
            "message": "No completed jobs found. Check logs and state.json",
            "failed_jobs": failed_jobs,
        }
        save_json(reports_dir / "no_results.json", note)
        logger.warning("No completed jobs found; wrote no_results.json")
        print("No completed jobs found. See reports/no_results.json")
        return

    python_exec = args.python_executable or sys.executable
    logger.info("Generating SHOA-COMBINED contribution plots")
    combined_plot_rows = generate_combined_plots(
        completed_jobs=completed_jobs,
        repo_root=repo_root,
        python_executable=python_exec,
        logs_dir=logs_dir,
        logger=logger,
    )
    write_csv(tables_dir / "combined_plot_jobs.csv", combined_plot_rows)
    logger.info("Wrote SHOA-COMBINED plot jobs table")

    fstar_cache: dict[tuple[int, int], float] = {}
    per_run_rows, grouped_errors, convergence_points = collect_experiment_data(
        completed_jobs,
        fstar_cache=fstar_cache,
    )

    write_csv(tables_dir / "per_run_errors.csv", per_run_rows)
    logger.info("Wrote per-run errors (%d rows)", len(per_run_rows))

    summary_rows = summarize_grouped_errors(grouped_errors)
    write_csv(tables_dir / "summary_stats.csv", summary_rows)
    logger.info("Wrote summary stats (%d rows)", len(summary_rows))

    dims = parse_int_csv(args.dims)
    algorithms_present = sorted({row["algorithm"] for row in summary_rows})
    for dim in dims:
        table_rows = build_mean_std_table(summary_rows=summary_rows, dim=dim, algorithms=algorithms_present)
        write_csv(tables_dir / f"mean_std_table_D{dim}.csv", table_rows)
        logger.info("Wrote mean/std table for D=%d", dim)

    ranking_rows, ranking_avg_rows = rank_shoa_vs_pso(grouped_errors)
    write_csv(tables_dir / "ranking_shoa_vs_pso_by_function.csv", ranking_rows)
    write_csv(tables_dir / "ranking_shoa_vs_pso_average.csv", ranking_avg_rows)
    logger.info("Wrote SHOA vs PSO ranking tables")

    wilcoxon_rows, wilcoxon_summary_rows = wilcoxon_shoa_vs_pso(grouped_errors)
    write_csv(tables_dir / "wilcoxon_shoa_vs_pso.csv", wilcoxon_rows)
    write_csv(tables_dir / "wins_ties_losses_shoa_vs_pso.csv", wilcoxon_summary_rows)
    logger.info("Wrote Wilcoxon and W/T/L tables")

    rep_functions = parse_int_csv(args.representative_functions)
    max_fes_by_dim = {
        10: max(int(args.max_fes_10), int(args.combined_max_fes)),
        20: max(int(args.max_fes_20), int(args.combined_max_fes)),
    }

    convergence_manifest = plot_convergence_curves(
        convergence_points=convergence_points,
        output_dir=plots_dir,
        dims=dims,
        representative_functions=rep_functions,
        max_fes_by_dim=max_fes_by_dim,
    )
    write_csv(tables_dir / "convergence_plots_manifest.csv", convergence_manifest)
    logger.info("Wrote convergence plots (%d entries)", len(convergence_manifest))

    boxplot_manifest = plot_boxplots_shoa_vs_pso(
        grouped_errors=grouped_errors,
        output_dir=plots_dir,
        dims=dims,
    )
    write_csv(tables_dir / "boxplots_manifest.csv", boxplot_manifest)
    logger.info("Wrote SHOA vs PSO boxplots (%d entries)", len(boxplot_manifest))

    notes_path = reports_dir / "statistical_notes.txt"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes = [
        "Statistical protocol notes:",
        "- Error metric used: abs(f(x_best)-f*) per run.",
        "- f* obtained from opfunu CEC2022 class metadata (f_global).",
        "- Wilcoxon signed-rank executed only for SHOA vs PSO as requested.",
        "- Friedman/Holm not executed because only two algorithms are used in statistical comparison;",
        "  scipy.stats.friedmanchisquare requires at least three samples.",
        "- Contribution plots are generated only for SHOA-COMBINED runs.",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    logger.info("Wrote statistical notes")

    report_manifest = {
        "output_root": str(output_root),
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "tables_dir": str(tables_dir.resolve()),
        "plots_dir": str(plots_dir.resolve()),
        "combined_plot_jobs": combined_plot_rows,
    }
    save_json(reports_dir / "report_manifest.json", report_manifest)
    logger.info("Wrote report manifest")
    logger.info("Orchestration finished | completed_jobs=%d failed_or_partial=%d", len(completed_jobs), len(failed_jobs))

    print("CEC2022 fail-safe orchestration complete")
    print(f"Completed jobs: {len(completed_jobs)}")
    print(f"Failed/partial jobs: {len(failed_jobs)}")
    print(f"Reports: {reports_dir}")


if __name__ == "__main__":
    main()
