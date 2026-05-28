"""Fail-safe orchestrator for TMLAP experiments with PSO, SHOA and SHOA-COMBINED.

Protocol:
- Instances 1, 2, 3: MaxFES = 5 000, 30 independent runs per algorithm
- Instance 4:        MaxFES = 5 000 AND MaxFES = 50 000, 30 runs per algorithm
- Metric: best_fitness (minimisation) + feasibility_rate
- Statistical comparison: SHOA vs PSO only (Wilcoxon signed-rank per instance × budget)
- Contribution plots: SHOA-COMBINED only
- No f* subtraction (TMLAP has no known analytical optimum)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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
from scipy.stats import wilcoxon


DEFAULT_MAX_FES = 5_000
DEFAULT_MAX_FES_EXTENDED = 50_000
DEFAULT_EXTENDED_INSTANCE = "4.instancia.txt"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_orchestrator_logger(logs_dir: Path, level_name: str) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger = logging.getLogger("tmlap_failsafe")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(logs_dir / "orchestrator.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# JobSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobSpec:
    algorithm: str
    instance_name: str
    max_fes: int
    runs: int
    seed: int
    max_iter: int
    output_dir: Path
    cwd: Path
    command: list[str]

    @property
    def job_id(self) -> str:
        stem = Path(self.instance_name).stem.replace(".", "_").replace(" ", "_")
        return f"{self.algorithm}_{stem}_fes{self.max_fes}"


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

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
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_int(value: str | int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raw = str(value).strip()
    return int(float(raw)) if raw else default


def _to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    raw = str(value).strip()
    return float(raw) if raw else default


# ---------------------------------------------------------------------------
# Run-directory discovery
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fail-safe job runner
# ---------------------------------------------------------------------------

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

    before = (
        set(str(p.resolve()) for p in job.output_dir.glob("run-*") if p.is_dir())
        if job.output_dir.exists()
        else set()
    )

    for attempt in range(attempts_done + 1, max_attempts + 1):
        job_state["status"] = "running"
        job_state["attempts"] = attempt
        job_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        log_file = logs_dir / f"{job.job_id}.attempt{attempt}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting %s (attempt %d/%d) | instance=%s runs=%d max_fes=%d max_iter=%d",
            job.job_id,
            attempt,
            max_attempts,
            job.instance_name,
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


# ---------------------------------------------------------------------------
# Iteration budget helpers
# ---------------------------------------------------------------------------

def compute_pso_max_iter(max_fes: int, particles: int) -> int:
    """nfev ≈ particles (init) + particles * iter"""
    if particles <= 0:
        raise ValueError("particles must be > 0")
    return max(1, (max_fes - particles) // particles)


def compute_shoa_max_iter(max_fes: int, pop: int) -> int:
    """nfev ≈ pop (init) + iter * (pop + pop//2)"""
    if pop <= 0:
        raise ValueError("pop must be > 0")
    per_iter = pop + pop // 2
    return max(1, (max_fes - pop) // max(1, per_iter))


def compute_combined_max_iter(max_fes: int, pop: int) -> int:
    """Adds buffer so --max-fes is the real stop criterion."""
    base = compute_shoa_max_iter(max_fes=max_fes, pop=pop)
    return int(base + 50)


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

def discover_instance_files(instance_dir: Path, exclude: list[str] | None = None) -> list[str]:
    """Return sorted list of .txt filenames found in instance_dir, optionally excluding some."""
    exclude_set = {e.lower() for e in (exclude or [])}
    found = sorted(p.name for p in instance_dir.glob("*.txt") if p.name.lower() not in exclude_set)
    if not found:
        raise ValueError(f"No .txt instance files found in {instance_dir}")
    return found


# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

def build_jobs(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    instance_filter: list[str] | None = None,
) -> list[JobSpec]:
    py_exec = args.python_executable or sys.executable

    # Reference directory: SHOA/tmlap (has exactly instances 1-4, no 5)
    shoa_tmlap = repo_root / "Final-Implementation" / "SHOA" / "tmlap"
    pso_tmlap  = repo_root / "Final-Implementation" / "PSO"  / "tmlap"
    comb_tmlap = repo_root / "Final-Implementation" / "SHOA-COMBINED" / "tmlap"

    all_instances = discover_instance_files(shoa_tmlap)
    if instance_filter is not None:
        all_instances = [i for i in all_instances if i in instance_filter]
        if not all_instances:
            raise ValueError(f"None of the requested instances were found. Requested: {instance_filter}")

    extended_instance = args.extended_instance.strip()
    max_fes_std      = int(args.max_fes)
    max_fes_ext      = int(args.max_fes_extended)

    # Build (instance, max_fes) pairs
    instance_budget_pairs: list[tuple[str, int]] = []
    for inst in all_instances:
        instance_budget_pairs.append((inst, max_fes_std))
        # Only add the extended run if the budget differs (avoid duplicate job_ids)
        if inst == extended_instance and max_fes_ext != max_fes_std:
            instance_budget_pairs.append((inst, max_fes_ext))

    jobs: list[JobSpec] = []

    for instance_name, max_fes in instance_budget_pairs:
        stem = Path(instance_name).stem.replace(".", "_").replace(" ", "_")
        tag = f"{stem}_fes{max_fes}"

        pso_iter      = compute_pso_max_iter(max_fes=max_fes, particles=args.pso_particles)
        shoa_iter     = compute_shoa_max_iter(max_fes=max_fes, pop=args.shoa_pop)
        combined_iter = compute_combined_max_iter(max_fes=max_fes, pop=args.combined_pop)

        pso_out  = output_root / "raw" / "PSO"          / tag
        shoa_out = output_root / "raw" / "SHOA"         / tag
        comb_out = output_root / "raw" / "SHOA-COMBINED" / tag

        # ---- PSO command ----
        pso_cmd = [
            py_exec, "run_pso_tmlap.py",
            "--mode", "light",
            "--instances", instance_name,
            "--particles", str(args.pso_particles),
            "--max-iter", str(pso_iter),
            "--runs", str(args.runs),
            "--seed", str(args.seed),
            "--w",  str(args.pso_w),
            "--c1", str(args.pso_c1),
            "--c2", str(args.pso_c2),
            "--output-dir", str(pso_out.resolve()),
            "--log-level", args.log_level,
        ]

        # ---- SHOA command ----
        shoa_cmd = [
            py_exec, "run_shoa_tmlap.py",
            "--instances", instance_name,
            "--pop", str(args.shoa_pop),
            "--max-iter", str(shoa_iter),
            "--runs", str(args.runs),
            "--seed", str(args.seed),
            "--output-dir", str(shoa_out.resolve()),
            "--log-level", args.log_level,
        ]

        # ---- SHOA-COMBINED command ----
        comb_cmd = [
            py_exec, "run_tmlap_combined.py",
            "--instances", instance_name,
            "--pop", str(args.combined_pop),
            "--max-iter", str(combined_iter),
            "--runs", str(args.runs),
            "--seed", str(args.seed),
            "--max-fes", str(max_fes),
            "--restart-enabled", "1" if args.combined_restart_enabled else "0",
            "--restart-percent", str(args.combined_restart_percent),
            "--restart-cooldown-fes-ratio", str(args.combined_restart_cooldown_ratio),
            "--restart-dominance-threshold", str(args.combined_restart_dominance_threshold),
            "--lime-min-samples", str(args.combined_lime_min_samples),
            "--stagnation-lime-selection-mode", args.combined_lime_selection_mode,
            "--progress-every", str(args.combined_progress_every),
            "--output-dir", str(comb_out.resolve()),
            "--log-level", args.log_level,
        ]

        jobs.append(JobSpec(
            algorithm="PSO",
            instance_name=instance_name,
            max_fes=max_fes,
            runs=args.runs,
            seed=args.seed,
            max_iter=pso_iter,
            output_dir=pso_out,
            cwd=pso_tmlap,
            command=pso_cmd,
        ))
        jobs.append(JobSpec(
            algorithm="SHOA",
            instance_name=instance_name,
            max_fes=max_fes,
            runs=args.runs,
            seed=args.seed,
            max_iter=shoa_iter,
            output_dir=shoa_out,
            cwd=shoa_tmlap,
            command=shoa_cmd,
        ))
        jobs.append(JobSpec(
            algorithm="SHOA-COMBINED",
            instance_name=instance_name,
            max_fes=max_fes,
            runs=args.runs,
            seed=args.seed,
            max_iter=combined_iter,
            output_dir=comb_out,
            cwd=comb_tmlap,
            command=comb_cmd,
        ))

    return jobs


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_experiment_data(
    completed_jobs: list[dict],
) -> tuple[
    list[dict],
    dict[tuple[str, str, int], list[dict]],
    dict[tuple[str, str, int, int], list[tuple[float, float]]],
]:
    """Read runs_raw.csv and full_output.csv from every completed job.

    Returns:
        per_run_rows: flat list with one row per run
        grouped_results: {(algorithm, instance_name, max_fes): [run_rows]}
        convergence_points: {(algorithm, instance_name, max_fes, run_number): [(fe, fitness)]}
    """
    per_run_rows: list[dict] = []
    grouped_results: dict[tuple[str, str, int], list[dict]] = {}
    convergence_points: dict[tuple[str, str, int, int], list[tuple[float, float]]] = {}

    for job in completed_jobs:
        algorithm   = str(job["algorithm"])
        instance    = str(job["instance_name"])
        max_fes     = int(job["max_fes"])
        run_dir     = Path(job["run_dir"])

        runs_raw = read_csv_rows(run_dir / "runs_raw.csv")
        for row in runs_raw:
            inst_name   = str(row.get("instance_name") or row.get("function_name") or instance)
            run_number  = _to_int(row.get("run_number"), 0)
            best_fitness = _to_float(row.get("best_fitness"), np.nan)
            feasible     = _to_int(row.get("feasible_best_solution"), 0)
            base_cost    = _to_float(row.get("base_cost"), np.nan)
            overflow     = _to_int(row.get("overflow_total"), 0)
            dist_viol    = _to_float(row.get("distance_violation"), 0.0)
            fes_used     = _to_int(row.get("fes_used") or row.get("final_fe"), 0)

            out_row = {
                "algorithm":     algorithm,
                "instance_name": inst_name,
                "max_fes_budget": max_fes,
                "run_number":    run_number,
                "best_fitness":  best_fitness,
                "base_cost":     base_cost,
                "feasible":      feasible,
                "overflow_total": overflow,
                "distance_violation": dist_viol,
                "fes_used":      fes_used,
                "run_dir":       str(run_dir.resolve()),
            }
            per_run_rows.append(out_row)
            grouped_results.setdefault((algorithm, inst_name, max_fes), []).append(out_row)

        full_output = read_csv_rows(run_dir / "full_output.csv")
        for row in full_output:
            inst_name  = str(row.get("instance_name") or row.get("function_name") or instance)
            run_number = _to_int(row.get("run_number"), 0)

            fe_raw = row.get("fe_estimate") or row.get("fe")
            fe_val = _to_float(fe_raw, np.nan)
            fitness_val = _to_float(row.get("best_fitness_so_far"), np.nan)

            key = (algorithm, inst_name, max_fes, run_number)
            convergence_points.setdefault(key, []).append((fe_val, fitness_val))

    # Sort and enforce monotone-decreasing (best-so-far) envelope per curve
    for key, values in list(convergence_points.items()):
        ordered = sorted(values, key=lambda item: item[0])
        cleaned: list[tuple[float, float]] = []
        best_fit = float("inf")
        for fe, fit in ordered:
            if not np.isfinite(fe) or not np.isfinite(fit):
                continue
            best_fit = min(best_fit, float(fit))
            cleaned.append((float(fe), float(best_fit)))
        convergence_points[key] = cleaned

    return per_run_rows, grouped_results, convergence_points


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_grouped_results(
    grouped_results: dict[tuple[str, str, int], list[dict]],
) -> list[dict]:
    rows: list[dict] = []
    for (algorithm, instance_name, max_fes), items in sorted(grouped_results.items()):
        fitness_vals = np.array([float(r["best_fitness"]) for r in items], dtype=float)
        feasible_vals = np.array([int(r["feasible"]) for r in items], dtype=float)

        if fitness_vals.size == 0:
            continue

        rows.append({
            "algorithm":        algorithm,
            "instance_name":    instance_name,
            "max_fes_budget":   max_fes,
            "runs_completed":   int(fitness_vals.size),
            "best":             float(np.min(fitness_vals)),
            "mean":             float(np.mean(fitness_vals)),
            "std":              float(np.std(fitness_vals, ddof=1)) if fitness_vals.size > 1 else 0.0,
            "median":           float(np.median(fitness_vals)),
            "worst":            float(np.max(fitness_vals)),
            "feasibility_rate": float(np.mean(feasible_vals)),
            "mean_pm_std":      (
                f"{np.mean(fitness_vals):.6e} +/- "
                f"{np.std(fitness_vals, ddof=1) if fitness_vals.size > 1 else 0.0:.6e}"
            ),
        })
    return rows


def build_comparison_table(
    summary_rows: list[dict],
    instance_names: list[str],
    algorithms: list[str],
    max_fes_budgets: list[int],
) -> list[dict]:
    """Mean±std table with one row per (instance, budget) and one column per algorithm."""
    by_key = {
        (str(r["instance_name"]), int(r["max_fes_budget"]), str(r["algorithm"])): r
        for r in summary_rows
    }
    rows: list[dict] = []
    for inst in instance_names:
        for budget in max_fes_budgets:
            row: dict = {"instance_name": inst, "max_fes_budget": budget}
            for algo in algorithms:
                item = by_key.get((inst, budget, algo))
                row[algo] = item["mean_pm_std"] if item is not None else "NA"
                row[f"{algo}_feasibility_rate"] = (
                    f"{item['feasibility_rate']:.3f}" if item is not None else "NA"
                )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_shoa_vs_pso(
    grouped_results: dict[tuple[str, str, int], list[dict]],
) -> tuple[list[dict], list[dict]]:
    ranking_rows: list[dict] = []
    avg_rows: list[dict] = []
    rank_acc: dict[tuple[str, int, str], list[float]] = {}

    instance_budgets = sorted({(inst, fes) for (_, inst, fes) in grouped_results.keys()})

    for inst, max_fes in instance_budgets:
        shoa = grouped_results.get(("SHOA", inst, max_fes), [])
        pso  = grouped_results.get(("PSO",  inst, max_fes), [])
        if not shoa or not pso:
            continue

        shoa_mean = float(np.mean([r["best_fitness"] for r in shoa]))
        pso_mean  = float(np.mean([r["best_fitness"] for r in pso]))

        if abs(shoa_mean - pso_mean) <= 1e-15:
            shoa_rank, pso_rank = 1.5, 1.5
        elif shoa_mean < pso_mean:
            shoa_rank, pso_rank = 1.0, 2.0
        else:
            shoa_rank, pso_rank = 2.0, 1.0

        ranking_rows.append({
            "instance_name":    inst,
            "max_fes_budget":   max_fes,
            "SHOA_mean":        shoa_mean,
            "PSO_mean":         pso_mean,
            "SHOA_rank":        shoa_rank,
            "PSO_rank":         pso_rank,
        })

        rank_acc.setdefault(("all", max_fes, "SHOA"), []).append(shoa_rank)
        rank_acc.setdefault(("all", max_fes, "PSO"),  []).append(pso_rank)

    for (group, max_fes, algo), values in sorted(rank_acc.items()):
        avg_rows.append({
            "group":          group,
            "max_fes_budget": max_fes,
            "algorithm":      algo,
            "average_rank":   float(np.mean(values)),
            "instances_count": len(values),
        })

    return ranking_rows, avg_rows


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------

def _paired_fitness(shoa_rows: list[dict], pso_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    shoa_by_run = {int(r["run_number"]): float(r["best_fitness"]) for r in shoa_rows}
    pso_by_run  = {int(r["run_number"]): float(r["best_fitness"]) for r in pso_rows}

    common = sorted(set(shoa_by_run).intersection(pso_by_run))
    if common:
        return (
            np.array([shoa_by_run[k] for k in common], dtype=float),
            np.array([pso_by_run[k]  for k in common], dtype=float),
        )

    sv = np.array(sorted(float(r["best_fitness"]) for r in shoa_rows), dtype=float)
    pv = np.array(sorted(float(r["best_fitness"]) for r in pso_rows),  dtype=float)
    n  = min(sv.size, pv.size)
    return sv[:n], pv[:n]


def wilcoxon_shoa_vs_pso(
    grouped_results: dict[tuple[str, str, int], list[dict]],
    alpha: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summary_counts: list[dict] = []

    instance_budgets = sorted({(inst, fes) for (_, inst, fes) in grouped_results.keys()})
    budgets = sorted({fes for (_, fes) in instance_budgets})

    for budget in budgets:
        wins = ties = losses = 0
        for inst, fes in instance_budgets:
            if fes != budget:
                continue
            shoa = grouped_results.get(("SHOA", inst, budget), [])
            pso  = grouped_results.get(("PSO",  inst, budget), [])
            if not shoa or not pso:
                continue

            x, y = _paired_fitness(shoa, pso)
            if x.size == 0:
                continue

            p_value = stat_value = 1.0
            try:
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", RuntimeWarning)
                    res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox",
                                   correction=False, method="auto")
                p_value  = float(res.pvalue)
                stat_value = float(res.statistic)
            except ValueError:
                p_value = 1.0
                stat_value = 0.0

            shoa_mean = float(np.mean(x))
            pso_mean  = float(np.mean(y))

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

            rows.append({
                "instance_name":       inst,
                "max_fes_budget":      budget,
                "n_pairs":             int(min(x.size, y.size)),
                "SHOA_mean_fitness":   shoa_mean,
                "PSO_mean_fitness":    pso_mean,
                "wilcoxon_statistic":  stat_value,
                "p_value":             p_value,
                "alpha":               alpha,
                "outcome_SHOA_vs_PSO": outcome,
            })

        summary_counts.append({
            "max_fes_budget": budget,
            "opponent":       "PSO",
            "wins_plus":      wins,
            "ties_equal":     ties,
            "losses_minus":   losses,
        })

    return rows, summary_counts


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    *,
    convergence_points: dict[tuple[str, str, int, int], list[tuple[float, float]]],
    output_dir: Path,
    max_fes_budgets: list[int],
    instance_names: list[str],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    algorithms = sorted({key[0] for key in convergence_points.keys()})

    for inst in instance_names:
        for budget in max_fes_budgets:
            grid = np.linspace(1, budget, 500)

            plt.figure(figsize=(10, 6))
            plotted_any = False

            for algo in algorithms:
                run_curves: list[np.ndarray] = []
                for (k_algo, k_inst, k_fes, _), series in convergence_points.items():
                    if k_algo != algo or k_inst != inst or k_fes != budget:
                        continue
                    if not series:
                        continue
                    fe  = np.array([s[0] for s in series], dtype=float)
                    fit = np.array([s[1] for s in series], dtype=float)
                    if fe.size == 0:
                        continue
                    curve = np.interp(grid, fe, fit, left=fit[0], right=fit[-1])
                    run_curves.append(curve)

                if not run_curves:
                    continue

                mat = np.vstack(run_curves)
                mean_c = np.mean(mat, axis=0)
                std_c  = np.std(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean_c)

                plt.plot(grid, mean_c, linewidth=2, label=f"{algo} mean")
                plt.fill_between(grid, mean_c - std_c, mean_c + std_c, alpha=0.15)
                plotted_any = True

            if not plotted_any:
                plt.close()
                continue

            safe_inst = Path(inst).stem
            plt.xlabel("Function Evaluations (FEs)")
            plt.ylabel("Best Fitness (cost)")
            plt.title(f"Convergence TMLAP – {safe_inst} – MaxFES={budget}")
            plt.grid(alpha=0.25)
            plt.legend(loc="best")
            plt.tight_layout()

            fname = f"convergence_{safe_inst}_fes{budget}.png"
            out_path = output_dir / fname
            plt.savefig(out_path, dpi=150)
            plt.close()

            manifest_rows.append({
                "plot_type":      "convergence",
                "instance_name":  inst,
                "max_fes_budget": budget,
                "path":           str(out_path.resolve()),
            })

    return manifest_rows


def plot_boxplots_shoa_vs_pso(
    *,
    grouped_results: dict[tuple[str, str, int], list[dict]],
    output_dir: Path,
    max_fes_budgets: list[int],
    instance_names: list[str],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for inst in instance_names:
        for budget in max_fes_budgets:
            shoa = grouped_results.get(("SHOA", inst, budget), [])
            pso  = grouped_results.get(("PSO",  inst, budget), [])
            if not shoa or not pso:
                continue

            shoa_vals = [float(r["best_fitness"]) for r in shoa]
            pso_vals  = [float(r["best_fitness"]) for r in pso]

            safe_inst = Path(inst).stem
            plt.figure(figsize=(8, 6))
            plt.boxplot([shoa_vals, pso_vals], tick_labels=["SHOA", "PSO"], showfliers=True)
            plt.ylabel("Best Fitness (cost)")
            plt.title(f"Boxplot TMLAP – {safe_inst} – MaxFES={budget} (SHOA vs PSO)")
            plt.grid(alpha=0.25)
            plt.tight_layout()

            fname = f"boxplot_{safe_inst}_fes{budget}_SHOA_vs_PSO.png"
            out_path = output_dir / fname
            plt.savefig(out_path, dpi=150)
            plt.close()

            manifest_rows.append({
                "plot_type":      "boxplot_shoa_vs_pso",
                "instance_name":  inst,
                "max_fes_budget": budget,
                "path":           str(out_path.resolve()),
            })

    return manifest_rows


# ---------------------------------------------------------------------------
# SHOA-COMBINED diagnostic plots
# ---------------------------------------------------------------------------

def generate_combined_plots(
    *,
    completed_jobs: list[dict],
    repo_root: Path,
    python_executable: str,
    logs_dir: Path,
    logger: logging.Logger,
) -> list[dict]:
    rows: list[dict] = []

    plot_script = repo_root / "Final-Implementation" / "SHOA-COMBINED" / "tmlap" / "plot_combined_run.py"
    plot_cwd    = plot_script.parent

    for job in completed_jobs:
        if str(job.get("algorithm")) != "SHOA-COMBINED":
            continue
        run_dir = Path(job.get("run_dir", ""))
        if not run_dir.exists():
            continue

        inst    = str(job.get("instance_name", "unknown"))
        max_fes = int(job.get("max_fes", 0))
        safe    = Path(inst).stem
        log_file = logs_dir / f"plot_SHOA-COMBINED_{safe}_fes{max_fes}.log"

        cmd = [
            python_executable,
            str(plot_script.name),
            "--run-dir", str(run_dir.resolve()),
        ]

        logger.info(
            "Generating SHOA-COMBINED plots for %s MaxFES=%d from %s",
            inst, max_fes, str(run_dir.resolve()),
        )
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

        rows.append({
            "algorithm":      "SHOA-COMBINED",
            "instance_name":  inst,
            "max_fes_budget": max_fes,
            "run_dir":        str(run_dir.resolve()),
            "return_code":    int(completed.returncode),
            "log_file":       str(log_file.resolve()),
            "plots_dir":      str((run_dir / "plots").resolve()),
        })
        logger.info(
            "Combined plots finished for %s MaxFES=%d with return_code=%d",
            inst, max_fes, int(completed.returncode),
        )

    return rows


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-safe TMLAP orchestrator for PSO, SHOA, SHOA-COMBINED"
    )

    parser.add_argument(
        "--instances", type=str, default="all",
        help="Instances to run: 'all' auto-discovers from SHOA/tmlap (excludes 5.instancia.txt)"
             " or comma-separated names, e.g. '1.instancia_simple.txt,2.instancia_mediana.txt'",
    )
    parser.add_argument("--runs", type=int, default=30, help="Independent runs per job")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    parser.add_argument("--max-fes", type=int, default=DEFAULT_MAX_FES,
                        help="MaxFEs for instances 1-3 (and first run of extended instance)")
    parser.add_argument("--max-fes-extended", type=int, default=DEFAULT_MAX_FES_EXTENDED,
                        help="MaxFEs for the extended run of the large instance")
    parser.add_argument("--extended-instance", type=str, default=DEFAULT_EXTENDED_INSTANCE,
                        help="Instance name that receives a second, larger budget run")

    parser.add_argument("--pso-particles", type=int, default=40)
    parser.add_argument("--pso-w",  type=float, default=0.7)
    parser.add_argument("--pso-c1", type=float, default=1.7)
    parser.add_argument("--pso-c2", type=float, default=1.7)

    parser.add_argument("--shoa-pop", type=int, default=30)

    parser.add_argument("--combined-pop", type=int, default=30)
    parser.add_argument("--combined-restart-enabled", action="store_true", default=True)
    parser.add_argument("--combined-restart-percent", type=float, default=7.0)
    parser.add_argument("--combined-restart-cooldown-ratio", type=float, default=0.04)
    parser.add_argument("--combined-restart-dominance-threshold", type=float, default=0.90)
    parser.add_argument("--combined-lime-min-samples", type=int, default=400)
    parser.add_argument(
        "--combined-lime-selection-mode", type=str, default="medoid",
        choices=["medoid", "selected_agents"],
    )
    parser.add_argument("--combined-progress-every", type=int, default=10)

    parser.add_argument("--retry", type=int, default=2, help="Retries per job on failure")
    parser.add_argument("--continue-on-failure", action="store_true", default=True)
    parser.add_argument(
        "--skip-execution", action="store_true",
        help="Skip runners; generate reports only from completed jobs in state.json",
    )

    parser.add_argument(
        "--output-root", type=str,
        default="Final-Implementation/experiments/tmlap_failsafe",
    )
    parser.add_argument("--python-executable", type=str, default="",
                        help="Python interpreter for sub-runners. Default: current interpreter")
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()

    repo_root    = Path(__file__).resolve().parents[1]
    output_root  = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logs_dir    = output_root / "logs"
    reports_dir = output_root / "reports"
    plots_dir   = reports_dir / "plots"
    tables_dir  = reports_dir / "tables"

    logger = setup_orchestrator_logger(logs_dir=logs_dir, level_name=args.log_level)
    logger.info("Starting TMLAP fail-safe orchestrator")
    logger.info("Output root: %s", str(output_root))
    logger.info(
        "Config | runs=%d seed=%d max_fes=%d max_fes_extended=%d "
        "extended_instance=%s skip_execution=%s retries=%d",
        args.runs, args.seed, args.max_fes, args.max_fes_extended,
        args.extended_instance, bool(args.skip_execution), args.retry,
    )

    state_path = output_root / "state.json"
    state = load_json(state_path)
    state.setdefault("metadata", {})
    state["metadata"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # If explicit instances passed on CLI, build a filter list
    instance_filter: list[str] | None = None
    if args.instances.strip().lower() != "all":
        shoa_tmlap_ref = repo_root / "Final-Implementation" / "SHOA" / "tmlap"
        available = {p.name for p in shoa_tmlap_ref.glob("*.txt")}
        selected: list[str] = []
        for token in args.instances.split(","):
            name = token.strip()
            if not name:
                continue
            if name in available:
                selected.append(name)
            else:
                raise ValueError(f"Instance not found in SHOA/tmlap: {name!r}")
        instance_filter = selected

    jobs = build_jobs(args=args, repo_root=repo_root, output_root=output_root,
                      instance_filter=instance_filter)
    logger.info("Built %d jobs", len(jobs))
    for j in jobs:
        logger.info("  job_id=%s | instance=%s | max_fes=%d | max_iter=%d",
                    j.job_id, j.instance_name, j.max_fes, j.max_iter)

    # Persist job metadata to state
    jobs_state = state.setdefault("jobs", {})
    for job in jobs:
        spec = jobs_state.setdefault(job.job_id, {})
        spec["algorithm"]     = job.algorithm
        spec["instance_name"] = job.instance_name
        spec["max_fes"]       = job.max_fes
        spec["max_iter"]      = job.max_iter
        spec["runs"]          = job.runs
        spec["seed"]          = job.seed
        spec["cwd"]           = str(job.cwd.resolve())
        spec["output_dir"]    = str(job.output_dir.resolve())

    save_json(state_path, state)
    logger.info("State initialised at %s", str(state_path.resolve()))

    # ---- Execution ----
    if not args.skip_execution:
        logger.info("Executing %d jobs sequentially", len(jobs))
        for job in jobs:
            state = load_json(state_path)
            run_job_fail_safe(
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
        logger.info("Execution skipped (--skip-execution); reporting from existing state")

    # ---- Collect completed / failed ----
    state = load_json(state_path)
    completed_jobs: list[dict] = []
    failed_jobs:    list[dict] = []

    for job in jobs:
        record = state.get("jobs", {}).get(job.job_id, {})
        row = {
            "job_id":        job.job_id,
            "algorithm":     job.algorithm,
            "instance_name": job.instance_name,
            "max_fes":       job.max_fes,
            "status":        record.get("status", "unknown"),
            "attempts":      record.get("attempts", 0),
            "return_code":   record.get("return_code", ""),
            "run_dir":       record.get("run_dir", ""),
            "log_file":      record.get("log_file", ""),
        }
        if row["status"] == "completed" and row["run_dir"]:
            completed_jobs.append(row)
        else:
            failed_jobs.append(row)

    write_csv(tables_dir / "job_status.csv", completed_jobs + failed_jobs)
    logger.info("Wrote job status table (%d completed, %d failed/partial)",
                len(completed_jobs), len(failed_jobs))

    if not completed_jobs:
        note = {"message": "No completed jobs. Check logs and state.json", "failed_jobs": failed_jobs}
        save_json(reports_dir / "no_results.json", note)
        logger.warning("No completed jobs found; wrote no_results.json")
        print("No completed jobs found. See reports/no_results.json")
        return

    # ---- SHOA-COMBINED diagnostic plots ----
    python_exec = args.python_executable or sys.executable
    logger.info("Generating SHOA-COMBINED diagnostic plots")
    combined_plot_rows = generate_combined_plots(
        completed_jobs=completed_jobs,
        repo_root=repo_root,
        python_executable=python_exec,
        logs_dir=logs_dir,
        logger=logger,
    )
    write_csv(tables_dir / "combined_plot_jobs.csv", combined_plot_rows)
    logger.info("Wrote SHOA-COMBINED plot jobs table (%d entries)", len(combined_plot_rows))

    # ---- Data collection ----
    per_run_rows, grouped_results, convergence_points = collect_experiment_data(completed_jobs)

    write_csv(tables_dir / "per_run_results.csv", per_run_rows)
    logger.info("Wrote per-run results (%d rows)", len(per_run_rows))

    # ---- Summary stats ----
    summary_rows = summarize_grouped_results(grouped_results)
    write_csv(tables_dir / "summary_stats.csv", summary_rows)
    logger.info("Wrote summary stats (%d rows)", len(summary_rows))

    # Derive instance list and budgets from completed data
    all_instances  = sorted({str(r["instance_name"])   for r in per_run_rows})
    all_budgets    = sorted({int(r["max_fes_budget"])   for r in per_run_rows})
    all_algorithms = sorted({str(r["algorithm"])        for r in per_run_rows})

    comparison_table = build_comparison_table(
        summary_rows=summary_rows,
        instance_names=all_instances,
        algorithms=all_algorithms,
        max_fes_budgets=all_budgets,
    )
    write_csv(tables_dir / "comparison_table_mean_std.csv", comparison_table)
    logger.info("Wrote comparison table (mean±std per instance × budget × algorithm)")

    # ---- Ranking ----
    ranking_rows, ranking_avg_rows = rank_shoa_vs_pso(grouped_results)
    write_csv(tables_dir / "ranking_shoa_vs_pso_by_instance.csv", ranking_rows)
    write_csv(tables_dir / "ranking_shoa_vs_pso_average.csv", ranking_avg_rows)
    logger.info("Wrote SHOA vs PSO ranking tables")

    # ---- Wilcoxon ----
    wilcoxon_rows, wilcoxon_summary = wilcoxon_shoa_vs_pso(grouped_results)
    write_csv(tables_dir / "wilcoxon_shoa_vs_pso.csv", wilcoxon_rows)
    write_csv(tables_dir / "wins_ties_losses_shoa_vs_pso.csv", wilcoxon_summary)
    logger.info("Wrote Wilcoxon and W/T/L tables")

    # ---- Convergence plots ----
    convergence_manifest = plot_convergence_curves(
        convergence_points=convergence_points,
        output_dir=plots_dir,
        max_fes_budgets=all_budgets,
        instance_names=all_instances,
    )
    write_csv(tables_dir / "convergence_plots_manifest.csv", convergence_manifest)
    logger.info("Wrote convergence plots (%d entries)", len(convergence_manifest))

    # ---- Boxplots ----
    boxplot_manifest = plot_boxplots_shoa_vs_pso(
        grouped_results=grouped_results,
        output_dir=plots_dir,
        max_fes_budgets=all_budgets,
        instance_names=all_instances,
    )
    write_csv(tables_dir / "boxplots_manifest.csv", boxplot_manifest)
    logger.info("Wrote SHOA vs PSO boxplots (%d entries)", len(boxplot_manifest))

    # ---- Notes ----
    notes_path = reports_dir / "statistical_notes.txt"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes = [
        "Statistical protocol notes (TMLAP):",
        "- Metric: best_fitness (total cost including penalty terms). Lower is better.",
        "- No f* subtraction: TMLAP has no known analytical optimum.",
        "- feasibility_rate: fraction of 30 runs that produced a feasible solution.",
        "- Wilcoxon signed-rank: two-sided, alpha=0.05, SHOA vs PSO only.",
        "- '+' = SHOA significantly better than PSO.",
        "- '-' = PSO significantly better than SHOA.",
        "- '≈' = no significant difference.",
        f"- Instance 4 ({DEFAULT_EXTENDED_INSTANCE}) has two budget runs:",
        f"    MaxFES={DEFAULT_MAX_FES} and MaxFES={DEFAULT_MAX_FES_EXTENDED}.",
        "- Contribution plots generated only for SHOA-COMBINED runs.",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    logger.info("Wrote statistical notes")

    # ---- Report manifest ----
    report_manifest = {
        "output_root":       str(output_root),
        "completed_jobs":    completed_jobs,
        "failed_jobs":       failed_jobs,
        "tables_dir":        str(tables_dir.resolve()),
        "plots_dir":         str(plots_dir.resolve()),
        "combined_plot_jobs": combined_plot_rows,
    }
    save_json(reports_dir / "report_manifest.json", report_manifest)
    logger.info("Wrote report manifest")
    logger.info(
        "Orchestration finished | completed=%d failed_or_partial=%d",
        len(completed_jobs), len(failed_jobs),
    )

    print("TMLAP fail-safe orchestration complete")
    print(f"Completed jobs : {len(completed_jobs)}")
    print(f"Failed/partial : {len(failed_jobs)}")
    print(f"Reports        : {reports_dir}")


if __name__ == "__main__":
    main()
