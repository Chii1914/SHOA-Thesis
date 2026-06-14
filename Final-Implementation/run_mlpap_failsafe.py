"""Fail-safe orchestrator for MLPAP experiments with PSO, SHOA and SHOA-COMBINED.

Protocol
--------
- Instances: 100 JSON files (S01-S20, M01-M20, L01-L20, XL01-XL20, 2XL01-2XL20)
- MaxFES budget is configurable per scale group (S/M/L/XL/2XL)
- Default runs: 30 independent runs per algorithm × instance
- Metrics: best_fitness (minimisation) + feasibility_rate + base_cost
- Statistical comparison: SHOA vs PSO (Wilcoxon signed-rank per instance)
- Contribution plots: SHOA-COMBINED only (via plot_combined_run.py)
- Additional plots: convergence curves, boxplots, feasibility bars, summary heatmap
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


# ---------------------------------------------------------------------------
# Default budgets per scale
# ---------------------------------------------------------------------------

DEFAULT_MAX_FES: dict[str, int] = {
    "S":   50_000,
    "M":  100_000,
    "L":  200_000,
    "XL": 500_000,
    "2XL": 1_000_000,
}

SCALE_ORDER = ["S", "M", "L", "XL", "2XL"]
SCALE_PREFIXES = ("2XL", "XL", "L", "M", "S")


def _instance_scale(name: str) -> str:
    stem = Path(name).stem.upper()
    for prefix in SCALE_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_orchestrator_logger(logs_dir: Path, level_name: str) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger = logging.getLogger("mlpap_failsafe")
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
    scale: str
    max_fes: int
    runs: int
    seed: int
    max_iter: int
    output_dir: Path
    cwd: Path
    command: list[str]

    @property
    def job_id(self) -> str:
        stem = Path(self.instance_name).stem.replace(".", "_")
        return f"{self.algorithm}_{stem}_fes{self.max_fes}"


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2, ensure_ascii=True)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    raw = str(value).strip()
    return int(float(raw)) if raw else default


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    raw = str(value).strip()
    return float(raw) if raw else default


# ---------------------------------------------------------------------------
# Run-directory discovery
# ---------------------------------------------------------------------------

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
    job_state  = jobs_state.setdefault(job.job_id, {})

    if job_state.get("status") == "completed":
        run_dir_str = job_state.get("run_dir", "")
        if run_dir_str and Path(run_dir_str).exists():
            logger.info("Skipping %s: already completed", job.job_id)
            return job_state

    max_attempts   = max(1, retries + 1)
    attempts_done  = int(job_state.get("attempts", 0))

    before = (
        {str(p.resolve()) for p in job.output_dir.glob("run-*") if p.is_dir()}
        if job.output_dir.exists() else set()
    )

    for attempt in range(attempts_done + 1, max_attempts + 1):
        job_state.update({
            "status": "running",
            "attempts": attempt,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        log_file = logs_dir / f"{job.job_id}.attempt{attempt}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting %s (attempt %d/%d) | scale=%s max_fes=%d",
            job.job_id, attempt, max_attempts, job.scale, job.max_fes,
        )

        with log_file.open("w", encoding="utf-8") as h:
            h.write(f"COMMAND:\n{' '.join(job.command)}\n\nCWD: {job.cwd}\n\n")
            h.flush()
            completed = subprocess.run(
                job.command, cwd=str(job.cwd),
                stdout=h, stderr=subprocess.STDOUT,
                text=True, check=False,
            )

        job_state["return_code"] = int(completed.returncode)
        job_state["log_file"]    = str(log_file.resolve())

        if completed.returncode == 0:
            run_dir = discover_new_or_latest_run_dir(job.output_dir, before)
            job_state.update({
                "status":      "completed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_dir":     str(run_dir.resolve()) if run_dir else "",
            })
            logger.info("Completed %s | run_dir=%s", job.job_id, job_state["run_dir"] or "<not-found>")
            return job_state

        job_state.update({
            "status":      "failed",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        logger.warning("Failed %s attempt %d/%d rc=%d", job.job_id, attempt, max_attempts, completed.returncode)
        if attempt < max_attempts:
            time.sleep(min(10, 2 * attempt))

    if not continue_on_failure:
        raise RuntimeError(f"Job {job.job_id} failed after {max_attempts} attempts")
    logger.error("Job %s exhausted retries", job.job_id)
    return job_state


# ---------------------------------------------------------------------------
# Budget / iteration helpers
# ---------------------------------------------------------------------------

def compute_pso_max_iter(max_fes: int, particles: int) -> int:
    return max(1, (max_fes - particles) // max(1, particles))


def compute_shoa_max_iter(max_fes: int, pop: int) -> int:
    per_iter = pop + pop // 2
    return max(1, (max_fes - pop) // max(1, per_iter))


def compute_combined_max_iter(max_fes: int, pop: int) -> int:
    return compute_shoa_max_iter(max_fes=max_fes, pop=pop) + 50


# ---------------------------------------------------------------------------
# Instance discovery
# ---------------------------------------------------------------------------

def discover_instances(
    instance_dir: Path,
    scales: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return sorted list of (filename, scale) tuples from instance_dir.

    Filters by scale prefixes and/or specific IDs if provided.
    """
    all_json = sorted(
        p for p in instance_dir.glob("*.json") if p.stem != "instances_index"
    )
    result: list[tuple[str, str]] = []
    for p in all_json:
        sc = _instance_scale(p.name)
        if scales and sc not in scales:
            continue
        if ids and p.stem.upper() not in {i.upper() for i in ids}:
            continue
        result.append((p.name, sc))
    return result


# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

def build_jobs(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    instance_dir: Path,
    instances: list[tuple[str, str]],
    fes_by_scale: dict[str, int],
) -> list[JobSpec]:
    py = args.python_executable or sys.executable

    shoa_dir  = repo_root / "Final-Implementation" / "SHOA"         / "mlpap"
    pso_dir   = repo_root / "Final-Implementation" / "PSO"          / "mlpap"
    comb_dir  = repo_root / "Final-Implementation" / "SHOA-COMBINED"/ "mlpap"

    jobs: list[JobSpec] = []

    for inst_name, scale in instances:
        max_fes = fes_by_scale.get(scale, DEFAULT_MAX_FES.get(scale, 50_000))
        tag     = f"{Path(inst_name).stem}_fes{max_fes}"

        pso_iter  = compute_pso_max_iter(max_fes=max_fes, particles=args.pso_particles)
        shoa_iter = compute_shoa_max_iter(max_fes=max_fes, pop=args.shoa_pop)
        comb_iter = compute_combined_max_iter(max_fes=max_fes, pop=args.combined_pop)

        pso_out  = output_root / "raw" / "PSO"           / tag
        shoa_out = output_root / "raw" / "SHOA"          / tag
        comb_out = output_root / "raw" / "SHOA-COMBINED" / tag

        inst_dir_str = str(instance_dir.resolve())

        pso_cmd = [
            py, "run_pso_mlpap.py",
            "--instances", inst_name,
            "--instance-dir", inst_dir_str,
            "--particles", str(args.pso_particles),
            "--max-iter",  str(pso_iter),
            "--runs",      str(args.runs),
            "--seed",      str(args.seed),
            "--w",         str(args.pso_w),
            "--c1",        str(args.pso_c1),
            "--c2",        str(args.pso_c2),
            "--output-dir", str(pso_out.resolve()),
            "--log-level",  args.log_level,
        ]

        shoa_cmd = [
            py, "run_shoa_mlpap.py",
            "--instances", inst_name,
            "--instance-dir", inst_dir_str,
            "--pop",      str(args.shoa_pop),
            "--max-iter", str(shoa_iter),
            "--runs",     str(args.runs),
            "--seed",     str(args.seed),
            "--output-dir", str(shoa_out.resolve()),
            "--log-level",  args.log_level,
        ]

        comb_cmd = [
            py, "run_mlpap_combined.py",
            "--instances", inst_name,
            "--instance-dir", inst_dir_str,
            "--pop",      str(args.combined_pop),
            "--max-iter", str(comb_iter),
            "--runs",     str(args.runs),
            "--seed",     str(args.seed),
            "--max-fes",  str(max_fes),
            "--restart-enabled",              "1" if args.combined_restart_enabled else "0",
            "--restart-percent",              str(args.combined_restart_percent),
            "--restart-cooldown-fes-ratio",   str(args.combined_restart_cooldown_ratio),
            "--restart-dominance-threshold",  str(args.combined_restart_dominance_threshold),
            "--lime-enabled",                 "1",
            "--lime-min-samples",             str(args.combined_lime_min_samples),
            "--stagnation-lime-selection-mode", args.combined_lime_selection_mode,
            "--progress-every",               str(args.combined_progress_every),
            "--output-dir", str(comb_out.resolve()),
            "--log-level",  args.log_level,
        ]

        for algo, cwd, cmd, out, n_iter in [
            ("PSO",          pso_dir,  pso_cmd,  pso_out,  pso_iter),
            ("SHOA",         shoa_dir, shoa_cmd, shoa_out, shoa_iter),
            ("SHOA-COMBINED",comb_dir, comb_cmd, comb_out, comb_iter),
        ]:
            jobs.append(JobSpec(
                algorithm=algo,
                instance_name=inst_name,
                scale=scale,
                max_fes=max_fes,
                runs=args.runs,
                seed=args.seed,
                max_iter=n_iter,
                output_dir=out,
                cwd=cwd,
                command=cmd,
            ))

    return jobs


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_experiment_data(completed_jobs: list[dict]):
    per_run_rows: list[dict] = []
    grouped:  dict[tuple[str, str, int], list[dict]] = {}
    conv_pts: dict[tuple[str, str, int, int], list[tuple[float, float]]] = {}

    for job in completed_jobs:
        algo     = str(job["algorithm"])
        inst     = str(job["instance_name"])
        max_fes  = int(job["max_fes"])
        scale    = str(job.get("scale", _instance_scale(inst)))
        run_dir  = Path(job["run_dir"])

        for row in read_csv_rows(run_dir / "runs_raw.csv"):
            inst_n   = str(row.get("instance_name") or row.get("function_name") or inst)
            run_num  = _to_int(row.get("run_number"), 0)
            bf       = _to_float(row.get("best_fitness"), np.nan)
            feasible = _to_int(row.get("feasible_best_solution"), 0)
            bc       = _to_float(row.get("base_cost"), np.nan)
            viol     = _to_float(row.get("violation_total"), 0.0)
            fes_used = _to_int(row.get("fes_used") or row.get("final_fe"), 0)

            out = {
                "algorithm":     algo,
                "instance_name": inst_n,
                "scale":         scale,
                "max_fes_budget":max_fes,
                "run_number":    run_num,
                "best_fitness":  bf,
                "base_cost":     bc,
                "feasible":      feasible,
                "violation_total": viol,
                "fes_used":      fes_used,
                "run_dir":       str(run_dir.resolve()),
            }
            per_run_rows.append(out)
            grouped.setdefault((algo, inst_n, max_fes), []).append(out)

        for row in read_csv_rows(run_dir / "full_output.csv"):
            inst_n  = str(row.get("instance_name") or row.get("function_name") or inst)
            run_num = _to_int(row.get("run_number"), 0)
            fe_val  = _to_float(row.get("fe_estimate") or row.get("fe"), np.nan)
            fit_val = _to_float(row.get("best_fitness_so_far"), np.nan)
            key = (algo, inst_n, max_fes, run_num)
            conv_pts.setdefault(key, []).append((fe_val, fit_val))

    # enforce monotone-decreasing best-so-far per curve
    for key, vals in list(conv_pts.items()):
        ordered = sorted(vals, key=lambda v: v[0])
        cleaned, best_f = [], float("inf")
        for fe, fit in ordered:
            if not (np.isfinite(fe) and np.isfinite(fit)):
                continue
            best_f = min(best_f, fit)
            cleaned.append((float(fe), float(best_f)))
        conv_pts[key] = cleaned

    return per_run_rows, grouped, conv_pts


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_grouped(grouped: dict) -> list[dict]:
    rows: list[dict] = []
    for (algo, inst, max_fes), items in sorted(grouped.items()):
        bf  = np.array([float(r["best_fitness"]) for r in items], dtype=float)
        feas = np.array([int(r["feasible"]) for r in items], dtype=float)
        bc  = np.array([float(r["base_cost"]) for r in items if np.isfinite(float(r["base_cost"]))])
        if bf.size == 0:
            continue
        scale = str(items[0].get("scale", _instance_scale(inst)))
        rows.append({
            "algorithm":        algo,
            "instance_name":    inst,
            "scale":            scale,
            "max_fes_budget":   max_fes,
            "runs_completed":   int(bf.size),
            "best":             float(np.min(bf)),
            "mean":             float(np.mean(bf)),
            "std":              float(np.std(bf, ddof=1)) if bf.size > 1 else 0.0,
            "median":           float(np.median(bf)),
            "worst":            float(np.max(bf)),
            "feasibility_rate": float(np.mean(feas)),
            "mean_base_cost":   float(np.mean(bc)) if bc.size > 0 else float("nan"),
            "mean_pm_std": (
                f"{np.mean(bf):.6e} +/- "
                f"{np.std(bf, ddof=1) if bf.size > 1 else 0.0:.6e}"
            ),
        })
    return rows


def build_comparison_table(
    summary_rows: list[dict],
    instance_names: list[str],
    algorithms: list[str],
) -> list[dict]:
    by_key = {
        (str(r["instance_name"]), str(r["algorithm"])): r
        for r in summary_rows
    }
    rows: list[dict] = []
    for inst in instance_names:
        row: dict = {"instance_name": inst, "scale": _instance_scale(inst)}
        for algo in algorithms:
            item = by_key.get((inst, algo))
            row[algo]                       = item["mean_pm_std"] if item else "NA"
            row[f"{algo}_feasibility_rate"] = f"{item['feasibility_rate']:.3f}" if item else "NA"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_shoa_vs_pso(grouped: dict) -> tuple[list[dict], list[dict]]:
    ranking, avg_acc = [], {}
    for (algo, inst, fes), _ in grouped.items():
        pass  # just enumerate keys

    instance_budgets = sorted({(inst, fes) for (_, inst, fes) in grouped})
    rank_acc: dict[tuple[str, int, str], list[float]] = {}

    for inst, fes in instance_budgets:
        shoa_items = grouped.get(("SHOA", inst, fes), [])
        pso_items  = grouped.get(("PSO",  inst, fes), [])
        if not shoa_items or not pso_items:
            continue
        sm = float(np.mean([r["best_fitness"] for r in shoa_items]))
        pm = float(np.mean([r["best_fitness"] for r in pso_items]))
        if abs(sm - pm) <= 1e-15:
            sr, pr = 1.5, 1.5
        elif sm < pm:
            sr, pr = 1.0, 2.0
        else:
            sr, pr = 2.0, 1.0
        ranking.append({
            "instance_name": inst, "max_fes_budget": fes,
            "SHOA_mean": sm, "PSO_mean": pm,
            "SHOA_rank": sr, "PSO_rank": pr,
        })
        rank_acc.setdefault(("all", fes, "SHOA"), []).append(sr)
        rank_acc.setdefault(("all", fes, "PSO"),  []).append(pr)

    avg_rows = [
        {"group": g, "max_fes_budget": fes, "algorithm": algo,
         "average_rank": float(np.mean(vals)), "instances_count": len(vals)}
        for (g, fes, algo), vals in sorted(rank_acc.items())
    ]
    return ranking, avg_rows


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------

def wilcoxon_shoa_vs_pso(grouped: dict, alpha: float = 0.05) -> tuple[list[dict], list[dict]]:
    rows, summary = [], []
    instance_budgets = sorted({(inst, fes) for (_, inst, fes) in grouped})
    budgets = sorted({fes for (_, fes) in instance_budgets})

    for budget in budgets:
        wins = ties = losses = 0
        for inst, fes in instance_budgets:
            if fes != budget:
                continue
            shoa = grouped.get(("SHOA", inst, budget), [])
            pso  = grouped.get(("PSO",  inst, budget), [])
            if not shoa or not pso:
                continue

            sb = {_to_int(r["run_number"]): float(r["best_fitness"]) for r in shoa}
            pb = {_to_int(r["run_number"]): float(r["best_fitness"]) for r in pso}
            common = sorted(set(sb) & set(pb))
            if common:
                x = np.array([sb[k] for k in common], dtype=float)
                y = np.array([pb[k] for k in common], dtype=float)
            else:
                x = np.array(sorted(float(r["best_fitness"]) for r in shoa), dtype=float)
                y = np.array(sorted(float(r["best_fitness"]) for r in pso),  dtype=float)
                n = min(x.size, y.size); x, y = x[:n], y[:n]

            p_val = stat_val = 1.0
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    res = wilcoxon(x, y, alternative="two-sided",
                                   zero_method="wilcox", correction=False, method="auto")
                p_val, stat_val = float(res.pvalue), float(res.statistic)
            except ValueError:
                pass

            sm, pm = float(np.mean(x)), float(np.mean(y))
            if p_val < alpha:
                if sm < pm:
                    outcome = "+"; wins += 1
                elif sm > pm:
                    outcome = "-"; losses += 1
                else:
                    outcome = "≈"; ties += 1
            else:
                outcome = "≈"; ties += 1

            rows.append({
                "instance_name": inst, "scale": _instance_scale(inst),
                "max_fes_budget": budget, "n_pairs": int(min(x.size, y.size)),
                "SHOA_mean_fitness": sm, "PSO_mean_fitness": pm,
                "wilcoxon_statistic": stat_val, "p_value": p_val,
                "alpha": alpha, "outcome_SHOA_vs_PSO": outcome,
            })

        summary.append({
            "max_fes_budget": budget, "opponent": "PSO",
            "wins_plus": wins, "ties_equal": ties, "losses_minus": losses,
        })
    return rows, summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    *,
    conv_pts: dict,
    output_dir: Path,
    instance_names: list[str],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    algorithms = sorted({key[0] for key in conv_pts})

    algo_styles = {
        "PSO":           {"color": "#e74c3c", "ls": "--"},
        "SHOA":          {"color": "#2980b9", "ls": "-"},
        "SHOA-COMBINED": {"color": "#27ae60", "ls": "-"},
    }

    budgets_per_inst: dict[str, set[int]] = {}
    for (algo, inst, fes, _) in conv_pts:
        budgets_per_inst.setdefault(inst, set()).add(fes)

    for inst in instance_names:
        for budget in sorted(budgets_per_inst.get(inst, set())):
            grid = np.linspace(1, budget, 500)
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted = False

            for algo in algorithms:
                run_curves: list[np.ndarray] = []
                for (ka, ki, kf, _), series in conv_pts.items():
                    if ka != algo or ki != inst or kf != budget or not series:
                        continue
                    fe  = np.array([s[0] for s in series])
                    fit = np.array([s[1] for s in series])
                    if fe.size == 0:
                        continue
                    run_curves.append(np.interp(grid, fe, fit, left=fit[0], right=fit[-1]))

                if not run_curves:
                    continue
                mat  = np.vstack(run_curves)
                mean = np.mean(mat, axis=0)
                std  = np.std(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
                sty  = algo_styles.get(algo, {"color": "gray", "ls": "-"})
                ax.plot(grid, mean, linewidth=2, label=f"{algo} (mean)", **{k: v for k, v in sty.items()})
                ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=sty["color"])
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            safe = Path(inst).stem
            ax.set_xlabel("Function Evaluations (FEs)")
            ax.set_ylabel("Best Fitness (penalized cost)")
            ax.set_title(f"MLPAP Convergence — {safe} — MaxFES={budget:,}")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fname = f"convergence_{safe}_fes{budget}.png"
            out = output_dir / fname
            fig.savefig(out, dpi=150)
            plt.close(fig)
            manifest.append({"plot_type": "convergence", "instance_name": inst,
                              "max_fes_budget": budget, "path": str(out)})
    return manifest


def plot_boxplots(
    *,
    grouped: dict,
    output_dir: Path,
    instance_names: list[str],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    algorithms = sorted({key[0] for key in grouped})

    for inst in instance_names:
        budgets = sorted({fes for (_, i, fes) in grouped if i == inst})
        for budget in budgets:
            data   = [
                [float(r["best_fitness"]) for r in grouped.get((algo, inst, budget), [])]
                for algo in algorithms
            ]
            labels = algorithms
            nonempty = [(d, l) for d, l in zip(data, labels) if d]
            if not nonempty:
                continue
            data_ne, labels_ne = zip(*nonempty)

            safe = Path(inst).stem
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(data_ne, tick_labels=labels_ne, showfliers=True)
            ax.set_ylabel("Best Fitness (penalized cost)")
            ax.set_title(f"MLPAP Boxplot — {safe} — MaxFES={budget:,}")
            ax.grid(alpha=0.25, axis="y")
            fig.tight_layout()
            fname = f"boxplot_{safe}_fes{budget}.png"
            out = output_dir / fname
            fig.savefig(out, dpi=150)
            plt.close(fig)
            manifest.append({"plot_type": "boxplot", "instance_name": inst,
                              "max_fes_budget": budget, "path": str(out)})
    return manifest


def plot_feasibility_by_scale(
    *,
    per_run_rows: list[dict],
    algorithms: list[str],
    output_dir: Path,
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, list[float]]] = {}
    for row in per_run_rows:
        sc   = str(row.get("scale", _instance_scale(str(row.get("instance_name", "")))))
        algo = str(row["algorithm"])
        data.setdefault(sc, {}).setdefault(algo, []).append(float(row["feasible"]))

    scales_present = [s for s in SCALE_ORDER if s in data]
    if not scales_present:
        return None

    x = np.arange(len(scales_present))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    algo_colors = {"PSO": "#e74c3c", "SHOA": "#2980b9", "SHOA-COMBINED": "#27ae60"}
    for i, algo in enumerate(algorithms):
        rates = [float(np.mean(data.get(sc, {}).get(algo, [0]))) for sc in scales_present]
        ax.bar(x + i * width, rates, width, label=algo, color=algo_colors.get(algo, "gray"), alpha=0.8)

    ax.set_xlabel("Instance Scale")
    ax.set_ylabel("Feasibility Rate")
    ax.set_title("MLPAP Feasibility Rate by Scale")
    ax.set_xticks(x + width)
    ax.set_xticklabels(scales_present)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = output_dir / "feasibility_by_scale.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_summary_heatmap(
    *,
    summary_rows: list[dict],
    algorithms: list[str],
    output_dir: Path,
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return None

    by_key = {(r["instance_name"], r["algorithm"]): r for r in summary_rows}
    instances_ordered = sorted(
        {r["instance_name"] for r in summary_rows},
        key=lambda n: (SCALE_ORDER.index(_instance_scale(n)) if _instance_scale(n) in SCALE_ORDER else 99, n),
    )

    matrix = np.full((len(instances_ordered), len(algorithms)), np.nan)
    for ri, inst in enumerate(instances_ordered):
        for ci, algo in enumerate(algorithms):
            item = by_key.get((inst, algo))
            if item:
                matrix[ri, ci] = float(item["mean"])

    # Normalize per row (instance) so all algorithms are comparable
    row_min  = np.nanmin(matrix, axis=1, keepdims=True)
    row_max  = np.nanmax(matrix, axis=1, keepdims=True)
    denom    = np.where(row_max - row_min > 0, row_max - row_min, 1.0)
    norm_mat = (matrix - row_min) / denom

    fig, ax = plt.subplots(figsize=(max(6, len(algorithms) * 2), max(8, len(instances_ordered) * 0.35)))
    im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=30, ha="right")
    ax.set_yticks(range(len(instances_ordered)))
    ax.set_yticklabels(instances_ordered, fontsize=7)
    ax.set_title("MLPAP — Mean Fitness Heatmap (row-normalized, green=best)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Normalized mean fitness")
    fig.tight_layout()
    out = output_dir / "summary_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_violin_by_scale(
    *,
    per_run_rows: list[dict],
    algorithms: list[str],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    by_scale: dict[str, dict[str, list[float]]] = {}
    for row in per_run_rows:
        sc   = str(row.get("scale", _instance_scale(str(row.get("instance_name", "")))))
        algo = str(row["algorithm"])
        by_scale.setdefault(sc, {}).setdefault(algo, []).append(float(row["best_fitness"]))

    for scale in SCALE_ORDER:
        if scale not in by_scale:
            continue
        data   = [by_scale[scale].get(algo, []) for algo in algorithms]
        labels = algorithms
        nonempty = [(d, l) for d, l in zip(data, labels) if len(d) > 1]
        if len(nonempty) < 2:
            continue
        data_ne, labels_ne = zip(*nonempty)

        fig, ax = plt.subplots(figsize=(8, 6))
        parts = ax.violinplot(data_ne, positions=range(len(data_ne)), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(labels_ne)))
        ax.set_xticklabels(labels_ne)
        ax.set_ylabel("Best Fitness")
        ax.set_title(f"MLPAP Violin — Scale {scale}")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        out = output_dir / f"violin_scale_{scale}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(str(out))
    return paths


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
    plot_script = repo_root / "Final-Implementation" / "SHOA-COMBINED" / "mlpap" / "plot_combined_run.py"
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
        log_f   = logs_dir / f"plot_SHOA-COMBINED_{safe}_fes{max_fes}.log"

        cmd = [python_executable, str(plot_script.name), "--run-dir", str(run_dir.resolve())]
        logger.info("Generating SHOA-COMBINED plots for %s MaxFES=%d", inst, max_fes)

        with log_f.open("w", encoding="utf-8") as h:
            h.write(f"COMMAND:\n{' '.join(cmd)}\n\n")
            h.flush()
            completed = subprocess.run(cmd, cwd=str(plot_cwd), stdout=h,
                                       stderr=subprocess.STDOUT, text=True, check=False)

        rows.append({
            "algorithm": "SHOA-COMBINED", "instance_name": inst, "max_fes_budget": max_fes,
            "run_dir": str(run_dir), "return_code": int(completed.returncode),
            "log_file": str(log_f), "plots_dir": str(run_dir / "plots"),
        })
    return rows


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-safe MLPAP orchestrator for PSO, SHOA, SHOA-COMBINED"
    )

    parser.add_argument("--instances", type=str, default="all",
                        help="'all' | scale prefix 'S,M' | specific IDs 'S01,S02'")
    parser.add_argument("--scales", type=str, default="",
                        help="Filter by scale: 'S,M,L,XL,2XL' (subset)")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    # Per-scale FES budget
    parser.add_argument("--max-fes-S",   type=int, default=DEFAULT_MAX_FES["S"])
    parser.add_argument("--max-fes-M",   type=int, default=DEFAULT_MAX_FES["M"])
    parser.add_argument("--max-fes-L",   type=int, default=DEFAULT_MAX_FES["L"])
    parser.add_argument("--max-fes-XL",  type=int, default=DEFAULT_MAX_FES["XL"])
    parser.add_argument("--max-fes-2XL", type=int, default=DEFAULT_MAX_FES["2XL"])

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
    parser.add_argument("--combined-lime-selection-mode", type=str, default="medoid",
                        choices=["medoid", "selected_agents"])
    parser.add_argument("--combined-progress-every", type=int, default=10)

    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--continue-on-failure", action="store_true", default=True)
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip runners; generate reports only from completed state.json")

    parser.add_argument("--instance-dir", type=str, default="",
                        help="Path to JSON instances folder. Default: <repo>/Instances/")
    parser.add_argument("--output-root", type=str,
                        default="Final-Implementation/experiments/mlpap_failsafe")
    parser.add_argument("--python-executable", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()

    repo_root   = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    instance_dir = Path(args.instance_dir).resolve() if args.instance_dir else (repo_root / "Instances").resolve()

    logs_dir    = output_root / "logs"
    reports_dir = output_root / "reports"
    plots_dir   = reports_dir / "plots"
    tables_dir  = reports_dir / "tables"

    logger = setup_orchestrator_logger(logs_dir=logs_dir, level_name=args.log_level)
    logger.info("Starting MLPAP fail-safe orchestrator")
    logger.info("Instance dir : %s", instance_dir)
    logger.info("Output root  : %s", output_root)

    # FES map
    fes_by_scale = {
        "S":   int(args.max_fes_S),
        "M":   int(args.max_fes_M),
        "L":   int(args.max_fes_L),
        "XL":  int(args.max_fes_XL),
        "2XL": int(args.max_fes_2XL),
    }
    logger.info("MaxFES by scale: %s", fes_by_scale)

    # Instance selection
    scale_filter = [s.strip().upper() for s in args.scales.split(",") if s.strip()] or None
    id_filter    = None
    inst_arg     = args.instances.strip()
    if inst_arg.lower() != "all":
        tokens = [t.strip().upper() for t in inst_arg.split(",") if t.strip()]
        ids    = [t for t in tokens if t not in SCALE_ORDER and not any(t == s for s in SCALE_ORDER)]
        scs    = [t for t in tokens if t in SCALE_ORDER]
        if ids:
            id_filter = ids
        if scs:
            scale_filter = list(set((scale_filter or []) + scs))

    instances = discover_instances(instance_dir, scales=scale_filter, ids=id_filter)
    if not instances:
        logger.error("No instances found for selection '%s' scales=%s", args.instances, scale_filter)
        sys.exit(1)
    logger.info("Selected %d instances: %s", len(instances), [i for i, _ in instances[:5]])

    state_path = output_root / "state.json"
    state = load_json(state_path)
    state.setdefault("metadata", {})["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    jobs = build_jobs(
        args=args, repo_root=repo_root, output_root=output_root,
        instance_dir=instance_dir, instances=instances, fes_by_scale=fes_by_scale,
    )
    logger.info("Built %d jobs (%d instances × 3 algorithms)", len(jobs), len(instances))

    jobs_state = state.setdefault("jobs", {})
    for job in jobs:
        spec = jobs_state.setdefault(job.job_id, {})
        spec.update({
            "algorithm": job.algorithm, "instance_name": job.instance_name,
            "scale": job.scale, "max_fes": job.max_fes,
            "max_iter": job.max_iter, "runs": job.runs, "seed": job.seed,
            "cwd": str(job.cwd.resolve()), "output_dir": str(job.output_dir.resolve()),
        })
    save_json(state_path, state)

    # ---- Execution ----
    if not args.skip_execution:
        logger.info("Executing %d jobs sequentially", len(jobs))
        for job in jobs:
            state = load_json(state_path)
            run_job_fail_safe(job, logs_dir=logs_dir, retries=args.retry,
                              state=state, continue_on_failure=bool(args.continue_on_failure),
                              logger=logger)
            save_json(state_path, state)
    else:
        logger.info("Execution skipped (--skip-execution)")

    # ---- Collect results ----
    state = load_json(state_path)
    completed_jobs, failed_jobs = [], []
    for job in jobs:
        record = state.get("jobs", {}).get(job.job_id, {})
        row = {
            "job_id": job.job_id, "algorithm": job.algorithm,
            "instance_name": job.instance_name, "scale": job.scale,
            "max_fes": job.max_fes, "status": record.get("status", "unknown"),
            "attempts": record.get("attempts", 0), "return_code": record.get("return_code", ""),
            "run_dir": record.get("run_dir", ""), "log_file": record.get("log_file", ""),
        }
        (completed_jobs if row["status"] == "completed" and row["run_dir"] else failed_jobs).append(row)

    write_csv(tables_dir / "job_status.csv", completed_jobs + failed_jobs)
    logger.info("Job status: %d completed, %d failed/partial", len(completed_jobs), len(failed_jobs))

    if not completed_jobs:
        save_json(reports_dir / "no_results.json", {"message": "No completed jobs", "failed": failed_jobs})
        logger.warning("No completed jobs. Exiting.")
        print("No completed jobs found.")
        return

    # ---- SHOA-COMBINED diagnostic plots ----
    py_exec = args.python_executable or sys.executable
    combined_plot_rows = generate_combined_plots(
        completed_jobs=completed_jobs, repo_root=repo_root,
        python_executable=py_exec, logs_dir=logs_dir, logger=logger,
    )
    write_csv(tables_dir / "combined_plot_jobs.csv", combined_plot_rows)

    # ---- Data collection ----
    per_run_rows, grouped, conv_pts = collect_experiment_data(completed_jobs)
    write_csv(tables_dir / "per_run_results.csv", per_run_rows)
    logger.info("Collected %d run rows", len(per_run_rows))

    # ---- Summary stats ----
    summary_rows = summarize_grouped(grouped)
    write_csv(tables_dir / "summary_stats.csv", summary_rows)

    all_instances  = sorted({str(r["instance_name"]) for r in per_run_rows},
                            key=lambda n: (SCALE_ORDER.index(_instance_scale(n))
                                          if _instance_scale(n) in SCALE_ORDER else 99, n))
    all_algorithms = sorted({str(r["algorithm"]) for r in per_run_rows})

    comparison = build_comparison_table(summary_rows, all_instances, all_algorithms)
    write_csv(tables_dir / "comparison_table_mean_std.csv", comparison)

    # ---- Ranking ----
    ranking_rows, ranking_avg = rank_shoa_vs_pso(grouped)
    write_csv(tables_dir / "ranking_shoa_vs_pso_by_instance.csv", ranking_rows)
    write_csv(tables_dir / "ranking_shoa_vs_pso_average.csv", ranking_avg)

    # ---- Wilcoxon ----
    wilcoxon_rows, wil_summary = wilcoxon_shoa_vs_pso(grouped)
    write_csv(tables_dir / "wilcoxon_shoa_vs_pso.csv", wilcoxon_rows)
    write_csv(tables_dir / "wins_ties_losses.csv", wil_summary)
    logger.info("Written statistical tables")

    # ---- Convergence plots ----
    conv_manifest = plot_convergence_curves(
        conv_pts=conv_pts, output_dir=plots_dir / "convergence",
        instance_names=all_instances,
    )
    write_csv(tables_dir / "convergence_plots_manifest.csv", conv_manifest)

    # ---- Boxplots ----
    box_manifest = plot_boxplots(
        grouped=grouped, output_dir=plots_dir / "boxplots",
        instance_names=all_instances,
    )
    write_csv(tables_dir / "boxplots_manifest.csv", box_manifest)

    # ---- Feasibility by scale ----
    feasibility_path = plot_feasibility_by_scale(
        per_run_rows=per_run_rows, algorithms=all_algorithms,
        output_dir=plots_dir / "analysis",
    )

    # ---- Violin plots ----
    violin_paths = plot_violin_by_scale(
        per_run_rows=per_run_rows, algorithms=all_algorithms,
        output_dir=plots_dir / "analysis",
    )

    # ---- Summary heatmap ----
    heatmap_path = plot_summary_heatmap(
        summary_rows=summary_rows, algorithms=all_algorithms,
        output_dir=plots_dir / "analysis",
    )

    logger.info("Convergence plots: %d | Boxplots: %d | Violin: %d",
                len(conv_manifest), len(box_manifest), len(violin_paths))

    # ---- Statistical notes ----
    notes = [
        "Statistical protocol notes (MLPAP):",
        "- Metric: best_fitness (penalized objective, lower is better).",
        "- base_cost: raw objective without penalty (reported separately).",
        "- feasibility_rate: fraction of runs with v(z) == 0.",
        "- Wilcoxon signed-rank: two-sided, alpha=0.05, SHOA vs PSO only.",
        "- '+' = SHOA significantly better than PSO.",
        "- '-' = PSO significantly better than SHOA.",
        "- '≈' = no significant difference.",
        "- LIME contribution plots generated only for SHOA-COMBINED runs.",
        f"- Instances: {len(instances)} total across scales S/M/L/XL/2XL.",
        f"- MaxFES by scale: {fes_by_scale}",
    ]
    (reports_dir / "statistical_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    # ---- Manifest ----
    save_json(reports_dir / "report_manifest.json", {
        "output_root":        str(output_root),
        "completed_jobs":     completed_jobs,
        "failed_jobs":        failed_jobs,
        "tables_dir":         str(tables_dir),
        "plots_dir":          str(plots_dir),
        "feasibility_plot":   feasibility_path,
        "heatmap_plot":       heatmap_path,
        "violin_plots":       violin_paths,
        "combined_plot_jobs": combined_plot_rows,
    })

    logger.info("Orchestration finished | completed=%d failed=%d", len(completed_jobs), len(failed_jobs))
    print(f"\nMLPAP fail-safe orchestration complete")
    print(f"  Completed jobs : {len(completed_jobs)}")
    print(f"  Failed/partial : {len(failed_jobs)}")
    print(f"  Reports        : {reports_dir}")
    print(f"  Plots          : {plots_dir}")


if __name__ == "__main__":
    main()
