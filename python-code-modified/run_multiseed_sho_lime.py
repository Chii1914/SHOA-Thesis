"""Run SHO+LIME multiple times with different seeds and export per-iteration logs."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

from SHO_LIME_Controller import SHOXAIConfig, SHO_with_lime_controller, build_opfunu_cec_objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed runner for SHO+LIME with per-iteration logging")
    parser.add_argument("--function", default="F112022", help="Opfunu class name, e.g. F112022")
    parser.add_argument("--ndim", type=int, default=20, help="Problem dimension")
    parser.add_argument("--runs", type=int, default=50, help="Number of runs (default: 50)")
    parser.add_argument("--seed-start", type=int, default=1, help="Initial seed")

    parser.add_argument("--pop-size", type=int, default=60)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--cooldown", type=int, default=30)
    parser.add_argument("--lime-samples", type=int, default=1200)
    parser.add_argument("--importance-threshold", type=float, default=0.15)
    parser.add_argument("--delta-tolerance", type=float, default=1e-8)
    parser.add_argument("--fidelity-threshold", type=float, default=0.4)
    parser.add_argument("--rescue-mode", choices=["levy_teleport", "leader_repulsion"], default="levy_teleport")

    parser.add_argument(
        "--output-dir",
        default="experiment_logs",
        help="Root directory where logs will be saved",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.function}_d{args.ndim}_runs{args.runs}_{timestamp}"
    out_root = Path(args.output_dir) / run_tag
    per_run_dir = out_root / "per_run_iteration_logs"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {out_root}")

    lb, ub, dim, objective, problem = build_opfunu_cec_objective(args.function, args.ndim)

    summary_rows: list[dict] = []
    all_iteration_path = out_root / "all_iterations.csv"

    all_iter_fieldnames = [
        "run_id",
        "seed",
        "iteration",
        "best_fitness",
        "window_size",
        "window_std",
        "trigger_candidate",
        "diagnostics_invoked",
        "diagnosis_status",
        "diagnosis_pred_delta",
        "diagnosis_fidelity",
        "rescue_applied",
        "cooldown_counter",
    ]

    with all_iteration_path.open("w", newline="", encoding="utf-8") as all_iter_file:
        all_iter_writer = csv.DictWriter(all_iter_file, fieldnames=all_iter_fieldnames)
        all_iter_writer.writeheader()

        for run_idx in range(1, args.runs + 1):
            seed = args.seed_start + run_idx - 1
            cfg = SHOXAIConfig(
                pop_size=args.pop_size,
                max_iter=args.max_iter,
                window_size=args.window_size,
                epsilon_stagnation=args.epsilon,
                cooldown_iters=args.cooldown,
                lime_num_samples=args.lime_samples,
                importance_threshold=args.importance_threshold,
                delta_tolerance=args.delta_tolerance,
                fidelity_threshold=args.fidelity_threshold,
                rescue_mode=args.rescue_mode,
                seed=seed,
            )

            start = time.time()
            result = SHO_with_lime_controller(objective, lb, ub, dim, cfg)
            runtime = time.time() - start

            per_run_rows: list[dict] = []
            for row in result.iteration_log:
                base_row = {
                    "run_id": run_idx,
                    "seed": seed,
                    "iteration": row["iteration"],
                    "best_fitness": row["best_fitness"],
                    "window_size": row["window_size"],
                    "window_std": row["window_std"],
                    "trigger_candidate": row["trigger_candidate"],
                    "diagnostics_invoked": row["diagnostics_invoked"],
                    "diagnosis_status": row["diagnosis_status"],
                    "diagnosis_pred_delta": row["diagnosis_pred_delta"],
                    "diagnosis_fidelity": row["diagnosis_fidelity"],
                    "rescue_applied": row["rescue_applied"],
                    "cooldown_counter": row["cooldown_counter"],
                }
                per_run_rows.append(base_row)
                all_iter_writer.writerow(base_row)

            run_iter_path = per_run_dir / f"run_{run_idx:03d}_seed_{seed}.csv"
            _write_csv(run_iter_path, per_run_rows, all_iter_fieldnames)

            positive_diag_count = sum(
                1 for diag in result.diagnostics_log if diag.get("status") == "POSITIVE_STAGNATION"
            )

            summary_rows.append(
                {
                    "run_id": run_idx,
                    "seed": seed,
                    "best_fitness": result.best_fitness,
                    "rescues_applied": result.rescue_count,
                    "diagnostics_invoked": len(result.diagnostics_log),
                    "positive_diagnosis": positive_diag_count,
                    "runtime_sec": runtime,
                }
            )

            print(
                f"[{run_idx:03d}/{args.runs}] seed={seed} best={result.best_fitness:.6e} "
                f"rescues={result.rescue_count} diagnostics={len(result.diagnostics_log)} time={runtime:.2f}s"
            )

    summary_path = out_root / "summary_runs.csv"
    summary_fieldnames = [
        "run_id",
        "seed",
        "best_fitness",
        "rescues_applied",
        "diagnostics_invoked",
        "positive_diagnosis",
        "runtime_sec",
    ]
    _write_csv(summary_path, summary_rows, summary_fieldnames)

    best_values = [float(row["best_fitness"]) for row in summary_rows]
    runtime_values = [float(row["runtime_sec"]) for row in summary_rows]

    stats_path = out_root / "stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(f"Function: {args.function}\n")
        f.write(f"Problem name: {problem.name}\n")
        f.write(f"Dimension used: {dim}\n")
        f.write(f"Runs: {args.runs}\n")
        f.write(f"Seed range: {args.seed_start}..{args.seed_start + args.runs - 1}\n")
        f.write("\n")
        f.write(f"Best fitness min: {min(best_values):.12e}\n")
        f.write(f"Best fitness max: {max(best_values):.12e}\n")
        f.write(f"Best fitness mean: {statistics.mean(best_values):.12e}\n")
        f.write(f"Best fitness median: {statistics.median(best_values):.12e}\n")
        f.write(f"Runtime mean (s): {statistics.mean(runtime_values):.4f}\n")
        f.write(f"Runtime median (s): {statistics.median(runtime_values):.4f}\n")

    print("\nCompleted.")
    print(f"Summary CSV: {summary_path}")
    print(f"All-iteration CSV: {all_iteration_path}")
    print(f"Per-run logs dir: {per_run_dir}")
    print(f"Stats file: {stats_path}")


if __name__ == "__main__":
    main()
