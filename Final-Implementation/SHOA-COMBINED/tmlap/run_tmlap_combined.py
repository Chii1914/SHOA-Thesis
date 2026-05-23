"""Minimal SHOA-COMBINED runner for TMLAP instances."""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np

from global_explanations import aggregate_global_feature_explanations
from SHO_HYBRID_Controller import SHO_HYBRID, estimate_max_fes
from tmlap_problem import TMLAPObjective, parse_instance_selection
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json

LOGGER = logging.getLogger("SHOA-COMBINED-TMLAP")


LIME_FIELDNAMES = [
    "run_id",
    "timestamp",
    "function_name",
    "dimension",
    "diagnosis_id",
    "iteration",
    "target_type",
    "feature",
    "mean_weight",
    "abs_mean_weight",
    "trigger_source",
    "trigger_event",
    "trigger_iteration",
]

GLOBAL_FIELDNAMES = [
    "run_id",
    "function_name",
    "dimension",
    "target_type",
    "feature",
    "window",
    "window_size",
    "If",
    "Sf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHOA-COMBINED on TMLAP instances")

    parser.add_argument("--instances", type=str, default="all", help="all | comma-separated file names")
    parser.add_argument("--instance-dir", type=str, default=".")
    parser.add_argument("--pop", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lime-enabled", type=int, default=0, choices=[0, 1], help="Enable LIME explanations")
    parser.add_argument("--lime-min-samples", type=int, default=400)
    parser.add_argument(
        "--stagnation-lime-selection-mode",
        type=str,
        default="medoid",
        choices=["selected_agents", "medoid"],
    )
    parser.add_argument("--lime-selection-mode", type=str, default=None, choices=["selected_agents", "medoid"])

    parser.add_argument("--min-sfes-ratio", type=float, default=0.04)
    parser.add_argument("--max-fes", type=int, default=0)

    parser.add_argument("--restart-enabled", type=int, default=1, choices=[0, 1])
    parser.add_argument("--restart-percent", type=float, default=7.0)
    parser.add_argument("--restart-cooldown-fes-ratio", type=float, default=None)
    parser.add_argument("--restart-dominance-threshold", type=float, default=0.90)

    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.restart_percent < 5.0 or args.restart_percent > 10.0:
        raise ValueError("--restart-percent must be in [5, 10]")
    if args.restart_dominance_threshold <= 0.0 or args.restart_dominance_threshold > 1.0:
        raise ValueError("--restart-dominance-threshold must be in (0, 1]")
    if args.restart_cooldown_fes_ratio is not None and args.restart_cooldown_fes_ratio <= 0.0:
        raise ValueError("--restart-cooldown-fes-ratio must be > 0 when provided")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    validate_args(args)

    instance_paths = parse_instance_selection(args.instances, args.instance_dir)

    run_id, ts, run_dir = create_run_directory(args.output_dir)

    runs_raw_rows: list[dict] = []
    full_output_rows: list[dict] = []
    lime_rows_all: list[dict] = []
    global_rows_all: list[dict] = []
    stagnation_history_rows: list[dict] = []
    stagnation_event_rows: list[dict] = []

    effective_selection_mode = (
        str(args.lime_selection_mode).strip().lower()
        if args.lime_selection_mode is not None
        else str(args.stagnation_lime_selection_mode).strip().lower()
    )

    base_max_fes = int(args.max_fes) if int(args.max_fes) > 0 else estimate_max_fes(pop=args.pop, max_iter=args.max_iter)
    warmup_fes = max(1, int(round(0.05 * base_max_fes)))
    restart_cooldown_ratio = args.restart_cooldown_fes_ratio if args.restart_cooldown_fes_ratio is not None else args.min_sfes_ratio
    restart_cooldown_fes = max(1, int(round(float(restart_cooldown_ratio) * base_max_fes)))

    for instance_path in instance_paths:
        instance_name = instance_path.name
        LOGGER.info("Running %s", instance_name)

        for run_number in range(1, args.runs + 1):
            run_seed = args.seed + run_number + (abs(hash(instance_name)) % 10000)
            np.random.seed(run_seed)

            objective = TMLAPObjective(instance_path)
            lower_bound, upper_bound = objective.get_bounds()

            run_metadata = {
                "run_id": run_id,
                "timestamp": ts,
                "function_name": instance_name,
                "dimension": int(objective.dimension),
            }

            start = time.perf_counter()
            result = SHO_HYBRID(
                pop=args.pop,
                max_iter=args.max_iter,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dim=objective.dimension,
                fobj=objective,
                run_metadata=run_metadata,
                random_state=run_seed,
                min_samples_before_lime=args.lime_min_samples,
                lime_selection_mode=effective_selection_mode,
                min_sfes_ratio=args.min_sfes_ratio,
                max_fes=base_max_fes,
                warmup_fes=warmup_fes,
                restart_enabled=bool(args.restart_enabled),
                restart_percent=args.restart_percent,
                restart_cooldown_fes=restart_cooldown_fes,
                restart_dominance_threshold=args.restart_dominance_threshold,
                enable_lime=bool(args.lime_enabled),
                enable_stagnation=True,
                progress_every=args.progress_every,
                verbose=(not args.quiet),
            )
            elapsed = time.perf_counter() - start

            best_position = np.asarray(result["best_position"], dtype=float)
            assignment = objective.decode(best_position)
            _, feasible_solution, base_cost, overflow_total, distance_violation = objective.evaluate_assignment(assignment)

            full_rows_run = result.get("full_output_rows", [])
            lime_rows_run = result.get("lime_contribution_rows", [])
            history_rows_run = result.get("stagnation_history_rows", [])
            events_rows_run = result.get("stagnation_event_rows", [])
            stagnation_meta = result.get("stagnation_meta", {})

            for row in full_rows_run:
                row["run_number"] = int(run_number)
            for row in lime_rows_run:
                row["run_number"] = int(run_number)
            for row in history_rows_run:
                row["run_number"] = int(run_number)
            for row in events_rows_run:
                row["run_number"] = int(run_number)

            full_output_rows.extend(full_rows_run)
            lime_rows_all.extend(lime_rows_run)
            stagnation_history_rows.extend(history_rows_run)
            stagnation_event_rows.extend(events_rows_run)

            run_events_count = sum(1 for row in events_rows_run if str(row.get("event", "")) in {"stagnation_start", "recovered"})

            runs_raw_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": ts,
                    "function_name": instance_name,
                    "run_number": run_number,
                    "dimension": int(objective.dimension),
                    "hubs": int(objective.data.n_hubs),
                    "seed": run_seed,
                    "pop": args.pop,
                    "max_iter": args.max_iter,
                    "best_fitness": float(result["best_fitness"]),
                    "base_cost": float(base_cost),
                    "feasible_best_solution": int(feasible_solution),
                    "overflow_total": int(overflow_total),
                    "distance_violation": float(distance_violation),
                    "lime_enabled": int(bool(args.lime_enabled)),
                    "diagnosis_count": int(sum(1 for row in lime_rows_run if int(row.get("diagnosis_id", 0)) > 0)),
                    "stagnation_events": int(run_events_count),
                    "final_fe": int(stagnation_meta.get("final_fe", getattr(objective, "nfev", 0))),
                    "stop_reason": str(stagnation_meta.get("stop_reason", "max_iter")),
                    "elapsed_seconds": float(elapsed),
                }
            )

            LOGGER.info(
                "%s run %d/%d -> best %.6e | feasible=%s | base_cost=%.3f | events=%d | %.2fs",
                instance_name,
                run_number,
                args.runs,
                float(result["best_fitness"]),
                "yes" if feasible_solution else "no",
                float(base_cost),
                run_events_count,
                elapsed,
            )

    if lime_rows_all:
        grouped_for_global: dict[str, list[dict]] = {}
        for row in lime_rows_all:
            key = f"{row.get('function_name','unknown')}::{row.get('run_number',1)}"
            grouped_for_global.setdefault(key, []).append(row)

        for key, rows in grouped_for_global.items():
            function_name = str(rows[0].get("function_name", "unknown"))
            dim = int(rows[0].get("dimension", 0))
            global_rows = aggregate_global_feature_explanations(
                contribution_rows=rows,
                run_id=run_id,
                function_name=function_name,
                dimension=dim,
            )
            run_number = int(rows[0].get("run_number", 1))
            for row in global_rows:
                row["run_number"] = run_number
            global_rows_all.extend(global_rows)

    summary_rows = summarize_by_function(runs_raw_rows)

    config_payload = {
        "run_id": run_id,
        "timestamp": ts,
        "algorithm": "SHOA-COMBINED",
        "problem": "TMLAP",
        "config": {
            "instances": [p.name for p in instance_paths],
            "population": args.pop,
            "max_iter": args.max_iter,
            "runs": args.runs,
            "seed": args.seed,
            "lime_enabled": int(bool(args.lime_enabled)),
            "lime_min_samples": args.lime_min_samples,
            "lime_selection_mode": effective_selection_mode,
            "min_sfes_ratio": args.min_sfes_ratio,
            "max_fes": base_max_fes,
            "warmup_fes": warmup_fes,
            "restart": {
                "enabled": int(bool(args.restart_enabled)),
                "percent": float(args.restart_percent),
                "cooldown_fes_ratio": float(restart_cooldown_ratio),
                "cooldown_fes": int(restart_cooldown_fes),
                "dominance_threshold": float(args.restart_dominance_threshold),
                "fallback_formula": "x = LB + (UB - LB) * rand[0,1]",
            },
        },
    }

    write_json(run_dir / "config_used.json", config_payload)
    write_csv(run_dir / "runs_raw.csv", runs_raw_rows)
    write_csv(run_dir / "full_output.csv", full_output_rows)
    write_csv(run_dir / "lime_contributions.csv", lime_rows_all, fieldnames=LIME_FIELDNAMES)
    write_csv(run_dir / "global_feature_explanations.csv", global_rows_all, fieldnames=GLOBAL_FIELDNAMES)
    write_csv(run_dir / "stagnation_history.csv", stagnation_history_rows)
    write_csv(run_dir / "stagnation_events.csv", stagnation_event_rows)
    write_csv(run_dir / "summary_by_function.csv", summary_rows)

    LOGGER.info("Artifacts written to %s", pathlib.Path(run_dir).resolve())


if __name__ == "__main__":
    main()
