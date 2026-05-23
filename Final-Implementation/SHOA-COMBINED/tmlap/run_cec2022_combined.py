"""Benchmark runner for SHOA-COMBINED (online LIME + stagnation)."""

from __future__ import annotations

import argparse
import logging
import math
import pathlib
import time

from global_explanations import aggregate_global_feature_explanations
from opfunu_wrapper import CEC2022FunctionWrapper, parse_function_ids
from SHO_HYBRID_Controller import SHO_HYBRID, estimate_max_fes
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json


LOGGER = logging.getLogger("SHOA-COMBINED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHOA-COMBINED benchmark on CEC2022")

    parser.add_argument("--functions", type=str, default="all", help="all | comma-separated ids (1-12)")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--lime-every",
        type=int,
        default=None,
        help="Legacy cadence field kept for compatibility (not active in stagnation_start_only trigger policy).",
    )
    parser.add_argument("--lime-min-samples", type=int, default=1000, help="Minimum accumulated samples required before firing LIME at stagnation_start.")
    parser.add_argument(
        "--stagnation-lime-selection-mode",
        type=str,
        default="medoid",
        choices=["selected_agents", "medoid"],
        help="Agent selection mode used when LIME is triggered by stagnation_start.",
    )
    parser.add_argument(
        "--lime-selection-mode",
        type=str,
        default=None,
        choices=["selected_agents", "medoid"],
        help="Legacy alias for --stagnation-lime-selection-mode.",
    )

    parser.add_argument(
        "--min-sfes-ratio",
        type=float,
        default=0.04,
        help="MinSFEs ratio used by stagnation detector. MinSFEs = ceil(max_fes * ratio).",
    )
    parser.add_argument(
        "--max-fes",
        type=int,
        default=0,
        help="Global FE budget. Use 0 to auto-estimate from SHO config.",
    )
    parser.add_argument(
        "--restart-enabled",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable stagnation-triggered restart policy (1=yes, 0=no).",
    )
    parser.add_argument(
        "--restart-percent",
        type=float,
        default=7.0,
        help="Restart percentage of population at stagnation_start. Must be in [5, 10].",
    )
    parser.add_argument(
        "--restart-cooldown-fes-ratio",
        type=float,
        default=None,
        help="Cooldown ratio over MaxFEs for restart attempts. If omitted, uses min_sfes_ratio.",
    )
    parser.add_argument(
        "--restart-dominance-threshold",
        type=float,
        default=0.90,
        help="LIME fused relative-importance threshold used to pick feature mutator.",
    )

    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
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

    effective_selection_mode = (
        args.lime_selection_mode
        if args.lime_selection_mode is not None
        else args.stagnation_lime_selection_mode
    )

    root_dir = pathlib.Path(__file__).resolve().parent
    output_base = (root_dir / args.output_dir).resolve()
    run_id, timestamp, run_output_dir = create_run_directory(output_base)

    if args.lime_every is None:
        effective_lime_every = max(1, int(math.ceil(args.max_iter * 0.05)))
        lime_every_strategy = "auto_5pct_max_iter"
        lime_every_input = None
    else:
        effective_lime_every = int(args.lime_every)
        lime_every_strategy = "user_fixed"
        lime_every_input = int(args.lime_every)

    function_ids = parse_function_ids(args.functions)
    effective_max_fes = int(args.max_fes) if int(args.max_fes) > 0 else estimate_max_fes(pop=args.pop, max_iter=args.max_iter)
    warmup_fes = max(1, int(math.ceil(effective_max_fes * 0.05)))
    cooldown_ratio = args.restart_cooldown_fes_ratio if args.restart_cooldown_fes_ratio is not None else args.min_sfes_ratio
    restart_cooldown_fes = max(0, int(math.ceil(effective_max_fes * float(cooldown_ratio))))

    LOGGER.info(
        "Starting SHOA-COMBINED run | run_id=%s dim=%d pop=%d max_iter=%d runs=%d functions=%s",
        run_id,
        args.dim,
        args.pop,
        args.max_iter,
        args.runs,
        function_ids,
    )
    LOGGER.info(
        "LIME config | trigger=stagnation_start_only min_samples=%d mode=%s",
        args.lime_min_samples,
        effective_selection_mode,
    )
    LOGGER.info(
        "LIME cadence (legacy) | every=%d strategy=%s active=no",
        effective_lime_every,
        lime_every_strategy,
    )
    LOGGER.info(
        "Stagnation config | min_sfes_ratio=%.4f max_fes=%d",
        args.min_sfes_ratio,
        args.max_fes,
    )
    LOGGER.info(
        "Restart config | enabled=%d percent=%.2f warmup_fes=%d cooldown_fes=%d dominance_threshold=%.2f",
        args.restart_enabled,
        args.restart_percent,
        warmup_fes,
        restart_cooldown_fes,
        args.restart_dominance_threshold,
    )

    runs_raw: list[dict] = []
    all_full_output_rows: list[dict] = []
    all_lime_contrib_rows: list[dict] = []
    all_global_rows: list[dict] = []
    all_stagnation_history_rows: list[dict] = []
    all_stagnation_event_rows: list[dict] = []

    global_start = time.perf_counter()

    for fid in function_ids:
        func_name = f"F{fid}"
        LOGGER.info("Processing %s", func_name)

        for run_number in range(1, args.runs + 1):
            seed = args.seed + 1000 * fid + run_number
            run_start = time.perf_counter()

            func_wrapper = CEC2022FunctionWrapper(function_id=fid, dimension=args.dim)
            lower_bound, upper_bound = func_wrapper.get_bounds()

            run_meta = {
                "run_id": run_id,
                "timestamp": timestamp,
                "function_name": func_name,
                "dimension": args.dim,
                "run_number": run_number,
            }

            LOGGER.info("%s run %d/%d | seed=%d", func_name, run_number, args.runs, seed)

            result = SHO_HYBRID(
                pop=args.pop,
                max_iter=args.max_iter,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dim=args.dim,
                fobj=func_wrapper,
                run_metadata=run_meta,
                random_state=seed,
                lime_every=effective_lime_every,
                min_samples_before_lime=args.lime_min_samples,
                lime_selection_mode=effective_selection_mode,
                min_sfes_ratio=args.min_sfes_ratio,
                max_fes=args.max_fes,
                warmup_fes=warmup_fes,
                restart_enabled=bool(args.restart_enabled),
                restart_percent=args.restart_percent,
                restart_cooldown_fes=restart_cooldown_fes,
                restart_dominance_threshold=args.restart_dominance_threshold,
                enable_lime=True,
                enable_stagnation=True,
                log_callback=LOGGER.info,
                progress_every=args.progress_every,
                verbose=(not args.quiet),
            )

            elapsed = time.perf_counter() - run_start
            stagnation_meta = result.get("stagnation_meta", {})
            stagnation_events_count = sum(
                1
                for row in result.get("stagnation_event_rows", [])
                if str(row.get("event", "")).strip() in {"stagnation_start", "recovered"}
            )

            run_row = {
                "run_id": run_id,
                "timestamp": timestamp,
                "function_name": func_name,
                "dimension": args.dim,
                "run_number": run_number,
                "seed": seed,
                "best_fitness": float(result["best_fitness"]),
                "diagnoses_count": len({
                    int(row["diagnosis_id"])
                    for row in result["lime_contribution_rows"]
                    if int(row.get("diagnosis_id", 0)) > 0
                }),
                "lime_contribution_rows": len(result["lime_contribution_rows"]),
                "stagnation_events": stagnation_events_count,
                "final_fe": int(stagnation_meta.get("final_fe", getattr(func_wrapper, "nfev", 0))),
                "max_fes": int(stagnation_meta.get("max_fes", 0)),
                "stop_reason": str(stagnation_meta.get("stop_reason", "unknown")),
                "elapsed_sec": elapsed,
            }
            runs_raw.append(run_row)

            for row in result["full_output_rows"]:
                row = dict(row)
                row["run_number"] = run_number
                all_full_output_rows.append(row)

            for row in result["lime_contribution_rows"]:
                row = dict(row)
                row["run_number"] = run_number
                all_lime_contrib_rows.append(row)

            run_global_rows = aggregate_global_feature_explanations(
                contribution_rows=result["lime_contribution_rows"],
                run_id=run_id,
                function_name=func_name,
                dimension=args.dim,
                windows=(50, 100),
            )
            for row in run_global_rows:
                row = dict(row)
                row["run_number"] = run_number
                all_global_rows.append(row)

            for row in result.get("stagnation_history_rows", []):
                row = dict(row)
                row["run_number"] = run_number
                all_stagnation_history_rows.append(row)

            for row in result.get("stagnation_event_rows", []):
                row = dict(row)
                row["run_number"] = run_number
                all_stagnation_event_rows.append(row)

            LOGGER.info(
                "%s run %d/%d done | best=%.6e diagnoses=%d events=%d fe=%d/%d stop=%s elapsed=%.2fs",
                func_name,
                run_number,
                args.runs,
                run_row["best_fitness"],
                run_row["diagnoses_count"],
                run_row["stagnation_events"],
                run_row["final_fe"],
                run_row["max_fes"],
                run_row["stop_reason"],
                elapsed,
            )

    summary_rows = summarize_by_function(runs_raw)

    config = {
        "algorithm": "SHOA-COMBINED",
        "run_id": run_id,
        "timestamp": timestamp,
        "functions": function_ids,
        "dimension": args.dim,
        "pop": args.pop,
        "max_iter": args.max_iter,
        "runs": args.runs,
        "seed_base": args.seed,
        "lime": {
            "enabled": 1,
            "trigger_policy": "stagnation_start_only",
            "accumulate_only_outside_stagnation": 1,
            "pause_during_stagnation": 1,
            "resume_after_recovered": 1,
            "history_window": "global_from_start_excluding_stagnated_iterations",
            "lime_every": effective_lime_every,
            "lime_every_input": lime_every_input,
            "lime_every_strategy": lime_every_strategy,
            "lime_every_active": 0,
            "lime_min_samples": args.lime_min_samples,
            "selection_mode": effective_selection_mode,
            "stagnation_lime_selection_mode": effective_selection_mode,
            "legacy_lime_selection_mode_input": args.lime_selection_mode,
        },
        "stagnation": {
            "enabled": 1,
            "min_sfes_ratio": args.min_sfes_ratio,
            "max_fes": args.max_fes,
        },
        "restart": {
            "enabled": int(bool(args.restart_enabled)),
            "policy": "stagnation_start_immediate",
            "selection": "worst_fitness_preserve_elite",
            "restart_percent": float(args.restart_percent),
            "restart_range_required": "[5,10]",
            "cooldown_unit": "fes",
            "cooldown_fes_ratio": float(cooldown_ratio),
            "cooldown_fes": int(restart_cooldown_fes),
            "warmup_fes_ratio": 0.05,
            "warmup_fes": int(warmup_fes),
            "lime_dominance_threshold": float(args.restart_dominance_threshold),
            "lime_dominance_metric": "relative_importance_per_diagnosis_fused_targets",
            "lime_targets_fusion": ["classification_improved", "regression_y_reg"],
            "fallback_formula": "LB + (UB - LB) * rand[0,1]",
            "fallback_scope": "restart_subset",
        },
    }

    write_json(run_output_dir / "config_used.json", config)

    write_csv(run_output_dir / "runs_raw.csv", runs_raw)
    write_csv(run_output_dir / "full_output.csv", all_full_output_rows)
    write_csv(run_output_dir / "lime_contributions.csv", all_lime_contrib_rows)
    write_csv(run_output_dir / "global_feature_explanations.csv", all_global_rows)
    write_csv(run_output_dir / "stagnation_history.csv", all_stagnation_history_rows)
    write_csv(run_output_dir / "stagnation_events.csv", all_stagnation_event_rows)
    write_csv(run_output_dir / "summary_by_function.csv", summary_rows)

    total_elapsed = time.perf_counter() - global_start
    LOGGER.info("Finished SHOA-COMBINED | elapsed=%.2fs", total_elapsed)
    LOGGER.info("Output directory: %s", run_output_dir)

    print("\nArtifacts generated:")
    print(f"- {run_output_dir / 'config_used.json'}")
    print(f"- {run_output_dir / 'runs_raw.csv'}")
    print(f"- {run_output_dir / 'full_output.csv'}")
    print(f"- {run_output_dir / 'lime_contributions.csv'}")
    print(f"- {run_output_dir / 'global_feature_explanations.csv'}")
    print(f"- {run_output_dir / 'stagnation_history.csv'}")
    print(f"- {run_output_dir / 'stagnation_events.csv'}")
    print(f"- {run_output_dir / 'summary_by_function.csv'}")


if __name__ == "__main__":
    main()
