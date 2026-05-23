"""Run SHOA-LIME on CEC2022 and export run-timestamp artifacts."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

from SHO_LIME_Controller import SHO_LIME
from global_explanations import aggregate_global_feature_explanations
from opfunu_wrapper import CEC2022FunctionWrapper, parse_function_ids
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json


LOGGER = logging.getLogger("SHOA-LIME")


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SHOA-LIME on CEC2022")
    parser.add_argument("--functions", type=str, default="all", help="all | 1-12 | 1,3,5")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lime-every",
        type=int,
        default=None,
        help="Ejecutar LIME cada N iteraciones (default: 5%% de max_ite)",
    )
    parser.add_argument("--lime-min-samples", type=int, default=1000, help="Número mínimo de muestras antes de ejecutar LIME")
    parser.add_argument("--progress-every", type=int, default=1, help="Log de progreso cada N iteraciones")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument("--quiet", action="store_true", help="Desactiva log por iteracion dentro de SHO_LIME")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    parser.add_argument(
        "--lime-selection-mode",
        type=str,
        default="medoid",
        choices=["selected_agents", "medoid"],
        help="Modo de explicacion LIME: todos los agentes seleccionados o medoid del grupo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    max_iter_safe = max(1, int(args.max_iter))

    if args.lime_every is None:
        # Auto mode: compute once at startup as 5% of max_iter.
        effective_lime_every = max(1, int(math.ceil(max_iter_safe * 0.05)))
    else:
        effective_lime_every = max(1, int(args.lime_every))

    function_ids = parse_function_ids(args.functions)
    run_id, timestamp, run_dir = create_run_directory(args.output_root)
    total_jobs = args.runs * len(function_ids)
    current_job = 0

    LOGGER.info(
        "Inicio benchmark | run_id=%s functions=%s dim=%d pop=%d max_iter=%d runs=%d lime_every=%d lime_mode=%s",
        run_id,
        function_ids,
        args.dimension,
        args.pop_size,
        args.max_iter,
        args.runs,
        effective_lime_every,
        args.lime_selection_mode,
    )

    all_run_rows: list[dict] = []
    all_full_output_rows: list[dict] = []
    all_contribution_rows: list[dict] = []
    all_global_rows: list[dict] = []

    min_samples_before_lime = args.lime_min_samples if args.lime_min_samples > 0 else None

    for run_number in range(1, args.runs + 1):
        for function_id in function_ids:
            current_job += 1
            function_wrapper = CEC2022FunctionWrapper(function_id=function_id, dimension=args.dimension)
            lower_bound, upper_bound = function_wrapper.get_bounds()
            LOGGER.info(
                "Job %d/%d | run=%d/%d function=%s",
                current_job,
                total_jobs,
                run_number,
                args.runs,
                function_wrapper.name,
            )

            function_start = time.perf_counter()

            result = SHO_LIME(
                pop=args.pop_size,
                max_iter=args.max_iter,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dim=args.dimension,
                fobj=function_wrapper,
                run_metadata={
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "function_name": function_wrapper.name,
                    "dimension": args.dimension,
                },
                random_state=args.seed + run_number * 100 + function_id,
                lime_every=effective_lime_every,
                min_samples_before_lime=min_samples_before_lime,
                lime_selection_mode=args.lime_selection_mode,
                progress_every=max(1, args.progress_every),
                verbose=not args.quiet,
                log_callback=LOGGER.info,
            )

            function_elapsed = time.perf_counter() - function_start
            LOGGER.info(
                "Fin function=%s | best=%.6e nfev=%d diagnosticos=%d tiempo=%.2fs",
                function_wrapper.name,
                result["best_fitness"],
                function_wrapper.nfev,
                len({row["diagnosis_id"] for row in result["lime_contribution_rows"]}),
                function_elapsed,
            )

            run_row = {
                "run_id": run_id,
                "timestamp": timestamp,
                "run_number": run_number,
                "function_name": function_wrapper.name,
                "dimension": args.dimension,
                "best_fitness": result["best_fitness"],
                "best_position": result["best_position"],
                "nfev": function_wrapper.nfev,
                "diagnoses_count": len({row["diagnosis_id"] for row in result["lime_contribution_rows"]}),
            }
            all_run_rows.append(run_row)

            for row in result["full_output_rows"]:
                row["run_number"] = run_number
                all_full_output_rows.append(row)

            for row in result["lime_contribution_rows"]:
                row["run_number"] = run_number
                all_contribution_rows.append(row)

            global_rows = aggregate_global_feature_explanations(
                contribution_rows=result["lime_contribution_rows"],
                run_id=run_id,
                function_name=function_wrapper.name,
                dimension=args.dimension,
            )
            for row in global_rows:
                row["run_number"] = run_number
                all_global_rows.append(row)

            LOGGER.info(
                "Global explanations function=%s | filas=%d",
                function_wrapper.name,
                len(global_rows),
            )

    summary_rows = summarize_by_function(all_run_rows)

    write_csv(run_dir / "runs_raw.csv", all_run_rows)
    write_csv(run_dir / "full_output.csv", all_full_output_rows)
    write_csv(run_dir / "lime_contributions.csv", all_contribution_rows)
    write_csv(run_dir / "global_feature_explanations.csv", all_global_rows)
    write_csv(run_dir / "summary_by_function.csv", summary_rows)
    LOGGER.info(
        "CSV escritos | runs=%d full_output=%d contrib=%d global=%d summary=%d",
        len(all_run_rows),
        len(all_full_output_rows),
        len(all_contribution_rows),
        len(all_global_rows),
        len(summary_rows),
    )

    write_json(
        run_dir / "config_used.json",
        {
            "algorithm": "SHOA-LIME",
            "run_id": run_id,
            "timestamp": timestamp,
            "functions": function_ids,
            "dimension": args.dimension,
            "pop_size": args.pop_size,
            "max_iter": args.max_iter,
            "runs": args.runs,
            "seed": args.seed,
            "lime_every": effective_lime_every,
            "lime_every_input": args.lime_every,
            "lime_every_strategy": "5_percent_of_max_iter_computed_once_at_start_when_not_provided",
            "lime_min_samples": args.lime_min_samples,
            "lime_selection_mode": args.lime_selection_mode,
            "progress_every": args.progress_every,
            "log_level": args.log_level,
            "quiet": args.quiet,
            "selection": {
                "base": "10_percent_pop",
                "distribution": "4-3-2-1",
                "minimum_per_category": 1,
                "overflow_policy": "allow_above_10_percent_if_needed",
                "lime_selection_mode": args.lime_selection_mode,
                "modes": {
                    "selected_agents": "explica todos los agentes seleccionados",
                    "medoid": "explica un representante medoid de los agentes seleccionados",
                },
            },
            "targets": ["classification_improved", "regression_y_reg"],
            "artifacts": [
                "config_used.json",
                "runs_raw.csv",
                "full_output.csv",
                "lime_contributions.csv",
                "global_feature_explanations.csv",
                "summary_by_function.csv",
            ],
        },
    )

    LOGGER.info("Benchmark terminado | run_dir=%s", run_dir)

    print(f"Run directory: {run_dir}")
    print("Artifacts created:")
    print("- config_used.json")
    print("- runs_raw.csv")
    print("- full_output.csv")
    print("- lime_contributions.csv")
    print("- global_feature_explanations.csv")
    print("- summary_by_function.csv")


if __name__ == "__main__":
    main()
