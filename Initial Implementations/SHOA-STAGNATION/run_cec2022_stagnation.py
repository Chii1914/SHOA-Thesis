"""Run SHOA-STAGNATION on CEC2022 and export run-timestamp artifacts."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from SHO_STAGNATION_Controller import SHO_STAGNATION, estimate_max_fes
from opfunu_wrapper import CEC2022FunctionWrapper, parse_function_ids
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json


LOGGER = logging.getLogger("SHOA-STAGNATION")


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SHOA-STAGNATION on CEC2022")
    parser.add_argument("--functions", type=str, default="all", help="all | 1-12 | 1,3,5")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-fes", type=int, default=0, help="0 => auto estimate from pop/max_iter")
    parser.add_argument("--min-sfes-ratio", type=float, default=0.04, help="Paper-style: 0.02, 0.04, 0.10")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1, help="Log de progreso cada N iteraciones")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument("--quiet", action="store_true", help="Desactiva log por iteracion dentro de SHO_STAGNATION")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    function_ids = parse_function_ids(args.functions)
    run_id, timestamp, run_dir = create_run_directory(args.output_root)
    total_jobs = args.runs * len(function_ids)
    current_job = 0

    LOGGER.info(
        "Inicio benchmark | run_id=%s functions=%s dim=%d pop=%d max_iter=%d runs=%d min_sfes_ratio=%.4f",
        run_id,
        function_ids,
        args.dimension,
        args.pop_size,
        args.max_iter,
        args.runs,
        args.min_sfes_ratio,
    )

    all_run_rows: list[dict] = []
    all_full_output_rows: list[dict] = []
    all_stagnation_history_rows: list[dict] = []
    all_stagnation_event_rows: list[dict] = []

    for run_number in range(1, args.runs + 1):
        for function_id in function_ids:
            current_job += 1
            function_wrapper = CEC2022FunctionWrapper(function_id=function_id, dimension=args.dimension)
            lower_bound, upper_bound = function_wrapper.get_bounds()

            effective_max_fes = int(args.max_fes) if int(args.max_fes) > 0 else estimate_max_fes(args.pop_size, args.max_iter)

            LOGGER.info(
                "Job %d/%d | run=%d/%d function=%s max_fes=%d",
                current_job,
                total_jobs,
                run_number,
                args.runs,
                function_wrapper.name,
                effective_max_fes,
            )

            function_start = time.perf_counter()
            result = SHO_STAGNATION(
                pop=args.pop_size,
                max_iter=args.max_iter,
                max_fes=effective_max_fes,
                min_sfes_ratio=args.min_sfes_ratio,
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
                progress_every=max(1, args.progress_every),
                verbose=not args.quiet,
                log_callback=LOGGER.info,
            )
            function_elapsed = time.perf_counter() - function_start

            stagnation_events = result["stagnation_event_rows"]
            first_stagnation_iter = None
            for row in stagnation_events:
                if row["event"] == "stagnation_start":
                    first_stagnation_iter = int(row["iteration"])
                    break

            LOGGER.info(
                "Fin function=%s | best=%.6e nfev=%d events=%d first_stag_iter=%s tiempo=%.2fs",
                function_wrapper.name,
                result["best_fitness"],
                function_wrapper.nfev,
                len(stagnation_events),
                str(first_stagnation_iter),
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
                "max_fes": result["stagnation_meta"]["max_fes"],
                "min_sfes_ratio": result["stagnation_meta"]["min_sfes_ratio"],
                "min_sfes": result["stagnation_meta"]["min_sfes"],
                "stagnation_events_count": len(stagnation_events),
                "first_stagnation_iteration": first_stagnation_iter,
                "stop_reason": result["stagnation_meta"]["stop_reason"],
            }
            all_run_rows.append(run_row)

            for row in result["full_output_rows"]:
                row["run_number"] = run_number
                all_full_output_rows.append(row)

            for row in result["stagnation_history_rows"]:
                row["run_number"] = run_number
                all_stagnation_history_rows.append(row)

            for row in result["stagnation_event_rows"]:
                row["run_number"] = run_number
                all_stagnation_event_rows.append(row)

    summary_rows = summarize_by_function(all_run_rows)

    write_csv(run_dir / "runs_raw.csv", all_run_rows)
    write_csv(run_dir / "full_output.csv", all_full_output_rows)
    write_csv(run_dir / "stagnation_history.csv", all_stagnation_history_rows)
    write_csv(run_dir / "stagnation_events.csv", all_stagnation_event_rows)
    write_csv(run_dir / "summary_by_function.csv", summary_rows)

    LOGGER.info(
        "CSV escritos | runs=%d full_output=%d stagnation_history=%d stagnation_events=%d summary=%d",
        len(all_run_rows),
        len(all_full_output_rows),
        len(all_stagnation_history_rows),
        len(all_stagnation_event_rows),
        len(summary_rows),
    )

    write_json(
        run_dir / "config_used.json",
        {
            "algorithm": "SHOA-STAGNATION",
            "run_id": run_id,
            "timestamp": timestamp,
            "functions": function_ids,
            "dimension": args.dimension,
            "pop_size": args.pop_size,
            "max_iter": args.max_iter,
            "max_fes": args.max_fes,
            "runs": args.runs,
            "seed": args.seed,
            "progress_every": args.progress_every,
            "log_level": args.log_level,
            "quiet": args.quiet,
            "stagnation": {
                "method": "MinSFEs_MaxFEs_paper_style",
                "min_sfes_ratio": args.min_sfes_ratio,
                "auto_max_fes_if_zero": estimate_max_fes(args.pop_size, args.max_iter),
            },
            "artifacts": [
                "config_used.json",
                "runs_raw.csv",
                "full_output.csv",
                "stagnation_history.csv",
                "stagnation_events.csv",
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
    print("- stagnation_history.csv")
    print("- stagnation_events.csv")
    print("- summary_by_function.csv")


if __name__ == "__main__":
    main()
