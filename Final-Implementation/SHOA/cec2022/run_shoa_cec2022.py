"""Minimal SHOA runner for CEC2022 benchmarks."""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np

from SHO import SHO
from opfunu_wrapper import CEC2022FunctionWrapper, parse_function_ids
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json

LOGGER = logging.getLogger("SHOA-CEC2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal SHOA on CEC2022")
    parser.add_argument("--functions", type=str, default="all", help="all | comma-separated ids (1-12)")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--pop", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    run_id, ts, run_dir = create_run_directory(args.output_dir)

    function_ids = parse_function_ids(args.functions)

    runs_raw_rows: list[dict] = []
    full_output_rows: list[dict] = []

    for fid in function_ids:
        function_name = f"F{fid}2022"
        LOGGER.info("Running %s", function_name)

        for run_number in range(1, args.runs + 1):
            run_seed = args.seed + fid * 1000 + run_number
            np.random.seed(run_seed)

            fobj = CEC2022FunctionWrapper(function_id=fid, dimension=args.dim)
            lower_bound, upper_bound = fobj.get_bounds()

            start = time.perf_counter()
            best_fitness, best_position, convergence_curve, _, _, _ = SHO(
                args.pop,
                args.max_iter,
                lower_bound,
                upper_bound,
                args.dim,
                fobj,
            )
            elapsed = time.perf_counter() - start

            fes_used = int(getattr(fobj, "nfev", 0))

            runs_raw_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": ts,
                    "function_name": function_name,
                    "function_id": fid,
                    "dimension": args.dim,
                    "run_number": run_number,
                    "seed": run_seed,
                    "pop": args.pop,
                    "max_iter": args.max_iter,
                    "best_fitness": float(best_fitness),
                    "fes_used": fes_used,
                    "elapsed_seconds": float(elapsed),
                    "best_position": ";".join(f"{float(v):.8e}" for v in np.asarray(best_position, dtype=float).reshape(-1)),
                }
            )

            curve = np.asarray(convergence_curve, dtype=float).reshape(-1)
            curve_len = max(1, int(curve.size))
            for idx, value in enumerate(curve, start=1):
                fe_estimate = int(round((idx / curve_len) * fes_used))
                full_output_rows.append(
                    {
                        "run_id": run_id,
                        "timestamp": ts,
                        "function_name": function_name,
                        "function_id": fid,
                        "dimension": args.dim,
                        "run_number": run_number,
                        "iteration": idx,
                        "best_fitness_so_far": float(value),
                        "fe_estimate": fe_estimate,
                    }
                )

            LOGGER.info(
                "%s run %d/%d -> best %.6e | fes=%d | %.2fs",
                function_name,
                run_number,
                args.runs,
                float(best_fitness),
                fes_used,
                elapsed,
            )

    summary_rows = summarize_by_function(runs_raw_rows)

    config_payload = {
        "run_id": run_id,
        "timestamp": ts,
        "algorithm": "SHOA",
        "problem": "CEC2022",
        "config": {
            "functions": function_ids,
            "dimension": args.dim,
            "population": args.pop,
            "max_iter": args.max_iter,
            "runs": args.runs,
            "seed": args.seed,
        },
    }

    write_json(run_dir / "config_used.json", config_payload)
    write_csv(run_dir / "runs_raw.csv", runs_raw_rows)
    write_csv(run_dir / "full_output.csv", full_output_rows)
    write_csv(run_dir / "summary_by_function.csv", summary_rows)

    LOGGER.info("Artifacts written to %s", pathlib.Path(run_dir).resolve())


if __name__ == "__main__":
    main()
