"""Minimal SHOA runner for TMLAP instances."""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np

from SHO import SHO
from tmlap_problem import TMLAPObjective, parse_instance_selection
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json

LOGGER = logging.getLogger("SHOA-TMLAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal SHOA on TMLAP")
    parser.add_argument("--instances", type=str, default="all", help="all | comma-separated file names")
    parser.add_argument("--instance-dir", type=str, default=".")
    parser.add_argument("--pop", type=int, default=40)
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

    instance_paths = parse_instance_selection(args.instances, args.instance_dir)

    run_id, ts, run_dir = create_run_directory(args.output_dir)

    runs_raw_rows: list[dict] = []
    full_output_rows: list[dict] = []

    for instance_path in instance_paths:
        instance_name = instance_path.name
        LOGGER.info("Running %s", instance_name)

        for run_number in range(1, args.runs + 1):
            run_seed = args.seed + run_number + (abs(hash(instance_name)) % 10000)
            np.random.seed(run_seed)

            objective = TMLAPObjective(instance_path)
            lower_bound, upper_bound = objective.get_bounds()

            start = time.perf_counter()
            best_fitness, best_position, convergence_curve, _, _, _ = SHO(
                args.pop,
                args.max_iter,
                lower_bound,
                upper_bound,
                objective.dimension,
                objective,
            )
            elapsed = time.perf_counter() - start

            assignment = objective.decode(np.asarray(best_position, dtype=float))
            _, feasible_solution, base_cost, overflow_total, distance_violation = objective.evaluate_assignment(assignment)

            fes_used = int(getattr(objective, "nfev", 0))

            runs_raw_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": ts,
                    "function_name": instance_name,
                    "instance_name": instance_name,
                    "run_number": run_number,
                    "seed": run_seed,
                    "dimension": objective.dimension,
                    "hubs": int(objective.data.n_hubs),
                    "pop": args.pop,
                    "max_iter": args.max_iter,
                    "best_fitness": float(best_fitness),
                    "base_cost": float(base_cost),
                    "feasible_best_solution": int(feasible_solution),
                    "overflow_total": int(overflow_total),
                    "distance_violation": float(distance_violation),
                    "fes_used": fes_used,
                    "elapsed_seconds": float(elapsed),
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
                        "function_name": instance_name,
                        "instance_name": instance_name,
                        "run_number": run_number,
                        "iteration": idx,
                        "best_fitness_so_far": float(value),
                        "fe_estimate": fe_estimate,
                    }
                )

            LOGGER.info(
                "%s run %d/%d -> best %.6e | feasible=%s | base_cost=%.3f | fes=%d | %.2fs",
                instance_name,
                run_number,
                args.runs,
                float(best_fitness),
                "yes" if feasible_solution else "no",
                float(base_cost),
                fes_used,
                elapsed,
            )

    summary_rows = summarize_by_function(runs_raw_rows)

    config_payload = {
        "run_id": run_id,
        "timestamp": ts,
        "algorithm": "SHOA",
        "problem": "TMLAP",
        "config": {
            "instances": [p.name for p in instance_paths],
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
