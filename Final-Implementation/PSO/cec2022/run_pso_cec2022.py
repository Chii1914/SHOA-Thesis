"""Minimal continuous PSO runner for CEC2022 benchmarks."""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np

from opfunu_wrapper import CEC2022FunctionWrapper, parse_function_ids
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json

LOGGER = logging.getLogger("PSO-CEC2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal continuous PSO on CEC2022")
    parser.add_argument("--functions", type=str, default="all", help="all | comma-separated ids (1-12)")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w", type=float, default=0.7)
    parser.add_argument("--c1", type=float, default=1.7)
    parser.add_argument("--c2", type=float, default=1.7)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _to_bound_vec(value: float | np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(dim, arr.item(), dtype=float)
    return arr


def run_continuous_pso(
    *,
    fobj,
    dim: int,
    lower_bound,
    upper_bound,
    particles: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    lb = _to_bound_vec(lower_bound, dim)
    ub = _to_bound_vec(upper_bound, dim)

    positions = rng.uniform(lb, ub, size=(particles, dim))
    v_scale = np.maximum(1e-12, ub - lb)
    velocities = rng.uniform(-v_scale, v_scale, size=(particles, dim)) * 0.1

    pbest_pos = positions.copy()
    pbest_fit = np.array([fobj(ind) for ind in positions], dtype=float)

    g_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    convergence = np.zeros(max_iter, dtype=float)

    for t in range(max_iter):
        r1 = rng.random((particles, dim))
        r2 = rng.random((particles, dim))

        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        vmax = 0.2 * v_scale
        velocities = np.clip(velocities, -vmax, vmax)

        positions = np.clip(positions + velocities, lb, ub)

        fitness = np.array([fobj(ind) for ind in positions], dtype=float)
        improved = fitness < pbest_fit
        pbest_fit = np.where(improved, fitness, pbest_fit)
        pbest_pos[improved] = positions[improved]

        g_idx = int(np.argmin(pbest_fit))
        candidate = float(pbest_fit[g_idx])
        if candidate < gbest_fit:
            gbest_fit = candidate
            gbest_pos = pbest_pos[g_idx].copy()

        convergence[t] = gbest_fit

    return gbest_fit, gbest_pos, convergence


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
            fobj = CEC2022FunctionWrapper(function_id=fid, dimension=args.dim)
            lower_bound, upper_bound = fobj.get_bounds()

            start = time.perf_counter()
            best_fitness, best_position, convergence_curve = run_continuous_pso(
                fobj=fobj,
                dim=args.dim,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                particles=args.particles,
                max_iter=args.max_iter,
                w=args.w,
                c1=args.c1,
                c2=args.c2,
                seed=run_seed,
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
                    "particles": args.particles,
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
        "algorithm": "PSO",
        "problem": "CEC2022",
        "config": {
            "functions": function_ids,
            "dimension": args.dim,
            "particles": args.particles,
            "max_iter": args.max_iter,
            "runs": args.runs,
            "seed": args.seed,
            "w": args.w,
            "c1": args.c1,
            "c2": args.c2,
        },
    }

    write_json(run_dir / "config_used.json", config_payload)
    write_csv(run_dir / "runs_raw.csv", runs_raw_rows)
    write_csv(run_dir / "full_output.csv", full_output_rows)
    write_csv(run_dir / "summary_by_function.csv", summary_rows)

    LOGGER.info("Artifacts written to %s", pathlib.Path(run_dir).resolve())


if __name__ == "__main__":
    main()
