"""PSO runner for TMLAP with lightweight and legacy execution modes."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import time
from pathlib import Path

import numpy as np

from tmlap_problem import TMLAPObjective, parse_instance_selection
from utils_logging import create_run_directory, summarize_by_function, write_csv, write_json

LOGGER = logging.getLogger("PSO-TMLAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSO on TMLAP (lightweight or legacy)")
    parser.add_argument("--mode", choices=["light", "legacy"], default="light")

    parser.add_argument("--instances", type=str, default="all", help="all | comma-separated file names")
    parser.add_argument("--instance-dir", type=str, default=str(Path(__file__).parent))
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


def _load_legacy_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("pso_tmlap_legacy", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_legacy(args: argparse.Namespace) -> None:
    script_path = Path(__file__).with_name("PSO-TMLAP.py")
    module = _load_legacy_module(script_path)

    solver = module.PSO()
    solver.max_iter = int(args.max_iter)
    solver.n_particles = int(args.particles)
    solver.solve()


def run_continuous_pso(
    *,
    fobj,
    dim: int,
    lower_bound: float,
    upper_bound: float,
    particles: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    lb = np.full(dim, float(lower_bound), dtype=float)
    ub = np.full(dim, float(upper_bound), dtype=float)

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


def _run_light(args: argparse.Namespace) -> None:
    run_id, ts, run_dir = create_run_directory(args.output_dir)
    instance_paths = parse_instance_selection(args.instances, args.instance_dir)

    runs_raw_rows: list[dict] = []
    full_output_rows: list[dict] = []

    for instance_path in instance_paths:
        instance_name = instance_path.name
        LOGGER.info("Running %s", instance_name)

        for run_number in range(1, args.runs + 1):
            run_seed = args.seed + run_number + (abs(hash(instance_name)) % 10000)

            objective = TMLAPObjective(instance_path)
            lower_bound, upper_bound = objective.get_bounds()

            start = time.perf_counter()
            best_fitness, best_position, convergence_curve = run_continuous_pso(
                fobj=objective,
                dim=objective.dimension,
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
                    "particles": args.particles,
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
        "algorithm": "PSO",
        "problem": "TMLAP",
        "mode": "light",
        "config": {
            "instances": [p.name for p in instance_paths],
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


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.mode == "legacy":
        _run_legacy(args)
        return

    _run_light(args)


if __name__ == "__main__":
    main()
