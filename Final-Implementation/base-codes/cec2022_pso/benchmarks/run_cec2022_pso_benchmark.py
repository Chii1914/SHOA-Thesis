"""Benchmark runner for PSO across CEC2022 functions.

Output contract is aligned with SHOA and SHO+LIME CEC2022 runners:
- runs_raw.csv
- full_output.csv
- lime_contributions.csv (header-only for compatibility)
- summary_by_function.csv
- ranking_by_dimension.csv
- config_used.json
- skipped_cases.csv (if needed)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CEC2022_CLASS_NAMES = tuple(f"F{i}2022" for i in range(1, 13))


def _bounds_vector(value, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(ndim, float(arr[0]), dtype=float)
    if arr.size != ndim:
        raise ValueError(f"Bounds size mismatch: expected {ndim}, got {arr.size}")
    return arr.astype(float)


def build_opfunu_cec_objective(function_class_name: str, ndim: int):
    try:
        import opfunu
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise RuntimeError("Opfunu requiere pkg_resources; instala setuptools==75.8.0") from exc
        raise RuntimeError("No se pudo importar opfunu. Ejecuta: pip install opfunu") from exc

    funcs = opfunu.get_functions_by_classname(function_class_name)
    if not funcs:
        raise ValueError(f"No existe la clase {function_class_name} en opfunu")

    cls = funcs[0]
    try:
        problem = cls(ndim=ndim)
    except Exception as exc:
        raise ValueError(f"No se pudo instanciar {function_class_name} con ndim={ndim}: {exc}") from exc

    lb_vec = _bounds_vector(problem.lb, problem.ndim)
    ub_vec = _bounds_vector(problem.ub, problem.ndim)

    def objective(x: np.ndarray) -> float:
        return float(problem.evaluate(np.asarray(x, dtype=float).reshape(-1)))

    return lb_vec, ub_vec, int(problem.ndim), objective, problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta benchmark PSO sobre CEC2022 (F1..F12) con protocolo "
            "comparable contra SHOA y SHO+LIME."
        )
    )
    parser.add_argument(
        "--functions",
        default="all",
        help="Funciones separadas por coma (ej: F12022,F22022) o 'all'",
    )
    parser.add_argument(
        "--dims",
        default="10",
        help="Dimensiones separadas por coma (por defecto: 10)",
    )
    parser.add_argument("--runs", type=int, default=30, help="Corridas por funcion/dimension")
    parser.add_argument("--seed-start", type=int, default=1, help="Semilla inicial")

    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)

    parser.add_argument("--inertia", type=float, default=0.72)
    parser.add_argument("--cognitive", type=float, default=1.70)
    parser.add_argument("--social", type=float, default=1.70)
    parser.add_argument("--v-max-frac", type=float, default=0.25)

    parser.add_argument(
        "--output-dir",
        default="benchmark_logs",
        help="Directorio raiz de salida",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Sufijo opcional para la carpeta de salida",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Detener ejecucion si falla alguna funcion/dimension",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_functions(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(CEC2022_CLASS_NAMES)

    requested = _parse_csv_list(raw)
    invalid = [name for name in requested if name not in CEC2022_CLASS_NAMES]
    if invalid:
        allowed = ", ".join(CEC2022_CLASS_NAMES)
        bad = ", ".join(invalid)
        raise ValueError(f"Funciones invalidas: {bad}. Disponibles: {allowed}")

    return requested


def _resolve_dims(raw: str) -> list[int]:
    dims = [int(item) for item in _parse_csv_list(raw)]
    if not dims:
        raise ValueError("Debes indicar al menos una dimension en --dims")
    if any(dim <= 0 for dim in dims):
        raise ValueError("Todas las dimensiones en --dims deben ser > 0")
    return dims


def _as_float_scalar(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _run_pso(
    objective,
    lb: np.ndarray,
    ub: np.ndarray,
    dim: int,
    pop_size: int,
    max_iter: int,
    inertia: float,
    cognitive: float,
    social: float,
    v_max_frac: float,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    if lb.size == 1:
        lb = np.full(dim, lb[0], dtype=float)
    if ub.size == 1:
        ub = np.full(dim, ub[0], dtype=float)

    span = np.maximum(ub - lb, 1e-12)
    vmax = v_max_frac * span

    pos = rng.uniform(lb, ub, size=(pop_size, dim))
    vel = rng.uniform(-vmax, vmax, size=(pop_size, dim))

    pbest_pos = pos.copy()
    pbest_fit = np.array([objective(p) for p in pbest_pos], dtype=float)

    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    convergence = np.zeros(max_iter, dtype=float)

    for iteration in range(max_iter):
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        vel = (
            inertia * vel
            + cognitive * r1 * (pbest_pos - pos)
            + social * r2 * (gbest_pos.reshape(1, -1) - pos)
        )
        vel = np.clip(vel, -vmax, vmax)

        pos = np.clip(pos + vel, lb, ub)
        fit = np.array([objective(p) for p in pos], dtype=float)

        improve_mask = fit < pbest_fit
        pbest_pos[improve_mask] = pos[improve_mask]
        pbest_fit[improve_mask] = fit[improve_mask]

        current_best_idx = int(np.argmin(pbest_fit))
        current_best_fit = float(pbest_fit[current_best_idx])
        if current_best_fit < gbest_fit:
            gbest_fit = current_best_fit
            gbest_pos = pbest_pos[current_best_idx].copy()

        convergence[iteration] = gbest_fit

    return gbest_fit, gbest_pos, convergence


def main() -> None:
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("--runs debe ser > 0")
    if args.max_iter <= 0:
        raise ValueError("--max-iter debe ser > 0")
    if args.pop_size < 4:
        raise ValueError("--pop-size debe ser >= 4")
    if args.v_max_frac <= 0:
        raise ValueError("--v-max-frac debe ser > 0")

    functions = _resolve_functions(args.functions)
    dims = _resolve_dims(args.dims)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_tag = "-".join(str(d) for d in dims)
    run_name = f"cec2022_pso_d{dim_tag}_r{args.runs}_{stamp}"
    if args.tag.strip():
        run_name = f"{run_name}_{args.tag.strip()}"

    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {out_root}")
    print(f"Functions: {functions}")
    print(f"Dimensions: {dims}")
    print(f"Runs per function/dim: {args.runs}")

    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    skipped_rows: list[dict] = []

    full_output_path = out_root / "full_output.csv"
    full_output_fieldnames = [
        "function",
        "dimension",
        "run_id",
        "seed",
        "iteration",
        "best_fitness",
        "global_optimum",
        "error_to_optimum",
        "window_size",
        "window_std",
        "trigger_candidate",
        "diagnostics_invoked",
        "diagnosis_status",
        "diagnosis_pred_delta",
        "diagnosis_fidelity",
        "rescue_applied",
        "rescue_count_cumulative",
        "cooldown_counter",
    ]

    lime_contributions_path = out_root / "lime_contributions.csv"
    lime_contributions_fieldnames = [
        "function",
        "dimension",
        "run_id",
        "seed",
        "diagnosis_id",
        "diagnosis_iteration",
        "diagnosis_status",
        "pred_delta",
        "fidelity",
        "strong_stochastic_importance",
        "low_expected_improvement",
        "weight_r1",
        "weight_mag_browniano",
        "weight_mag_levy",
        "weight_r2",
        "weight_mag_predacion",
        "abs_weight_r1",
        "abs_weight_mag_browniano",
        "abs_weight_mag_levy",
        "abs_weight_r2",
        "abs_weight_mag_predacion",
    ]

    total_jobs = len(functions) * len(dims) * args.runs
    finished_jobs = 0

    with (
        full_output_path.open("w", newline="", encoding="utf-8") as full_output_file,
        lime_contributions_path.open("w", newline="", encoding="utf-8") as lime_contributions_file,
    ):
        full_output_writer = csv.DictWriter(full_output_file, fieldnames=full_output_fieldnames)
        full_output_writer.writeheader()

        lime_contributions_writer = csv.DictWriter(
            lime_contributions_file, fieldnames=lime_contributions_fieldnames
        )
        lime_contributions_writer.writeheader()

        for dim in dims:
            for function_name in functions:
                print(f"\n=== Benchmark {function_name} | dim={dim} ===")

                try:
                    lb, ub, used_dim, objective, problem = build_opfunu_cec_objective(function_name, dim)
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "function": function_name,
                            "requested_dim": dim,
                            "error": str(exc),
                        }
                    )
                    print(f"SKIP {function_name} dim={dim}: {exc}")
                    if args.stop_on_error:
                        raise
                    continue

                f_global = _as_float_scalar(getattr(problem, "f_global", np.nan))
                per_case_best: list[float] = []
                per_case_runtime: list[float] = []
                per_case_rescues: list[int] = []
                per_case_diagnostics: list[int] = []
                per_case_positive_diag: list[int] = []

                for run_idx in range(1, args.runs + 1):
                    seed = args.seed_start + run_idx - 1

                    t0 = time.time()
                    best_fitness, _best_position, convergence_curve = _run_pso(
                        objective=objective,
                        lb=lb,
                        ub=ub,
                        dim=used_dim,
                        pop_size=args.pop_size,
                        max_iter=args.max_iter,
                        inertia=args.inertia,
                        cognitive=args.cognitive,
                        social=args.social,
                        v_max_frac=args.v_max_frac,
                        seed=seed,
                    )
                    runtime = time.time() - t0

                    best_fitness = float(best_fitness)
                    convergence_curve = np.asarray(convergence_curve, dtype=float).reshape(-1)
                    error_to_optimum = best_fitness - f_global if np.isfinite(f_global) else float("nan")

                    run_rows.append(
                        {
                            "function": function_name,
                            "dimension": used_dim,
                            "run_id": run_idx,
                            "seed": seed,
                            "best_fitness": best_fitness,
                            "global_optimum": f_global,
                            "error_to_optimum": error_to_optimum,
                            "rescues_applied": 0,
                            "diagnostics_invoked": 0,
                            "diagnostics_invocation_count": 0,
                            "positive_diagnosis_count": 0,
                            "runtime_sec": runtime,
                            "diagnostics_invocation_iterations": "",
                        }
                    )

                    for iteration, iter_best_fitness in enumerate(convergence_curve, start=1):
                        iter_best_fitness = float(iter_best_fitness)
                        iter_error_to_optimum = (
                            iter_best_fitness - f_global if np.isfinite(f_global) else float("nan")
                        )

                        full_output_writer.writerow(
                            {
                                "function": function_name,
                                "dimension": used_dim,
                                "run_id": run_idx,
                                "seed": seed,
                                "iteration": iteration,
                                "best_fitness": iter_best_fitness,
                                "global_optimum": f_global,
                                "error_to_optimum": iter_error_to_optimum,
                                "window_size": 0,
                                "window_std": float("nan"),
                                "trigger_candidate": False,
                                "diagnostics_invoked": False,
                                "diagnosis_status": "NONE",
                                "diagnosis_pred_delta": float("nan"),
                                "diagnosis_fidelity": float("nan"),
                                "rescue_applied": False,
                                "rescue_count_cumulative": 0,
                                "cooldown_counter": 0,
                            }
                        )

                    per_case_best.append(best_fitness)
                    per_case_runtime.append(runtime)
                    per_case_rescues.append(0)
                    per_case_diagnostics.append(0)
                    per_case_positive_diag.append(0)

                    finished_jobs += 1
                    print(
                        f"[{finished_jobs}/{total_jobs}] {function_name} d={used_dim} run={run_idx}/{args.runs} "
                        f"seed={seed} best={best_fitness:.6e} rescues=0 diag=0 time={runtime:.2f}s"
                    )

                summary_rows.append(
                    {
                        "function": function_name,
                        "dimension": used_dim,
                        "runs": args.runs,
                        "global_optimum": f_global,
                        "best": float(min(per_case_best)),
                        "worst": float(max(per_case_best)),
                        "mean": float(statistics.mean(per_case_best)),
                        "median": float(statistics.median(per_case_best)),
                        "std": _compute_std(per_case_best),
                        "mean_error_to_optimum": (
                            float(statistics.mean([value - f_global for value in per_case_best]))
                            if np.isfinite(f_global)
                            else float("nan")
                        ),
                        "mean_runtime_sec": float(statistics.mean(per_case_runtime)),
                        "mean_rescues": float(statistics.mean(per_case_rescues)),
                        "mean_diagnostics_invoked": float(statistics.mean(per_case_diagnostics)),
                        "mean_positive_diagnosis": float(statistics.mean(per_case_positive_diag)),
                    }
                )

    rank_rows: list[dict] = []
    for dim in dims:
        dim_rows = [row for row in summary_rows if int(row["dimension"]) == int(dim)]
        dim_rows_sorted = sorted(dim_rows, key=lambda row: float(row["mean"]))
        for rank, row in enumerate(dim_rows_sorted, start=1):
            rank_rows.append(
                {
                    "dimension": dim,
                    "rank": rank,
                    "function": row["function"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "best": row["best"],
                    "worst": row["worst"],
                }
            )

    runs_path = out_root / "runs_raw.csv"
    summary_path = out_root / "summary_by_function.csv"
    ranking_path = out_root / "ranking_by_dimension.csv"
    skipped_path = out_root / "skipped_cases.csv"
    config_path = out_root / "config_used.json"

    if run_rows:
        _write_csv(
            runs_path,
            run_rows,
            [
                "function",
                "dimension",
                "run_id",
                "seed",
                "best_fitness",
                "global_optimum",
                "error_to_optimum",
                "rescues_applied",
                "diagnostics_invoked",
                "diagnostics_invocation_count",
                "positive_diagnosis_count",
                "runtime_sec",
                "diagnostics_invocation_iterations",
            ],
        )

    if summary_rows:
        _write_csv(
            summary_path,
            summary_rows,
            [
                "function",
                "dimension",
                "runs",
                "global_optimum",
                "best",
                "worst",
                "mean",
                "median",
                "std",
                "mean_error_to_optimum",
                "mean_runtime_sec",
                "mean_rescues",
                "mean_diagnostics_invoked",
                "mean_positive_diagnosis",
            ],
        )

    if rank_rows:
        _write_csv(
            ranking_path,
            rank_rows,
            ["dimension", "rank", "function", "mean", "std", "best", "worst"],
        )

    if skipped_rows:
        _write_csv(skipped_path, skipped_rows, ["function", "requested_dim", "error"])

    config_payload = {
        "timestamp": stamp,
        "algorithm": "PSO",
        "functions": functions,
        "dims": dims,
        "runs": args.runs,
        "seed_start": args.seed_start,
        "paper_like_protocol": {
            "pop_size": args.pop_size,
            "max_iter": args.max_iter,
        },
        "pso_controller": {
            "inertia": args.inertia,
            "cognitive": args.cognitive,
            "social": args.social,
            "v_max_frac": args.v_max_frac,
        },
        "traceability_files": {
            "full_output_csv": full_output_path.name,
            "lime_contributions_csv": lime_contributions_path.name,
        },
        "notes": {
            "diagnostics": "No aplica en PSO",
            "rescues": "No aplica en PSO",
            "lime_contributions": "Archivo creado vacio (solo header) para compatibilidad",
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print("\nBenchmark PSO CEC2022 terminado.")
    if run_rows:
        print(f"Runs raw CSV: {runs_path}")
    print(f"Full output CSV: {full_output_path}")
    print(f"LIME contributions CSV (vacio por compatibilidad): {lime_contributions_path}")
    if summary_rows:
        print(f"Summary CSV: {summary_path}")
        print(f"Ranking CSV: {ranking_path}")
    if skipped_rows:
        print(f"Skipped cases CSV: {skipped_path}")
    print(f"Config JSON: {config_path}")


if __name__ == "__main__":
    main()
