"""Benchmark runner for pure SHOA across CEC2022 functions (no LIME)."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from base.SHO_LIME_Controller import build_opfunu_cec_objective

CEC2022_CLASS_NAMES = tuple(f"F{i}2022" for i in range(1, 13))


def _load_sho_function():
    current_dir = Path(__file__).resolve().parent
    candidate_roots = [current_dir.parent.parent, current_dir.parent]
    sho_path = None

    for root in candidate_roots:
        candidate = root / "python-code" / "SHO.py"
        if candidate.exists():
            sho_path = candidate
            break

    if sho_path is None:
        checked = ", ".join(str(root / "python-code" / "SHO.py") for root in candidate_roots)
        raise FileNotFoundError(f"No se encontro el modulo base SHO. Rutas revisadas: {checked}")

    helper_dir = str(sho_path.parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)

    spec = importlib.util.spec_from_file_location("sho_core_module_for_cec_benchmark", sho_path)
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo crear el spec para cargar SHO.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sho_fn = getattr(module, "SHO", None)
    if sho_fn is None:
        raise AttributeError("El modulo SHO.py no expone la funcion SHO")

    return sho_fn


SHO = _load_sho_function()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta benchmark SHOA puro sobre CEC2022 (F1..F12) "
            "con protocolo tipo paper de SHO por defecto."
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

    # Protocolo paper-like SHO (defaults)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)

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


def main() -> None:
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("--runs debe ser > 0")
    if args.max_iter <= 0:
        raise ValueError("--max-iter debe ser > 0")
    if args.pop_size < 4:
        raise ValueError("--pop-size debe ser >= 4")

    functions = _resolve_functions(args.functions)
    dims = _resolve_dims(args.dims)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_tag = "-".join(str(d) for d in dims)
    run_name = f"cec2022_shoa_puro_d{dim_tag}_r{args.runs}_{stamp}"
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

    # Keep the same output structure as SHO+LIME benchmark for easy side-by-side comparison.
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

        # SHOA puro no invoca LIME, se deja archivo vacio con header por compatibilidad.
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

                    np.random.seed(seed)

                    t0 = time.time()
                    best_fitness, _, convergence_curve, *_ = SHO(
                        pop=args.pop_size,
                        Max_iter=args.max_iter,
                        LB=lb,
                        UB=ub,
                        Dim=used_dim,
                        fobj=objective,
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
        "algorithm": "SHOA_PURO",
        "functions": functions,
        "dims": dims,
        "runs": args.runs,
        "seed_start": args.seed_start,
        "paper_like_protocol": {
            "pop_size": args.pop_size,
            "max_iter": args.max_iter,
        },
        "traceability_files": {
            "full_output_csv": full_output_path.name,
            "lime_contributions_csv": lime_contributions_path.name,
        },
        "notes": {
            "diagnostics": "No aplica en SHOA puro",
            "rescues": "No aplica en SHOA puro",
            "lime_contributions": "Archivo creado vacio (solo header) para compatibilidad",
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print("\nBenchmark SHOA puro terminado.")
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
