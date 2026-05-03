"""Benchmark runner for SHO + LIME across CEC2022 functions."""

from __future__ import annotations

import argparse
import csv
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

from base.SHO_LIME_Controller import SHOXAIConfig, SHO_with_lime_controller, build_opfunu_cec_objective

CEC2022_CLASS_NAMES = tuple(f"F{i}2022" for i in range(1, 13))
LIME_FEATURE_NAMES = ("r1", "mag_browniano", "mag_levy", "r2", "mag_predacion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta benchmark SHO+LIME sobre CEC2022 (F1..F12) con protocolo "
            "SHO y parametros agresivos por defecto para el controlador LIME."
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

    # Configuracion SHO + LIME (editable)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=8e-4)
    parser.add_argument("--cooldown", type=int, default=6)
    parser.add_argument("--lime-samples", type=int, default=1200)
    parser.add_argument("--importance-threshold", type=float, default=0.035)
    parser.add_argument("--delta-tolerance", type=float, default=8e-5)
    parser.add_argument("--fidelity-threshold", type=float, default=0.10)
    parser.add_argument(
        "--rescue-mode",
        choices=["levy_teleport", "leader_repulsion"],
        default="leader_repulsion",
    )
    parser.add_argument("--rescue-eta", type=float, default=1.0)
    parser.add_argument("--rescue-levy-scale", type=float, default=0.30)
    parser.add_argument("--rescue-patience-iters", type=int, default=25)
    parser.add_argument("--rescue-min-improvement", type=float, default=0.0)
    parser.add_argument(
        "--enforce-elite-archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserva e inyecta la mejor solucion historica para evitar amnesia",
    )

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
    if args.rescue_patience_iters <= 0:
        raise ValueError("--rescue-patience-iters debe ser > 0")

    functions = _resolve_functions(args.functions)
    dims = _resolve_dims(args.dims)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_tag = "-".join(str(d) for d in dims)
    run_name = f"cec2022_sholime_d{dim_tag}_r{args.runs}_{stamp}"
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
        "rescue_trial_active",
        "rescue_rollback_applied",
        "rollback_count_cumulative",
        "cooldown_counter",
        "elite_best_fitness",
        "diagnosis_id",
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

    lime_contributions_path = out_root / "lime_contributions.csv"
    lime_contributions_fieldnames = [
        "function",
        "dimension",
        "run_id",
        "seed",
        "max_iter",
        "pop_size",
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
                    cfg = SHOXAIConfig(
                        pop_size=args.pop_size,
                        max_iter=args.max_iter,
                        window_size=args.window_size,
                        epsilon_stagnation=args.epsilon,
                        cooldown_iters=args.cooldown,
                        lime_num_samples=args.lime_samples,
                        importance_threshold=args.importance_threshold,
                        delta_tolerance=args.delta_tolerance,
                        fidelity_threshold=args.fidelity_threshold,
                        rescue_mode=args.rescue_mode,
                        rescue_eta=args.rescue_eta,
                        rescue_levy_scale=args.rescue_levy_scale,
                        rescue_patience_iters=args.rescue_patience_iters,
                        rescue_min_improvement=args.rescue_min_improvement,
                        enforce_elite_archive=bool(args.enforce_elite_archive),
                        seed=seed,
                    )

                    t0 = time.time()
                    result = SHO_with_lime_controller(objective, lb, ub, used_dim, cfg)
                    runtime = time.time() - t0

                    positive_diag_count = sum(
                        1 for diag in result.diagnostics_log if diag.get("status") == "POSITIVE_STAGNATION"
                    )
                    diagnostics_invocation_count = len(result.diagnostics_invocation_iterations)
                    best_fitness = float(result.best_fitness)
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
                            "rescues_applied": int(result.rescue_count),
                            "diagnostics_invoked": len(result.diagnostics_log),
                            "diagnostics_invocation_count": diagnostics_invocation_count,
                            "positive_diagnosis_count": positive_diag_count,
                            "runtime_sec": runtime,
                            "diagnostics_invocation_iterations": ";".join(
                                str(i) for i in result.diagnostics_invocation_iterations
                            ),
                        }
                    )

                    diagnosis_by_iteration: dict[int, dict] = {}

                    for diag_idx, diag in enumerate(result.diagnostics_log, start=1):
                        weights = diag.get("weights", {})
                        w_r1 = float(weights.get("r1", 0.0))
                        w_brown = float(weights.get("mag_browniano", 0.0))
                        w_levy = float(weights.get("mag_levy", 0.0))
                        w_r2 = float(weights.get("r2", 0.0))
                        w_pred = float(weights.get("mag_predacion", 0.0))

                        diagnosis_iteration = int(diag.get("iteration", 0))
                        strong_stochastic_importance = bool(
                            diag.get("strong_stochastic_importance", False)
                        )
                        low_expected_improvement = bool(diag.get("low_expected_improvement", False))

                        diagnosis_by_iteration[diagnosis_iteration] = {
                            "diagnosis_id": diag_idx,
                            "strong_stochastic_importance": strong_stochastic_importance,
                            "low_expected_improvement": low_expected_improvement,
                            "weight_r1": w_r1,
                            "weight_mag_browniano": w_brown,
                            "weight_mag_levy": w_levy,
                            "weight_r2": w_r2,
                            "weight_mag_predacion": w_pred,
                            "abs_weight_r1": abs(w_r1),
                            "abs_weight_mag_browniano": abs(w_brown),
                            "abs_weight_mag_levy": abs(w_levy),
                            "abs_weight_r2": abs(w_r2),
                            "abs_weight_mag_predacion": abs(w_pred),
                        }

                        lime_contributions_writer.writerow(
                            {
                                "function": function_name,
                                "dimension": used_dim,
                                "run_id": run_idx,
                                "seed": seed,
                                "max_iter": args.max_iter,
                                "pop_size": args.pop_size,
                                "diagnosis_id": diag_idx,
                                "diagnosis_iteration": diagnosis_iteration,
                                "diagnosis_status": str(diag.get("status", "UNKNOWN")),
                                "pred_delta": float(diag.get("pred_delta", np.nan)),
                                "fidelity": float(diag.get("fidelity", np.nan)),
                                "strong_stochastic_importance": strong_stochastic_importance,
                                "low_expected_improvement": low_expected_improvement,
                                "weight_r1": w_r1,
                                "weight_mag_browniano": w_brown,
                                "weight_mag_levy": w_levy,
                                "weight_r2": w_r2,
                                "weight_mag_predacion": w_pred,
                                "abs_weight_r1": abs(w_r1),
                                "abs_weight_mag_browniano": abs(w_brown),
                                "abs_weight_mag_levy": abs(w_levy),
                                "abs_weight_r2": abs(w_r2),
                                "abs_weight_mag_predacion": abs(w_pred),
                            }
                        )

                    rescue_count_cumulative = 0
                    for iter_row in result.iteration_log:
                        if bool(iter_row.get("rescue_applied", False)):
                            rescue_count_cumulative += 1

                        iteration = int(iter_row.get("iteration", 0))
                        iter_best_fitness = float(iter_row.get("best_fitness", np.nan))
                        iter_error_to_optimum = (
                            iter_best_fitness - f_global if np.isfinite(f_global) else float("nan")
                        )
                        iter_diag = diagnosis_by_iteration.get(iteration, {})

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
                                "window_size": int(iter_row.get("window_size", 0)),
                                "window_std": float(iter_row.get("window_std", np.nan)),
                                "trigger_candidate": bool(iter_row.get("trigger_candidate", False)),
                                "diagnostics_invoked": bool(iter_row.get("diagnostics_invoked", False)),
                                "diagnosis_status": str(iter_row.get("diagnosis_status", "NONE")),
                                "diagnosis_pred_delta": float(iter_row.get("diagnosis_pred_delta", np.nan)),
                                "diagnosis_fidelity": float(iter_row.get("diagnosis_fidelity", np.nan)),
                                "rescue_applied": bool(iter_row.get("rescue_applied", False)),
                                "rescue_count_cumulative": rescue_count_cumulative,
                                "rescue_trial_active": bool(iter_row.get("rescue_trial_active", False)),
                                "rescue_rollback_applied": bool(
                                    iter_row.get("rescue_rollback_applied", False)
                                ),
                                "rollback_count_cumulative": int(
                                    iter_row.get("rollback_count_cumulative", 0)
                                ),
                                "cooldown_counter": int(iter_row.get("cooldown_counter", 0)),
                                "elite_best_fitness": float(
                                    iter_row.get("elite_best_fitness", iter_best_fitness)
                                ),
                                "diagnosis_id": int(iter_diag.get("diagnosis_id", 0)),
                                "strong_stochastic_importance": bool(
                                    iter_diag.get("strong_stochastic_importance", False)
                                ),
                                "low_expected_improvement": bool(
                                    iter_diag.get("low_expected_improvement", False)
                                ),
                                "weight_r1": float(iter_diag.get("weight_r1", np.nan)),
                                "weight_mag_browniano": float(
                                    iter_diag.get("weight_mag_browniano", np.nan)
                                ),
                                "weight_mag_levy": float(iter_diag.get("weight_mag_levy", np.nan)),
                                "weight_r2": float(iter_diag.get("weight_r2", np.nan)),
                                "weight_mag_predacion": float(
                                    iter_diag.get("weight_mag_predacion", np.nan)
                                ),
                                "abs_weight_r1": float(iter_diag.get("abs_weight_r1", np.nan)),
                                "abs_weight_mag_browniano": float(
                                    iter_diag.get("abs_weight_mag_browniano", np.nan)
                                ),
                                "abs_weight_mag_levy": float(iter_diag.get("abs_weight_mag_levy", np.nan)),
                                "abs_weight_r2": float(iter_diag.get("abs_weight_r2", np.nan)),
                                "abs_weight_mag_predacion": float(
                                    iter_diag.get("abs_weight_mag_predacion", np.nan)
                                ),
                            }
                        )

                    per_case_best.append(best_fitness)
                    per_case_runtime.append(runtime)
                    per_case_rescues.append(int(result.rescue_count))
                    per_case_diagnostics.append(len(result.diagnostics_log))
                    per_case_positive_diag.append(positive_diag_count)

                    finished_jobs += 1
                    print(
                        f"[{finished_jobs}/{total_jobs}] {function_name} d={used_dim} run={run_idx}/{args.runs} "
                        f"seed={seed} best={best_fitness:.6e} rescues={result.rescue_count} "
                        f"diag={len(result.diagnostics_log)} time={runtime:.2f}s"
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
        "functions": functions,
        "dims": dims,
        "runs": args.runs,
        "seed_start": args.seed_start,
        "paper_like_protocol": {
            "pop_size": args.pop_size,
            "max_iter": args.max_iter,
        },
        "sholime_controller": {
            "window_size": args.window_size,
            "epsilon_stagnation": args.epsilon,
            "cooldown_iters": args.cooldown,
            "lime_num_samples": args.lime_samples,
            "importance_threshold": args.importance_threshold,
            "delta_tolerance": args.delta_tolerance,
            "fidelity_threshold": args.fidelity_threshold,
            "rescue_mode": args.rescue_mode,
            "rescue_eta": args.rescue_eta,
            "rescue_levy_scale": args.rescue_levy_scale,
            "rescue_patience_iters": args.rescue_patience_iters,
            "rescue_min_improvement": args.rescue_min_improvement,
            "enforce_elite_archive": bool(args.enforce_elite_archive),
        },
        "traceability_files": {
            "full_output_csv": full_output_path.name,
            "lime_contributions_csv": lime_contributions_path.name,
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print("\nBenchmark terminado.")
    if run_rows:
        print(f"Runs raw CSV: {runs_path}")
    print(f"Full output CSV: {full_output_path}")
    print(f"LIME contributions CSV: {lime_contributions_path}")
    if summary_rows:
        print(f"Summary CSV: {summary_path}")
        print(f"Ranking CSV: {ranking_path}")
    if skipped_rows:
        print(f"Skipped cases CSV: {skipped_path}")
    print(f"Config JSON: {config_path}")


if __name__ == "__main__":
    main()
