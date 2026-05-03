"""Benchmark runner for SHO + LIME on TMLAP instances.

This script mirrors the output schema used in the CEC benchmark runner:
- runs_raw.csv
- full_output.csv
- lime_contributions.csv
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
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHOLIME_ROOT = PROJECT_ROOT / "SHO + LIME"
if str(SHOLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOLIME_ROOT))

from base.SHO_LIME_Controller import SHOXAIConfig, SHO_with_lime_controller

INSTANCE_REQUIRED_KEYS = (
    "n_clientes",
    "n_hubs",
    "distancias",
    "costos_fijos",
    "capacidad",
    "D_max",
)


class TMLAPProblem:
    """TMLAP objective over latent continuous vectors."""

    def __init__(
        self,
        n_clientes: int,
        n_hubs: int,
        distancias,
        costos_fijos,
        capacidad,
        d_max: float,
    ) -> None:
        self.n_clients = int(n_clientes)
        self.n_hubs = int(n_hubs)
        self.distancias = [[float(value) for value in row] for row in distancias]
        self.costs = [float(value) for value in costos_fijos]
        self.capacidad = [int(value) for value in capacidad]
        self.D_max = float(d_max)

        self._validate()

    def _validate(self) -> None:
        if self.n_clients <= 0:
            raise ValueError("n_clientes debe ser > 0")
        if self.n_hubs <= 0:
            raise ValueError("n_hubs debe ser > 0")

        if len(self.distancias) != self.n_clients:
            raise ValueError("distancias debe tener n_clientes filas")
        for idx, row in enumerate(self.distancias):
            if len(row) != self.n_hubs:
                raise ValueError(f"distancias fila {idx} debe tener n_hubs columnas")

        if len(self.costs) != self.n_hubs:
            raise ValueError("costos_fijos debe tener n_hubs elementos")
        if len(self.capacidad) != self.n_hubs:
            raise ValueError("capacidad debe tener n_hubs elementos")

        if sum(self.capacidad) < self.n_clients:
            raise ValueError("La suma de capacidad debe ser >= n_clientes")

    def _feasible_hubs(self, client_idx: int) -> list[int]:
        return [
            hub_idx
            for hub_idx in range(self.n_hubs)
            if self.distancias[client_idx][hub_idx] <= self.D_max
        ]

    def check(self, assignment) -> bool:
        if len(assignment) != self.n_clients:
            return False

        usage = [0] * self.n_hubs
        for client_idx in range(self.n_clients):
            hub_idx = int(assignment[client_idx])
            if hub_idx < 0 or hub_idx >= self.n_hubs:
                return False
            if self.distancias[client_idx][hub_idx] > self.D_max:
                return False
            usage[hub_idx] += 1
            if usage[hub_idx] > self.capacidad[hub_idx]:
                return False

        return True

    def fit(self, assignment) -> float:
        opened = [0] * self.n_hubs
        total = 0.0

        for client_idx in range(self.n_clients):
            hub_idx = int(assignment[client_idx])
            opened[hub_idx] = 1
            total += self.distancias[client_idx][hub_idx]

        for hub_idx in range(self.n_hubs):
            if opened[hub_idx] == 1:
                total += self.costs[hub_idx]

        return float(total)

    def repair_from_latent(self, latent) -> list[int] | None:
        z = np.asarray(latent, dtype=float).reshape(-1)
        if z.size != self.n_clients:
            return None

        feasible_by_client = [self._feasible_hubs(client_idx) for client_idx in range(self.n_clients)]
        if any(len(options) == 0 for options in feasible_by_client):
            return None

        remaining = self.capacidad.copy()
        assignment = [-1] * self.n_clients

        # Assign most constrained clients first.
        client_order = sorted(range(self.n_clients), key=lambda client_idx: len(feasible_by_client[client_idx]))

        for client_idx in client_order:
            options = sorted(
                feasible_by_client[client_idx],
                key=lambda hub_idx: (abs(z[client_idx] - hub_idx), self.distancias[client_idx][hub_idx]),
            )

            chosen = None
            for hub_idx in options:
                if remaining[hub_idx] > 0:
                    chosen = hub_idx
                    break

            if chosen is None:
                fallback = [hub_idx for hub_idx in feasible_by_client[client_idx] if remaining[hub_idx] > 0]
                if not fallback:
                    return None
                chosen = min(
                    fallback,
                    key=lambda hub_idx: (self.distancias[client_idx][hub_idx], abs(z[client_idx] - hub_idx)),
                )

            assignment[client_idx] = chosen
            remaining[chosen] -= 1

        if not self.check(assignment):
            return None

        return assignment

    def objective_from_latent(self, latent) -> float:
        assignment = self.repair_from_latent(latent)
        if assignment is None:
            return 1e9
        return self.fit(assignment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta benchmark SHO+LIME sobre instancias TMLAP definidas en txt "
            "con salida compatible con benchmark CEC."
        )
    )

    parser.add_argument(
        "--instances",
        default="all",
        help="Instancias separadas por coma (nombre .txt o ruta) o 'all'",
    )
    parser.add_argument(
        "--instance-dir",
        default=".",
        help="Directorio donde estan los txt de instancias",
    )

    parser.add_argument("--runs", type=int, default=30, help="Corridas por instancia")
    parser.add_argument("--seed-start", type=int, default=1, help="Semilla inicial")

    # Paper-like SHO defaults
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)

    # SHO + LIME controller
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
        default="results/benchmark_logs",
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
        help="Detener ejecucion si falla alguna instancia",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_instance_dir(raw_dir: str) -> Path:
    candidate = Path(raw_dir)
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent / candidate).resolve()

    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"No existe el directorio de instancias: {candidate}")

    return candidate


def _resolve_instances(raw_instances: str, instance_dir: Path) -> list[Path]:
    if raw_instances.strip().lower() == "all":
        discovered = sorted(path for path in instance_dir.glob("*.txt") if path.is_file())
        if not discovered:
            raise ValueError(f"No se encontraron .txt en {instance_dir}")
        return discovered

    resolved: list[Path] = []
    for token in _parse_csv_list(raw_instances):
        candidates = [
            Path(token),
            instance_dir / token,
            instance_dir / f"{token}.txt",
        ]

        picked = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                picked = candidate.resolve()
                break

        if picked is None:
            raise FileNotFoundError(
                f"No se pudo resolver la instancia '{token}'. Revisa ruta o nombre en {instance_dir}"
            )

        resolved.append(picked)

    return resolved


def _load_instance_payload(instance_path: Path) -> dict:
    holder = SimpleNamespace()
    source = instance_path.read_text(encoding="utf-8")

    # Files contain simple assignments like: self.n_clientes = ...
    exec(compile(source, str(instance_path), "exec"), {"__builtins__": {}}, {"self": holder})

    payload = {}
    for key in INSTANCE_REQUIRED_KEYS:
        if not hasattr(holder, key):
            raise ValueError(f"La instancia {instance_path.name} no define '{key}'")
        payload[key] = getattr(holder, key)

    return payload


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

    instance_dir = _resolve_instance_dir(args.instance_dir)
    instances = _resolve_instances(args.instances, instance_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tmlap_sholime_r{args.runs}_{stamp}"
    if args.tag.strip():
        run_name = f"{run_name}_{args.tag.strip()}"

    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {out_root}")
    print(f"Instance dir: {instance_dir}")
    print(f"Instances: {[path.name for path in instances]}")
    print(f"Runs per instance: {args.runs}")

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

    total_jobs = len(instances) * args.runs
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

        for instance_path in instances:
            instance_name = instance_path.stem
            print(f"\n=== Benchmark {instance_name} ===")

            try:
                payload = _load_instance_payload(instance_path)
                problem = TMLAPProblem(
                    n_clientes=payload["n_clientes"],
                    n_hubs=payload["n_hubs"],
                    distancias=payload["distancias"],
                    costos_fijos=payload["costos_fijos"],
                    capacidad=payload["capacidad"],
                    d_max=payload["D_max"],
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "function": instance_name,
                        "requested_dim": "N/A",
                        "error": f"{instance_path.name}: {exc}",
                    }
                )
                print(f"SKIP {instance_name}: {exc}")
                if args.stop_on_error:
                    raise
                continue

            lb = 0.0
            ub = float(problem.n_hubs - 1)
            used_dim = int(problem.n_clients)
            f_global = float("nan")

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
                result = SHO_with_lime_controller(problem.objective_from_latent, lb, ub, used_dim, cfg)
                runtime = time.time() - t0

                positive_diag_count = sum(
                    1 for diag in result.diagnostics_log if diag.get("status") == "POSITIVE_STAGNATION"
                )
                diagnostics_invocation_count = len(result.diagnostics_invocation_iterations)
                best_fitness = float(result.best_fitness)
                error_to_optimum = float("nan")

                run_rows.append(
                    {
                        "function": instance_name,
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
                            str(value) for value in result.diagnostics_invocation_iterations
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
                            "function": instance_name,
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
                    iter_error_to_optimum = float("nan")
                    iter_diag = diagnosis_by_iteration.get(iteration, {})

                    full_output_writer.writerow(
                        {
                            "function": instance_name,
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
                    f"[{finished_jobs}/{total_jobs}] {instance_name} d={used_dim} run={run_idx}/{args.runs} "
                    f"seed={seed} best={best_fitness:.6e} rescues={result.rescue_count} "
                    f"diag={len(result.diagnostics_log)} time={runtime:.2f}s"
                )

            summary_rows.append(
                {
                    "function": instance_name,
                    "dimension": used_dim,
                    "runs": args.runs,
                    "global_optimum": f_global,
                    "best": float(min(per_case_best)),
                    "worst": float(max(per_case_best)),
                    "mean": float(statistics.mean(per_case_best)),
                    "median": float(statistics.median(per_case_best)),
                    "std": _compute_std(per_case_best),
                    "mean_error_to_optimum": float("nan"),
                    "mean_runtime_sec": float(statistics.mean(per_case_runtime)),
                    "mean_rescues": float(statistics.mean(per_case_rescues)),
                    "mean_diagnostics_invoked": float(statistics.mean(per_case_diagnostics)),
                    "mean_positive_diagnosis": float(statistics.mean(per_case_positive_diag)),
                }
            )

    rank_rows: list[dict] = []
    dimensions = sorted({int(row["dimension"]) for row in summary_rows})
    for dim in dimensions:
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
        "instances": [path.name for path in instances],
        "instance_dir": str(instance_dir),
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

    print("\nBenchmark TMLAP terminado.")
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
