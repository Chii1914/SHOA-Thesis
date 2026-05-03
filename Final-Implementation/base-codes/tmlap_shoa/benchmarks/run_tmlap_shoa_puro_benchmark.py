"""Benchmark runner for SHOA (pure) on TMLAP instances.

Output contract is aligned with run_tmlap_sholime_benchmark.py:
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
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_CODE_ROOT = PROJECT_ROOT / "python-code"
if str(PYTHON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_CODE_ROOT))

INSTANCE_REQUIRED_KEYS = (
    "n_clientes",
    "n_hubs",
    "distancias",
    "costos_fijos",
    "capacidad",
    "D_max",
)


def _load_sho_function():
    sho_path = PYTHON_CODE_ROOT / "SHO.py"
    if not sho_path.exists():
        raise FileNotFoundError(f"No se encontro el modulo SHO base en: {sho_path}")

    spec = importlib.util.spec_from_file_location("sho_core_module", sho_path)
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo crear el spec para cargar SHO.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sho_fn = getattr(module, "SHO", None)
    if sho_fn is None:
        raise AttributeError("El modulo SHO.py no expone la funcion SHO")

    return sho_fn


SHO = _load_sho_function()


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
            "Ejecuta benchmark SHOA puro sobre instancias TMLAP definidas en txt "
            "con salida compatible con benchmark CEC/SHO+LIME."
        )
    )

    parser.add_argument("--instances", default="all", help="Instancias separadas por coma o 'all'")
    parser.add_argument("--instance-dir", default=".", help="Directorio de instancias .txt")

    parser.add_argument("--runs", type=int, default=30, help="Corridas por instancia")
    parser.add_argument("--seed-start", type=int, default=1, help="Semilla inicial")

    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=500)

    parser.add_argument("--output-dir", default="results/benchmark_logs", help="Directorio raiz de salida")
    parser.add_argument("--tag", default="", help="Sufijo opcional para la carpeta de salida")
    parser.add_argument("--stop-on-error", action="store_true", help="Detener ejecucion en primera falla")
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


def _run_sho(objective, lb: float, ub: float, dim: int, pop_size: int, max_iter: int, seed: int):
    np.random.seed(seed)

    best_fitness, best_latent, convergence_curve, *_ = SHO(
        pop=pop_size,
        Max_iter=max_iter,
        LB=lb,
        UB=ub,
        Dim=dim,
        fobj=objective,
    )

    best_fitness = float(best_fitness)
    best_latent = np.asarray(best_latent, dtype=float)
    curve = np.asarray(convergence_curve, dtype=float).reshape(-1)

    if curve.size < max_iter:
        if curve.size == 0:
            curve = np.full(max_iter, best_fitness, dtype=float)
        else:
            curve = np.concatenate([curve, np.full(max_iter - curve.size, curve[-1], dtype=float)])
    elif curve.size > max_iter:
        curve = curve[:max_iter]

    return best_fitness, best_latent, curve


def main() -> None:
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("--runs debe ser > 0")
    if args.max_iter <= 0:
        raise ValueError("--max-iter debe ser > 0")
    if args.pop_size < 4:
        raise ValueError("--pop-size debe ser >= 4")

    instance_dir = _resolve_instance_dir(args.instance_dir)
    instances = _resolve_instances(args.instances, instance_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tmlap_shoa_puro_r{args.runs}_{stamp}"
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
        lime_contributions_path.open("w", newline="", encoding="utf-8") as lime_file,
    ):
        full_output_writer = csv.DictWriter(full_output_file, fieldnames=full_output_fieldnames)
        full_output_writer.writeheader()

        lime_writer = csv.DictWriter(lime_file, fieldnames=lime_contributions_fieldnames)
        lime_writer.writeheader()

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

            for run_idx in range(1, args.runs + 1):
                seed = args.seed_start + run_idx - 1

                t0 = time.time()
                best_fitness, best_latent, convergence_curve = _run_sho(
                    objective=problem.objective_from_latent,
                    lb=lb,
                    ub=ub,
                    dim=used_dim,
                    pop_size=args.pop_size,
                    max_iter=args.max_iter,
                    seed=seed,
                )
                runtime = time.time() - t0

                _ = problem.repair_from_latent(best_latent)

                run_rows.append(
                    {
                        "function": instance_name,
                        "dimension": used_dim,
                        "run_id": run_idx,
                        "seed": seed,
                        "best_fitness": float(best_fitness),
                        "global_optimum": f_global,
                        "error_to_optimum": float("nan"),
                        "rescues_applied": 0,
                        "diagnostics_invoked": 0,
                        "diagnostics_invocation_count": 0,
                        "positive_diagnosis_count": 0,
                        "runtime_sec": runtime,
                        "diagnostics_invocation_iterations": "",
                    }
                )

                for iteration in range(1, args.max_iter + 1):
                    iter_best_fitness = float(convergence_curve[iteration - 1])
                    full_output_writer.writerow(
                        {
                            "function": instance_name,
                            "dimension": used_dim,
                            "run_id": run_idx,
                            "seed": seed,
                            "iteration": iteration,
                            "best_fitness": iter_best_fitness,
                            "global_optimum": f_global,
                            "error_to_optimum": float("nan"),
                            "window_size": 0,
                            "window_std": float("nan"),
                            "trigger_candidate": False,
                            "diagnostics_invoked": False,
                            "diagnosis_status": "NONE",
                            "diagnosis_pred_delta": float("nan"),
                            "diagnosis_fidelity": float("nan"),
                            "rescue_applied": False,
                            "rescue_count_cumulative": 0,
                            "rescue_trial_active": False,
                            "rescue_rollback_applied": False,
                            "rollback_count_cumulative": 0,
                            "cooldown_counter": 0,
                            "elite_best_fitness": iter_best_fitness,
                            "diagnosis_id": 0,
                            "strong_stochastic_importance": False,
                            "low_expected_improvement": False,
                            "weight_r1": float("nan"),
                            "weight_mag_browniano": float("nan"),
                            "weight_mag_levy": float("nan"),
                            "weight_r2": float("nan"),
                            "weight_mag_predacion": float("nan"),
                            "abs_weight_r1": float("nan"),
                            "abs_weight_mag_browniano": float("nan"),
                            "abs_weight_mag_levy": float("nan"),
                            "abs_weight_r2": float("nan"),
                            "abs_weight_mag_predacion": float("nan"),
                        }
                    )

                per_case_best.append(float(best_fitness))
                per_case_runtime.append(runtime)

                finished_jobs += 1
                print(
                    f"[{finished_jobs}/{total_jobs}] {instance_name} d={used_dim} run={run_idx}/{args.runs} "
                    f"seed={seed} best={best_fitness:.6e} time={runtime:.2f}s"
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
                    "mean_rescues": 0.0,
                    "mean_diagnostics_invoked": 0.0,
                    "mean_positive_diagnosis": 0.0,
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
        "shoa_puro_controller": {
            "algorithm": "SHO",
        },
        "traceability_files": {
            "full_output_csv": full_output_path.name,
            "lime_contributions_csv": lime_contributions_path.name,
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print("\nBenchmark TMLAP SHOA puro terminado.")
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
