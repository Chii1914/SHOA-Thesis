from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent
BASE_CODES_DIR = ROOT_DIR / "base-codes"

LARGE_INSTANCES = (
    {
        "filename": "4.instancia_very_large_500_hubs.txt",
        "n_clientes": 500,
        "n_hubs": 500,
        "seed": 500_321,
    },
    {
        "filename": "5.instancia_very_large_1000_hubs.txt",
        "n_clientes": 1000,
        "n_hubs": 1000,
        "seed": 1000_321,
    },
)

PROFILE_MODE_PATTERN = re.compile(r"(soft|medium|hard)_(leader_repulsion|levy_teleport)")


@dataclass
class Job:
    name: str
    suite: str
    algorithm: str
    cwd: Path
    command: list[str]


@dataclass
class JobResult:
    name: str
    suite: str
    algorithm: str
    cwd: Path
    command: list[str]
    start_ts: str
    end_ts: str
    duration_sec: float
    exit_code: int
    status: str
    output_dirs: list[Path]


class Logger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_file.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line)
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orquestador secuencial de pruebas finales (CEC2022 + TMLAP) con "
            "analisis estadistico SHOA vs SHOA+LIME."
        )
    )
    parser.add_argument("--suite", choices=["all", "cec2022", "tmlap"], default="all")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia")
    parser.add_argument("--output-root", default="final-test-runs", help="Directorio de salida")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Ejecutable de Python para lanzar benchmarks",
    )
    parser.add_argument(
        "--regenerate-instances",
        action="store_true",
        help="Regenera instancias grandes aunque ya existan",
    )
    parser.add_argument(
        "--skip-instance-generation",
        action="store_true",
        help="No genera instancias grandes TMLAP",
    )
    parser.add_argument("--skip-stats", action="store_true", help="No ejecutar bloque estadistico")
    parser.add_argument("--fail-fast", action="store_true", help="Detener pipeline en primer error")
    parser.add_argument("--dry-run", action="store_true", help="No ejecutar comandos")
    return parser.parse_args()


def build_jobs(python_executable: str, suite: str) -> list[Job]:
    jobs: list[Job] = []

    if suite in {"all", "cec2022"}:
        jobs.extend(
            [
                Job(
                    name="cec2022_shoa_full",
                    suite="cec2022",
                    algorithm="shoa",
                    cwd=BASE_CODES_DIR / "cec2022_shoa",
                    command=[
                        python_executable,
                        "benchmarks/run_cec2022_benchmark_shoa_puro.py",
                        "--functions",
                        "all",
                        "--dims",
                        "10",
                        "--runs",
                        "30",
                        "--seed-start",
                        "1",
                        "--pop-size",
                        "30",
                        "--max-iter",
                        "500",
                        "--output-dir",
                        "benchmark_logs",
                        "--tag",
                        "full",
                    ],
                ),
                Job(
                    name="cec2022_pso_full",
                    suite="cec2022",
                    algorithm="pso",
                    cwd=BASE_CODES_DIR / "cec2022_pso",
                    command=[
                        python_executable,
                        "benchmarks/run_cec2022_pso_benchmark.py",
                        "--functions",
                        "all",
                        "--dims",
                        "10",
                        "--runs",
                        "30",
                        "--seed-start",
                        "1",
                        "--pop-size",
                        "30",
                        "--max-iter",
                        "500",
                        "--output-dir",
                        "benchmark_logs",
                        "--tag",
                        "full",
                    ],
                ),
                Job(
                    name="cec2022_sholime_profiles",
                    suite="cec2022",
                    algorithm="sholime",
                    cwd=BASE_CODES_DIR / "cec2022_sholime",
                    command=[
                        python_executable,
                        "benchmarks/run_sholime_profiles_matrix.py",
                        "--profiles",
                        "soft,medium,hard",
                        "--modes",
                        "leader_repulsion,levy_teleport",
                    ],
                ),
            ]
        )

    if suite in {"all", "tmlap"}:
        jobs.extend(
            [
                Job(
                    name="tmlap_shoa_full",
                    suite="tmlap",
                    algorithm="shoa",
                    cwd=BASE_CODES_DIR / "tmlap_shoa",
                    command=[
                        python_executable,
                        "benchmarks/run_tmlap_shoa_puro_benchmark.py",
                        "--instances",
                        "all",
                        "--instance-dir",
                        ".",
                        "--runs",
                        "30",
                        "--seed-start",
                        "1",
                        "--pop-size",
                        "30",
                        "--max-iter",
                        "500",
                        "--output-dir",
                        "results/benchmark_logs",
                        "--tag",
                        "full",
                    ],
                ),
                Job(
                    name="tmlap_pso_full",
                    suite="tmlap",
                    algorithm="pso",
                    cwd=BASE_CODES_DIR / "tmlap_pso",
                    command=[
                        python_executable,
                        "benchmarks/run_tmlap_pso_benchmark.py",
                        "--instances",
                        "all",
                        "--instance-dir",
                        ".",
                        "--runs",
                        "30",
                        "--seed-start",
                        "1",
                        "--pop-size",
                        "30",
                        "--max-iter",
                        "500",
                        "--output-dir",
                        "results/benchmark_logs",
                        "--tag",
                        "full",
                    ],
                ),
                Job(
                    name="tmlap_sholime_full",
                    suite="tmlap",
                    algorithm="sholime",
                    cwd=BASE_CODES_DIR / "tmlap_sholime",
                    command=[
                        python_executable,
                        "benchmarks/run_tmlap_sholime_benchmark.py",
                        "--instances",
                        "all",
                        "--instance-dir",
                        ".",
                        "--runs",
                        "30",
                        "--seed-start",
                        "1",
                        "--pop-size",
                        "30",
                        "--max-iter",
                        "500",
                        "--output-dir",
                        "results/benchmark_logs",
                        "--tag",
                        "full",
                    ],
                ),
            ]
        )

    return jobs


def _snapshot_output_dirs(base_dir: Path) -> set[Path]:
    candidates = [
        base_dir / "benchmark_logs",
        base_dir / "results" / "benchmark_logs",
    ]
    out: set[Path] = set()
    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir():
                out.add(child.resolve())
    return out


def _run_job(job: Job, logger: Logger, dry_run: bool) -> JobResult:
    before_dirs = _snapshot_output_dirs(job.cwd)
    start = time.time()
    start_ts = datetime.now().isoformat(timespec="seconds")

    logger.write(f"[JOB START] {job.name}")
    logger.write(f"[JOB CWD] {job.cwd}")
    logger.write(f"[JOB CMD] {' '.join(job.command)}")

    if dry_run:
        exit_code = 0
        logger.write(f"[JOB DRY-RUN] {job.name} no ejecutado.")
    else:
        process = subprocess.Popen(
            job.command,
            cwd=job.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logger.write(f"[{job.name}] {line.rstrip()}")
        exit_code = process.wait()

    after_dirs = _snapshot_output_dirs(job.cwd)
    new_dirs = sorted(after_dirs - before_dirs)

    duration_sec = time.time() - start
    end_ts = datetime.now().isoformat(timespec="seconds")
    status = "success" if exit_code == 0 else "failed"

    logger.write(
        f"[JOB END] {job.name} status={status} exit_code={exit_code} duration_sec={duration_sec:.2f}"
    )
    if new_dirs:
        logger.write(f"[JOB OUTPUTS] {job.name} -> {', '.join(str(path) for path in new_dirs)}")
    else:
        logger.write(f"[JOB OUTPUTS] {job.name} -> sin nuevas carpetas detectadas")

    return JobResult(
        name=job.name,
        suite=job.suite,
        algorithm=job.algorithm,
        cwd=job.cwd,
        command=job.command,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_sec=duration_sec,
        exit_code=exit_code,
        status=status,
        output_dirs=new_dirs,
    )


def _build_large_instance_file(path: Path, n_clientes: int, n_hubs: int, seed: int) -> None:
    rng = random.Random(seed)
    d_max = 12

    # Capacidad minima para garantizar factibilidad global.
    base_cap = max(1, n_clientes // n_hubs)
    capacidad = [base_cap for _ in range(n_hubs)]
    remaining = n_clientes - sum(capacidad)
    idx = 0
    while remaining > 0:
        capacidad[idx % n_hubs] += 1
        idx += 1
        remaining -= 1

    # Costos heterogeneos para mantener dificultad en apertura de hubs.
    costos = [rng.randint(40, 180) for _ in range(n_hubs)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"self.n_clientes = {n_clientes}\n")
        f.write(f"self.n_hubs = {n_hubs}\n\n")
        f.write("# Instancia generada automaticamente para stress test grande y factible\n")
        f.write("# Factibilidad garantizada por construccion (cobertura + capacidad).\n\n")

        f.write("self.distancias = [\n")
        for client_idx in range(n_clientes):
            row = [rng.randint(45, 180) for _ in range(n_hubs)]

            # Asegura al menos un hub factible por cliente.
            primary_hub = client_idx % n_hubs
            row[primary_hub] = rng.randint(1, d_max - 2)

            # Un segundo hub factible agrega alternativas locales.
            secondary_hub = (client_idx * 37 + 11) % n_hubs
            if secondary_hub != primary_hub:
                row[secondary_hub] = rng.randint(2, d_max)

            row_txt = ", ".join(str(value) for value in row)
            f.write(f"    [{row_txt}],\n")
        f.write("]\n\n")

        costos_txt = ", ".join(str(value) for value in costos)
        capacidad_txt = ", ".join(str(value) for value in capacidad)

        f.write(f"self.costos_fijos = [{costos_txt}]\n")
        f.write(f"self.capacidad = [{capacidad_txt}]\n")
        f.write(f"self.D_max = {d_max}\n")


def _validate_instance_file(path: Path) -> tuple[bool, str]:
    holder = SimpleNamespace()
    raw_code = path.read_text(encoding="utf-8")

    try:
        exec(compile(raw_code, str(path), "exec"), {"__builtins__": {}}, {"self": holder})
    except Exception as exc:
        return False, f"exec_error:{exc}"

    required_keys = ["n_clientes", "n_hubs", "distancias", "costos_fijos", "capacidad", "D_max"]
    for key in required_keys:
        if not hasattr(holder, key):
            return False, f"missing_key:{key}"

    n_clientes = int(holder.n_clientes)
    n_hubs = int(holder.n_hubs)
    distancias = holder.distancias
    costos = holder.costos_fijos
    capacidad = holder.capacidad
    d_max = float(holder.D_max)

    if len(distancias) != n_clientes:
        return False, "invalid_distancias_rows"
    if any(len(row) != n_hubs for row in distancias):
        return False, "invalid_distancias_cols"
    if len(costos) != n_hubs:
        return False, "invalid_costos_len"
    if len(capacidad) != n_hubs:
        return False, "invalid_capacidad_len"
    if sum(int(v) for v in capacidad) < n_clientes:
        return False, "capacity_below_clients"

    # Cobertura minima por cliente: al menos un hub alcanzable.
    for row in distancias:
        if min(float(v) for v in row) > d_max:
            return False, "client_without_reachable_hub"

    return True, "ok"


def ensure_large_tmlap_instances(regenerate: bool, logger: Logger) -> None:
    benchmark_dirs = [
        BASE_CODES_DIR / "tmlap_shoa" / "benchmarks",
        BASE_CODES_DIR / "tmlap_pso" / "benchmarks",
        BASE_CODES_DIR / "tmlap_sholime" / "benchmarks",
    ]

    for benchmark_dir in benchmark_dirs:
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        for spec in LARGE_INSTANCES:
            instance_path = benchmark_dir / spec["filename"]

            if instance_path.exists() and not regenerate:
                ok, reason = _validate_instance_file(instance_path)
                if ok:
                    logger.write(f"[INSTANCES] Reutilizando {instance_path.name} en {benchmark_dir}")
                    continue
                logger.write(
                    f"[INSTANCES] {instance_path.name} invalida ({reason}). Se regenera automaticamente."
                )

            logger.write(f"[INSTANCES] Generando {instance_path.name} en {benchmark_dir}")
            _build_large_instance_file(
                path=instance_path,
                n_clientes=int(spec["n_clientes"]),
                n_hubs=int(spec["n_hubs"]),
                seed=int(spec["seed"]),
            )
            ok, reason = _validate_instance_file(instance_path)
            if not ok:
                raise RuntimeError(f"Instancia generada invalida: {instance_path} ({reason})")


def _extract_profile_mode_from_dir(output_dir: Path) -> str:
    match = PROFILE_MODE_PATTERN.search(output_dir.name)
    if match is None:
        return "none"
    profile, rescue_mode = match.group(1), match.group(2)
    return f"{profile}_{rescue_mode}"


def _load_runs_raw(output_dir: Path, suite: str, algorithm: str) -> pd.DataFrame:
    csv_path = output_dir / "runs_raw.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required = ["function", "dimension", "run_id", "seed", "best_fitness"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df[required].copy()
    out["suite"] = suite
    out["algorithm"] = algorithm
    out["profile_mode"] = (
        _extract_profile_mode_from_dir(output_dir)
        if suite == "cec2022" and algorithm == "sholime"
        else "none"
    )
    out["output_dir"] = str(output_dir)

    out["dimension"] = pd.to_numeric(out["dimension"], errors="coerce")
    out["run_id"] = pd.to_numeric(out["run_id"], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["best_fitness"] = pd.to_numeric(out["best_fitness"], errors="coerce")

    out = out.dropna(subset=["dimension", "run_id", "seed", "best_fitness"])
    out["dimension"] = out["dimension"].astype(int)
    out["run_id"] = out["run_id"].astype(int)
    out["seed"] = out["seed"].astype(int)

    return out


def _shapiro(values: np.ndarray) -> tuple[float, str]:
    if values.size < 3:
        return float("nan"), "insufficient_n"
    if np.allclose(values, values[0]):
        return float("nan"), "constant"

    try:
        _, p_value = stats.shapiro(values)
        return float(p_value), "ok"
    except Exception as exc:
        return float("nan"), f"error:{type(exc).__name__}"


def _trend_from_delta(delta: float) -> str:
    if np.isnan(delta):
        return "sin_datos"
    if delta < 0:
        return "SHOLIME mejor (menor fitness)"
    if delta > 0:
        return "SHOA mejor (menor fitness)"
    return "Empate"


def _build_statistical_results(
    shoa_df: pd.DataFrame,
    sholime_df: pd.DataFrame,
    suite: str,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normality_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    if shoa_df.empty or sholime_df.empty:
        return normality_rows, comparison_rows

    if suite == "cec2022":
        grouping = ["function", "dimension", "profile_mode"]
    else:
        grouping = ["function", "dimension"]

    for key_values, group in sholime_df.groupby(grouping, dropna=False):
        if suite == "cec2022":
            function, dimension, profile_mode = key_values
            base = shoa_df[
                (shoa_df["function"] == function)
                & (shoa_df["dimension"] == int(dimension))
            ]
        else:
            function, dimension = key_values
            profile_mode = "none"
            base = shoa_df[
                (shoa_df["function"] == function)
                & (shoa_df["dimension"] == int(dimension))
            ]

        treatment = group.copy()

        shoa_values = base["best_fitness"].to_numpy(dtype=float)
        sholime_values = treatment["best_fitness"].to_numpy(dtype=float)

        paired = pd.merge(
            base[["run_id", "seed", "best_fitness"]],
            treatment[["run_id", "seed", "best_fitness"]],
            on=["run_id", "seed"],
            suffixes=("_shoa", "_sholime"),
        )

        paired_count = int(paired.shape[0])
        delta_values = (
            paired["best_fitness_sholime"].to_numpy(dtype=float)
            - paired["best_fitness_shoa"].to_numpy(dtype=float)
            if paired_count > 0
            else np.array([], dtype=float)
        )

        p_shoa, shoa_status = _shapiro(shoa_values)
        p_sholime, sholime_status = _shapiro(sholime_values)
        p_delta, delta_status = _shapiro(delta_values)

        normality_rows.append(
            {
                "suite": suite,
                "function": function,
                "dimension": int(dimension),
                "profile_mode": profile_mode,
                "n_shoa": int(shoa_values.size),
                "n_sholime": int(sholime_values.size),
                "n_pairs": paired_count,
                "shoa_shapiro_p": p_shoa,
                "shoa_status": shoa_status,
                "sholime_shapiro_p": p_sholime,
                "sholime_status": sholime_status,
                "delta_shapiro_p": p_delta,
                "delta_status": delta_status,
            }
        )

        test_used = ""
        statistic = float("nan")
        p_value = float("nan")
        notes = ""

        if paired_count > 0:
            test_used = "wilcoxon_paired"
            x_vals = paired["best_fitness_shoa"].to_numpy(dtype=float)
            y_vals = paired["best_fitness_sholime"].to_numpy(dtype=float)

            if np.allclose(x_vals, y_vals):
                statistic = 0.0
                p_value = 1.0
                notes = "all_pairs_equal"
            else:
                try:
                    stat_result = stats.wilcoxon(x_vals, y_vals, alternative="two-sided")
                    statistic = float(stat_result.statistic)
                    p_value = float(stat_result.pvalue)
                except Exception as exc:
                    notes = f"wilcoxon_error:{type(exc).__name__}"

            median_delta = float(np.median(delta_values)) if delta_values.size > 0 else float("nan")
            mean_delta = float(np.mean(delta_values)) if delta_values.size > 0 else float("nan")
        else:
            test_used = "mannwhitneyu"
            if shoa_values.size > 0 and sholime_values.size > 0:
                try:
                    stat_result = stats.mannwhitneyu(
                        shoa_values,
                        sholime_values,
                        alternative="two-sided",
                    )
                    statistic = float(stat_result.statistic)
                    p_value = float(stat_result.pvalue)
                except Exception as exc:
                    notes = f"mannwhitneyu_error:{type(exc).__name__}"
            else:
                notes = "insufficient_data"

            median_delta = float(np.median(sholime_values) - np.median(shoa_values)) if (
                shoa_values.size > 0 and sholime_values.size > 0
            ) else float("nan")
            mean_delta = float(np.mean(sholime_values) - np.mean(shoa_values)) if (
                shoa_values.size > 0 and sholime_values.size > 0
            ) else float("nan")

        decision = "Rechazar H0" if np.isfinite(p_value) and p_value < alpha else "No rechazar H0"
        trend = _trend_from_delta(median_delta)

        comparison_rows.append(
            {
                "suite": suite,
                "function": function,
                "dimension": int(dimension),
                "profile_mode": profile_mode,
                "n_shoa": int(shoa_values.size),
                "n_sholime": int(sholime_values.size),
                "n_pairs": paired_count,
                "test_used": test_used,
                "statistic": statistic,
                "p_value": p_value,
                "decision_alpha": decision,
                "median_delta_sholime_minus_shoa": median_delta,
                "mean_delta_sholime_minus_shoa": mean_delta,
                "trend": trend,
                "notes": notes,
            }
        )

    return normality_rows, comparison_rows


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(num):
        return "NaN"
    return f"{num:.{digits}g}"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No hay filas para mostrar."

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")

    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(_format_float(value, digits=6))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def run_statistics(
    results: list[JobResult],
    alpha: float,
    run_root: Path,
    logger: Logger,
) -> None:
    stats_root = run_root / "statistics"
    stats_root.mkdir(parents=True, exist_ok=True)

    cec_shoa_frames: list[pd.DataFrame] = []
    cec_sholime_frames: list[pd.DataFrame] = []
    tmlap_shoa_frames: list[pd.DataFrame] = []
    tmlap_sholime_frames: list[pd.DataFrame] = []

    for result in results:
        if result.status != "success":
            continue

        for output_dir in result.output_dirs:
            loaded = _load_runs_raw(output_dir=output_dir, suite=result.suite, algorithm=result.algorithm)
            if loaded.empty:
                continue

            if result.suite == "cec2022" and result.algorithm == "shoa":
                cec_shoa_frames.append(loaded)
            elif result.suite == "cec2022" and result.algorithm == "sholime":
                cec_sholime_frames.append(loaded)
            elif result.suite == "tmlap" and result.algorithm == "shoa":
                tmlap_shoa_frames.append(loaded)
            elif result.suite == "tmlap" and result.algorithm == "sholime":
                tmlap_sholime_frames.append(loaded)

    normality_all: list[dict[str, Any]] = []
    comparisons_all: list[dict[str, Any]] = []

    if cec_shoa_frames and cec_sholime_frames:
        cec_shoa_df = pd.concat(cec_shoa_frames, ignore_index=True)
        cec_sholime_df = pd.concat(cec_sholime_frames, ignore_index=True)
        normality_rows, comparison_rows = _build_statistical_results(
            cec_shoa_df,
            cec_sholime_df,
            suite="cec2022",
            alpha=alpha,
        )
        normality_all.extend(normality_rows)
        comparisons_all.extend(comparison_rows)
    else:
        logger.write("[STATS] No hay datos suficientes para CEC SHOA vs SHOA+LIME")

    if tmlap_shoa_frames and tmlap_sholime_frames:
        tmlap_shoa_df = pd.concat(tmlap_shoa_frames, ignore_index=True)
        tmlap_sholime_df = pd.concat(tmlap_sholime_frames, ignore_index=True)
        normality_rows, comparison_rows = _build_statistical_results(
            tmlap_shoa_df,
            tmlap_sholime_df,
            suite="tmlap",
            alpha=alpha,
        )
        normality_all.extend(normality_rows)
        comparisons_all.extend(comparison_rows)
    else:
        logger.write("[STATS] No hay datos suficientes para TMLAP SHOA vs SHOA+LIME")

    normality_path = stats_root / "normality_shapiro_shoa_vs_sholime.csv"
    comparisons_path = stats_root / "comparisons_shoa_vs_sholime.csv"
    report_path = stats_root / "reporte_estadistico_shoa_vs_sholime.md"

    pd.DataFrame(normality_all).to_csv(normality_path, index=False)
    pd.DataFrame(comparisons_all).to_csv(comparisons_path, index=False)

    report_lines: list[str] = []
    report_lines.append("# Reporte Estadistico: SHOA vs SHOA+LIME")
    report_lines.append("")
    report_lines.append(f"Fecha: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append("")
    report_lines.append("## 1) Objetivo")
    report_lines.append("")
    report_lines.append("Evaluar si la diferencia de rendimiento entre SHOA y SHOA+LIME es estadisticamente significativa en CEC2022 y TMLAP.")
    report_lines.append("")
    report_lines.append("## 2) Metodologia")
    report_lines.append("")
    report_lines.append(f"- Nivel de significancia: alpha = {alpha}")
    report_lines.append("- Normalidad: Shapiro-Wilk en SHOA, SHOA+LIME y delta (cuando hay pares).")
    report_lines.append("- Contraste principal: Wilcoxon pareado cuando hay emparejamiento por run_id/seed.")
    report_lines.append("- Fallback: Mann-Whitney U cuando no hay pares validos.")
    report_lines.append("")

    report_lines.append("## 3) Normalidad (Shapiro-Wilk)")
    report_lines.append("")
    normality_cols = [
        "suite",
        "function",
        "dimension",
        "profile_mode",
        "n_shoa",
        "n_sholime",
        "n_pairs",
        "shoa_shapiro_p",
        "shoa_status",
        "sholime_shapiro_p",
        "sholime_status",
        "delta_shapiro_p",
        "delta_status",
    ]
    report_lines.append(_markdown_table(normality_all, normality_cols))
    report_lines.append("")

    report_lines.append("## 4) Comparacion SHOA vs SHOA+LIME")
    report_lines.append("")
    comparison_cols = [
        "suite",
        "function",
        "dimension",
        "profile_mode",
        "n_shoa",
        "n_sholime",
        "n_pairs",
        "test_used",
        "statistic",
        "p_value",
        "decision_alpha",
        "median_delta_sholime_minus_shoa",
        "mean_delta_sholime_minus_shoa",
        "trend",
        "notes",
    ]
    report_lines.append(_markdown_table(comparisons_all, comparison_cols))
    report_lines.append("")

    reject_count = sum(1 for row in comparisons_all if row.get("decision_alpha") == "Rechazar H0")
    total_count = len(comparisons_all)
    report_lines.append("## 5) Resumen")
    report_lines.append("")
    report_lines.append(f"- Comparaciones evaluadas: {total_count}")
    report_lines.append(f"- Casos con rechazo de H0: {reject_count}")
    report_lines.append(f"- Casos sin rechazo de H0: {max(total_count - reject_count, 0)}")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    logger.write(f"[STATS] Normalidad -> {normality_path}")
    logger.write(f"[STATS] Comparaciones -> {comparisons_path}")
    logger.write(f"[STATS] Reporte -> {report_path}")


def write_manifest(results: list[JobResult], run_root: Path) -> None:
    csv_path = run_root / "job_manifest.csv"
    json_path = run_root / "job_manifest.json"

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "name": result.name,
                "suite": result.suite,
                "algorithm": result.algorithm,
                "cwd": str(result.cwd),
                "command": " ".join(result.command),
                "start_ts": result.start_ts,
                "end_ts": result.end_ts,
                "duration_sec": result.duration_sec,
                "exit_code": result.exit_code,
                "status": result.status,
                "output_dirs": ";".join(str(path) for path in result.output_dirs),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "suite",
                "algorithm",
                "cwd",
                "command",
                "start_ts",
                "end_ts",
                "duration_sec",
                "exit_code",
                "status",
                "output_dirs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_payload = []
    for result in results:
        json_payload.append(
            {
                "name": result.name,
                "suite": result.suite,
                "algorithm": result.algorithm,
                "cwd": str(result.cwd),
                "command": result.command,
                "start_ts": result.start_ts,
                "end_ts": result.end_ts,
                "duration_sec": result.duration_sec,
                "exit_code": result.exit_code,
                "status": result.status,
                "output_dirs": [str(path) for path in result.output_dirs],
            }
        )

    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def write_summary(results: list[JobResult], run_root: Path) -> None:
    summary_path = run_root / "start-tests-summary.md"
    total = len(results)
    ok = sum(1 for result in results if result.status == "success")
    failed = total - ok

    lines: list[str] = []
    lines.append("# Start-Tests Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Total jobs: {total}")
    lines.append(f"- Success: {ok}")
    lines.append(f"- Failed: {failed}")
    lines.append("")
    lines.append("## Jobs")
    lines.append("")
    lines.append("| name | suite | algorithm | status | exit_code | duration_sec | output_dirs |")
    lines.append("|---|---|---|---|---:|---:|---|")

    for result in results:
        output_dirs_text = "<br>".join(str(path) for path in result.output_dirs) if result.output_dirs else "-"
        lines.append(
            f"| {result.name} | {result.suite} | {result.algorithm} | {result.status} "
            f"| {result.exit_code} | {result.duration_sec:.2f} | {output_dirs_text} |"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (ROOT_DIR / args.output_root / f"start_tests_{timestamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    logger = Logger(run_root / "start_tests.log")
    logger.write("Inicio de start-tests.py (modo secuencial, sin paralelismo)")
    logger.write(f"Suite: {args.suite}")
    logger.write(f"Alpha: {args.alpha}")
    logger.write(f"Dry-run: {args.dry_run}")

    try:
        if args.suite in {"all", "tmlap"} and not args.skip_instance_generation:
            if args.dry_run:
                logger.write("[INSTANCES] Dry-run activo: no se generan instancias grandes.")
            else:
                ensure_large_tmlap_instances(regenerate=args.regenerate_instances, logger=logger)

        jobs = build_jobs(python_executable=args.python_executable, suite=args.suite)
        logger.write(f"Jobs programados: {len(jobs)}")

        results: list[JobResult] = []
        for idx, job in enumerate(jobs, start=1):
            logger.write(f"Ejecutando job {idx}/{len(jobs)}: {job.name}")
            result = _run_job(job=job, logger=logger, dry_run=args.dry_run)
            results.append(result)

            if result.status == "failed" and args.fail_fast:
                logger.write("Fail-fast habilitado: deteniendo pipeline por error.")
                break

        write_manifest(results=results, run_root=run_root)
        write_summary(results=results, run_root=run_root)

        if not args.skip_stats and not args.dry_run:
            run_statistics(results=results, alpha=args.alpha, run_root=run_root, logger=logger)
        elif args.skip_stats:
            logger.write("[STATS] Omitido por flag --skip-stats")
        else:
            logger.write("[STATS] Omitido en dry-run")

        logger.write(f"Manifiesto CSV: {run_root / 'job_manifest.csv'}")
        logger.write(f"Manifiesto JSON: {run_root / 'job_manifest.json'}")
        logger.write(f"Resumen: {run_root / 'start-tests-summary.md'}")
        logger.write("Pipeline finalizado")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
