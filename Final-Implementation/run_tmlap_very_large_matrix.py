from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent
BASE_CODES_DIR = ROOT_DIR / "base-codes"

DEFAULT_INSTANCES = [
    "4.instancia_very_large_500_hubs.txt",
    "5.instancia_very_large_1000_hubs.txt",
]


@dataclass
class Job:
    name: str
    algorithm: str
    profile_mode: str
    cwd: Path
    command: list[str]


@dataclass
class JobResult:
    name: str
    algorithm: str
    profile_mode: str
    cwd: Path
    command: list[str]
    start_ts: str
    end_ts: str
    duration_sec: float
    exit_code: int
    status: str
    output_dirs: list[Path]


@dataclass
class AnalysisResult:
    algorithm: str
    benchmark_root: Path
    output_dir: Path
    run_names: list[str]
    status: str
    exit_code: int


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
            "Runner secuencial TMLAP very_large/very_very_large para PSO, SHOA y "
            "SHOA+LIME (soft/medium/hard x leader_repulsion/levy_teleport), "
            "con analisis descriptivo y estadistico pairwise."
        )
    )

    parser.add_argument(
        "--instances",
        default=",".join(DEFAULT_INSTANCES),
        help="Instancias separadas por coma",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=2000)

    parser.add_argument("--profiles", default="soft,medium,hard")
    parser.add_argument("--modes", default="leader_repulsion,levy_teleport")
    parser.add_argument(
        "--sholime-config",
        default="base-codes/tmlap_sholime/benchmarks/sholime_tmlap_profiles_config.json",
        help="Ruta al config de perfiles SHOLIME relativa a Final-Implementation",
    )

    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-root", default="final-test-runs")
    parser.add_argument("--tag-prefix", default="tmlap_vl_vvl")

    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-descriptive-analysis", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _to_flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _append_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        if key == "enforce_elite_archive":
            cmd.append("--enforce-elite-archive" if value else "--no-enforce-elite-archive")
            return
        cmd.append(_to_flag(key))
        cmd.append(str(value).lower())
        return

    if isinstance(value, (list, tuple)):
        cmd.append(_to_flag(key))
        cmd.append(",".join(str(item) for item in value))
        return

    cmd.append(_to_flag(key))
    cmd.append(str(value))


def _load_sholime_profile_config(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"No existe config SHOLIME: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_args = dict(payload.get("base_args", {}))
    profiles = dict(payload.get("profiles", {}))

    if not profiles:
        raise ValueError(f"Config SHOLIME sin perfiles: {config_path}")

    return base_args, profiles


def _build_job_command(script_rel: str, python_executable: str, args_map: dict[str, Any]) -> list[str]:
    cmd = [python_executable, script_rel]

    for key, value in args_map.items():
        _append_arg(cmd, key, value)

    return cmd


def build_jobs(args: argparse.Namespace) -> list[Job]:
    instances_csv = ",".join(_parse_csv(args.instances))
    profiles = _parse_csv(args.profiles)
    modes = _parse_csv(args.modes)

    if not profiles:
        raise ValueError("Debes indicar al menos un perfil en --profiles")
    if not modes:
        raise ValueError("Debes indicar al menos un modo en --modes")

    sholime_config_path = (ROOT_DIR / args.sholime_config).resolve()
    sholime_base, sholime_profiles = _load_sholime_profile_config(sholime_config_path)

    missing_profiles = [profile for profile in profiles if profile not in sholime_profiles]
    if missing_profiles:
        available = ", ".join(sorted(sholime_profiles.keys()))
        raise ValueError(
            f"Perfiles no definidos en config SHOLIME: {', '.join(missing_profiles)}. "
            f"Disponibles: {available}"
        )

    valid_modes = {"leader_repulsion", "levy_teleport"}
    invalid_modes = [mode for mode in modes if mode not in valid_modes]
    if invalid_modes:
        raise ValueError(f"Modos invalidos: {', '.join(invalid_modes)}")

    tag_common = f"{args.tag_prefix}_r{args.runs}_i{args.max_iter}"

    jobs: list[Job] = []

    jobs.append(
        Job(
            name="tmlap_pso_very_large",
            algorithm="pso",
            profile_mode="none",
            cwd=BASE_CODES_DIR / "tmlap_pso",
            command=_build_job_command(
                "benchmarks/run_tmlap_pso_benchmark.py",
                args.python_executable,
                {
                    "instances": instances_csv,
                    "instance_dir": ".",
                    "runs": args.runs,
                    "seed_start": args.seed_start,
                    "pop_size": args.pop_size,
                    "max_iter": args.max_iter,
                    "output_dir": "results/benchmark_logs",
                    "tag": f"{tag_common}_pso",
                },
            ),
        )
    )

    jobs.append(
        Job(
            name="tmlap_shoa_very_large",
            algorithm="shoa",
            profile_mode="none",
            cwd=BASE_CODES_DIR / "tmlap_shoa",
            command=_build_job_command(
                "benchmarks/run_tmlap_shoa_puro_benchmark.py",
                args.python_executable,
                {
                    "instances": instances_csv,
                    "instance_dir": ".",
                    "runs": args.runs,
                    "seed_start": args.seed_start,
                    "pop_size": args.pop_size,
                    "max_iter": args.max_iter,
                    "output_dir": "results/benchmark_logs",
                    "tag": f"{tag_common}_shoa",
                },
            ),
        )
    )

    for profile in profiles:
        profile_overrides = dict(sholime_profiles[profile])

        for mode in modes:
            merged = {**sholime_base, **profile_overrides}
            merged["instances"] = instances_csv
            merged["instance_dir"] = "."
            merged["runs"] = args.runs
            merged["seed_start"] = args.seed_start
            merged["pop_size"] = args.pop_size
            merged["max_iter"] = args.max_iter
            merged["rescue_mode"] = mode
            merged["output_dir"] = "results/benchmark_logs"
            merged["tag"] = f"{tag_common}_sholime_{profile}_{mode}"

            profile_mode = f"{profile}_{mode}"
            jobs.append(
                Job(
                    name=f"tmlap_sholime_{profile_mode}",
                    algorithm="sholime",
                    profile_mode=profile_mode,
                    cwd=BASE_CODES_DIR / "tmlap_sholime",
                    command=_build_job_command(
                        "benchmarks/run_tmlap_sholime_benchmark.py",
                        args.python_executable,
                        merged,
                    ),
                )
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


def _run_subprocess_with_stream(
    command: list[str],
    cwd: Path,
    logger: Logger,
    prefix: str,
    dry_run: bool,
) -> int:
    logger.write(f"[{prefix}] CWD={cwd}")
    logger.write(f"[{prefix}] CMD={shlex.join(command)}")

    if dry_run:
        logger.write(f"[{prefix}] DRY-RUN: comando no ejecutado")
        return 0

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    for line in process.stdout:
        logger.write(f"[{prefix}] {line.rstrip()}")

    return int(process.wait())


def run_job(job: Job, logger: Logger, dry_run: bool) -> JobResult:
    before_dirs = _snapshot_output_dirs(job.cwd)
    start = time.time()
    start_ts = datetime.now().isoformat(timespec="seconds")

    logger.write(f"[JOB START] {job.name}")
    exit_code = _run_subprocess_with_stream(job.command, job.cwd, logger, job.name, dry_run)

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
        algorithm=job.algorithm,
        profile_mode=job.profile_mode,
        cwd=job.cwd,
        command=job.command,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_sec=duration_sec,
        exit_code=exit_code,
        status=status,
        output_dirs=new_dirs,
    )


def run_descriptive_analysis(
    results: list[JobResult],
    python_executable: str,
    run_root: Path,
    logger: Logger,
    dry_run: bool,
) -> list[AnalysisResult]:
    analysis_results: list[AnalysisResult] = []

    algo_specs = {
        "pso": {
            "cwd": BASE_CODES_DIR / "tmlap_pso",
            "script": "benchmarks/analyze_tmlap_benchmark_logs.py",
            "benchmark_root": BASE_CODES_DIR / "tmlap_pso" / "results" / "benchmark_logs",
        },
        "shoa": {
            "cwd": BASE_CODES_DIR / "tmlap_shoa",
            "script": "benchmarks/analyze_tmlap_benchmark_logs.py",
            "benchmark_root": BASE_CODES_DIR / "tmlap_shoa" / "results" / "benchmark_logs",
        },
        "sholime": {
            "cwd": BASE_CODES_DIR / "tmlap_sholime",
            "script": "benchmarks/analyze_tmlap_benchmark_logs.py",
            "benchmark_root": BASE_CODES_DIR / "tmlap_sholime" / "results" / "benchmark_logs",
        },
    }

    runs_by_algo: dict[str, list[Path]] = {"pso": [], "shoa": [], "sholime": []}
    for result in results:
        if result.status != "success":
            continue
        if result.algorithm not in runs_by_algo:
            continue
        runs_by_algo[result.algorithm].extend(result.output_dirs)

    for algorithm, spec in algo_specs.items():
        run_dirs = sorted(set(runs_by_algo[algorithm]))
        run_names = [path.name for path in run_dirs]

        if not run_names:
            logger.write(f"[ANALYSIS] {algorithm}: sin corridas para analizar")
            analysis_results.append(
                AnalysisResult(
                    algorithm=algorithm,
                    benchmark_root=spec["benchmark_root"],
                    output_dir=run_root / "analysis" / "descriptive" / algorithm,
                    run_names=[],
                    status="skipped",
                    exit_code=0,
                )
            )
            continue

        output_dir = (run_root / "analysis" / "descriptive" / algorithm).resolve()
        command = [
            python_executable,
            spec["script"],
            "--benchmark-root",
            str(spec["benchmark_root"]),
            "--runs",
            ",".join(run_names),
            "--output-dir",
            str(output_dir),
        ]

        logger.write(f"[ANALYSIS START] {algorithm} runs={len(run_names)}")
        exit_code = _run_subprocess_with_stream(
            command=command,
            cwd=spec["cwd"],
            logger=logger,
            prefix=f"analysis_{algorithm}",
            dry_run=dry_run,
        )
        status = "success" if exit_code == 0 else "failed"
        logger.write(f"[ANALYSIS END] {algorithm} status={status} exit_code={exit_code}")

        analysis_results.append(
            AnalysisResult(
                algorithm=algorithm,
                benchmark_root=spec["benchmark_root"],
                output_dir=output_dir,
                run_names=run_names,
                status=status,
                exit_code=exit_code,
            )
        )

    return analysis_results


def _load_runs_raw(output_dir: Path, algorithm: str, profile_mode: str) -> pd.DataFrame:
    csv_path = output_dir / "runs_raw.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required = ["function", "dimension", "run_id", "seed", "best_fitness"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df[required].copy()
    out["algorithm"] = algorithm
    out["profile_mode"] = profile_mode
    out["output_dir"] = str(output_dir)

    out["dimension"] = pd.to_numeric(out["dimension"], errors="coerce")
    out["run_id"] = pd.to_numeric(out["run_id"], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["best_fitness"] = pd.to_numeric(out["best_fitness"], errors="coerce")

    out = out.dropna(subset=["dimension", "run_id", "seed", "best_fitness"])
    if out.empty:
        return pd.DataFrame()

    out["dimension"] = out["dimension"].astype(int)
    out["run_id"] = out["run_id"].astype(int)
    out["seed"] = out["seed"].astype(int)
    out["function"] = out["function"].astype(str)

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


def _trend_from_delta(delta: float, algo_a: str, algo_b: str) -> str:
    if np.isnan(delta):
        return "sin_datos"
    if delta < 0:
        return f"{algo_b} mejor (menor fitness)"
    if delta > 0:
        return f"{algo_a} mejor (menor fitness)"
    return "Empate"


def _build_pairwise_stats(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    comparison: str,
    algo_a: str,
    algo_b: str,
    profile_mode: str,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normality_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    keys_a = set(zip(df_a["function"], df_a["dimension"])) if not df_a.empty else set()
    keys_b = set(zip(df_b["function"], df_b["dimension"])) if not df_b.empty else set()
    all_keys = sorted(keys_a.union(keys_b), key=lambda item: (str(item[0]), int(item[1])))

    for function, dimension in all_keys:
        subset_a = df_a[(df_a["function"] == function) & (df_a["dimension"] == int(dimension))]
        subset_b = df_b[(df_b["function"] == function) & (df_b["dimension"] == int(dimension))]

        values_a = subset_a["best_fitness"].to_numpy(dtype=float)
        values_b = subset_b["best_fitness"].to_numpy(dtype=float)

        paired = pd.merge(
            subset_a[["run_id", "seed", "best_fitness"]],
            subset_b[["run_id", "seed", "best_fitness"]],
            on=["run_id", "seed"],
            suffixes=("_a", "_b"),
        )

        paired_count = int(paired.shape[0])
        delta_values = (
            paired["best_fitness_b"].to_numpy(dtype=float)
            - paired["best_fitness_a"].to_numpy(dtype=float)
            if paired_count > 0
            else np.array([], dtype=float)
        )

        p_a, status_a = _shapiro(values_a)
        p_b, status_b = _shapiro(values_b)
        p_delta, status_delta = _shapiro(delta_values)

        normality_rows.append(
            {
                "comparison": comparison,
                "algorithm_a": algo_a,
                "algorithm_b": algo_b,
                "function": function,
                "dimension": int(dimension),
                "profile_mode": profile_mode,
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
                "n_pairs": paired_count,
                "a_shapiro_p": p_a,
                "a_status": status_a,
                "b_shapiro_p": p_b,
                "b_status": status_b,
                "delta_shapiro_p": p_delta,
                "delta_status": status_delta,
            }
        )

        test_used = ""
        statistic = float("nan")
        p_value = float("nan")
        notes = ""

        if paired_count > 0:
            test_used = "wilcoxon_paired"
            x_vals = paired["best_fitness_a"].to_numpy(dtype=float)
            y_vals = paired["best_fitness_b"].to_numpy(dtype=float)

            if np.allclose(x_vals, y_vals):
                statistic = 0.0
                p_value = 1.0
                notes = "all_pairs_equal"
            else:
                try:
                    result = stats.wilcoxon(x_vals, y_vals, alternative="two-sided")
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                except Exception as exc:
                    notes = f"wilcoxon_error:{type(exc).__name__}"

            median_delta = float(np.median(delta_values)) if delta_values.size > 0 else float("nan")
            mean_delta = float(np.mean(delta_values)) if delta_values.size > 0 else float("nan")
        else:
            test_used = "mannwhitneyu"
            if values_a.size > 0 and values_b.size > 0:
                try:
                    result = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                except Exception as exc:
                    notes = f"mannwhitneyu_error:{type(exc).__name__}"
            else:
                notes = "insufficient_data"

            if values_a.size > 0 and values_b.size > 0:
                median_delta = float(np.median(values_b) - np.median(values_a))
                mean_delta = float(np.mean(values_b) - np.mean(values_a))
            else:
                median_delta = float("nan")
                mean_delta = float("nan")

        decision = "Rechazar H0" if np.isfinite(p_value) and p_value < alpha else "No rechazar H0"
        trend = _trend_from_delta(median_delta, algo_a=algo_a, algo_b=algo_b)

        comparison_rows.append(
            {
                "comparison": comparison,
                "algorithm_a": algo_a,
                "algorithm_b": algo_b,
                "function": function,
                "dimension": int(dimension),
                "profile_mode": profile_mode,
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
                "n_pairs": paired_count,
                "test_used": test_used,
                "statistic": statistic,
                "p_value": p_value,
                "decision_alpha": decision,
                "median_delta_b_minus_a": median_delta,
                "mean_delta_b_minus_a": mean_delta,
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
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(_format_float(value, digits=6))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def run_pairwise_statistics(
    results: list[JobResult],
    alpha: float,
    run_root: Path,
    logger: Logger,
) -> None:
    stats_root = run_root / "analysis" / "statistics"
    stats_root.mkdir(parents=True, exist_ok=True)

    frames = []
    for result in results:
        if result.status != "success":
            continue
        for output_dir in result.output_dirs:
            loaded = _load_runs_raw(
                output_dir=output_dir,
                algorithm=result.algorithm,
                profile_mode=result.profile_mode,
            )
            if not loaded.empty:
                frames.append(loaded)

    if not frames:
        logger.write("[STATS] Sin datos para estadistica pairwise")
        pd.DataFrame().to_csv(stats_root / "normality_shapiro_pairwise.csv", index=False)
        pd.DataFrame().to_csv(stats_root / "comparisons_pairwise.csv", index=False)
        (stats_root / "reporte_estadistico_pairwise.md").write_text(
            "# Reporte Estadistico Pairwise\n\nNo hay datos para analizar.",
            encoding="utf-8",
        )
        return

    all_df = pd.concat(frames, ignore_index=True)

    pso_df = all_df[all_df["algorithm"] == "pso"].copy()
    shoa_df = all_df[all_df["algorithm"] == "shoa"].copy()
    sholime_df = all_df[all_df["algorithm"] == "sholime"].copy()

    normality_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    # PSO vs SHOA (sin perfil)
    n_rows, c_rows = _build_pairwise_stats(
        df_a=pso_df,
        df_b=shoa_df,
        comparison="pso_vs_shoa",
        algo_a="pso",
        algo_b="shoa",
        profile_mode="none",
        alpha=alpha,
    )
    normality_rows.extend(n_rows)
    comparison_rows.extend(c_rows)

    # SHOA vs SHOLIME por profile_mode
    for profile_mode, sholime_subset in sholime_df.groupby("profile_mode", dropna=False):
        n_rows, c_rows = _build_pairwise_stats(
            df_a=shoa_df,
            df_b=sholime_subset,
            comparison="shoa_vs_sholime",
            algo_a="shoa",
            algo_b="sholime",
            profile_mode=str(profile_mode),
            alpha=alpha,
        )
        normality_rows.extend(n_rows)
        comparison_rows.extend(c_rows)

    # PSO vs SHOLIME por profile_mode
    for profile_mode, sholime_subset in sholime_df.groupby("profile_mode", dropna=False):
        n_rows, c_rows = _build_pairwise_stats(
            df_a=pso_df,
            df_b=sholime_subset,
            comparison="pso_vs_sholime",
            algo_a="pso",
            algo_b="sholime",
            profile_mode=str(profile_mode),
            alpha=alpha,
        )
        normality_rows.extend(n_rows)
        comparison_rows.extend(c_rows)

    normality_path = stats_root / "normality_shapiro_pairwise.csv"
    comparisons_path = stats_root / "comparisons_pairwise.csv"
    report_path = stats_root / "reporte_estadistico_pairwise.md"

    pd.DataFrame(normality_rows).to_csv(normality_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)

    summary_by_comp = {}
    for row in comparison_rows:
        key = f"{row.get('comparison')}::{row.get('profile_mode')}"
        if key not in summary_by_comp:
            summary_by_comp[key] = {"total": 0, "reject": 0}
        summary_by_comp[key]["total"] += 1
        if row.get("decision_alpha") == "Rechazar H0":
            summary_by_comp[key]["reject"] += 1

    report_lines: list[str] = []
    report_lines.append("# Reporte Estadistico Pairwise: TMLAP")
    report_lines.append("")
    report_lines.append(f"Fecha: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append("")
    report_lines.append("## 1) Objetivo")
    report_lines.append("")
    report_lines.append(
        "Evaluar diferencias de rendimiento en TMLAP (instancias very_large y very_very_large) "
        "entre PSO, SHOA y SHOA+LIME con comparaciones pairwise."
    )
    report_lines.append("")
    report_lines.append("## 2) Metodologia")
    report_lines.append("")
    report_lines.append(f"- Nivel de significancia: alpha = {alpha}")
    report_lines.append("- Normalidad: Shapiro-Wilk en muestras A/B y delta cuando hay pares.")
    report_lines.append("- Test principal: Wilcoxon pareado si hay emparejamiento por run_id/seed.")
    report_lines.append("- Fallback: Mann-Whitney U cuando no hay pares validos.")
    report_lines.append("")

    report_lines.append("## 3) Normalidad")
    report_lines.append("")
    normality_cols = [
        "comparison",
        "algorithm_a",
        "algorithm_b",
        "function",
        "dimension",
        "profile_mode",
        "n_a",
        "n_b",
        "n_pairs",
        "a_shapiro_p",
        "a_status",
        "b_shapiro_p",
        "b_status",
        "delta_shapiro_p",
        "delta_status",
    ]
    report_lines.append(_markdown_table(normality_rows, normality_cols))
    report_lines.append("")

    report_lines.append("## 4) Comparaciones")
    report_lines.append("")
    comparison_cols = [
        "comparison",
        "algorithm_a",
        "algorithm_b",
        "function",
        "dimension",
        "profile_mode",
        "n_a",
        "n_b",
        "n_pairs",
        "test_used",
        "statistic",
        "p_value",
        "decision_alpha",
        "median_delta_b_minus_a",
        "mean_delta_b_minus_a",
        "trend",
        "notes",
    ]
    report_lines.append(_markdown_table(comparison_rows, comparison_cols))
    report_lines.append("")

    report_lines.append("## 5) Resumen por Comparacion")
    report_lines.append("")
    report_lines.append("| comparison_profile | total_groups | reject_h0 | no_reject_h0 |")
    report_lines.append("|---|---:|---:|---:|")
    for key in sorted(summary_by_comp.keys()):
        total = int(summary_by_comp[key]["total"])
        reject = int(summary_by_comp[key]["reject"])
        no_reject = max(total - reject, 0)
        report_lines.append(f"| {key} | {total} | {reject} | {no_reject} |")

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
                "algorithm": result.algorithm,
                "profile_mode": result.profile_mode,
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

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "algorithm",
                "profile_mode",
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

    payload = [
        {
            "name": result.name,
            "algorithm": result.algorithm,
            "profile_mode": result.profile_mode,
            "cwd": str(result.cwd),
            "command": result.command,
            "start_ts": result.start_ts,
            "end_ts": result.end_ts,
            "duration_sec": result.duration_sec,
            "exit_code": result.exit_code,
            "status": result.status,
            "output_dirs": [str(path) for path in result.output_dirs],
        }
        for result in results
    ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_analysis_manifest(analysis_results: list[AnalysisResult], run_root: Path) -> None:
    csv_path = run_root / "analysis_manifest.csv"
    json_path = run_root / "analysis_manifest.json"

    rows = []
    for item in analysis_results:
        rows.append(
            {
                "algorithm": item.algorithm,
                "benchmark_root": str(item.benchmark_root),
                "output_dir": str(item.output_dir),
                "run_names": ";".join(item.run_names),
                "status": item.status,
                "exit_code": item.exit_code,
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "benchmark_root",
                "output_dir",
                "run_names",
                "status",
                "exit_code",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = [
        {
            "algorithm": item.algorithm,
            "benchmark_root": str(item.benchmark_root),
            "output_dir": str(item.output_dir),
            "run_names": item.run_names,
            "status": item.status,
            "exit_code": item.exit_code,
        }
        for item in analysis_results
    ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(
    results: list[JobResult],
    analysis_results: list[AnalysisResult],
    run_root: Path,
) -> None:
    summary_path = run_root / "run_tmlap_very_large_summary.md"

    total = len(results)
    ok = sum(1 for result in results if result.status == "success")
    failed = total - ok

    lines: list[str] = []
    lines.append("# TMLAP Very Large Matrix Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Total benchmark jobs: {total}")
    lines.append(f"- Benchmark success: {ok}")
    lines.append(f"- Benchmark failed: {failed}")
    lines.append("")

    lines.append("## Benchmark Jobs")
    lines.append("")
    lines.append("| name | algorithm | profile_mode | status | exit_code | duration_sec | output_dirs |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for result in results:
        output_dirs = "<br>".join(str(path) for path in result.output_dirs) if result.output_dirs else "-"
        lines.append(
            f"| {result.name} | {result.algorithm} | {result.profile_mode} | {result.status} "
            f"| {result.exit_code} | {result.duration_sec:.2f} | {output_dirs} |"
        )

    lines.append("")
    lines.append("## Descriptive Analysis")
    lines.append("")
    lines.append("| algorithm | status | exit_code | runs | output_dir |")
    lines.append("|---|---|---:|---|---|")
    for item in analysis_results:
        runs = "<br>".join(item.run_names) if item.run_names else "-"
        lines.append(
            f"| {item.algorithm} | {item.status} | {item.exit_code} | {runs} | {item.output_dir} |"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (ROOT_DIR / args.output_root / f"tmlap_very_large_matrix_{timestamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    logger = Logger(run_root / "run_tmlap_very_large_matrix.log")
    logger.write("Inicio run_tmlap_very_large_matrix.py (secuencial)")
    logger.write(f"Instances: {args.instances}")
    logger.write(f"Runs: {args.runs}")
    logger.write(f"Max iter: {args.max_iter}")
    logger.write(f"Profiles: {args.profiles}")
    logger.write(f"Modes: {args.modes}")
    logger.write(f"Dry-run: {args.dry_run}")

    try:
        jobs = build_jobs(args)
        logger.write(f"Jobs programados: {len(jobs)}")

        results: list[JobResult] = []
        for idx, job in enumerate(jobs, start=1):
            logger.write(f"Ejecutando benchmark job {idx}/{len(jobs)}: {job.name}")
            result = run_job(job=job, logger=logger, dry_run=args.dry_run)
            results.append(result)

            if result.status == "failed" and args.fail_fast:
                logger.write("Fail-fast activo: deteniendo ejecucion por error")
                break

        write_manifest(results=results, run_root=run_root)

        analysis_results: list[AnalysisResult] = []
        if args.skip_descriptive_analysis:
            logger.write("[ANALYSIS] Omitido por flag --skip-descriptive-analysis")
        else:
            analysis_results = run_descriptive_analysis(
                results=results,
                python_executable=args.python_executable,
                run_root=run_root,
                logger=logger,
                dry_run=args.dry_run,
            )
            write_analysis_manifest(analysis_results=analysis_results, run_root=run_root)

        if args.skip_stats:
            logger.write("[STATS] Omitido por flag --skip-stats")
        elif args.dry_run:
            logger.write("[STATS] Omitido en dry-run")
        else:
            run_pairwise_statistics(
                results=results,
                alpha=args.alpha,
                run_root=run_root,
                logger=logger,
            )

        write_summary(results=results, analysis_results=analysis_results, run_root=run_root)

        logger.write(f"Manifiesto benchmark CSV: {run_root / 'job_manifest.csv'}")
        logger.write(f"Manifiesto benchmark JSON: {run_root / 'job_manifest.json'}")
        if analysis_results:
            logger.write(f"Manifiesto analysis CSV: {run_root / 'analysis_manifest.csv'}")
            logger.write(f"Manifiesto analysis JSON: {run_root / 'analysis_manifest.json'}")
        logger.write(f"Resumen: {run_root / 'run_tmlap_very_large_summary.md'}")
        logger.write("Pipeline finalizado")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
