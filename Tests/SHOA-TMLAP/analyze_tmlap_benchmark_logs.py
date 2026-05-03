"""Analyze TMLAP benchmark logs with per-iteration statistics and plots.

Expected per-run files (same schema as SHO+LIME benchmark runner):
- full_output.csv
- runs_raw.csv
- summary_by_function.csv (optional)
- lime_contributions.csv (optional)
- config_used.json (optional)

Outputs:
- analysis_<timestamp>/
  - per_run/<run_name>/csv/*.csv
  - per_run/<run_name>/plots/*.png
  - global/csv/*.csv
  - global/plots/*.png
  - analysis_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "No se pudo importar pandas. Instala dependencias con: pip install pandas matplotlib numpy"
    ) from exc


FEATURE_COLUMNS = [
    "weight_r1",
    "weight_mag_browniano",
    "weight_mag_levy",
    "weight_r2",
    "weight_mag_predacion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza benchmark logs TMLAP (stats + graficos por iteracion y por perfil)"
    )
    parser.add_argument(
        "--benchmark-root",
        default="benchmark_logs",
        help="Carpeta raiz que contiene corridas tmlap_sholime_*",
    )
    parser.add_argument(
        "--runs",
        default="all",
        help="Lista separada por coma de carpetas run a analizar o 'all'",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directorio de salida. Si se omite, se crea dentro de benchmark-root",
    )
    parser.add_argument(
        "--no-show-fliers",
        action="store_true",
        help="Oculta outliers en boxplots",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_root(raw_root: str) -> Path:
    root = Path(raw_root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"No existe benchmark root: {root}")
    return root


def _discover_run_dirs(root: Path, runs_arg: str) -> list[Path]:
    if runs_arg.strip().lower() == "all":
        discovered = sorted(
            [
                path
                for path in root.iterdir()
                if path.is_dir() and (path / "full_output.csv").exists() and (path / "runs_raw.csv").exists()
            ]
        )
        if not discovered:
            raise ValueError(f"No se encontraron corridas validas en: {root}")
        return discovered

    selected: list[Path] = []
    for token in _parse_csv_list(runs_arg):
        candidate = root / token
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"No existe corrida: {candidate}")
        if not (candidate / "full_output.csv").exists() or not (candidate / "runs_raw.csv").exists():
            raise ValueError(f"Corrida incompleta (faltan CSV): {candidate}")
        selected.append(candidate)
    return selected


def _infer_profile(run_name: str) -> str:
    lname = run_name.lower()
    for profile in ("soft", "medium", "hard"):
        if lname.endswith(f"_{profile}"):
            return profile
    return "unknown"


def _infer_algorithm(run_name: str) -> str:
    lname = run_name.lower()
    if lname.startswith("tmlap_sholime_"):
        return "sholime"
    if lname.startswith("tmlap_pso_"):
        return "pso"
    if lname.startswith("tmlap_shoa_puro_"):
        return "shoa_puro"
    return "unknown"


def _safe_std(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    value = float(series.std(ddof=1))
    if np.isnan(value):
        return 0.0
    return value


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _iteration_stats(full_df: pd.DataFrame) -> pd.DataFrame:
    grouped = full_df.groupby(["function", "dimension", "iteration"], dropna=False, as_index=False)

    rows = []
    for (function, dimension, iteration), group in grouped:
        values = group["best_fitness"].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "function": function,
                "dimension": int(dimension),
                "iteration": int(iteration),
                "count": int(values.shape[0]),
                "best": float(values.min()),
                "worst": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": _safe_std(values),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
        )

    return pd.DataFrame(rows).sort_values(["function", "dimension", "iteration"]).reset_index(drop=True)


def _lime_iteration_stats(lime_df: pd.DataFrame) -> pd.DataFrame:
    if lime_df.empty:
        return pd.DataFrame()

    required = ["function", "dimension", "diagnosis_iteration"]
    for column in required:
        if column not in lime_df.columns:
            return pd.DataFrame()

    grouped = lime_df.groupby(["function", "dimension", "diagnosis_iteration"], dropna=False, as_index=False)

    rows = []
    for (function, dimension, iteration), group in grouped:
        row = {
            "function": function,
            "dimension": int(dimension),
            "diagnosis_iteration": int(iteration),
            "count": int(group.shape[0]),
        }

        if "pop_size" in group.columns:
            pop_size_values = pd.to_numeric(group["pop_size"], errors="coerce").dropna()
            row["pop_size"] = int(pop_size_values.iloc[0]) if not pop_size_values.empty else np.nan
        else:
            row["pop_size"] = np.nan

        if "max_iter" in group.columns:
            max_iter_values = pd.to_numeric(group["max_iter"], errors="coerce").dropna()
            row["max_iter"] = int(max_iter_values.iloc[0]) if not max_iter_values.empty else np.nan
        else:
            row["max_iter"] = np.nan

        for feature in FEATURE_COLUMNS:
            if feature in group.columns:
                values = pd.to_numeric(group[feature], errors="coerce")
                row[f"mean_{feature}"] = float(values.mean()) if not values.dropna().empty else np.nan
                row[f"mean_abs_{feature}"] = (
                    float(values.abs().mean()) if not values.dropna().empty else np.nan
                )
            else:
                row[f"mean_{feature}"] = np.nan
                row[f"mean_abs_{feature}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["function", "dimension", "diagnosis_iteration"]).reset_index(drop=True)


def _plot_convergence(iter_stats: pd.DataFrame, function: str, dimension: int, out_path: Path) -> None:
    subset = iter_stats[(iter_stats["function"] == function) & (iter_stats["dimension"] == dimension)]
    if subset.empty:
        return

    x = subset["iteration"].to_numpy(dtype=int)
    mean = subset["mean"].to_numpy(dtype=float)
    std = subset["std"].to_numpy(dtype=float)
    best = subset["best"].to_numpy(dtype=float)
    worst = subset["worst"].to_numpy(dtype=float)

    plt.figure(figsize=(12, 6))
    plt.plot(x, mean, color="#1f77b4", linewidth=2.0, label="mean(best_fitness)")
    plt.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.20, label="mean ± std")
    plt.fill_between(x, best, worst, color="#ff7f0e", alpha=0.15, label="best..worst")
    plt.xlabel("iteration")
    plt.ylabel("best_fitness")
    plt.title(f"Convergence: {function} (dim={dimension})")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_iteration_boxplot(
    full_df: pd.DataFrame,
    function: str,
    dimension: int,
    out_path: Path,
    show_fliers: bool,
) -> None:
    subset = full_df[(full_df["function"] == function) & (full_df["dimension"] == dimension)]
    if subset.empty:
        return

    grouped = subset.groupby("iteration")["best_fitness"].apply(list)
    if grouped.empty:
        return

    positions = grouped.index.to_numpy(dtype=int)
    values = grouped.to_list()

    plt.figure(figsize=(max(14, len(positions) * 0.04), 6))
    plt.boxplot(
        values,
        positions=positions,
        widths=0.5,
        showfliers=show_fliers,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#1f77b4", "linewidth": 0.8},
        whiskerprops={"color": "#1f77b4", "linewidth": 0.8},
        capprops={"color": "#1f77b4", "linewidth": 0.8},
        medianprops={"color": "#d62728", "linewidth": 1.0},
    )
    plt.xlabel("iteration")
    plt.ylabel("best_fitness across runs")
    plt.title(f"Boxplot by iteration: {function} (dim={dimension})")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_lime_abs(lime_stats: pd.DataFrame, function: str, dimension: int, out_path: Path) -> None:
    if lime_stats.empty:
        return
    required = {"function", "dimension", "diagnosis_iteration"}
    if not required.issubset(set(lime_stats.columns)):
        return

    subset = lime_stats[(lime_stats["function"] == function) & (lime_stats["dimension"] == dimension)]
    if subset.empty:
        return

    x = subset["diagnosis_iteration"].to_numpy(dtype=int)

    title_meta = [f"dim={dimension}"]
    if "pop_size" in subset.columns:
        pop_size_values = pd.to_numeric(subset["pop_size"], errors="coerce").dropna()
        if not pop_size_values.empty:
            title_meta.append(f"pop={int(pop_size_values.iloc[0])}")
    if "max_iter" in subset.columns:
        max_iter_values = pd.to_numeric(subset["max_iter"], errors="coerce").dropna()
        if not max_iter_values.empty:
            title_meta.append(f"iters={int(max_iter_values.iloc[0])}")

    plt.figure(figsize=(12, 6))
    for feature in FEATURE_COLUMNS:
        col = f"mean_abs_{feature}"
        if col not in subset.columns:
            continue
        y = subset[col].to_numpy(dtype=float)
        plt.plot(x, y, linewidth=1.8, label=feature)

    plt.xlabel("diagnosis_iteration")
    plt.ylabel("mean abs contribution")
    plt.title(f"LIME abs contribution by iteration: {function} ({', '.join(title_meta)})")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_final_boxplot_by_function(runs_raw: pd.DataFrame, out_path: Path, show_fliers: bool) -> None:
    if runs_raw.empty:
        return

    grouped = runs_raw.groupby("function")["best_fitness"].apply(list)
    if grouped.empty:
        return

    labels = grouped.index.to_list()
    values = grouped.to_list()

    plt.figure(figsize=(max(8, len(labels) * 1.3), 6))
    plt.boxplot(
        values,
        tick_labels=labels,
        showfliers=show_fliers,
        patch_artist=True,
        boxprops={"facecolor": "#c7e9c0", "edgecolor": "#2ca02c", "linewidth": 0.8},
        whiskerprops={"color": "#2ca02c", "linewidth": 0.8},
        capprops={"color": "#2ca02c", "linewidth": 0.8},
        medianprops={"color": "#d62728", "linewidth": 1.0},
    )
    plt.ylabel("final best_fitness")
    plt.title("Final fitness distribution by instance")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_profile_final_boxplot(
    combined_runs_raw: pd.DataFrame,
    function: str,
    out_path: Path,
    show_fliers: bool,
) -> None:
    subset = combined_runs_raw[combined_runs_raw["function"] == function]
    if subset.empty:
        return

    grouped = subset.groupby("profile")["best_fitness"].apply(list)
    if grouped.empty:
        return

    labels = grouped.index.to_list()
    values = grouped.to_list()

    plt.figure(figsize=(8, 6))
    plt.boxplot(
        values,
        tick_labels=labels,
        showfliers=show_fliers,
        patch_artist=True,
        boxprops={"facecolor": "#fdd0a2", "edgecolor": "#ff7f0e", "linewidth": 0.8},
        whiskerprops={"color": "#ff7f0e", "linewidth": 0.8},
        capprops={"color": "#ff7f0e", "linewidth": 0.8},
        medianprops={"color": "#d62728", "linewidth": 1.0},
    )
    plt.ylabel("final best_fitness")
    plt.title(f"Profile comparison (final): {function}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_profile_convergence(combined_full: pd.DataFrame, function: str, out_path: Path) -> None:
    subset = combined_full[combined_full["function"] == function]
    if subset.empty:
        return

    plt.figure(figsize=(12, 6))

    for profile, profile_df in subset.groupby("profile"):
        grouped = profile_df.groupby("iteration")["best_fitness"]
        mean = grouped.mean()
        std = grouped.std(ddof=1).fillna(0.0)

        x = mean.index.to_numpy(dtype=int)
        y = mean.to_numpy(dtype=float)
        s = std.to_numpy(dtype=float)

        plt.plot(x, y, linewidth=2.0, label=f"{profile} mean")
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("iteration")
    plt.ylabel("best_fitness")
    plt.title(f"Profile convergence comparison: {function}")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _run_analysis_for_single_run(
    run_dir: Path,
    out_root: Path,
    show_fliers: bool,
) -> dict:
    run_name = run_dir.name
    profile = _infer_profile(run_name)
    algorithm = _infer_algorithm(run_name)

    per_run_root = out_root / "per_run" / run_name
    csv_dir = per_run_root / "csv"
    plots_dir = per_run_root / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(run_dir / "full_output.csv")
    runs_raw_df = pd.read_csv(run_dir / "runs_raw.csv")

    lime_path = run_dir / "lime_contributions.csv"
    lime_df = pd.read_csv(lime_path) if lime_path.exists() else pd.DataFrame()

    full_df = _to_numeric(full_df, ["dimension", "run_id", "seed", "iteration", "best_fitness"])
    full_df = full_df.dropna(subset=["dimension", "run_id", "seed", "iteration", "best_fitness"])
    full_df["dimension"] = full_df["dimension"].astype(int)
    full_df["run_id"] = full_df["run_id"].astype(int)
    full_df["seed"] = full_df["seed"].astype(int)
    full_df["iteration"] = full_df["iteration"].astype(int)
    full_df["function"] = full_df["function"].astype(str)

    runs_raw_df = _to_numeric(runs_raw_df, ["dimension", "run_id", "seed", "best_fitness"])
    runs_raw_df = runs_raw_df.dropna(subset=["dimension", "run_id", "seed", "best_fitness"])
    runs_raw_df["dimension"] = runs_raw_df["dimension"].astype(int)
    runs_raw_df["run_id"] = runs_raw_df["run_id"].astype(int)
    runs_raw_df["seed"] = runs_raw_df["seed"].astype(int)
    runs_raw_df["function"] = runs_raw_df["function"].astype(str)

    if not lime_df.empty:
        lime_df = _to_numeric(
            lime_df,
            ["dimension", "run_id", "seed", "diagnosis_iteration"] + FEATURE_COLUMNS,
        )
        lime_df = lime_df.dropna(subset=["dimension", "run_id", "seed", "diagnosis_iteration"])
        lime_df["dimension"] = lime_df["dimension"].astype(int)
        lime_df["run_id"] = lime_df["run_id"].astype(int)
        lime_df["seed"] = lime_df["seed"].astype(int)
        lime_df["diagnosis_iteration"] = lime_df["diagnosis_iteration"].astype(int)
        lime_df["function"] = lime_df["function"].astype(str)

    summary_ext = (
        runs_raw_df.groupby(["function", "dimension"], as_index=False)["best_fitness"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .reset_index()
    )
    summary_ext = summary_ext.rename(
        columns={
            "count": "runs",
            "min": "best",
            "max": "worst",
            "std": "std",
        }
    )
    summary_ext["std"] = summary_ext["std"].fillna(0.0)

    iter_stats = _iteration_stats(full_df)
    lime_stats = _lime_iteration_stats(lime_df)

    summary_ext.to_csv(csv_dir / "summary_extended.csv", index=False)
    iter_stats.to_csv(csv_dir / "iteration_stats.csv", index=False)
    lime_stats.to_csv(csv_dir / "lime_iteration_stats.csv", index=False)

    _plot_final_boxplot_by_function(runs_raw_df, plots_dir / "boxplot_final_by_instance.png", show_fliers)

    for (function, dimension), _ in runs_raw_df.groupby(["function", "dimension"]):
        safe_name = function.replace("/", "_").replace(" ", "_")
        _plot_convergence(
            iter_stats,
            function=function,
            dimension=int(dimension),
            out_path=plots_dir / f"convergence_{safe_name}.png",
        )
        _plot_iteration_boxplot(
            full_df,
            function=function,
            dimension=int(dimension),
            out_path=plots_dir / f"boxplot_by_iteration_{safe_name}.png",
            show_fliers=show_fliers,
        )
        if algorithm not in {"pso", "shoa_puro"}:
            _plot_lime_abs(
                lime_stats,
                function=function,
                dimension=int(dimension),
                out_path=plots_dir / f"lime_abs_contrib_{safe_name}.png",
            )

    config_payload = {}
    config_path = run_dir / "config_used.json"
    if config_path.exists():
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config_payload = {}

    return {
        "run_name": run_name,
        "algorithm": algorithm,
        "profile": profile,
        "run_dir": run_dir,
        "summary_ext": summary_ext,
        "iter_stats": iter_stats,
        "full_df": full_df,
        "runs_raw_df": runs_raw_df,
        "lime_stats": lime_stats,
        "config": config_payload,
    }


def _build_global_outputs(run_reports: list[dict], out_root: Path, show_fliers: bool) -> None:
    global_csv = out_root / "global" / "csv"
    global_plots = out_root / "global" / "plots"
    global_csv.mkdir(parents=True, exist_ok=True)
    global_plots.mkdir(parents=True, exist_ok=True)

    combined_runs = []
    combined_full = []
    combined_iter = []

    for report in run_reports:
        rr = report["runs_raw_df"].copy()
        rr["run_name"] = report["run_name"]
        rr["algorithm"] = report["algorithm"]
        rr["profile"] = report["profile"]
        combined_runs.append(rr)

        ff = report["full_df"].copy()
        ff["run_name"] = report["run_name"]
        ff["algorithm"] = report["algorithm"]
        ff["profile"] = report["profile"]
        combined_full.append(ff)

        it = report["iter_stats"].copy()
        it["run_name"] = report["run_name"]
        it["algorithm"] = report["algorithm"]
        it["profile"] = report["profile"]
        combined_iter.append(it)

    if not combined_runs:
        return

    combined_runs_df = pd.concat(combined_runs, ignore_index=True)
    combined_full_df = pd.concat(combined_full, ignore_index=True)
    combined_iter_df = pd.concat(combined_iter, ignore_index=True)

    combined_runs_df.to_csv(global_csv / "combined_runs_raw.csv", index=False)
    combined_full_df.to_csv(global_csv / "combined_full_output.csv", index=False)
    combined_iter_df.to_csv(global_csv / "combined_iteration_stats_per_run.csv", index=False)

    profile_summary = (
        combined_runs_df.groupby(["profile", "function", "dimension"], as_index=False)["best_fitness"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "runs", "min": "best", "max": "worst"})
    )
    profile_summary["std"] = profile_summary["std"].fillna(0.0)
    profile_summary.to_csv(global_csv / "profile_summary_by_instance.csv", index=False)

    for function in sorted(combined_runs_df["function"].unique()):
        safe_name = function.replace("/", "_").replace(" ", "_")
        _plot_profile_final_boxplot(
            combined_runs_df,
            function=function,
            out_path=global_plots / f"profile_boxplot_final_{safe_name}.png",
            show_fliers=show_fliers,
        )
        _plot_profile_convergence(
            combined_full_df,
            function=function,
            out_path=global_plots / f"profile_convergence_{safe_name}.png",
        )


def _build_report_markdown(run_reports: list[dict], out_root: Path) -> None:
    lines = []
    lines.append("# TMLAP Benchmark Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for report in run_reports:
        lines.append(f"## {report['run_name']}")
        lines.append("")
        lines.append(f"- algorithm: {report['algorithm']}")
        lines.append(f"- profile: {report['profile']}")
        lines.append(f"- source: {report['run_dir']}")

        if report["summary_ext"].empty:
            lines.append("- no summary data")
            lines.append("")
            continue

        summary = report["summary_ext"].copy()
        summary = summary[["function", "dimension", "runs", "best", "worst", "mean", "median", "std"]]

        lines.append("")
        lines.append("| function | dim | runs | best | worst | mean | median | std |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in summary.iterrows():
            lines.append(
                "| {function} | {dimension} | {runs} | {best:.6f} | {worst:.6f} | {mean:.6f} | {median:.6f} | {std:.6f} |".format(
                    function=row["function"],
                    dimension=int(row["dimension"]),
                    runs=int(row["runs"]),
                    best=float(row["best"]),
                    worst=float(row["worst"]),
                    mean=float(row["mean"]),
                    median=float(row["median"]),
                    std=float(row["std"]),
                )
            )
        lines.append("")

    (out_root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    root = _resolve_root(args.benchmark_root)
    run_dirs = _discover_run_dirs(root, args.runs)

    if args.output_dir.strip():
        out_root = Path(args.output_dir)
        if not out_root.is_absolute():
            out_root = (Path(__file__).resolve().parent / out_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = root / f"analysis_{stamp}"

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark root: {root}")
    print(f"Run dirs: {[path.name for path in run_dirs]}")
    print(f"Output analysis dir: {out_root}")

    show_fliers = not args.no_show_fliers

    run_reports = []
    for run_dir in run_dirs:
        print(f"\nAnalyzing run: {run_dir.name}")
        report = _run_analysis_for_single_run(run_dir, out_root, show_fliers)
        run_reports.append(report)

    _build_global_outputs(run_reports, out_root, show_fliers)
    _build_report_markdown(run_reports, out_root)

    print("\nAnalysis completed.")
    print(f"Report: {out_root / 'analysis_report.md'}")
    print(f"Per-run outputs: {out_root / 'per_run'}")
    print(f"Global outputs: {out_root / 'global'}")


if __name__ == "__main__":
    main()
