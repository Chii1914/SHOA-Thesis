#!/usr/bin/env python3
"""Analiza datasets de CEC2022 y TMLAP en la carpeta analizis.

Genera:
- Estadísticas de fitness final por función (best, worst, mean, std)
- Curvas de convergencia por función para cada dataset
- Consolidados para TMLAP (tablas y convergencia comparativa por función)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RowKey = Tuple[str, int, str]
FuncDimKey = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza CSV de CEC2022 y TMLAP para obtener stats de fitness "
            "y graficos de convergencia."
        )
    )
    parser.add_argument(
        "--input-root",
        default="analizis",
        help="Carpeta raiz de entrada que contiene CEC2022/ y TMLAP/",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Carpeta de salida. Si se omite, se crea automaticamente dentro "
            "de input-root con timestamp."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="DPI para graficos PNG",
    )
    return parser.parse_args()


def safe_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: str) -> float | None:
    try:
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def safe_std(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    return float(np.std(np.array(data, dtype=float), ddof=1))


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def dataset_name_from_path(csv_path: Path) -> str:
    stem = csv_path.stem
    suffixes = ["combined_full_output", "full_output", "_output"]
    lowered = stem.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.rstrip("_-")
    return stem or csv_path.stem


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_single_csv(csv_path: Path) -> tuple[
    Dict[FuncDimKey, List[float]],
    Dict[FuncDimKey, Dict[int, List[float]]],
    int,
]:
    """Retorna finales por funcion, convergencia por iteracion y filas validas."""
    final_by_run: Dict[RowKey, Tuple[int, float]] = {}
    conv_values: Dict[FuncDimKey, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    valid_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            function = (row.get("function") or "").strip()
            dimension = safe_int(row.get("dimension", ""))
            run_id = (row.get("run_id") or row.get("seed") or "").strip()
            iteration = safe_int(row.get("iteration", ""))
            best_fitness = safe_float(row.get("best_fitness", ""))

            if not function or dimension is None or not run_id or iteration is None or best_fitness is None:
                continue

            valid_rows += 1
            func_dim = (function, dimension)
            conv_values[func_dim][iteration].append(best_fitness)

            key = (function, dimension, run_id)
            prev = final_by_run.get(key)
            if prev is None or iteration > prev[0]:
                final_by_run[key] = (iteration, best_fitness)

    final_values: Dict[FuncDimKey, List[float]] = defaultdict(list)
    for function, dimension, _run_id in final_by_run.keys():
        _iteration, fit = final_by_run[(function, dimension, _run_id)]
        final_values[(function, dimension)].append(fit)

    return final_values, conv_values, valid_rows


def build_final_stats_rows(dataset: str, final_values: Dict[FuncDimKey, List[float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (function, dimension), values in sorted(final_values.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "function": function,
                "dimension": dimension,
                "n_runs": int(arr.size),
                "best": float(np.min(arr)),
                "worst": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": safe_std(arr.tolist()),
            }
        )
    return rows


def build_iteration_stats_rows(
    dataset: str,
    conv_values: Dict[FuncDimKey, Dict[int, List[float]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (function, dimension), by_iter in sorted(conv_values.items(), key=lambda x: (x[0][0], x[0][1])):
        for iteration in sorted(by_iter.keys()):
            values = np.array(by_iter[iteration], dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "dimension": dimension,
                    "iteration": iteration,
                    "n": int(values.size),
                    "best": float(np.min(values)),
                    "worst": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": safe_std(values.tolist()),
                }
            )
    return rows


def plot_dataset_convergence(
    dataset: str,
    conv_values: Dict[FuncDimKey, Dict[int, List[float]]],
    out_dir: Path,
    dpi: int,
) -> None:
    ensure_dir(out_dir)
    for (function, dimension), by_iter in sorted(conv_values.items(), key=lambda x: (x[0][0], x[0][1])):
        iterations = sorted(by_iter.keys())
        if not iterations:
            continue

        means: List[float] = []
        stds: List[float] = []
        bests: List[float] = []
        worsts: List[float] = []

        for it in iterations:
            values = np.array(by_iter[it], dtype=float)
            means.append(float(np.mean(values)))
            stds.append(safe_std(values.tolist()))
            bests.append(float(np.min(values)))
            worsts.append(float(np.max(values)))

        x = np.array(iterations, dtype=int)
        mean_arr = np.array(means, dtype=float)
        std_arr = np.array(stds, dtype=float)
        best_arr = np.array(bests, dtype=float)
        worst_arr = np.array(worsts, dtype=float)

        plt.figure(figsize=(11, 5.5))
        plt.plot(x, mean_arr, color="#1f77b4", linewidth=2.0, label="mean(best_fitness)")
        plt.fill_between(
            x,
            mean_arr - std_arr,
            mean_arr + std_arr,
            color="#1f77b4",
            alpha=0.18,
            label="mean +/- std",
        )
        plt.fill_between(x, best_arr, worst_arr, color="#ff7f0e", alpha=0.12, label="best..worst")
        plt.xlabel("iteration")
        plt.ylabel("best_fitness")
        plt.title(f"Convergencia | {dataset} | {function} (d={dimension})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()

        out_name = f"{sanitize_filename(function)}_d{dimension}.png"
        plt.savefig(out_dir / out_name, dpi=dpi)
        plt.close()


def plot_consolidated_convergence(
    group_name: str,
    dataset_conv_map: Dict[str, Dict[FuncDimKey, Dict[int, List[float]]]],
    out_dir: Path,
    dpi: int,
) -> None:
    ensure_dir(out_dir)

    all_keys: set[FuncDimKey] = set()
    for conv_data in dataset_conv_map.values():
        all_keys.update(conv_data.keys())

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#ff7f0e",
    ]
    datasets_sorted = sorted(dataset_conv_map.keys())

    for function, dimension in sorted(all_keys, key=lambda x: (x[0], x[1])):
        plt.figure(figsize=(11, 5.8))
        plotted_any = False

        for idx, dataset in enumerate(datasets_sorted):
            conv_data = dataset_conv_map[dataset]
            by_iter = conv_data.get((function, dimension))
            if not by_iter:
                continue

            iterations = sorted(by_iter.keys())
            x = np.array(iterations, dtype=int)
            means = np.array([float(np.mean(by_iter[it])) for it in iterations], dtype=float)
            stds = np.array([safe_std(by_iter[it]) for it in iterations], dtype=float)
            color = colors[idx % len(colors)]

            plt.plot(x, means, color=color, linewidth=2.0, label=f"{dataset}: mean")
            plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.12)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("iteration")
        plt.ylabel("best_fitness")
        plt.title(f"Convergencia consolidada | {group_name} | {function} (d={dimension})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()

        out_name = f"{sanitize_filename(function)}_d{dimension}.png"
        plt.savefig(out_dir / out_name, dpi=dpi)
        plt.close()


def build_group_pivot_rows(group_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    datasets = sorted({str(row["dataset"]) for row in group_rows})
    grouped: Dict[FuncDimKey, Dict[str, Dict[str, object]]] = defaultdict(dict)

    for row in group_rows:
        key = (str(row["function"]), int(row["dimension"]))
        grouped[key][str(row["dataset"])] = row

    pivot_rows: List[Dict[str, object]] = []
    for (function, dimension), dataset_map in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out: Dict[str, object] = {
            "function": function,
            "dimension": dimension,
        }
        for dataset in datasets:
            row = dataset_map.get(dataset)
            out[f"n_runs__{dataset}"] = "" if row is None else row["n_runs"]
            out[f"best__{dataset}"] = "" if row is None else row["best"]
            out[f"worst__{dataset}"] = "" if row is None else row["worst"]
            out[f"mean__{dataset}"] = "" if row is None else row["mean"]
            out[f"std__{dataset}"] = "" if row is None else row["std"]
        pivot_rows.append(out)
    return pivot_rows


def analyze_group(
    group_name: str,
    group_dir: Path,
    output_root: Path,
    dpi: int,
    make_consolidated: bool,
) -> Dict[str, object]:
    csv_files = sorted(group_dir.glob("*.csv"))
    if not csv_files:
        return {
            "group": group_name,
            "datasets": 0,
            "files": 0,
            "rows_valid": 0,
            "output_dir": str(output_root),
            "note": "Sin CSV para analizar",
        }

    ensure_dir(output_root)
    dataset_conv_map: Dict[str, Dict[FuncDimKey, Dict[int, List[float]]]] = {}
    group_stats_rows: List[Dict[str, object]] = []
    total_valid_rows = 0

    for csv_file in csv_files:
        dataset = dataset_name_from_path(csv_file)
        dataset_dir = output_root / sanitize_filename(dataset)
        ensure_dir(dataset_dir)

        final_values, conv_values, valid_rows = analyze_single_csv(csv_file)
        total_valid_rows += valid_rows

        final_stats_rows = build_final_stats_rows(dataset, final_values)
        iter_stats_rows = build_iteration_stats_rows(dataset, conv_values)
        group_stats_rows.extend(final_stats_rows)
        dataset_conv_map[dataset] = conv_values

        write_csv(
            dataset_dir / "fitness_summary_final.csv",
            ["dataset", "function", "dimension", "n_runs", "best", "worst", "mean", "std"],
            final_stats_rows,
        )
        write_csv(
            dataset_dir / "convergence_iteration_stats.csv",
            [
                "dataset",
                "function",
                "dimension",
                "iteration",
                "n",
                "best",
                "worst",
                "mean",
                "std",
            ],
            iter_stats_rows,
        )

        plot_dataset_convergence(
            dataset=dataset,
            conv_values=conv_values,
            out_dir=dataset_dir / "convergence",
            dpi=dpi,
        )

    write_csv(
        output_root / "fitness_summary_consolidated_long.csv",
        ["dataset", "function", "dimension", "n_runs", "best", "worst", "mean", "std"],
        group_stats_rows,
    )

    pivot_rows = build_group_pivot_rows(group_stats_rows)
    if pivot_rows:
        pivot_headers = list(pivot_rows[0].keys())
        write_csv(output_root / "fitness_summary_consolidated_pivot.csv", pivot_headers, pivot_rows)

    if make_consolidated:
        plot_consolidated_convergence(
            group_name=group_name,
            dataset_conv_map=dataset_conv_map,
            out_dir=output_root / "convergence_consolidated",
            dpi=dpi,
        )

    return {
        "group": group_name,
        "datasets": len(dataset_conv_map),
        "files": len(csv_files),
        "rows_valid": total_valid_rows,
        "output_dir": str(output_root),
    }


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (Path.cwd() / input_root).resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"No existe input-root: {input_root}")

    if args.output_dir.strip():
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_root / f"analysis_outputs_{timestamp}"
    ensure_dir(output_dir)

    cec_dir = input_root / "CEC2022"
    tmlap_dir = input_root / "TMLAP"

    summaries: List[Dict[str, object]] = []
    if cec_dir.exists() and cec_dir.is_dir():
        summaries.append(
            analyze_group(
                group_name="CEC2022",
                group_dir=cec_dir,
                output_root=output_dir / "CEC2022",
                dpi=args.dpi,
                make_consolidated=False,
            )
        )
    else:
        summaries.append(
            {
                "group": "CEC2022",
                "datasets": 0,
                "files": 0,
                "rows_valid": 0,
                "output_dir": str(output_dir / "CEC2022"),
                "note": "Carpeta CEC2022 no encontrada",
            }
        )

    if tmlap_dir.exists() and tmlap_dir.is_dir():
        summaries.append(
            analyze_group(
                group_name="TMLAP",
                group_dir=tmlap_dir,
                output_root=output_dir / "TMLAP",
                dpi=args.dpi,
                make_consolidated=True,
            )
        )
    else:
        summaries.append(
            {
                "group": "TMLAP",
                "datasets": 0,
                "files": 0,
                "rows_valid": 0,
                "output_dir": str(output_dir / "TMLAP"),
                "note": "Carpeta TMLAP no encontrada",
            }
        )

    report_lines = [
        "# Resumen de analisis",
        "",
        f"Input root: {input_root}",
        f"Output root: {output_dir}",
        "",
    ]
    for item in summaries:
        report_lines.append(f"## {item['group']}")
        report_lines.append(f"- datasets analizados: {item['datasets']}")
        report_lines.append(f"- archivos CSV: {item['files']}")
        report_lines.append(f"- filas validas procesadas: {item['rows_valid']}")
        report_lines.append(f"- salida: {item['output_dir']}")
        if "note" in item:
            report_lines.append(f"- nota: {item['note']}")
        report_lines.append("")

    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Analisis finalizado.")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_dir}")
    print(f"Reporte: {report_path}")

    for item in summaries:
        print(
            f"[{item['group']}] datasets={item['datasets']} files={item['files']} "
            f"rows_valid={item['rows_valid']} out={item['output_dir']}"
        )


if __name__ == "__main__":
    main()
