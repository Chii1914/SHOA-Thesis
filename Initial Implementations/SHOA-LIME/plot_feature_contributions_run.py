"""Generate feature-contribution charts for a single run-timestamp directory."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SHOA-LIME feature contributions for one run")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run-YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--top-k-temporal", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_bar(features: list[str], values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    order = sorted(range(len(features)), key=lambda idx: abs(values[idx]), reverse=True)
    x_labels = [features[idx] for idx in order]
    y_vals = [values[idx] for idx in order]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(x_labels)), y_vals)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_temporal(
    diagnosis_ids: list[int],
    series: dict[str, list[float]],
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    for feature, values in series.items():
        plt.plot(diagnosis_ids, values, label=feature)

    plt.xlabel("diagnosis_id")
    plt.ylabel("mean_weight")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_plots_for_target(rows: list[dict], target_type: str, plots_dir: Path, top_k: int) -> None:
    target_rows = [row for row in rows if row["target_type"] == target_type]
    if not target_rows:
        return

    by_feature_signed: dict[str, list[float]] = defaultdict(list)
    by_feature_abs: dict[str, list[float]] = defaultdict(list)

    by_diag_feature: dict[tuple[int, str], list[float]] = defaultdict(list)
    diagnosis_set: set[int] = set()

    for row in target_rows:
        feature = row["feature"]
        signed = float(row["mean_weight"])
        absolute = float(row["mean_abs_weight"])
        diag_id = int(row["diagnosis_id"])

        by_feature_signed[feature].append(signed)
        by_feature_abs[feature].append(absolute)

        by_diag_feature[(diag_id, feature)].append(signed)
        diagnosis_set.add(diag_id)

    features = sorted(by_feature_signed.keys())
    signed_means = [sum(by_feature_signed[f]) / len(by_feature_signed[f]) for f in features]
    abs_means = [sum(by_feature_abs[f]) / len(by_feature_abs[f]) for f in features]

    _plot_bar(
        features=features,
        values=signed_means,
        title=f"Signed Feature Contribution - {target_type}",
        ylabel="mean signed weight",
        out_path=plots_dir / f"signed_mean_{target_type}.png",
    )

    _plot_bar(
        features=features,
        values=abs_means,
        title=f"Absolute Feature Contribution - {target_type}",
        ylabel="mean absolute weight",
        out_path=plots_dir / f"absolute_mean_{target_type}.png",
    )

    # Temporal chart with top-k by absolute contribution.
    feature_strength = {feature: abs(value) for feature, value in zip(features, signed_means)}
    top_features = sorted(feature_strength, key=feature_strength.get, reverse=True)[:top_k]
    diagnosis_ids = sorted(diagnosis_set)

    temporal_series: dict[str, list[float]] = {}
    for feature in top_features:
        values: list[float] = []
        for diag_id in diagnosis_ids:
            slot = by_diag_feature.get((diag_id, feature), [])
            values.append(sum(slot) / len(slot) if slot else 0.0)
        temporal_series[feature] = values

    _plot_temporal(
        diagnosis_ids=diagnosis_ids,
        series=temporal_series,
        title=f"Temporal Feature Contribution - {target_type}",
        out_path=plots_dir / f"temporal_{target_type}.png",
    )


def plot_global_if_sf(global_rows: list[dict], plots_dir: Path) -> None:
    if not global_rows:
        return

    all_rows = [row for row in global_rows if row.get("window") == "all"]
    if not all_rows:
        return

    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_target[row["target_type"]].append(row)

    for target_type, rows in by_target.items():
        features = [row["feature"] for row in rows]
        if_vals = [float(row["If"]) for row in rows]
        sf_vals = [float(row["Sf"]) for row in rows]

        _plot_bar(
            features=features,
            values=if_vals,
            title=f"Global If - {target_type}",
            ylabel="If",
            out_path=plots_dir / f"global_if_{target_type}.png",
        )

        _plot_bar(
            features=features,
            values=sf_vals,
            title=f"Global Sf - {target_type}",
            ylabel="Sf",
            out_path=plots_dir / f"global_sf_{target_type}.png",
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    contribution_rows = _read_csv_rows(run_dir / "lime_contributions.csv")
    global_rows = _read_csv_rows(run_dir / "global_feature_explanations.csv")

    if not contribution_rows:
        raise FileNotFoundError(f"No contribution rows found in {run_dir / 'lime_contributions.csv'}")

    target_types = sorted({row["target_type"] for row in contribution_rows})
    for target_type in target_types:
        build_plots_for_target(contribution_rows, target_type, plots_dir, top_k=args.top_k_temporal)

    plot_global_if_sf(global_rows, plots_dir)

    print(f"Plots created in: {plots_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
