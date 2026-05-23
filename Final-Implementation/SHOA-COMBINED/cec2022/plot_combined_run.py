"""Generate LIME contribution and stagnation/convergence plots for one SHOA-COMBINED run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SHOA-COMBINED diagnostics for one run")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run-YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--top-k-temporal", type=int, default=8, help="Top-K features in temporal contribution plots")
    parser.add_argument("--target-fitness", type=float, default=0.0, help="Target/goal fitness shown as horizontal line")
    parser.add_argument("--full-log-file", type=str, default="", help="Optional output path for full log report (.txt)")
    parser.add_argument("--log-y", action="store_true", help="Use logarithmic y-axis for convergence")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_int(value: str | int | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    raw = str(value).strip()
    if not raw:
        return default
    return int(float(raw))


def _to_float(value: str | float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float):
        return value
    raw = str(value).strip()
    if not raw:
        return default
    return float(raw)


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


def _build_lime_target_plots(
    rows: list[dict],
    target_type: str,
    plots_dir: Path,
    top_k: int,
    file_prefix: str = "",
    title_prefix: str = "",
) -> None:
    target_rows = [row for row in rows if row.get("target_type") == target_type]
    if not target_rows:
        return

    by_feature_signed: dict[str, list[float]] = defaultdict(list)
    by_feature_abs: dict[str, list[float]] = defaultdict(list)

    by_diag_feature: dict[tuple[int, str], list[float]] = defaultdict(list)
    diagnosis_set: set[int] = set()

    for row in target_rows:
        feature = str(row["feature"])
        signed = _to_float(row.get("mean_weight"), 0.0)
        absolute = _to_float(row.get("mean_abs_weight"), 0.0)
        diag_id = _to_int(row.get("diagnosis_id"), 0)

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
        title=f"{title_prefix}Signed Feature Contribution - {target_type}".strip(),
        ylabel="mean signed weight",
        out_path=plots_dir / f"{file_prefix}signed_mean_{target_type}.png",
    )

    _plot_bar(
        features=features,
        values=abs_means,
        title=f"{title_prefix}Absolute Feature Contribution - {target_type}".strip(),
        ylabel="mean absolute weight",
        out_path=plots_dir / f"{file_prefix}absolute_mean_{target_type}.png",
    )

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
        title=f"{title_prefix}Temporal Feature Contribution - {target_type}".strip(),
        out_path=plots_dir / f"{file_prefix}temporal_{target_type}.png",
    )


def _plot_global_if_sf(global_rows: list[dict], plots_dir: Path) -> None:
    if not global_rows:
        return

    all_rows = [row for row in global_rows if row.get("window") == "all"]
    if not all_rows:
        return

    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_target[str(row.get("target_type", "unknown"))].append(row)

    for target_type, rows in by_target.items():
        features = [str(row.get("feature", "unknown")) for row in rows]
        if_vals = [_to_float(row.get("If"), 0.0) for row in rows]
        sf_vals = [_to_float(row.get("Sf"), 0.0) for row in rows]

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


def _compute_stagnation_spans(rows: list[dict]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    active_start: int | None = None
    previous_iteration: int | None = None

    for row in rows:
        iteration = _to_int(row.get("iteration"), 0)
        stagnated = _to_int(row.get("stagnated"), 0) == 1

        if stagnated and active_start is None:
            active_start = iteration
        elif (not stagnated) and active_start is not None:
            end_it = previous_iteration if previous_iteration is not None else iteration
            spans.append((active_start, end_it))
            active_start = None

        previous_iteration = iteration

    if active_start is not None and previous_iteration is not None:
        spans.append((active_start, previous_iteration))

    return spans


def _build_stagnation_events(
    rows: list[dict],
    events_rows: list[dict],
) -> tuple[
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
]:
    best_by_iteration: dict[int, float] = {}
    for row in rows:
        iteration = _to_int(row.get("iteration"), 0)
        best_by_iteration[iteration] = _to_float(row.get("best_fitness_so_far"), 0.0)

    start_events: list[tuple[int, float]] = []
    recovered_events: list[tuple[int, float]] = []

    for event_row in events_rows:
        event_name = str(event_row.get("event", "")).strip().lower()
        iteration = _to_int(event_row.get("iteration"), 0)
        y = best_by_iteration.get(iteration)
        if y is None:
            continue
        if event_name == "stagnation_start":
            start_events.append((iteration, y))
        elif event_name == "recovered":
            recovered_events.append((iteration, y))

    lime_events: list[tuple[int, float]] = []
    restart_lime_events: list[tuple[int, float]] = []
    restart_fallback_events: list[tuple[int, float]] = []
    restart_blocked_events: list[tuple[int, float]] = []
    for row in rows:
        iteration = _to_int(row.get("iteration"), 0)
        y_val = _to_float(row.get("best_fitness_so_far"), 0.0)

        if _to_int(row.get("lime_triggered"), 0) == 1:
            lime_events.append((iteration, y_val))

        if _to_int(row.get("restart_executed"), 0) == 1:
            restart_source = str(row.get("restart_source", "")).strip().lower()
            if restart_source == "lime_mutator":
                restart_lime_events.append((iteration, y_val))
            else:
                restart_fallback_events.append((iteration, y_val))
        elif _to_int(row.get("restart_blocked_by_cooldown"), 0) == 1:
            restart_blocked_events.append((iteration, y_val))

    return (
        start_events,
        recovered_events,
        lime_events,
        restart_lime_events,
        restart_fallback_events,
        restart_blocked_events,
    )


def _plot_convergence_stagnation(
    rows: list[dict],
    events_rows: list[dict],
    title: str,
    out_path: Path,
    log_y: bool,
    target_fitness: float,
) -> None:
    ordered = sorted(rows, key=lambda row: _to_int(row.get("iteration"), 0))
    if not ordered:
        return

    iterations = [_to_int(row.get("iteration"), 0) for row in ordered]
    best_curve = [_to_float(row.get("best_fitness_so_far"), 0.0) for row in ordered]

    spans = _compute_stagnation_spans(ordered)
    (
        start_events,
        recovered_events,
        lime_events,
        restart_lime_events,
        restart_fallback_events,
        restart_blocked_events,
    ) = _build_stagnation_events(ordered, events_rows)

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, best_curve, color="tab:blue", linewidth=2, label="best_fitness_so_far")
    plt.axhline(target_fitness, color="tab:purple", linestyle="--", linewidth=1.5, label=f"target_fitness={target_fitness:g}")

    for idx, (start_it, end_it) in enumerate(spans):
        label = "stagnated interval" if idx == 0 else None
        plt.axvspan(start_it, end_it, color="tab:red", alpha=0.12, label=label)

    if start_events:
        x_vals = [x for x, _ in start_events]
        y_vals = [y for _, y in start_events]
        plt.scatter(
            x_vals,
            y_vals,
            marker="v",
            color="tab:red",
            s=55,
            zorder=3,
            label="stagnation_start",
        )

    if recovered_events:
        x_vals = [x for x, _ in recovered_events]
        y_vals = [y for _, y in recovered_events]
        plt.scatter(
            x_vals,
            y_vals,
            marker="o",
            facecolors="none",
            edgecolors="tab:green",
            s=60,
            zorder=3,
            label="recovered",
        )

    if lime_events:
        x_vals = [x for x, _ in lime_events]
        y_vals = [y for _, y in lime_events]
        plt.scatter(
            x_vals,
            y_vals,
            marker="x",
            color="tab:orange",
            s=50,
            zorder=3,
            label="lime_trigger",
        )

    if restart_lime_events:
        x_vals = [x for x, _ in restart_lime_events]
        y_vals = [y for _, y in restart_lime_events]
        plt.scatter(
            x_vals,
            y_vals,
            marker="*",
            color="tab:cyan",
            s=85,
            zorder=3,
            label="restart_lime_mutator",
        )

    if restart_fallback_events:
        x_vals = [x for x, _ in restart_fallback_events]
        y_vals = [y for _, y in restart_fallback_events]
        plt.scatter(
            x_vals,
            y_vals,
            marker="s",
            facecolors="none",
            edgecolors="tab:brown",
            s=65,
            zorder=3,
            label="restart_random_fallback",
        )

    if restart_blocked_events:
        x_vals = [x for x, _ in restart_blocked_events]
        y_vals = [y for _, y in restart_blocked_events]
        plt.scatter(
            x_vals,
            y_vals,
            marker="P",
            color="tab:gray",
            s=45,
            zorder=3,
            label="restart_blocked_cooldown",
        )

    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("best fitness")

    if log_y:
        plt.yscale("log")

    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _collect_reset_diagnosis_sets(rows: list[dict]) -> tuple[set[int], set[int], set[int]]:
    all_reset: set[int] = set()
    lime_feature_reset: set[int] = set()
    no_feature_reset: set[int] = set()

    for row in rows:
        if _to_int(row.get("restart_executed"), 0) != 1:
            continue

        diagnosis_id = _to_int(row.get("diagnosis_id"), 0)
        if diagnosis_id <= 0:
            continue

        all_reset.add(diagnosis_id)
        if str(row.get("restart_source", "")).strip().lower() == "lime_mutator":
            lime_feature_reset.add(diagnosis_id)
        else:
            no_feature_reset.add(diagnosis_id)

    return all_reset, lime_feature_reset, no_feature_reset


def _plot_restart_source_counts(rows: list[dict], plots_dir: Path) -> None:
    counts = {
        "lime_mutator": 0,
        "random_fallback": 0,
        "blocked_cooldown": 0,
    }

    for row in rows:
        if _to_int(row.get("restart_executed"), 0) == 1:
            source = str(row.get("restart_source", "")).strip().lower()
            if source == "lime_mutator":
                counts["lime_mutator"] += 1
            else:
                counts["random_fallback"] += 1
        elif _to_int(row.get("restart_blocked_by_cooldown"), 0) == 1:
            counts["blocked_cooldown"] += 1

    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title("Restart Outcomes")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "restart_outcomes.png", dpi=150)
    plt.close()


def _write_full_log(
    full_rows: list[dict],
    events_rows: list[dict],
    target_fitness: float,
    out_path: Path,
) -> None:
    ordered = sorted(full_rows, key=lambda row: (_to_int(row.get("run_number"), 1), _to_int(row.get("iteration"), 0)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("SHOA-COMBINED FULL LOG REPORT\n")
        handle.write("============================\n")
        handle.write(f"target_fitness={target_fitness}\n")
        handle.write(f"iterations_logged={len(ordered)}\n")
        handle.write(f"stagnation_events_logged={len(events_rows)}\n\n")

        handle.write("ITERATION LOG\n")
        handle.write("-------------\n")
        for row in ordered:
            handle.write(
                " | ".join(
                    [
                        f"fn={row.get('function_name', 'unknown')}",
                        f"run={_to_int(row.get('run_number'), 1)}",
                        f"iter={_to_int(row.get('iteration'), 0)}",
                        f"fe={_to_int(row.get('fe'), 0)}",
                        f"best={_to_float(row.get('best_fitness_so_far'), 0.0):.8e}",
                        f"stagnated={_to_int(row.get('stagnated'), 0)}",
                        f"event={row.get('event', '')}",
                        f"lime_triggered={_to_int(row.get('lime_triggered'), 0)}",
                        f"diag={_to_int(row.get('diagnosis_id'), 0)}",
                        f"restart_attempted={_to_int(row.get('restart_attempted'), 0)}",
                        f"restart_executed={_to_int(row.get('restart_executed'), 0)}",
                        f"restart_source={row.get('restart_source', '')}",
                        f"restart_feature={row.get('restart_feature', '')}",
                        f"restart_feature_score={_to_float(row.get('restart_feature_score'), 0.0):.4f}",
                        f"restart_count={_to_int(row.get('restart_count'), 0)}",
                        f"warmup_active={_to_int(row.get('warmup_active'), 0)}",
                    ]
                )
                + "\n"
            )

        handle.write("\nEVENT LOG\n")
        handle.write("---------\n")
        ordered_events = sorted(events_rows, key=lambda row: (_to_int(row.get("run_number"), 1), _to_int(row.get("iteration"), 0)))
        for row in ordered_events:
            handle.write(
                " | ".join(
                    [
                        f"fn={row.get('function_name', 'unknown')}",
                        f"run={_to_int(row.get('run_number'), 1)}",
                        f"iter={_to_int(row.get('iteration'), 0)}",
                        f"event={row.get('event', '')}",
                        f"fe={_to_int(row.get('fe'), 0)}",
                        f"sfes={_to_int(row.get('sfes'), 0)}",
                        f"restart_executed={_to_int(row.get('restart_executed'), 0)}",
                        f"restart_source={row.get('restart_source', '')}",
                        f"restart_feature={row.get('restart_feature', '')}",
                        f"restart_feature_score={_to_float(row.get('restart_feature_score'), 0.0):.4f}",
                    ]
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    contribution_rows = _read_csv_rows(run_dir / "lime_contributions.csv")
    global_rows = _read_csv_rows(run_dir / "global_feature_explanations.csv")
    full_output_rows = _read_csv_rows(run_dir / "full_output.csv")
    stagnation_events = _read_csv_rows(run_dir / "stagnation_events.csv")

    if not full_output_rows:
        raise FileNotFoundError(f"No full_output rows found in {run_dir / 'full_output.csv'}")

    if contribution_rows:
        target_types = sorted({str(row.get("target_type", "unknown")) for row in contribution_rows})
        for target_type in target_types:
            _build_lime_target_plots(contribution_rows, target_type, plots_dir, top_k=args.top_k_temporal)
        _plot_global_if_sf(global_rows, plots_dir)
        print(f"LIME contribution plots created in: {plots_dir}")
    else:
        print("No LIME contribution rows found. Skipping contribution plots.")

    grouped_runs: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in full_output_rows:
        function_name = str(row.get("function_name", "unknown")).strip() or "unknown"
        run_number = _to_int(row.get("run_number"), 1)
        grouped_runs[(function_name, run_number)].append(row)

    grouped_events: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in stagnation_events:
        function_name = str(row.get("function_name", "unknown")).strip() or "unknown"
        run_number = _to_int(row.get("run_number"), 1)
        grouped_events[(function_name, run_number)].append(row)

    for (function_name, run_number), rows in sorted(grouped_runs.items()):
        event_rows = grouped_events.get((function_name, run_number), [])
        title = f"Convergence + Stagnation + LIME Triggers - {function_name} (run {run_number})"
        filename = f"combined_convergence_{function_name}_run{run_number}.png"
        out_path = plots_dir / filename
        _plot_convergence_stagnation(
            rows=rows,
            events_rows=event_rows,
            title=title,
            out_path=out_path,
            log_y=args.log_y,
            target_fitness=args.target_fitness,
        )

        if contribution_rows:
            reset_all, reset_lime, reset_no_feature = _collect_reset_diagnosis_sets(rows)

            if reset_all:
                reset_rows_all = [r for r in contribution_rows if _to_int(r.get("diagnosis_id"), 0) in reset_all]
                if reset_rows_all:
                    for target_type in sorted({str(r.get("target_type", "unknown")) for r in reset_rows_all}):
                        _build_lime_target_plots(
                            reset_rows_all,
                            target_type,
                            plots_dir,
                            top_k=args.top_k_temporal,
                            file_prefix="reset_all_",
                            title_prefix="Reset Moments - ",
                        )

            if reset_lime:
                reset_rows_lime = [r for r in contribution_rows if _to_int(r.get("diagnosis_id"), 0) in reset_lime]
                if reset_rows_lime:
                    for target_type in sorted({str(r.get("target_type", "unknown")) for r in reset_rows_lime}):
                        _build_lime_target_plots(
                            reset_rows_lime,
                            target_type,
                            plots_dir,
                            top_k=args.top_k_temporal,
                            file_prefix="reset_with_feature_",
                            title_prefix="Reset using Feature - ",
                        )

            if reset_no_feature:
                reset_rows_no_feature = [r for r in contribution_rows if _to_int(r.get("diagnosis_id"), 0) in reset_no_feature]
                if reset_rows_no_feature:
                    for target_type in sorted({str(r.get("target_type", "unknown")) for r in reset_rows_no_feature}):
                        _build_lime_target_plots(
                            reset_rows_no_feature,
                            target_type,
                            plots_dir,
                            top_k=args.top_k_temporal,
                            file_prefix="reset_without_feature_",
                            title_prefix="Reset without Feature (fallback) - ",
                        )

            _plot_restart_source_counts(rows, plots_dir)

    print(f"Convergence/stagnation plots created in: {plots_dir}")

    if args.full_log_file.strip():
        full_log_path = Path(args.full_log_file).resolve()
    else:
        full_log_path = plots_dir / "full_log_report.txt"

    _write_full_log(
        full_rows=full_output_rows,
        events_rows=stagnation_events,
        target_fitness=args.target_fitness,
        out_path=full_log_path,
    )
    print(f"Full log report created at: {full_log_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
