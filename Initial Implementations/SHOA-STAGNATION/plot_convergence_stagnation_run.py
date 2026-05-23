"""Plot convergence curves and stagnation marks for one SHOA-STAGNATION run directory."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SHOA-STAGNATION convergence for one run")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run-YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis for best fitness",
    )
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


def _build_events(rows: list[dict], external_events: list[dict]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    best_by_iteration: dict[int, float] = {}
    for row in rows:
        iteration = _to_int(row.get("iteration"), 0)
        best = _to_float(row.get("best_fitness_so_far"), 0.0)
        best_by_iteration[iteration] = best

    start_events: list[tuple[int, float]] = []
    recovered_events: list[tuple[int, float]] = []

    if external_events:
        for event_row in external_events:
            event_name = str(event_row.get("event", "")).strip().lower()
            iteration = _to_int(event_row.get("iteration"), 0)
            y = best_by_iteration.get(iteration)
            if y is None:
                continue
            if event_name == "stagnation_start":
                start_events.append((iteration, y))
            elif event_name == "recovered":
                recovered_events.append((iteration, y))
        return start_events, recovered_events

    # Fallback: derive events from full_output event column.
    for row in rows:
        event_name = str(row.get("event", "")).strip().lower()
        if event_name not in {"stagnation_start", "recovered"}:
            continue
        iteration = _to_int(row.get("iteration"), 0)
        y = _to_float(row.get("best_fitness_so_far"), 0.0)
        if event_name == "stagnation_start":
            start_events.append((iteration, y))
        else:
            recovered_events.append((iteration, y))

    return start_events, recovered_events


def _plot_single_series(
    rows: list[dict],
    events_rows: list[dict],
    title: str,
    out_path: Path,
    log_y: bool,
) -> None:
    ordered = sorted(rows, key=lambda row: _to_int(row.get("iteration"), 0))
    if not ordered:
        return

    iterations = [_to_int(row.get("iteration"), 0) for row in ordered]
    best_curve = [_to_float(row.get("best_fitness_so_far"), 0.0) for row in ordered]

    spans = _compute_stagnation_spans(ordered)
    start_events, recovered_events = _build_events(ordered, events_rows)

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, best_curve, color="tab:blue", linewidth=2, label="best_fitness_so_far")

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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    full_output = _read_csv_rows(run_dir / "full_output.csv")
    if not full_output:
        raise FileNotFoundError(f"No full_output rows found in {run_dir / 'full_output.csv'}")

    events_all = _read_csv_rows(run_dir / "stagnation_events.csv")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in full_output:
        function_name = str(row.get("function_name", "unknown")).strip() or "unknown"
        run_number = _to_int(row.get("run_number"), 1)
        grouped[(function_name, run_number)].append(row)

    grouped_events: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in events_all:
        function_name = str(row.get("function_name", "unknown")).strip() or "unknown"
        run_number = _to_int(row.get("run_number"), 1)
        grouped_events[(function_name, run_number)].append(row)

    for (function_name, run_number), rows in sorted(grouped.items()):
        event_rows = grouped_events.get((function_name, run_number), [])
        title = f"Convergence with Stagnation - {function_name} (run {run_number})"
        filename = f"convergence_{function_name}_run{run_number}.png"
        out_path = plots_dir / filename
        _plot_single_series(
            rows=rows,
            events_rows=event_rows,
            title=title,
            out_path=out_path,
            log_y=args.log_y,
        )

    print(f"Convergence plots created in: {plots_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
