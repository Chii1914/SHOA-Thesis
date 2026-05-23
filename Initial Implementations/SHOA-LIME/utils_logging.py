"""Utilities for run-timestamp output management."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence


def create_run_directory(output_root: str | Path) -> tuple[str, str, Path]:
    now = datetime.utcnow()
    run_id = now.strftime("run-%Y-%m-%d-%H-%M-%S")
    iso_timestamp = now.isoformat(timespec="seconds") + "Z"

    root = Path(output_root)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, iso_timestamp, run_dir


def write_json(path: str | Path, payload: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_csv(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by_function(run_rows: Iterable[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["function_name"])].append(float(row["best_fitness"]))

    summary: list[dict] = []
    for function_name in sorted(grouped):
        values = grouped[function_name]
        summary.append(
            {
                "function_name": function_name,
                "runs_completed": len(values),
                "mean_best_fitness": mean(values),
                "std_best_fitness": pstdev(values) if len(values) > 1 else 0.0,
                "min_best_fitness": min(values),
                "max_best_fitness": max(values),
            }
        )
    return summary
