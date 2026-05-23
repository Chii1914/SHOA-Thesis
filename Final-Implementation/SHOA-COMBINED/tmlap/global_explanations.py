"""Global aggregation for feature contributions (point 3 in features spec)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def aggregate_global_feature_explanations(
    contribution_rows: Iterable[dict],
    run_id: str,
    function_name: str,
    dimension: int,
    windows: tuple[int, ...] = (50, 100),
) -> list[dict]:
    """Build global explanations with If, Sf and temporal windows.

    The input should contain rows with fields: diagnosis_id, target_type, feature, mean_weight.
    """

    buckets: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)

    for row in contribution_rows:
        target_type = str(row["target_type"])
        feature = str(row["feature"])
        diagnosis_id = int(row["diagnosis_id"])
        mean_weight = float(row["mean_weight"])
        buckets[(target_type, feature)].append((diagnosis_id, mean_weight))

    output: list[dict] = []

    for (target_type, feature), values in sorted(buckets.items()):
        values_sorted = sorted(values, key=lambda item: item[0])
        weights = [item[1] for item in values_sorted]

        n_total = len(weights)
        if n_total == 0:
            continue

        if_value = sum(abs(w) for w in weights) / n_total
        sf_value = sum(weights) / n_total

        output.append(
            {
                "run_id": run_id,
                "function_name": function_name,
                "dimension": dimension,
                "target_type": target_type,
                "feature": feature,
                "window": "all",
                "window_size": n_total,
                "If": if_value,
                "Sf": sf_value,
            }
        )

        for window in windows:
            selected = weights[-window:] if n_total > window else weights
            win_size = len(selected)
            if win_size == 0:
                continue

            if_window = sum(abs(w) for w in selected) / win_size
            sf_window = sum(selected) / win_size

            output.append(
                {
                    "run_id": run_id,
                    "function_name": function_name,
                    "dimension": dimension,
                    "target_type": target_type,
                    "feature": feature,
                    "window": f"last_{window}",
                    "window_size": win_size,
                    "If": if_window,
                    "Sf": sf_window,
                }
            )

    return output
