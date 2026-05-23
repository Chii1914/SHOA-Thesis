"""Adapters for CEC2022 functions from opfunu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from opfunu.cec_based import cec2022


@dataclass
class CEC2022FunctionWrapper:
    """Callable wrapper compatible with SHO objective signature."""

    function_id: int
    dimension: int

    def __post_init__(self) -> None:
        if not 1 <= self.function_id <= 12:
            raise ValueError("function_id must be in [1, 12] for CEC2022.")

        class_name = f"F{self.function_id}2022"
        func_class = getattr(cec2022, class_name)
        self._func = func_class(ndim=self.dimension)
        self.name = class_name
        self.nfev = 0

    def __call__(self, x: np.ndarray) -> float:
        self.nfev += 1
        return float(self._func.evaluate(np.asarray(x, dtype=float)))

    def get_bounds(self) -> tuple[float, float]:
        return -100.0, 100.0

    def reset_counter(self) -> None:
        self.nfev = 0


def parse_function_ids(raw: str) -> List[int]:
    """Parse function selection syntax like 'all', '1,2,3', '1-12'."""
    value = raw.strip().lower()
    if value == "all":
        return list(range(1, 13))

    result: set[int] = set()
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", maxsplit=1)
            start_i = int(start_s)
            end_i = int(end_s)
            if start_i > end_i:
                start_i, end_i = end_i, start_i
            result.update(range(start_i, end_i + 1))
        else:
            result.add(int(token))

    filtered = sorted(fid for fid in result if 1 <= fid <= 12)
    if not filtered:
        raise ValueError("No valid CEC2022 function IDs found. Use 1-12, comma list, or 'all'.")
    return filtered


def build_wrappers(function_ids: Iterable[int], dimension: int) -> List[CEC2022FunctionWrapper]:
    return [CEC2022FunctionWrapper(function_id=fid, dimension=dimension) for fid in function_ids]
