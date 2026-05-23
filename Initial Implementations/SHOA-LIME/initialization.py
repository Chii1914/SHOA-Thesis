"""Population initialization helper for SHOA variants."""

from __future__ import annotations

import numpy as np


def initialization(pop: int, dim: int, upper_bound, lower_bound) -> np.ndarray:
    ub = np.asarray(upper_bound, dtype=float).reshape(-1)
    lb = np.asarray(lower_bound, dtype=float).reshape(-1)

    boundary_no = ub.size

    if boundary_no == 1:
        return np.random.rand(pop, dim) * (ub.item() - lb.item()) + lb.item()

    population = np.zeros((pop, dim), dtype=float)
    for i in range(dim):
        population[:, i] = np.random.rand(pop) * (ub[i] - lb[i]) + lb[i]
    return population
