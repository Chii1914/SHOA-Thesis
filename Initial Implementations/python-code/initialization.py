"""Population initialization helper for the SHO algorithm."""

from __future__ import annotations

import numpy as np


def initialization(pop: int, Dim: int, UB, LB) -> np.ndarray:
    ub = np.asarray(UB, dtype=float).reshape(-1)
    lb = np.asarray(LB, dtype=float).reshape(-1)

    boundary_no = ub.size

    if boundary_no == 1:
        return np.random.rand(pop, Dim) * (ub.item() - lb.item()) + lb.item()

    population = np.zeros((pop, Dim), dtype=float)
    for i in range(Dim):
        population[:, i] = np.random.rand(pop) * (ub[i] - lb[i]) + lb[i]
    return population
