"""Levy-flight random coefficient generator."""

from __future__ import annotations

from math import gamma, pi, sin

import numpy as np


def levy(pop: int, m: int, omega: float) -> np.ndarray:
    num = gamma(1 + omega) * sin(pi * omega / 2)
    den = gamma((1 + omega) / 2) * omega * (2 ** ((omega - 1) / 2))
    sigma_u = (num / den) ** (1 / omega)
    u = np.random.normal(0, sigma_u, (pop, m))
    v = np.random.normal(0, 1, (pop, m))
    return u / (np.abs(v) ** (1 / omega))
