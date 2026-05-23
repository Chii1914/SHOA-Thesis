"""Benchmark test functions used by the Sea-Horse Optimizer (SHO)."""

from __future__ import annotations

from typing import Callable, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def BenchmarkFunctions(F: str) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], int, Callable[[ArrayLike], float]]:
    if F == "F1":
        fobj = F1
        LB = -100
        UB = 100
        Dim = 30
    elif F == "F2":
        fobj = F2
        LB = -10
        UB = 10
        Dim = 30
    elif F == "F3":
        fobj = F3
        LB = -100
        UB = 100
        Dim = 30
    elif F == "F4":
        fobj = F4
        LB = -100
        UB = 100
        Dim = 30
    elif F == "F5":
        fobj = F5
        LB = -30
        UB = 30
        Dim = 30
    elif F == "F6":
        fobj = F6
        LB = -100
        UB = 100
        Dim = 30
    elif F == "F7":
        fobj = F7
        LB = -1.28
        UB = 1.28
        Dim = 30
    elif F == "F8":
        fobj = F8
        LB = -500
        UB = 500
        Dim = 30
    elif F == "F9":
        fobj = F9
        LB = -5.12
        UB = 5.12
        Dim = 30
    elif F == "F10":
        fobj = F10
        LB = -32
        UB = 32
        Dim = 30
    elif F == "F11":
        fobj = F11
        LB = -600
        UB = 600
        Dim = 30
    elif F == "F12":
        fobj = F12
        LB = -50
        UB = 50
        Dim = 30
    elif F == "F13":
        fobj = F13
        LB = -50
        UB = 50
        Dim = 30
    elif F == "F14":
        fobj = F14
        LB = -65.536
        UB = 65.536
        Dim = 2
    elif F == "F15":
        fobj = F15
        LB = -5
        UB = 5
        Dim = 4
    elif F == "F16":
        fobj = F16
        LB = -5
        UB = 5
        Dim = 2
    elif F == "F17":
        fobj = F17
        LB = np.array([-5.0, 0.0])
        UB = np.array([10.0, 15.0])
        Dim = 2
    elif F == "F18":
        fobj = F18
        LB = -2
        UB = 2
        Dim = 2
    elif F == "F19":
        fobj = F19
        LB = 0
        UB = 1
        Dim = 3
    elif F == "F20":
        fobj = F20
        LB = 0
        UB = 1
        Dim = 6
    elif F == "F21":
        fobj = F21
        LB = 0
        UB = 10
        Dim = 4
    elif F == "F22":
        fobj = F22
        LB = 0
        UB = 10
        Dim = 4
    elif F == "F23":
        fobj = F23
        LB = 0
        UB = 10
        Dim = 4
    else:
        raise ValueError(f"Unknown benchmark function name: {F}")

    return LB, UB, Dim, fobj


def _vec(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def F1(x: ArrayLike) -> float:
    x = _vec(x)
    return float(np.sum(x**2))


def F2(x: ArrayLike) -> float:
    x = _vec(x)
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def F3(x: ArrayLike) -> float:
    x = _vec(x)
    return float(sum(np.sum(x[:i]) ** 2 for i in range(1, x.size + 1)))


def F4(x: ArrayLike) -> float:
    x = _vec(x)
    return float(np.max(np.abs(x)))


def F5(x: ArrayLike) -> float:
    x = _vec(x)
    return float(np.sum(100 * (x[1:] - (x[:-1] ** 2)) ** 2 + (x[:-1] - 1) ** 2))


def F6(x: ArrayLike) -> float:
    x = _vec(x)
    return float(np.sum(np.abs(x + 0.5) ** 2))


def F7(x: ArrayLike) -> float:
    x = _vec(x)
    idx = np.arange(1, x.size + 1)
    return float(np.sum(idx * (x**4)) + np.random.rand())


def F8(x: ArrayLike) -> float:
    x = _vec(x)
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))


def F9(x: ArrayLike) -> float:
    x = _vec(x)
    dim = x.size
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim)


def F10(x: ArrayLike) -> float:
    x = _vec(x)
    dim = x.size
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.e)


def F11(x: ArrayLike) -> float:
    x = _vec(x)
    dim = x.size
    return float(np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1)


def F12(x: ArrayLike) -> float:
    x = _vec(x)
    dim = x.size
    term1 = 10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2)
    term2 = np.sum((((x[:-1] + 1) / 4) ** 2) * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2)))
    term3 = ((x[-1] + 1) / 4) ** 2
    return float((np.pi / dim) * (term1 + term2 + term3) + np.sum(Ufun(x, 10, 100, 4)))


def F13(x: ArrayLike) -> float:
    x = _vec(x)
    dim = x.size
    term1 = np.sin(3 * np.pi * x[0]) ** 2
    term2 = np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
    term3 = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
    return float(0.1 * (term1 + term2 + term3) + np.sum(Ufun(x, 5, 100, 4)))


def F14(x: ArrayLike) -> float:
    x = _vec(x)
    aS = np.array(
        [
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32],
        ],
        dtype=float,
    )
    bS = np.sum((x.reshape(-1, 1) - aS) ** 6, axis=0)
    return float((1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))) ** (-1))


def F15(x: ArrayLike) -> float:
    x = _vec(x)
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246], dtype=float)
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    num = x[0] * (bK**2 + x[1] * bK)
    den = bK**2 + x[2] * bK + x[3]
    return float(np.sum((aK - num / den) ** 2))


def F16(x: ArrayLike) -> float:
    x = _vec(x)
    return float(4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4))


def F17(x: ArrayLike) -> float:
    x = _vec(x)
    return float((x[1] - (x[0] ** 2) * 5.1 / (4 * np.pi**2) + 5 / np.pi * x[0] - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)


def F18(x: ArrayLike) -> float:
    x = _vec(x)
    t1 = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
    t2 = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2)
    return float(t1 * t2)


def F19(x: ArrayLike) -> float:
    x = _vec(x)
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], dtype=float)
    cH = np.array([1, 1.2, 3, 3.2], dtype=float)
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]], dtype=float)
    o = 0.0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :]) ** 2)))
    return float(o)


def F20(x: ArrayLike) -> float:
    x = _vec(x)
    aH = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
        dtype=float,
    )
    cH = np.array([1, 1.2, 3, 3.2], dtype=float)
    pH = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ],
        dtype=float,
    )
    o = 0.0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :]) ** 2)))
    return float(o)


def F21(x: ArrayLike) -> float:
    return _shekel(x, m=5)


def F22(x: ArrayLike) -> float:
    return _shekel(x, m=7)


def F23(x: ArrayLike) -> float:
    return _shekel(x, m=10)


def _shekel(x: ArrayLike, m: int) -> float:
    x = _vec(x)
    aSH = np.array(
        [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ],
        dtype=float,
    )
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], dtype=float)

    o = 0.0
    for i in range(m):
        o -= 1.0 / (np.dot(x - aSH[i, :], x - aSH[i, :]) + cSH[i])
    return float(o)


def Ufun(x: ArrayLike, a: float, k: float, m: float) -> np.ndarray:
    x = _vec(x)
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)
