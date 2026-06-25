"""MLPAP instance loader and objective adapter for population optimizers.

Microhub Location and Pedestrian Access Problem (MLPAP).

Decision variables (internal):
    y[j] in {0,1}   — hub j is open
    x[c] in {0,...,m-1} — client c assigned to hub x[c]

Search space exposed to optimizers:
    Continuous vector v in [0,1]^n  (one value per client).
    Decode maps each v[c] to a preferred hub index, then applies a
    greedy demand-aware assignment that respects feasibility as much
    as possible before penalty evaluation.

Objective (Eq. 1 of the model):
    f(z) = facility_cost + weighted_assignment_cost
    facility_cost   = sum_j [ f_j*y_j + o_j * sum_c(q_c * x_{cj}) ]
    assignment_cost = sum_c [ w_c * d_{c,assigned[c]} ]

Penalization (Eq. 8):
    f_tilde(z) = f(z) + pi * v(z)
    v(z) = v1 + v2 + v3 + v4 + v5
    v1: upper capacity  sum_j max(0, load_j - L_j * y_j)
    v2: min utilization sum_j max(0, mu_j * y_j - load_j)
    v3: distance        sum_c max(0, d[c,assigned[c]] - D_max)
    v4: hub count low   max(0, P_min - sum(y))
    v5: hub count high  max(0, sum(y) - P_max)
    (Eqs. 2 and 4 are satisfied by construction of the decoder.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import logging as _logging

import numpy as np

_log = _logging.getLogger(__name__)

# Numba JIT for decode inner loop — graceful fallback when not installed
try:
    from numba import njit as _njit
    _USE_NUMBA = True
except ImportError:
    def _njit(**kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator
    _USE_NUMBA = False

_NUMBA_LOGGED = False


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MLPAPData:
    instance_id: str
    scale: str
    n_clients: int
    n_hubs: int
    fixed_costs: np.ndarray   # f_j  [m]
    op_costs: np.ndarray      # o_j  [m]
    capacity: np.ndarray      # L_j  [m]  demand-units upper bound
    min_util: np.ndarray      # mu_j [m]  demand-units lower bound
    demand: np.ndarray        # q_c  [n]
    priority: np.ndarray      # w_c  [n]
    distances: np.ndarray     # d_cj [n x m]
    d_max: float
    p_min: int
    p_max: int
    penalty: float            # pi


# ---------------------------------------------------------------------------
# Instance loader
# ---------------------------------------------------------------------------

def load_mlpap_instance(path: str | Path) -> MLPAPData:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    n = int(raw["n"])
    m = int(raw["m"])

    fixed_costs = np.asarray(raw["f"], dtype=float).reshape(-1)
    op_costs    = np.asarray(raw["o"], dtype=float).reshape(-1)
    capacity    = np.asarray(raw["L"], dtype=float).reshape(-1)
    min_util    = np.asarray(raw["mu"], dtype=float).reshape(-1)
    demand      = np.asarray(raw["q"], dtype=float).reshape(-1)
    priority    = np.asarray(raw["w"], dtype=float).reshape(-1)
    distances   = np.asarray(raw["d"], dtype=float)

    if distances.shape != (n, m):
        raise ValueError(
            f"{path.name}: distance matrix shape {distances.shape}, expected ({n}, {m})"
        )
    for arr, name, expected in [
        (fixed_costs, "f", m), (op_costs, "o", m),
        (capacity, "L", m),    (min_util, "mu", m),
        (demand, "q", n),      (priority, "w", n),
    ]:
        if arr.size != expected:
            raise ValueError(f"{path.name}: field '{name}' has {arr.size} elements, expected {expected}")

    return MLPAPData(
        instance_id=str(raw.get("instance_id", path.stem)),
        scale=str(raw.get("scale", "Unknown")),
        n_clients=n,
        n_hubs=m,
        fixed_costs=fixed_costs,
        op_costs=op_costs,
        capacity=capacity,
        min_util=min_util,
        demand=demand,
        priority=priority,
        distances=distances,
        d_max=float(raw["D_max"]),
        p_min=int(raw["P_min"]),
        p_max=int(raw["P_max"]),
        penalty=float(raw["pi"]),
    )


# ---------------------------------------------------------------------------
# JIT-compiled decode kernel
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _decode_jit(
    preferred: np.ndarray,     # int64[n]   preferred hub index per client
    n_hubs: np.int64,          # scalar
    fhubs_arr: np.ndarray,     # int64[n, max_fhubs]  -1-padded feasible hubs
    fhubs_len: np.ndarray,     # int64[n]             valid count per client
    client_order: np.ndarray,  # int64[n]             assignment order
    demand: np.ndarray,        # float64[n]
    capacity: np.ndarray,      # float64[m]
    distances: np.ndarray,     # float64[n, m]
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy demand-aware hub assignment — same semantics as the Python decode().

    Hub selection priority (lexicographic, same as hub_key sort):
        1. Prefer hubs with capacity headroom (has_room=1 before has_room=0)
        2. Among equal, prefer hub index closest to preferred (abs(j - pref))
        3. Among equal, prefer hub with shorter actual walking distance
    First-encountered wins on full tie (stable, matching Python Timsort).
    """
    n = client_order.shape[0]
    load = np.zeros(n_hubs, dtype=np.float64)
    assignment = np.empty(n, dtype=np.int64)

    for k in range(n):
        i = client_order[k]
        q_i = demand[i]
        pref = preferred[i]
        n_feas = fhubs_len[i]

        if n_feas > 0:
            best_j    = fhubs_arr[i, 0]
            best_hr   = np.int64(1) if load[best_j] + q_i <= capacity[best_j] else np.int64(0)
            best_diff = abs(best_j - pref)
            best_dist = distances[i, best_j]

            for idx in range(1, n_feas):
                j    = fhubs_arr[i, idx]
                hr   = np.int64(1) if load[j] + q_i <= capacity[j] else np.int64(0)
                diff = abs(j - pref)
                dist = distances[i, j]
                # Lexicographic comparison: (-hr, diff, dist)
                if (hr > best_hr
                        or (hr == best_hr and diff < best_diff)
                        or (hr == best_hr and diff == best_diff and dist < best_dist)):
                    best_j, best_hr, best_diff, best_dist = j, hr, diff, dist

            chosen = best_j

        else:
            # No feasible hub within D_max: assign to globally nearest (incurs v3 penalty)
            chosen = np.int64(0)
            mn = distances[i, 0]
            for jj in range(1, n_hubs):
                if distances[i, jj] < mn:
                    mn = distances[i, jj]
                    chosen = np.int64(jj)

        assignment[i] = chosen
        load[chosen] += q_i

    return load, assignment


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

class MLPAPObjective:
    """Continuous-space wrapper for the MLPAP.

    The optimizer works in [0, 1]^n_clients.  Each evaluation decodes the
    continuous vector into a discrete (y, assignment) pair and returns the
    penalized objective value f̃(z).
    """

    def __init__(
        self,
        instance_path: str | Path,
        penalty_scale: float | None = None,
    ) -> None:
        global _NUMBA_LOGGED
        if not _NUMBA_LOGGED:
            if _USE_NUMBA:
                _log.info("Numba JIT activo — _decode_jit compilado (cache=True)")
            else:
                _log.warning("Numba no instalado — _decode_jit corre en Python puro (más lento)")
            _NUMBA_LOGGED = True
        self.instance_path = Path(instance_path)
        self.data = load_mlpap_instance(self.instance_path)
        self.dimension = int(self.data.n_clients)
        self.penalty_scale = float(penalty_scale) if penalty_scale is not None else self.data.penalty
        self.nfev = 0

        d = self.data

        # Precompute feasible hub lists per client (hubs within D_max)
        self._feasible_hubs: list[list[int]] = [
            [j for j in range(d.n_hubs) if d.distances[i, j] <= d.d_max]
            for i in range(d.n_clients)
        ]
        # Stable client order: most constrained first (fewest feasible hubs)
        self._client_order: list[int] = sorted(
            range(d.n_clients),
            key=lambda i: (len(self._feasible_hubs[i]), i),
        )

        # --- Numba-compatible precomputed arrays ---
        _mf = max((len(fh) for fh in self._feasible_hubs), default=1)
        self._fhubs_arr = np.full((d.n_clients, _mf), -1, dtype=np.int64)
        self._fhubs_len = np.zeros(d.n_clients, dtype=np.int64)
        for _i, _fh in enumerate(self._feasible_hubs):
            _k = len(_fh)
            if _k:
                self._fhubs_arr[_i, :_k] = _fh
            self._fhubs_len[_i] = _k
        self._client_order_arr = np.array(self._client_order, dtype=np.int64)
        # Precomputed index range for vectorised evaluate_assignment
        self._arange_n = np.arange(d.n_clients, dtype=np.int64)
        # Ensure distances are C-contiguous float64 for numba
        self._distances_f64 = np.ascontiguousarray(d.distances, dtype=np.float64)
        self._capacity_f64  = np.ascontiguousarray(d.capacity,  dtype=np.float64)
        self._demand_f64    = np.ascontiguousarray(d.demand,    dtype=np.float64)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_bounds(self) -> tuple[float, float]:
        return 0.0, 1.0

    def decode(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map continuous [0,1]^n vector to (y[m], assignment[n])."""
        d = self.data
        x = np.clip(np.asarray(vector, dtype=float).reshape(-1), 0.0, 1.0)
        if x.size != self.dimension:
            raise ValueError(f"Vector size {x.size} != dimension {self.dimension}")

        preferred = np.clip(
            np.floor(x * d.n_hubs).astype(np.int64), 0, d.n_hubs - 1
        )

        load, assignment = _decode_jit(
            preferred,
            np.int64(d.n_hubs),
            self._fhubs_arr,
            self._fhubs_len,
            self._client_order_arr,
            self._demand_f64,
            self._capacity_f64,
            self._distances_f64,
        )

        y = (load > 0.0).astype(int)
        return y, assignment

    def evaluate_assignment(
        self,
        y: np.ndarray,
        assignment: np.ndarray,
    ) -> tuple[float, bool, float, float]:
        """Evaluate a decoded (y, assignment) pair.

        Returns
        -------
        fitness       : penalized objective f̃(z)
        feasible      : True if v(z) == 0
        base_cost     : raw objective f(z) without penalty
        violation     : total constraint violation v(z)
        """
        d = self.data
        y    = np.asarray(y,          dtype=int).reshape(-1)
        asgn = np.asarray(assignment, dtype=np.int64).reshape(-1)

        # Demand load per hub — scatter-add via np.bincount (replaces Python loop)
        load = np.bincount(asgn, weights=self._demand_f64, minlength=d.n_hubs)

        # Facility cost
        fixed_cost    = float(np.dot(d.fixed_costs, y))
        op_cost       = float(np.dot(d.op_costs, load * y))
        facility_cost = fixed_cost + op_cost

        # Weighted assignment cost + per-client assigned distance (vectorised)
        asgn_dist   = self._distances_f64[self._arange_n, asgn]   # shape [n]
        assign_cost = float(np.dot(d.priority, asgn_dist))
        base_cost   = facility_cost + assign_cost

        # Violations
        v1  = float(np.sum(np.maximum(0.0, load - d.capacity * y)))
        v2  = float(np.sum(np.maximum(0.0, d.min_util * y - load)))
        v3  = float(np.sum(np.maximum(0.0, asgn_dist - d.d_max)))
        n_a = int(np.sum(y))
        v4  = float(max(0, d.p_min - n_a))
        v5  = float(max(0, n_a - d.p_max))

        violation = v1 + v2 + v3 + v4 + v5
        fitness   = base_cost + self.penalty_scale * violation
        return float(fitness), (violation == 0.0), float(base_cost), float(violation)

    def __call__(self, vector: np.ndarray) -> float:
        self.nfev += 1
        y, assignment = self.decode(vector)
        fitness, _, _, _ = self.evaluate_assignment(y, assignment)
        return fitness


# ---------------------------------------------------------------------------
# Instance discovery helpers
# ---------------------------------------------------------------------------

_SCALE_PREFIXES = ("2XL", "XL", "L", "M", "S")  # order matters: 2XL before XL


def _instance_scale(name: str) -> str:
    stem = Path(name).stem.upper()
    for prefix in _SCALE_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return "UNKNOWN"


def available_instance_paths(instance_dir: str | Path) -> list[Path]:
    base = Path(instance_dir)
    return sorted(p for p in base.glob("*.json") if p.stem != "instances_index")


def parse_instance_selection(raw: str, instance_dir: str | Path) -> list[Path]:
    """Parse --instances argument.

    Accepted forms
    --------------
    "all"           → all JSON instances (sorted)
    "S"             → all Small instances
    "S,M"           → Small + Medium
    "S01"           → specific instance by stem (case-insensitive)
    "S01,S02,M03"   → comma-separated mix of scales and IDs
    """
    all_paths = available_instance_paths(instance_dir)
    if not all_paths:
        raise ValueError(f"No JSON instances found in {instance_dir}")

    value = raw.strip()
    if value.lower() == "all":
        return all_paths

    by_stem = {p.stem.upper(): p for p in all_paths}
    selected: list[Path] = []
    seen: set[str] = set()

    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        token_up = token.upper()

        # Scale prefix shortcut (e.g. "S", "XL")
        if token_up in _SCALE_PREFIXES:
            for p in all_paths:
                if _instance_scale(p.name) == token_up and p.stem.upper() not in seen:
                    selected.append(p)
                    seen.add(p.stem.upper())
            continue

        # Specific ID (e.g. "S01", "M03.json")
        stem = Path(token).stem.upper()
        if stem in by_stem:
            if stem not in seen:
                selected.append(by_stem[stem])
                seen.add(stem)
        else:
            raise ValueError(f"Unknown instance: '{token}' (not found in {instance_dir})")

    if not selected:
        raise ValueError(f"No valid instances selected from '{raw}'")

    return sorted(selected, key=lambda p: p.stem)
