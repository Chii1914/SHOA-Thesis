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

import numpy as np


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
        self.instance_path = Path(instance_path)
        self.data = load_mlpap_instance(self.instance_path)
        self.dimension = int(self.data.n_clients)
        # Use instance penalty unless caller overrides
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_bounds(self) -> tuple[float, float]:
        return 0.0, 1.0

    def decode(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map continuous [0,1]^n vector to (y[m], assignment[n]).

        Returns
        -------
        y          : int array [m], 1 if hub j is active
        assignment : int array [n], hub index assigned to each client
        """
        d = self.data
        x = np.clip(np.asarray(vector, dtype=float).reshape(-1), 0.0, 1.0)
        if x.size != self.dimension:
            raise ValueError(f"Vector size {x.size} != dimension {self.dimension}")

        preferred = np.clip(np.floor(x * d.n_hubs).astype(int), 0, d.n_hubs - 1)

        # Demand load per hub (accumulated during assignment)
        load = np.zeros(d.n_hubs, dtype=float)
        assignment = np.full(d.n_clients, -1, dtype=int)

        for i in self._client_order:
            q_i = d.demand[i]
            pref = int(preferred[i])
            feasible = self._feasible_hubs[i]

            chosen = -1
            if feasible:
                # Sort feasible hubs: prefer those closest to pref with capacity headroom
                def hub_key(j: int) -> tuple:
                    has_room = int(load[j] + q_i <= d.capacity[j])
                    return (-has_room, abs(j - pref), d.distances[i, j])

                for j in sorted(feasible, key=hub_key):
                    chosen = j
                    break  # take the best-ranked hub (may or may not have room)
            else:
                # No feasible hub: nearest globally (distance violation will be penalized)
                chosen = int(np.argmin(d.distances[i]))

            assignment[i] = chosen
            load[chosen] += q_i

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
        y = np.asarray(y, dtype=int).reshape(-1)
        asgn = np.asarray(assignment, dtype=int).reshape(-1)

        # --- compute demand load per hub ---
        load = np.zeros(d.n_hubs, dtype=float)
        for i, j in enumerate(asgn):
            if 0 <= j < d.n_hubs:
                load[j] += d.demand[i]

        # --- facility cost ---
        # fixed opening cost
        fixed_cost = float(np.dot(d.fixed_costs, y))
        # operational cost: o_j * (demand served by hub j)
        op_cost = float(np.dot(d.op_costs, load * y))
        facility_cost = fixed_cost + op_cost

        # --- weighted assignment cost ---
        assign_cost = 0.0
        for i, j in enumerate(asgn):
            if 0 <= j < d.n_hubs:
                assign_cost += d.priority[i] * d.distances[i, j]

        base_cost = facility_cost + assign_cost

        # --- violations ---
        # v1: upper capacity  (demand-units overflow)
        v1 = float(np.sum(np.maximum(0.0, load - d.capacity * y)))
        # v2: min utilization (demand-units under-use for active hubs)
        v2 = float(np.sum(np.maximum(0.0, d.min_util * y - load)))
        # v3: walking-distance violation
        v3 = 0.0
        for i, j in enumerate(asgn):
            if 0 <= j < d.n_hubs:
                excess = d.distances[i, j] - d.d_max
                if excess > 0:
                    v3 += excess
        # v4/v5: hub budget
        n_active = int(np.sum(y))
        v4 = float(max(0, d.p_min - n_active))
        v5 = float(max(0, n_active - d.p_max))

        violation = v1 + v2 + v3 + v4 + v5
        fitness = base_cost + self.penalty_scale * violation
        feasible = violation == 0.0

        return float(fitness), feasible, float(base_cost), float(violation)

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
