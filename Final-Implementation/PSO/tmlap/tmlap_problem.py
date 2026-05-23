"""TMLAP instance loader and objective adapter for population optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TMLAPData:
    n_clients: int
    n_hubs: int
    distances: np.ndarray
    fixed_costs: np.ndarray
    capacity: np.ndarray
    d_max: float


def load_tmlap_instance(path: str | Path) -> TMLAPData:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")

    namespace: dict = {}
    exec(text.replace("self.", ""), {}, namespace)

    n_clients = int(namespace["n_clientes"])
    n_hubs = int(namespace["n_hubs"])
    distances = np.asarray(namespace["distancias"], dtype=float)

    fixed_raw = namespace.get("costos_fijos", namespace.get("costs"))
    if fixed_raw is None:
        raise ValueError(f"Instance {file_path} does not provide costos_fijos/costs")

    fixed_costs = np.asarray(fixed_raw, dtype=float).reshape(-1)
    capacity = np.asarray(namespace["capacidad"], dtype=int).reshape(-1)
    d_max = float(namespace["D_max"])

    if distances.shape != (n_clients, n_hubs):
        raise ValueError(
            f"Invalid distance matrix shape in {file_path}: {distances.shape}, expected {(n_clients, n_hubs)}"
        )

    return TMLAPData(
        n_clients=n_clients,
        n_hubs=n_hubs,
        distances=distances,
        fixed_costs=fixed_costs,
        capacity=capacity,
        d_max=d_max,
    )


class TMLAPObjective:
    def __init__(self, instance_path: str | Path, penalty_scale: float = 1e5) -> None:
        self.instance_path = Path(instance_path)
        self.data = load_tmlap_instance(self.instance_path)
        self.dimension = int(self.data.n_clients)
        self.penalty_scale = float(penalty_scale)
        self.nfev = 0

        self._feasible_hubs: list[list[int]] = []
        for i in range(self.data.n_clients):
            hubs = [h for h in range(self.data.n_hubs) if self.data.distances[i, h] <= self.data.d_max]
            self._feasible_hubs.append(hubs)

    def get_bounds(self) -> tuple[float, float]:
        return 0.0, 1.0

    def decode(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=float).reshape(-1)
        if x.size != self.dimension:
            raise ValueError(f"Vector dimension {x.size} does not match instance dimension {self.dimension}")

        x = np.clip(x, 0.0, 1.0)
        preferred_hub = np.floor(x * self.data.n_hubs).astype(int)
        preferred_hub = np.clip(preferred_hub, 0, self.data.n_hubs - 1)

        remaining = self.data.capacity.astype(int).copy()
        assignment = np.full(self.data.n_clients, -1, dtype=int)

        client_order = sorted(range(self.data.n_clients), key=lambda i: len(self._feasible_hubs[i]) or 10**9)

        for i in client_order:
            preferred = int(preferred_hub[i])
            feasible = self._feasible_hubs[i]

            chosen = -1
            if feasible:
                for h in sorted(feasible, key=lambda hub: (abs(hub - preferred), self.data.distances[i, hub])):
                    if remaining[h] > 0:
                        chosen = h
                        break

                if chosen < 0:
                    chosen = min(feasible, key=lambda hub: (self.data.distances[i, hub], abs(hub - preferred)))
            else:
                chosen = min(
                    range(self.data.n_hubs),
                    key=lambda hub: (self.data.distances[i, hub], abs(hub - preferred), -remaining[hub]),
                )

            assignment[i] = chosen
            remaining[chosen] -= 1

        return assignment

    def evaluate_assignment(self, assignment: np.ndarray) -> tuple[float, bool, float, int, float]:
        assign = np.asarray(assignment, dtype=int).reshape(-1)
        if assign.size != self.data.n_clients:
            raise ValueError("Assignment size mismatch")

        usage = np.zeros(self.data.n_hubs, dtype=int)
        fixed_used = np.zeros(self.data.n_hubs, dtype=int)

        distance_cost = 0.0
        distance_violation = 0.0

        for i, h in enumerate(assign):
            if h < 0 or h >= self.data.n_hubs:
                distance_violation += self.data.d_max + 1.0
                continue

            usage[h] += 1
            fixed_used[h] = 1

            d = float(self.data.distances[i, h])
            distance_cost += d
            if d > self.data.d_max:
                distance_violation += d - self.data.d_max

        overflow = np.maximum(0, usage - self.data.capacity)
        overflow_total = int(np.sum(overflow))

        fixed_cost = float(np.sum(self.data.fixed_costs * fixed_used))
        base_cost = distance_cost + fixed_cost

        penalty = self.penalty_scale * (float(overflow_total) + float(distance_violation))
        fitness = base_cost + penalty
        feasible = overflow_total == 0 and distance_violation <= 0.0

        return fitness, feasible, base_cost, overflow_total, distance_violation

    def __call__(self, vector: np.ndarray) -> float:
        self.nfev += 1
        assignment = self.decode(vector)
        fitness, _, _, _, _ = self.evaluate_assignment(assignment)
        return float(fitness)


def available_instance_paths(instance_dir: str | Path) -> list[Path]:
    base = Path(instance_dir)
    return sorted(base.glob("*.txt"))


def parse_instance_selection(raw: str, instance_dir: str | Path) -> list[Path]:
    available = {p.name: p for p in available_instance_paths(instance_dir)}
    if not available:
        raise ValueError(f"No .txt instances found in {instance_dir}")

    value = raw.strip().lower()
    if value == "all":
        return [available[name] for name in sorted(available)]

    selected: list[Path] = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        if name in available:
            selected.append(available[name])
        elif f"{name}.txt" in available:
            selected.append(available[f"{name}.txt"])
        else:
            raise ValueError(f"Unknown instance selection: {name}")

    if not selected:
        raise ValueError("No valid instances selected")

    return selected
