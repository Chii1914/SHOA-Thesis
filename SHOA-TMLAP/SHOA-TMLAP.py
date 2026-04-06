import math
import random
import importlib.util
import sys
from pathlib import Path

import numpy as np

# Reuse the SHO core implementation from python-code/SHO.py.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def _load_sho_function():
    sho_path = PROJECT_ROOT / "python-code" / "SHO.py"
    if not sho_path.exists():
        raise FileNotFoundError(f"No se encontro el modulo base SHO en: {sho_path}")

    # SHO.py imports initialization.py and levy.py from the same folder.
    helper_dir = str(sho_path.parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)

    spec = importlib.util.spec_from_file_location("sho_core_module", sho_path)
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo crear el spec para cargar SHO.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sho_fn = getattr(module, "SHO", None)
    if sho_fn is None:
        raise AttributeError("El modulo SHO.py no expone la funcion SHO")

    return sho_fn


SHO = _load_sho_function()


class Problem:
    def __init__(self):
        self.n_clients = 6
        self.n_hubs = 3

        # Matriz de distancias cliente-hub
        self.distancias = [
            [5, 8, 11],
            [7, 4, 10],
            [6, 6, 6],
            [9, 5, 7],
            [8, 9, 5],
            [4, 7, 12],
        ]

        # Costo fijo por abrir cada hub
        self.costs = [20, 25, 15]

        # Capacidad maxima de atencion por hub
        self.capacidad = [2, 3, 2]

        # Distancia maxima tolerada cliente-hub
        self.D_max = 8

    def check(self, x):
        conteo = [0] * self.n_hubs
        for c in range(self.n_clients):
            h = int(x[c])
            if self.distancias[c][h] > self.D_max:
                return False
            conteo[h] += 1
            if conteo[h] > self.capacidad[h]:
                return False
        return True

    def fit(self, x):
        hubs = [0] * self.n_hubs
        for h in x:
            hubs[int(h)] = 1

        total = 0
        for c in range(self.n_clients):
            total += self.distancias[c][int(x[c])]

        for j in range(self.n_hubs):
            if hubs[j] == 1:
                total += self.costs[j]

        return total

    def _feasible_hubs(self, c):
        return [h for h in range(self.n_hubs) if self.distancias[c][h] <= self.D_max]

    def repair_from_latent(self, latent):
        z = np.asarray(latent, dtype=float).reshape(-1)
        if z.size != self.n_clients:
            return None

        feasible_by_client = [self._feasible_hubs(c) for c in range(self.n_clients)]
        if any(len(options) == 0 for options in feasible_by_client):
            return None

        remaining = self.capacidad.copy()
        assignment = [-1] * self.n_clients

        # Most constrained clients are assigned first.
        client_order = sorted(range(self.n_clients), key=lambda c: len(feasible_by_client[c]))

        for c in client_order:
            options = sorted(
                feasible_by_client[c],
                key=lambda h: (abs(z[c] - h), self.distancias[c][h]),
            )

            chosen = None
            for h in options:
                if remaining[h] > 0:
                    chosen = h
                    break

            if chosen is None:
                fallback = [h for h in feasible_by_client[c] if remaining[h] > 0]
                if not fallback:
                    return None
                chosen = min(fallback, key=lambda h: (self.distancias[c][h], abs(z[c] - h)))

            assignment[c] = chosen
            remaining[chosen] -= 1

        if not self.check(assignment):
            return None

        return assignment

    def objective_from_latent(self, latent):
        x = self.repair_from_latent(latent)
        if x is None:
            return 1e9
        return float(self.fit(x))


class SHOA_TMLAP:
    def __init__(self):
        self.problem = Problem()

        self.max_iter = 25
        self.n_agents = 10
        self.seed = 7

        self.lb = 0
        self.ub = self.problem.n_hubs - 1
        self.dim = self.problem.n_clients

        self.best_fitness = math.inf
        self.best_assignment = None
        self.best_latent = None
        self.convergence = None

    def solve(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

        best_fitness, best_latent, convergence_curve, *_ = SHO(
            pop=self.n_agents,
            Max_iter=self.max_iter,
            LB=self.lb,
            UB=self.ub,
            Dim=self.dim,
            fobj=self.problem.objective_from_latent,
        )

        self.best_fitness = float(best_fitness)
        self.best_latent = np.asarray(best_latent, dtype=float)
        self.convergence = np.asarray(convergence_curve, dtype=float)
        self.best_assignment = self.problem.repair_from_latent(self.best_latent)

        if self.best_assignment is None:
            raise RuntimeError("No se pudo decodificar una solucion factible desde el mejor estado latente.")

        self.show_results()

    def show_results(self):
        for t, value in enumerate(self.convergence, start=1):
            print(f"t: {t}, g_best fitness: {value:.4f}")

        hubs_open = sorted(set(self.best_assignment))
        print("\n=== Resultado final SHOA-TMLAP ===")
        print(f"Asignacion cliente->hub: {self.best_assignment}")
        print(f"Hubs abiertos: {hubs_open}")
        print(f"Fitness final: {self.best_fitness:.4f}")


if __name__ == "__main__":
    SHOA_TMLAP().solve()
