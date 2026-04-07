import argparse
from collections import deque
from dataclasses import dataclass, field
from math import gamma, pi, sin
from typing import Callable

import numpy as np

FEATURE_NAMES = ("r1", "mag_browniano", "mag_levy", "r2", "mag_predacion")


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


def levy(pop: int, m: int, omega: float) -> np.ndarray:
    num = gamma(1 + omega) * sin(pi * omega / 2)
    den = gamma((1 + omega) / 2) * omega * (2 ** ((omega - 1) / 2))
    sigma_u = (num / den) ** (1 / omega)
    u = np.random.normal(0, sigma_u, (pop, m))
    v = np.random.normal(0, 1, (pop, m))
    return u / (np.abs(v) ** (1 / omega))


@dataclass
class SeaHorseAgent:
    position: np.ndarray
    fitness: float
    flight_log: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))


@dataclass
class SHOXAIConfig:
    pop_size: int = 30
    max_iter: int = 500
    window_size: int = 15
    epsilon_stagnation: float = 1e-10
    cooldown_iters: int = 20
    lime_num_samples: int = 1500
    importance_threshold: float = 0.15
    delta_tolerance: float = 1e-8
    fidelity_threshold: float = 0.4
    synthetic_history_min: int = 150
    rescue_mode: str = "leader_repulsion"  # levy_teleport | leader_repulsion
    rescue_eta: float = 1.0
    rescue_levy_scale: float = 0.2
    seed: int | None = None


@dataclass
class SHOXAIResult:
    best_fitness: float
    best_position: np.ndarray
    convergence_curve: np.ndarray
    diagnostics_log: list[dict]
    diagnostics_invocation_iterations: list[int]
    rescue_count: int
    iteration_log: list[dict]


def _bounds_vector(bounds, dim: int) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(dim, arr.item(), dtype=float)
    return arr


def _alpha_schedule(iteration: int, max_iter: int) -> float:
    ratio = min(max(iteration / max_iter, 0.0), 1.0)
    return float((1.0 - ratio) ** (2.0 * ratio + 1e-12))


def _sample_decision_variables(rng: np.random.Generator, iteration: int, max_iter: int) -> np.ndarray:
    alpha = _alpha_schedule(iteration, max_iter)
    r1 = float(rng.normal(0.0, 1.0))
    mag_browniano = float(abs(rng.normal(0.0, 1.0)))
    mag_levy = float(abs(levy(1, 1, 1.5)[0, 0]))
    r2 = float(rng.random())
    mag_predacion = float(rng.random() * max(alpha, 1e-8))
    return np.array([r1, mag_browniano, mag_levy, r2, mag_predacion], dtype=float)


def _apply_motor_step(
    position: np.ndarray,
    elite: np.ndarray,
    decision: np.ndarray,
    rng: np.random.Generator,
    u: float = 0.05,
    v: float = 0.05,
    l: float = 0.05,
) -> np.ndarray:
    r1, mag_browniano, mag_levy, _, _ = decision
    dim = position.size

    if r1 > 0:
        theta = float(rng.random() * 2.0 * np.pi)
        row = u * np.exp(theta * v)
        helix = (row * np.cos(theta)) * (row * np.sin(theta)) * (row * theta)
        step_length = levy(1, dim, 1.5)[0] * mag_levy
        return position + step_length * (((elite - position) * helix) + elite)

    beta = rng.normal(0.0, 1.0, dim)
    return position + (rng.random() * l * mag_browniano) * beta * (position - beta * elite)


def _apply_predation_step(
    x_motor: np.ndarray,
    elite: np.ndarray,
    decision: np.ndarray,
    iteration: int,
    max_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    _, _, _, r2, mag_predacion = decision
    alpha = _alpha_schedule(iteration, max_iter)
    rand_scale = max(mag_predacion, 1e-12)

    if r2 >= 0.1:
        return alpha * (elite - rand_scale * x_motor) + (1.0 - alpha) * elite

    return (1.0 - alpha) * (x_motor - rand_scale * elite) + alpha * x_motor


def _best_agent(agents: list[SeaHorseAgent]) -> tuple[int, SeaHorseAgent]:
    fitness_values = np.array([agent.fitness for agent in agents], dtype=float)
    best_idx = int(np.argmin(fitness_values))
    return best_idx, agents[best_idx]


def _initialize_agents(
    pop_size: int,
    dim: int,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    objective: Callable[[np.ndarray], float],
) -> list[SeaHorseAgent]:
    positions = initialization(pop_size, dim, ub_vec, lb_vec)
    agents: list[SeaHorseAgent] = []
    for idx in range(pop_size):
        pos = positions[idx, :].astype(float)
        fit = float(objective(pos))
        agents.append(SeaHorseAgent(position=pos, fitness=fit))
    return agents


def _reproduction_and_selection(
    candidate_positions: np.ndarray,
    candidate_fitness: np.ndarray,
    objective: Callable[[np.ndarray], float],
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    pop_size: int,
    rng: np.random.Generator,
) -> list[SeaHorseAgent]:
    dim = candidate_positions.shape[1]
    order = np.argsort(candidate_fitness)
    sorted_pos = candidate_positions[order]

    half = pop_size // 2
    fathers = sorted_pos[:half]
    mothers = sorted_pos[half : half + half]
    pair_count = min(len(fathers), len(mothers))

    offspring = np.empty((pair_count, dim), dtype=float)
    for idx in range(pair_count):
        r3 = float(rng.random())
        offspring[idx] = r3 * fathers[idx] + (1.0 - r3) * mothers[idx]

    offspring = np.clip(offspring, lb_vec, ub_vec)
    offspring_fit = np.array([float(objective(x)) for x in offspring], dtype=float)

    all_positions = np.vstack([candidate_positions, offspring])
    all_fitness = np.concatenate([candidate_fitness, offspring_fit])
    best_idx = np.argsort(all_fitness)[:pop_size]

    agents: list[SeaHorseAgent] = []
    for idx in best_idx:
        agents.append(SeaHorseAgent(position=all_positions[idx].copy(), fitness=float(all_fitness[idx])))
    return agents


def _import_lime_explainer():
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("No se pudo importar lime. Ejecuta: pip install lime") from exc
    return LimeTabularExplainer


def _ensure_training_data(
    decision_history: deque[np.ndarray],
    cfg: SHOXAIConfig,
    rng: np.random.Generator,
    iteration: int,
) -> np.ndarray:
    rows = [np.asarray(row, dtype=float) for row in decision_history]
    training_data = np.array(rows, dtype=float) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=float)

    if training_data.shape[0] >= cfg.synthetic_history_min:
        return training_data

    missing = cfg.synthetic_history_min - training_data.shape[0]
    alpha = _alpha_schedule(iteration, cfg.max_iter)
    synthetic = np.empty((missing, len(FEATURE_NAMES)), dtype=float)
    synthetic[:, 0] = rng.normal(0.0, 1.0, missing)
    synthetic[:, 1] = np.abs(rng.normal(0.0, 1.0, missing))
    synthetic[:, 2] = np.clip(np.abs(levy(missing, 1, 1.5).reshape(-1)), 0.0, 5.0)
    synthetic[:, 3] = rng.random(missing)
    synthetic[:, 4] = rng.random(missing) * max(alpha, 1e-8)

    if training_data.size == 0:
        return synthetic
    return np.vstack([training_data, synthetic])


def _aggregate_feature_weights(exp_list: list[tuple[str, float]]) -> dict[str, float]:
    weights = {name: 0.0 for name in FEATURE_NAMES}
    for rule, weight in exp_list:
        for feature in FEATURE_NAMES:
            if feature in rule:
                weights[feature] += float(weight)
    return weights


def _build_lime_sim_context(rng: np.random.Generator, dim: int) -> dict[str, np.ndarray | float]:
    theta = float(rng.random() * 2.0 * np.pi)
    u = 0.05
    v = 0.05
    row = u * np.exp(theta * v)
    helix = float((row * np.cos(theta)) * (row * np.sin(theta)) * (row * theta))

    return {
        "levy_vec": levy(1, dim, 1.5)[0],
        "beta_vec": rng.normal(0.0, 1.0, dim),
        "helix": helix,
    }


def _simulate_jumps_vectorized(
    x0: np.ndarray,
    data_matrix: np.ndarray,
    alpha: float,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    context: dict[str, np.ndarray | float],
) -> np.ndarray:
    rows = np.atleast_2d(np.asarray(data_matrix, dtype=float))

    r1 = rows[:, 0:1]
    mag_browniano = rows[:, 1:2]
    mag_levy = rows[:, 2:3]
    r2 = rows[:, 3:4]
    mag_predacion = rows[:, 4:5]

    elite = x0

    levy_vec = np.asarray(context["levy_vec"], dtype=float)
    helix = float(context["helix"])

    x_espiral = x0 + (mag_levy * levy_vec) * (((elite - x0) * helix) + elite)

    beta_vec = np.asarray(context["beta_vec"], dtype=float)
    x_brown = x0 + mag_browniano * beta_vec * (x0 - beta_vec * elite)

    x_motor = np.where(r1 > 0, x_espiral, x_brown)
    x_motor = np.clip(x_motor, lb_vec, ub_vec)

    rand_scale = np.maximum(mag_predacion, 1e-12)

    pred_exito = alpha * (elite - rand_scale * x_motor) + (1.0 - alpha) * elite
    pred_fallo = (1.0 - alpha) * (x_motor - rand_scale * elite) + alpha * x_motor

    x_pred = np.where(r2 >= 0.1, pred_exito, pred_fallo)

    return np.clip(x_pred, lb_vec, ub_vec)


def _run_lime_diagnosis(
    g_best: SeaHorseAgent,
    objective: Callable[[np.ndarray], float],
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    decision_history: deque[np.ndarray],
    cfg: SHOXAIConfig,
    iteration: int,
    rng: np.random.Generator,
) -> dict:
    LimeTabularExplainer = _import_lime_explainer()

    training_data = _ensure_training_data(decision_history, cfg, rng, iteration)
    alpha = _alpha_schedule(iteration, cfg.max_iter)
    context = _build_lime_sim_context(rng, g_best.position.size)
    old_fitness = float(g_best.fitness)

    def lime_wrapper(data_matrix: np.ndarray) -> np.ndarray:
        candidatos_matriz = _simulate_jumps_vectorized(
            g_best.position, data_matrix, alpha, lb_vec, ub_vec, context
        )

        deltas = np.empty(candidatos_matriz.shape[0], dtype=float)
        for i in range(candidatos_matriz.shape[0]):
            deltas[i] = old_fitness - float(objective(candidatos_matriz[i]))

        return deltas

    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=list(FEATURE_NAMES),
        mode="regression",
        discretize_continuous=True,
        random_state=cfg.seed,
    )

    explanation = explainer.explain_instance(
        g_best.flight_log.astype(float),
        lime_wrapper,
        num_features=len(FEATURE_NAMES),
        num_samples=cfg.lime_num_samples,
    )

    weights = _aggregate_feature_weights(explanation.as_list())
    local_pred = getattr(explanation, "local_pred", np.array([0.0]))
    pred_delta = float(np.asarray(local_pred, dtype=float).reshape(-1)[0])
    fidelity = float(getattr(explanation, "score", np.nan))

    strong_stochastic_importance = (
        abs(weights["mag_browniano"]) > cfg.importance_threshold
        or abs(weights["mag_predacion"]) > cfg.importance_threshold
    )
    low_expected_improvement = pred_delta < cfg.delta_tolerance
    fidelity_ok = (not np.isfinite(fidelity)) or (fidelity >= cfg.fidelity_threshold)

    status = "POSITIVE_STAGNATION" if (strong_stochastic_importance and low_expected_improvement and fidelity_ok) else "FALSE_ALARM"

    return {
        "status": status,
        "weights": weights,
        "pred_delta": pred_delta,
        "fidelity": fidelity,
        "strong_stochastic_importance": strong_stochastic_importance,
        "low_expected_improvement": low_expected_improvement,
    }


def _apply_rescue_mutation(
    agents: list[SeaHorseAgent],
    best_idx: int,
    objective: Callable[[np.ndarray], float],
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    cfg: SHOXAIConfig,
    rng: np.random.Generator,
) -> list[SeaHorseAgent]:
    pop_size = len(agents)
    dim = agents[best_idx].position.size

    if cfg.rescue_mode == "levy_teleport":
        n_mutate = max(1, pop_size // 2)
        mutable_idx = [idx for idx in range(pop_size) if idx != best_idx]
        if not mutable_idx:
            mutable_idx = [best_idx]
        selected = rng.choice(mutable_idx, size=min(n_mutate, len(mutable_idx)), replace=False)
        leader_position = agents[best_idx].position.copy()

        for idx in selected:
            jump = levy(1, dim, 1.5)[0] * cfg.rescue_levy_scale
            new_pos = np.clip(leader_position + jump, lb_vec, ub_vec)
            agents[int(idx)].position = new_pos
            agents[int(idx)].fitness = float(objective(new_pos))

    elif cfg.rescue_mode == "leader_repulsion":
        centroid = np.mean(np.vstack([agent.position for agent in agents]), axis=0)
        direction = agents[best_idx].position - centroid
        norm = float(np.linalg.norm(direction))

        direction = rng.normal(0.0, 1.0, dim)
        norm = float(np.linalg.norm(direction))

        direction = direction / (norm + 1e-12)
        new_pos = np.clip(agents[best_idx].position + cfg.rescue_eta * direction, lb_vec, ub_vec)
        agents[best_idx].position = new_pos
        agents[best_idx].fitness = float(objective(new_pos))

    else:
        raise ValueError(f"Modo de rescate no soportado: {cfg.rescue_mode}")

    return agents


def SHO_with_lime_controller(
    objective: Callable[[np.ndarray], float],
    LB,
    UB,
    Dim: int,
    cfg: SHOXAIConfig,
) -> SHOXAIResult:
    if cfg.pop_size < 4:
        raise ValueError("pop_size debe ser >= 4")

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    lb_vec = _bounds_vector(LB, Dim)
    ub_vec = _bounds_vector(UB, Dim)

    agents = _initialize_agents(cfg.pop_size, Dim, lb_vec, ub_vec, objective)
    best_idx, g_best = _best_agent(agents)

    decision_history: deque[np.ndarray] = deque(maxlen=max(cfg.synthetic_history_min * 6, cfg.window_size * cfg.pop_size * 2))
    memory_window: deque[float] = deque(maxlen=cfg.window_size)

    convergence_curve = np.zeros(cfg.max_iter, dtype=float)
    diagnostics_log: list[dict] = []
    diagnostics_invocation_iterations: list[int] = []
    iteration_log: list[dict] = []
    cooldown_counter = 0
    rescue_count = 0

    for iteration in range(1, cfg.max_iter + 1):
        candidate_positions = np.empty((cfg.pop_size, Dim), dtype=float)
        candidate_fitness = np.empty(cfg.pop_size, dtype=float)

        for idx, agent in enumerate(agents):
            decision = _sample_decision_variables(rng, iteration, cfg.max_iter)
            agent.flight_log = decision.copy()
            decision_history.append(decision.copy())

            x_motor = _apply_motor_step(agent.position, g_best.position, decision, rng)
            x_motor = np.clip(x_motor, lb_vec, ub_vec)

            x_pred = _apply_predation_step(x_motor, g_best.position, decision, iteration, cfg.max_iter, rng)
            x_pred = np.clip(x_pred, lb_vec, ub_vec)

            candidate_positions[idx] = x_pred
            candidate_fitness[idx] = float(objective(x_pred))

        agents = _reproduction_and_selection(candidate_positions, candidate_fitness, objective, lb_vec, ub_vec, cfg.pop_size, rng)
        best_idx, g_best = _best_agent(agents)

        convergence_curve[iteration - 1] = g_best.fitness
        memory_window.append(g_best.fitness)

        window_std = float(np.std(memory_window)) if len(memory_window) > 1 else float("nan")
        trigger_candidate = bool(
            len(memory_window) == cfg.window_size
            and np.isfinite(window_std)
            and window_std < cfg.epsilon_stagnation
        )
        diagnostics_invoked = False
        diagnosis_status = "NONE"
        diagnosis_pred_delta = float("nan")
        diagnosis_fidelity = float("nan")
        rescue_applied = False

        if cooldown_counter > 0:
            cooldown_counter -= 1

        elif trigger_candidate:
            diagnostics_invocation_iterations.append(iteration)
            diagnosis = _run_lime_diagnosis(g_best, objective, lb_vec, ub_vec, decision_history, cfg, iteration, rng)
            diagnosis["iteration"] = iteration
            diagnostics_log.append(diagnosis)
            diagnostics_invoked = True
            diagnosis_status = str(diagnosis.get("status", "UNKNOWN"))
            diagnosis_pred_delta = float(diagnosis.get("pred_delta", np.nan))
            diagnosis_fidelity = float(diagnosis.get("fidelity", np.nan))

            if diagnosis_status == "POSITIVE_STAGNATION":
                agents = _apply_rescue_mutation(agents, best_idx, objective, lb_vec, ub_vec, cfg, rng)
                for agent in agents:
                    agent.fitness = float(objective(agent.position))
                best_idx, g_best = _best_agent(agents)

                convergence_curve[iteration - 1] = g_best.fitness
                memory_window.clear()
                cooldown_counter = cfg.cooldown_iters
                rescue_count += 1
                rescue_applied = True

        iteration_log.append(
            {
                "iteration": iteration,
                "best_fitness": float(convergence_curve[iteration - 1]),
                "window_size": len(memory_window),
                "window_std": window_std,
                "trigger_candidate": trigger_candidate,
                "diagnostics_invoked": diagnostics_invoked,
                "diagnosis_status": diagnosis_status,
                "diagnosis_pred_delta": diagnosis_pred_delta,
                "diagnosis_fidelity": diagnosis_fidelity,
                "rescue_applied": rescue_applied,
                "cooldown_counter": int(cooldown_counter),
            }
        )

    return SHOXAIResult(
        best_fitness=float(g_best.fitness),
        best_position=g_best.position.copy(),
        convergence_curve=convergence_curve,
        diagnostics_log=diagnostics_log,
        diagnostics_invocation_iterations=diagnostics_invocation_iterations,
        rescue_count=rescue_count,
        iteration_log=iteration_log,
    )


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

        self.best_seen_fitness = float("inf")
        self.best_seen_assignment = None

    def reset_tracking(self):
        self.best_seen_fitness = float("inf")
        self.best_seen_assignment = None

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

        fitness = float(self.fit(x))
        if fitness < self.best_seen_fitness:
            self.best_seen_fitness = fitness
            self.best_seen_assignment = list(x)

        return fitness


class SHOA_LIME_TMLAP:
    def __init__(self, args):
        self.problem = Problem()

        self.max_iter = int(args.max_iter)
        self.n_agents = int(args.population)
        self.seed = int(args.seed)

        self.lb = 0
        self.ub = self.problem.n_hubs - 1
        self.dim = self.problem.n_clients

        self.window_size = int(args.window_size)
        self.epsilon_stagnation = float(args.epsilon)
        self.cooldown_iters = int(args.cooldown)
        self.lime_samples = int(args.lime_samples)
        self.importance_threshold = float(args.importance_threshold)
        self.delta_tolerance = float(args.delta_tolerance)
        self.fidelity_threshold = float(args.fidelity_threshold)
        self.rescue_mode = str(args.rescue_mode)
        self.rescue_eta = float(args.rescue_eta)
        self.rescue_levy_scale = float(args.rescue_levy_scale)

        self.result = None
        self.best_assignment = None
        self.best_fitness_seen = None
        self.final_assignment = None

    def solve(self):
        np.random.seed(self.seed)

        self.problem.reset_tracking()

        cfg = SHOXAIConfig(
            pop_size=self.n_agents,
            max_iter=self.max_iter,
            window_size=self.window_size,
            epsilon_stagnation=self.epsilon_stagnation,
            cooldown_iters=self.cooldown_iters,
            lime_num_samples=self.lime_samples,
            importance_threshold=self.importance_threshold,
            delta_tolerance=self.delta_tolerance,
            fidelity_threshold=self.fidelity_threshold,
            rescue_mode=self.rescue_mode,
            rescue_eta=self.rescue_eta,
            rescue_levy_scale=self.rescue_levy_scale,
            seed=self.seed,
        )

        self.result = SHO_with_lime_controller(
            objective=self.problem.objective_from_latent,
            LB=self.lb,
            UB=self.ub,
            Dim=self.dim,
            cfg=cfg,
        )

        self.final_assignment = self.problem.repair_from_latent(self.result.best_position)

        if self.problem.best_seen_assignment is not None:
            self.best_assignment = list(self.problem.best_seen_assignment)
            self.best_fitness_seen = float(self.problem.best_seen_fitness)
        else:
            self.best_assignment = self.final_assignment
            self.best_fitness_seen = float(self.result.best_fitness)

        if self.best_assignment is None:
            raise RuntimeError("No se pudo decodificar una solucion factible durante la corrida SHOA+LIME.")

        self.show_results()

    def show_results(self):
        for t, value in enumerate(self.result.convergence_curve, start=1):
            print(f"t: {t}, g_best fitness: {float(value):.4f}")

        hubs_open = sorted(set(self.best_assignment))
        positive_diag_iters = [
            diag["iteration"]
            for diag in self.result.diagnostics_log
            if str(diag.get("status", "")) == "POSITIVE_STAGNATION"
        ]

        print("\n=== Resultado final SHOA+LIME-TMLAP ===")
        print(f"Asignacion cliente->hub: {self.best_assignment}")
        print(f"Hubs abiertos: {hubs_open}")
        print(f"Best fitness observado: {float(self.best_fitness_seen):.4f}")
        print(f"Fitness del estado final del controlador: {float(self.result.best_fitness):.4f}")
        print(f"Diagnostics invoked: {len(self.result.diagnostics_log)}")
        print(f"Diagnostics invocation iterations: {self.result.diagnostics_invocation_iterations}")
        print(f"Rescues applied: {self.result.rescue_count}")
        print(f"Positive diagnosis iterations: {positive_diag_iters}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resolver TMLAP con SHOA+LIME")

    parser.add_argument("--max-iter", type=int, default=25, help="Iteraciones maximas")
    parser.add_argument("--population", type=int, default=10, help="Tamano de poblacion")
    parser.add_argument("--seed", type=int, default=7, help="Semilla")

    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--cooldown", type=int, default=4)
    parser.add_argument("--lime-samples", type=int, default=800)
    parser.add_argument("--importance-threshold", type=float, default=0.05)
    parser.add_argument("--delta-tolerance", type=float, default=1e-6)
    parser.add_argument("--fidelity-threshold", type=float, default=0.15)
    parser.add_argument("--rescue-mode", choices=["levy_teleport", "leader_repulsion"], default="levy_teleport")
    parser.add_argument("--rescue-eta", type=float, default=1.0)
    parser.add_argument("--rescue-levy-scale", type=float, default=0.14)

    return parser.parse_args()


def main():
    args = parse_args()
    solver = SHOA_LIME_TMLAP(args)
    solver.solve()


if __name__ == "__main__":
    main()
