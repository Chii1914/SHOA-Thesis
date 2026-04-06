"""SHO with stagnation diagnosis and rescue using LIME and Opfunu."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from initialization import initialization
from levy import levy

FEATURE_NAMES = ("r1", "mag_browniano", "mag_levy", "r2", "mag_predacion")


@dataclass
class SeaHorseAgent:
    """Single sea-horse agent with local decision log."""

    position: np.ndarray
    fitness: float
    flight_log: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))


@dataclass
class SHOXAIConfig:
    """Configuration for SHO with XAI-based stagnation controller."""

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
    """Output payload for SHO with XAI controller."""

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
    # mag_predacion ya incluye el factor estocastico de la decision
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
    # Aplicamos un limite superior para evitar colas infinitas que rompan la varianza.
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


def _simulate_one_jump_from_decision(
    x0: np.ndarray,
    decision: np.ndarray,
    alpha: float,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    context: dict[str, np.ndarray | float],
) -> np.ndarray:
    r1, mag_browniano, mag_levy, r2, mag_predacion = decision
    elite = x0

    if r1 > 0:
        levy_vec = np.asarray(context["levy_vec"], dtype=float)
        helix = float(context["helix"])
        x_motor = x0 + (mag_levy * levy_vec) * (((elite - x0) * helix) + elite)
    else:
        beta_vec = np.asarray(context["beta_vec"], dtype=float)
        x_motor = x0 + mag_browniano * beta_vec * (x0 - beta_vec * elite)

    x_motor = np.clip(x_motor, lb_vec, ub_vec)
    rand_scale = max(mag_predacion, 1e-12)

    if r2 >= 0.1:
        x_pred = alpha * (elite - rand_scale * x_motor) + (1.0 - alpha) * elite
    else:
        x_pred = (1.0 - alpha) * (x_motor - rand_scale * elite) + alpha * x_motor

    return np.clip(x_pred, lb_vec, ub_vec)


def _simulate_jumps_vectorized(
    x0: np.ndarray,
    data_matrix: np.ndarray,
    alpha: float,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    context: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Calcula miles de saltos simultaneos usando Numpy broadcasting."""
    rows = np.atleast_2d(np.asarray(data_matrix, dtype=float))

    # Extraer columnas completas (N, 1) para broadcasting sobre dimensiones (N, dim)
    r1 = rows[:, 0:1]
    mag_browniano = rows[:, 1:2]
    mag_levy = rows[:, 2:3]
    r2 = rows[:, 3:4]
    mag_predacion = rows[:, 4:5]

    elite = x0  # (dim,)

    # --- FASE 1: MOTOR (Vectorizada) ---
    levy_vec = np.asarray(context["levy_vec"], dtype=float)  # (dim,)
    helix = float(context["helix"])

    # Calculamos ambas ramas para todas las muestras al mismo tiempo.
    x_espiral = x0 + (mag_levy * levy_vec) * (((elite - x0) * helix) + elite)

    beta_vec = np.asarray(context["beta_vec"], dtype=float)  # (dim,)
    x_brown = x0 + mag_browniano * beta_vec * (x0 - beta_vec * elite)

    # Elegimos usando np.where segun la condicion r1 > 0.
    x_motor = np.where(r1 > 0, x_espiral, x_brown)
    x_motor = np.clip(x_motor, lb_vec, ub_vec)

    # --- FASE 2: PREDACION (Vectorizada) ---
    rand_scale = np.maximum(mag_predacion, 1e-12)

    pred_exito = alpha * (elite - rand_scale * x_motor) + (1.0 - alpha) * elite
    pred_fallo = (1.0 - alpha) * (x_motor - rand_scale * elite) + alpha * x_motor

    # Elegimos usando np.where segun r2 >= 0.1.
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
        # 1. Calculamos todas las posiciones de un solo golpe matricial.
        candidatos_matriz = _simulate_jumps_vectorized(
            g_best.position, data_matrix, alpha, lb_vec, ub_vec, context
        )

        # 2. Evaluamos la funcion objetivo para cada candidato.
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

        # Si estamos estancados, todos estan agrupados.
        # La mejor repulsion en un estancamiento severo es puramente aleatoria
        # en lugar de mantener la pequena inercia lineal que los llevo al pozo.
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
    """Run SHO with trigger, LIME diagnosis and autonomous rescue."""
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


def build_opfunu_cec_objective(function_class_name: str, ndim: int):
    """Build objective callable and bounds from an Opfunu class name (e.g. F12022)."""
    try:
        import opfunu
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise RuntimeError("Opfunu requiere pkg_resources; instala setuptools==75.8.0") from exc
        raise RuntimeError("No se pudo importar opfunu. Ejecuta: pip install opfunu") from exc

    funcs = opfunu.get_functions_by_classname(function_class_name)
    if not funcs:
        raise ValueError(f"No existe la clase {function_class_name} en opfunu")

    cls = funcs[0]
    try:
        problem = cls(ndim=ndim)
    except Exception as exc:
        raise ValueError(f"No se pudo instanciar {function_class_name} con ndim={ndim}: {exc}") from exc

    lb_vec = _bounds_vector(problem.lb, problem.ndim)
    ub_vec = _bounds_vector(problem.ub, problem.ndim)

    def objective(x: np.ndarray) -> float:
        return float(problem.evaluate(np.asarray(x, dtype=float).reshape(-1)))

    return lb_vec, ub_vec, int(problem.ndim), objective, problem
