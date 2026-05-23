"""Sea-Horse Optimizer with paper-style stagnation detection (no LIME)."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from initialization import initialization
from levy import levy
from stagnation_detector import StagnationDetector


def _bounds_vector(bounds: Any, dim: int) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(dim, arr.item(), dtype=float)
    return arr


def estimate_max_fes(pop: int, max_iter: int) -> int:
    """Estimated FE budget based on SHO evaluation pattern.

    Initial evaluations: pop
    Per iteration: pop (predation candidates) + pop//2 (offspring)
    """

    return int(pop + max_iter * (pop + pop // 2))


def SHO_STAGNATION(
    pop: int,
    max_iter: int,
    lower_bound,
    upper_bound,
    dim: int,
    fobj,
    run_metadata: dict,
    min_sfes_ratio: float = 0.04,
    max_fes: int = 0,
    random_state: int = 42,
    log_callback: Callable[[str], None] | None = None,
    progress_every: int = 1,
    verbose: bool = True,
) -> dict:
    progress_every = max(1, int(progress_every))
    effective_max_fes = int(max_fes) if int(max_fes) > 0 else estimate_max_fes(pop=pop, max_iter=max_iter)

    if log_callback is None:

        def default_log_callback(message: str) -> None:
            print(message, flush=True)

        log_fn = default_log_callback
    else:
        log_fn = log_callback

    def _log(message: str) -> None:
        if verbose:
            log_fn(message)

    np.random.seed(random_state)

    sea_horses = initialization(pop, dim, upper_bound, lower_bound)

    sea_horses_fitness = np.zeros(pop, dtype=float)
    fitness_history = np.zeros((pop, max_iter), dtype=float)
    population_history = np.zeros((pop, dim, max_iter), dtype=float)
    convergence_curve = np.zeros(max_iter, dtype=float)
    trajectories = np.zeros((pop, max_iter), dtype=float)

    for i in range(pop):
        sea_horses_fitness[i] = fobj(sea_horses[i, :])
        fitness_history[i, 0] = sea_horses_fitness[i]
        population_history[i, :, 0] = sea_horses[i, :]
    trajectories[:, 0] = sea_horses[:, 0]

    sorted_indexes = np.argsort(sea_horses_fitness)
    target_position = sea_horses[sorted_indexes[0], :].copy()
    target_fitness = float(sea_horses_fitness[sorted_indexes[0]])
    convergence_curve[0] = target_fitness

    current_fe = int(getattr(fobj, "nfev", pop))
    detector = StagnationDetector(max_fes=effective_max_fes, min_sfes_ratio=min_sfes_ratio)
    detector.initialize(best_fitness=target_fitness, fe=current_fe, iteration=0)

    _log(
        f"[{run_metadata['function_name']}] inicio | pop={pop} dim={dim} max_iter={max_iter} "
        f"max_fes={effective_max_fes} min_sfes={detector.min_sfes} "
        f"best_inicial={target_fitness:.6e}"
    )

    lb_vec = _bounds_vector(lower_bound, dim)
    ub_vec = _bounds_vector(upper_bound, dim)

    u = 0.05
    v = 0.05
    l = 0.05

    t = 1
    full_output_rows: list[dict] = []
    stagnation_history_rows: list[dict] = []
    stagnation_event_rows: list[dict] = []

    while t < max_iter + 1 and int(getattr(fobj, "nfev", 0)) < effective_max_fes:
        iteration_start = time.perf_counter()
        previous_population = sea_horses.copy()
        previous_fitness = sea_horses_fitness.copy()
        elite_position = target_position.copy()

        beta = np.random.randn(pop, dim)
        elite = np.tile(elite_position, (pop, 1))

        # Motor behavior
        r1 = np.random.randn(pop)
        step_length = levy(pop, dim, 1.5)
        sea_horses_new1 = np.zeros_like(sea_horses)

        for i in range(pop):
            for j in range(dim):
                if r1[i] > 0:
                    rand_theta = np.random.rand()
                    theta = rand_theta * 2 * np.pi
                    row = u * np.exp(theta * v)
                    x = row * np.cos(theta)
                    y = row * np.sin(theta)
                    z = row * theta
                    sea_horses_new1[i, j] = previous_population[i, j] + step_length[i, j] * (
                        (elite[i, j] - previous_population[i, j]) * x * y * z + elite[i, j]
                    )
                else:
                    browniano_term = (
                        np.random.rand()
                        * l
                        * beta[i, j]
                        * (previous_population[i, j] - beta[i, j] * elite[i, j])
                    )
                    sea_horses_new1[i, j] = previous_population[i, j] + browniano_term

        sea_horses_new1 = np.clip(sea_horses_new1, lb_vec, ub_vec)

        # Predation behavior
        sea_horses_new2 = np.zeros_like(sea_horses)
        r2 = np.random.rand(pop)
        alpha = (1 - t / max_iter) ** (2 * t / max_iter)

        for i in range(pop):
            for j in range(dim):
                if r2[i] >= 0.1:
                    stochastic = np.random.rand() * sea_horses_new1[i, j]
                    sea_horses_new2[i, j] = alpha * (elite[i, j] - stochastic) + (1 - alpha) * elite[i, j]
                else:
                    stochastic = np.random.rand() * elite[i, j]
                    sea_horses_new2[i, j] = (1 - alpha) * (sea_horses_new1[i, j] - stochastic) + alpha * sea_horses_new1[i, j]

        sea_horses_new2 = np.clip(sea_horses_new2, lb_vec, ub_vec)
        sea_horses_fitness1 = np.array([fobj(ind) for ind in sea_horses_new2], dtype=float)

        delta_f = sea_horses_fitness1 - previous_fitness
        improved_count = int(np.sum(delta_f < 0))
        diversity_metric = float(np.mean(np.linalg.norm(previous_population - np.mean(previous_population, axis=0), axis=1)))

        # Reproductive behavior (parity with original SHO)
        index = np.argsort(sea_horses_fitness1)
        half = pop // 2
        sea_horses_father = sea_horses_new2[index[:half], :]
        sea_horses_mother = sea_horses_new2[index[half:pop], :]

        si = np.zeros((half, dim), dtype=float)
        for k in range(half):
            r3 = np.random.rand()
            si[k, :] = r3 * sea_horses_father[k, :] + (1 - r3) * sea_horses_mother[k, :]

        sea_horses_offspring = np.clip(si, lb_vec, ub_vec)
        sea_horses_fitness2 = np.array([fobj(ind) for ind in sea_horses_offspring], dtype=float)

        # Selection
        sea_horses_fitness_all = np.concatenate([sea_horses_fitness1, sea_horses_fitness2])
        sea_horses_new = np.vstack([sea_horses_new2, sea_horses_offspring])

        sorted_indexes = np.argsort(sea_horses_fitness_all)
        sea_horses = sea_horses_new[sorted_indexes[:pop], :]

        sortfitbestn = sea_horses_fitness_all[sorted_indexes[:pop]]
        sea_horses_fitness = sortfitbestn.copy()
        fitness_history[:, t - 1] = sortfitbestn
        population_history[:, :, t - 1] = sea_horses
        trajectories[:, t - 1] = sea_horses[:, 0]

        if sortfitbestn[0] < target_fitness:
            target_position = sea_horses[0, :].copy()
            target_fitness = float(sortfitbestn[0])

        convergence_curve[t - 1] = target_fitness

        current_fe = int(getattr(fobj, "nfev", current_fe))
        snapshot = detector.update(best_fitness=target_fitness, fe=current_fe, iteration=t)

        if snapshot.event == "stagnation_start":
            _log(
                f"[{run_metadata['function_name']}] ESTANCAMIENTO detectado | "
                f"iter={t} fe={snapshot.fe} sfes={snapshot.sfes} min_sfes={snapshot.min_sfes}"
            )
        elif snapshot.event == "recovered":
            _log(
                f"[{run_metadata['function_name']}] recuperado de estancamiento | "
                f"iter={t} fe={snapshot.fe} sfes={snapshot.sfes}"
            )

        if snapshot.event != "none":
            stagnation_event_rows.append(
                {
                    "run_id": run_metadata["run_id"],
                    "timestamp": run_metadata["timestamp"],
                    "function_name": run_metadata["function_name"],
                    "dimension": run_metadata["dimension"],
                    "iteration": t,
                    "event": snapshot.event,
                    "fe": snapshot.fe,
                    "sfes": snapshot.sfes,
                    "min_sfes": snapshot.min_sfes,
                    "last_improvement_fe": snapshot.last_improvement_fe,
                }
            )

        full_output_rows.append(
            {
                "run_id": run_metadata["run_id"],
                "timestamp": run_metadata["timestamp"],
                "function_name": run_metadata["function_name"],
                "dimension": run_metadata["dimension"],
                "iteration": t,
                "best_fitness_so_far": float(target_fitness),
                "population_avg_fitness": float(np.mean(sea_horses_fitness1)),
                "diversity_metric": diversity_metric,
                "improved_agents": improved_count,
                "fe": snapshot.fe,
                "max_fes": detector.max_fes,
                "last_improvement_fe": snapshot.last_improvement_fe,
                "sfes": snapshot.sfes,
                "min_sfes": snapshot.min_sfes,
                "stagnated": int(snapshot.stagnated),
                "event": snapshot.event,
            }
        )

        stagnation_history_rows.append(
            {
                "run_id": run_metadata["run_id"],
                "timestamp": run_metadata["timestamp"],
                "function_name": run_metadata["function_name"],
                "dimension": run_metadata["dimension"],
                "iteration": t,
                "fe": snapshot.fe,
                "best_fitness": float(snapshot.best_fitness),
                "last_improvement_fe": snapshot.last_improvement_fe,
                "sfes": snapshot.sfes,
                "min_sfes": snapshot.min_sfes,
                "stagnated": int(snapshot.stagnated),
                "improved": int(snapshot.improved),
                "event": snapshot.event,
            }
        )

        should_log = (
            t == 1
            or t == max_iter
            or (t % progress_every == 0)
            or (snapshot.event != "none")
        )
        if should_log:
            elapsed = time.perf_counter() - iteration_start
            _log(
                f"[{run_metadata['function_name']}] iter {t}/{max_iter} | "
                f"best={target_fitness:.6e} avg={float(np.mean(sea_horses_fitness1)):.6e} "
                f"improved={improved_count}/{pop} fe={snapshot.fe}/{detector.max_fes} "
                f"sfes={snapshot.sfes}/{snapshot.min_sfes} stagnated={'si' if snapshot.stagnated else 'no'} "
                f"dt={elapsed:.2f}s"
            )

        t += 1

    final_fe = int(getattr(fobj, "nfev", current_fe))
    stop_reason = "max_fes" if final_fe >= effective_max_fes else "max_iter"

    _log(
        f"[{run_metadata['function_name']}] fin | best={target_fitness:.6e} fe={final_fe} "
        f"stagnation_events={len(stagnation_event_rows)} stop={stop_reason}"
    )

    return {
        "best_fitness": float(target_fitness),
        "best_position": target_position.tolist(),
        "convergence_curve": convergence_curve.tolist(),
        "trajectories": trajectories.tolist(),
        "fitness_history": fitness_history.tolist(),
        "population_history": population_history.tolist(),
        "full_output_rows": full_output_rows,
        "stagnation_history_rows": stagnation_history_rows,
        "stagnation_event_rows": stagnation_event_rows,
        "stagnation_meta": {
            "max_fes": int(detector.max_fes),
            "min_sfes_ratio": float(detector.min_sfes_ratio),
            "min_sfes": int(detector.min_sfes),
            "final_fe": int(final_fe),
            "stop_reason": stop_reason,
        },
    }
