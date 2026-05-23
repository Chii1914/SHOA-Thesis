"""Sea-Horse Optimizer with per-agent LIME diagnostics."""

from __future__ import annotations

import time
from typing import Callable
from typing import Any

import numpy as np

from initialization import initialization
from levy import levy
from lime_diagnostic import explain_selected_agents, select_agents_for_lime


FEATURE_COLUMNS = [
    "r1",
    "r2",
    "alpha",
    "theta_mean",
    "theta_active",
    "mag_levy_mean",
    "mag_levy_max",
    "mag_levy_std",
    "mag_browniano_mean",
    "mag_browniano_max",
    "mag_browniano_std",
    "mag_predacion_mean",
    "mag_predacion_max",
    "mag_predacion_std",
    "distance_to_elite",
    "delta_position_norm",
]


def _bounds_vector(bounds: Any, dim: int) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(dim, arr.item(), dtype=float)
    return arr


def _safe_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))


def SHO_LIME(
    pop: int,
    max_iter: int,
    lower_bound,
    upper_bound,
    dim: int,
    fobj,
    run_metadata: dict,
    random_state: int = 42,
    lime_every: int = 1,
    min_samples_before_lime: int | None = None,
    lime_selection_mode: str = "medoid",
    log_callback: Callable[[str], None] | None = None,
    progress_every: int = 1,
    verbose: bool = True,
) -> dict:
    if min_samples_before_lime is None:
        min_samples_before_lime = max(pop * 2, 30)
    lime_selection_mode = str(lime_selection_mode).strip().lower()
    if lime_selection_mode not in {"selected_agents", "medoid"}:
        raise ValueError("lime_selection_mode must be 'selected_agents' or 'medoid'")

    progress_every = max(1, int(progress_every))

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
    rng = np.random.default_rng(random_state)

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

    _log(
        f"[{run_metadata['function_name']}] inicio | pop={pop} dim={dim} "
        f"max_iter={max_iter} best_inicial={target_fitness:.6e}"
    )

    lb_vec = _bounds_vector(lower_bound, dim)
    ub_vec = _bounds_vector(upper_bound, dim)

    u = 0.05
    v = 0.05
    l = 0.05

    t = 1
    diagnosis_id = 0

    agent_samples: list[dict] = []
    full_output_rows: list[dict] = []
    contribution_rows: list[dict] = []

    while t < max_iter + 1:
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

        theta_means = np.zeros(pop, dtype=float)
        theta_active = np.zeros(pop, dtype=float)
        mag_levy_stats = np.zeros((pop, 3), dtype=float)
        mag_browniano_stats = np.zeros((pop, 3), dtype=float)

        for i in range(pop):
            theta_values: list[float] = []
            levy_values: list[float] = []
            browniano_values: list[float] = []

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
                    theta_values.append(float(theta))
                    levy_values.append(float(abs(step_length[i, j])))
                else:
                    browniano_term = (
                        np.random.rand()
                        * l
                        * beta[i, j]
                        * (previous_population[i, j] - beta[i, j] * elite[i, j])
                    )
                    sea_horses_new1[i, j] = previous_population[i, j] + browniano_term
                    browniano_values.append(float(abs(browniano_term)))

            if theta_values:
                theta_means[i] = float(np.mean(theta_values))
                theta_active[i] = 1.0

            mag_levy_stats[i, :] = np.array(_safe_stats(levy_values), dtype=float)
            mag_browniano_stats[i, :] = np.array(_safe_stats(browniano_values), dtype=float)

        sea_horses_new1 = np.clip(sea_horses_new1, lb_vec, ub_vec)

        # Predation behavior
        sea_horses_new2 = np.zeros_like(sea_horses)
        r2 = np.random.rand(pop)
        alpha = (1 - t / max_iter) ** (2 * t / max_iter)

        mag_predacion_stats = np.zeros((pop, 3), dtype=float)

        for i in range(pop):
            predacion_values: list[float] = []
            for j in range(dim):
                if r2[i] >= 0.1:
                    stochastic = np.random.rand() * sea_horses_new1[i, j]
                    sea_horses_new2[i, j] = alpha * (elite[i, j] - stochastic) + (1 - alpha) * elite[i, j]
                else:
                    stochastic = np.random.rand() * elite[i, j]
                    sea_horses_new2[i, j] = (1 - alpha) * (sea_horses_new1[i, j] - stochastic) + alpha * sea_horses_new1[i, j]
                predacion_values.append(float(abs(stochastic)))

            mag_predacion_stats[i, :] = np.array(_safe_stats(predacion_values), dtype=float)

        sea_horses_new2 = np.clip(sea_horses_new2, lb_vec, ub_vec)
        sea_horses_fitness1 = np.array([fobj(ind) for ind in sea_horses_new2], dtype=float)

        distance_to_elite = np.linalg.norm(previous_population - elite, axis=1)
        delta_position_norm = np.linalg.norm(sea_horses_new2 - previous_population, axis=1)
        delta_f = sea_horses_fitness1 - previous_fitness
        improved_count = int(np.sum(delta_f < 0))

        current_iteration_samples: list[dict] = []
        for i in range(pop):
            row = {
                "iteration": t,
                "agent_id": i,
                "r1": float(r1[i]),
                "r2": float(r2[i]),
                "alpha": float(alpha),
                "theta_mean": float(theta_means[i]),
                "theta_active": float(theta_active[i]),
                "mag_levy_mean": float(mag_levy_stats[i, 0]),
                "mag_levy_max": float(mag_levy_stats[i, 1]),
                "mag_levy_std": float(mag_levy_stats[i, 2]),
                "mag_browniano_mean": float(mag_browniano_stats[i, 0]),
                "mag_browniano_max": float(mag_browniano_stats[i, 1]),
                "mag_browniano_std": float(mag_browniano_stats[i, 2]),
                "mag_predacion_mean": float(mag_predacion_stats[i, 0]),
                "mag_predacion_max": float(mag_predacion_stats[i, 1]),
                "mag_predacion_std": float(mag_predacion_stats[i, 2]),
                "distance_to_elite": float(distance_to_elite[i]),
                "delta_position_norm": float(delta_position_norm[i]),
                "fitness_old": float(previous_fitness[i]),
                "fitness_new": float(sea_horses_fitness1[i]),
                "delta_f": float(delta_f[i]),
                "improved": int(delta_f[i] < 0),
                "y_reg": float(-delta_f[i]),
            }
            current_iteration_samples.append(row)

        agent_samples.extend(current_iteration_samples)

        selection_result = select_agents_for_lime(
            fitness_new=sea_horses_fitness1,
            delta_f=delta_f,
            distance_to_elite=distance_to_elite,
            rng=rng,
        )

        lime_triggered = False
        if t % lime_every == 0 and len(agent_samples) >= min_samples_before_lime:
            diagnosis_id += 1
            lime_rows = explain_selected_agents(
                all_samples=agent_samples,
                current_samples=current_iteration_samples,
                selected_indices=selection_result.selected_unique,
                feature_columns=FEATURE_COLUMNS,
                run_metadata={
                    "run_id": run_metadata["run_id"],
                    "timestamp": run_metadata["timestamp"],
                    "function_name": run_metadata["function_name"],
                    "dimension": run_metadata["dimension"],
                },
                diagnosis_id=diagnosis_id,
                iteration=t,
                random_state=random_state + diagnosis_id,
                selection_mode=lime_selection_mode,
            )
            if lime_rows:
                lime_triggered = True
                contribution_rows.extend(lime_rows)

        diversity_metric = float(np.mean(np.linalg.norm(previous_population - np.mean(previous_population, axis=0), axis=1)))

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
                "selected_agents": len(selection_result.selected_unique),
                "elite_selected": len(selection_result.category_indices.get("elite_high_impact", [])),
                "diverse_selected": len(selection_result.category_indices.get("diverse", [])),
                "outliers_selected": len(selection_result.category_indices.get("outliers", [])),
                "random_selected": len(selection_result.category_indices.get("random", [])),
                "lime_selection_mode": lime_selection_mode,
                "lime_triggered": int(lime_triggered),
                "diagnosis_id": diagnosis_id if lime_triggered else 0,
            }
        )

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

        should_log = (
            t == 1
            or t == max_iter
            or (t % progress_every == 0)
            or lime_triggered
        )
        if should_log:
            elapsed = time.perf_counter() - iteration_start
            lime_status = "si" if lime_triggered else "no"
            diagnosis_text = f" id={diagnosis_id}" if lime_triggered else ""
            _log(
                f"[{run_metadata['function_name']}] iter {t}/{max_iter} | "
                f"best={target_fitness:.6e} avg={float(np.mean(sea_horses_fitness1)):.6e} "
                f"improved={improved_count}/{pop} selected={len(selection_result.selected_unique)} "
                f"lime={lime_status}{diagnosis_text} mode={lime_selection_mode} dt={elapsed:.2f}s"
            )

        t += 1

    _log(
        f"[{run_metadata['function_name']}] fin | best={target_fitness:.6e} "
        f"diagnosticos={diagnosis_id} muestras={len(agent_samples)}"
    )

    return {
        "best_fitness": float(target_fitness),
        "best_position": target_position.tolist(),
        "convergence_curve": convergence_curve.tolist(),
        "trajectories": trajectories.tolist(),
        "fitness_history": fitness_history.tolist(),
        "population_history": population_history.tolist(),
        "agent_samples": agent_samples,
        "full_output_rows": full_output_rows,
        "lime_contribution_rows": contribution_rows,
    }
