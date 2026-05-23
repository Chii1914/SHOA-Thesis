"""Sea-Horse Optimizer with online LIME diagnostics + stagnation detection."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from initialization import initialization
from levy import levy
from lime_diagnostic import explain_selected_agents, select_agents_for_lime
from stagnation_detector import StagnationDetector, StagnationSnapshot


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


def estimate_max_fes(pop: int, max_iter: int) -> int:
    """Estimated FE budget based on SHO evaluation pattern."""
    return int(pop + max_iter * (pop + pop // 2))


def _build_no_stagnation_snapshot(fe: int, best_fitness: float, iteration: int) -> StagnationSnapshot:
    return StagnationSnapshot(
        fe=int(fe),
        best_fitness=float(best_fitness),
        last_improvement_fe=int(fe),
        sfes=0,
        min_sfes=0,
        stagnated=False,
        improved=False,
        event="none",
        iteration=int(iteration),
    )


def _fused_lime_dominance(lime_rows: list[dict]) -> tuple[str | None, float, dict[str, float]]:
    """Compute dominant feature score by fusing relative importances across LIME targets."""
    if not lime_rows:
        return None, 0.0, {}

    by_target_feature: dict[str, dict[str, float]] = {}
    for row in lime_rows:
        target_type = str(row.get("target_type", ""))
        feature = str(row.get("feature", ""))
        value = float(row.get("mean_abs_weight", 0.0))
        if target_type not in by_target_feature:
            by_target_feature[target_type] = {}
        by_target_feature[target_type][feature] = by_target_feature[target_type].get(feature, 0.0) + max(0.0, value)

    fused: dict[str, float] = {}
    channel_count: dict[str, int] = {}

    for feature_map in by_target_feature.values():
        total = sum(feature_map.values())
        if total <= 0.0:
            continue
        for feature, raw in feature_map.items():
            rel = raw / total
            fused[feature] = fused.get(feature, 0.0) + rel
            channel_count[feature] = channel_count.get(feature, 0) + 1

    if not fused:
        return None, 0.0, {}

    for feature in list(fused.keys()):
        fused[feature] = fused[feature] / float(channel_count[feature])

    dominant_feature = max(fused, key=fused.get)
    return dominant_feature, float(fused[dominant_feature]), fused


def _random_restart_vector(lb_vec: np.ndarray, ub_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fallback restart formula required by policy: LB + (UB-LB) * rand[0,1]."""
    return lb_vec + (ub_vec - lb_vec) * rng.random(lb_vec.shape[0])


def _mutate_with_feature(
    vector: np.ndarray,
    feature: str,
    elite: np.ndarray,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    span = ub_vec - lb_vec
    feature_name = str(feature)

    if feature_name.startswith("mag_levy"):
        candidate = vector + 0.03 * span * levy(1, vector.size, 1.5)[0]
    elif feature_name.startswith("mag_browniano"):
        candidate = vector + rng.normal(loc=0.0, scale=0.05, size=vector.size) * span
    elif feature_name.startswith("mag_predacion") or feature_name == "alpha":
        pull = rng.uniform(0.35, 0.85)
        candidate = (1.0 - pull) * vector + pull * elite + rng.normal(loc=0.0, scale=0.015, size=vector.size) * span
    elif feature_name in {"distance_to_elite", "delta_position_norm", "theta_mean", "theta_active", "r1", "r2"}:
        direction = vector - elite
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = rng.normal(loc=0.0, scale=1.0, size=vector.size)
            norm = float(np.linalg.norm(direction))
        direction = direction / (norm + 1e-12)
        step = rng.uniform(0.05, 0.25)
        candidate = vector + step * direction * span + rng.normal(loc=0.0, scale=0.01, size=vector.size) * span
    else:
        candidate = _random_restart_vector(lb_vec=lb_vec, ub_vec=ub_vec, rng=rng)

    return np.clip(candidate, lb_vec, ub_vec)


def _select_restart_indices(sea_horses_fitness: np.ndarray, restart_count: int) -> list[int]:
    pop = int(sea_horses_fitness.size)
    if pop <= 1 or restart_count <= 0:
        return []

    elite_idx = int(np.argmin(sea_horses_fitness))
    worst_first = list(np.argsort(sea_horses_fitness)[::-1])
    non_elite = [idx for idx in worst_first if idx != elite_idx]
    count = min(max(0, restart_count), len(non_elite))
    return [int(idx) for idx in non_elite[:count]]


def SHO_HYBRID(
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
    min_sfes_ratio: float = 0.04,
    max_fes: int = 0,
    warmup_fes: int = 0,
    restart_enabled: bool = True,
    restart_percent: float = 7.0,
    restart_cooldown_fes: int = 0,
    restart_dominance_threshold: float = 0.90,
    enable_lime: bool = True,
    enable_stagnation: bool = True,
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
    effective_max_fes = int(max_fes) if int(max_fes) > 0 else estimate_max_fes(pop=pop, max_iter=max_iter)
    warmup_fes_effective = int(warmup_fes) if int(warmup_fes) > 0 else max(1, int(math.ceil(effective_max_fes * 0.05)))
    restart_percent_effective = float(restart_percent)
    if restart_percent_effective < 5.0 or restart_percent_effective > 10.0:
        raise ValueError("restart_percent must be in [5, 10]")
    restart_cooldown_fes_effective = max(0, int(restart_cooldown_fes))
    restart_dominance_threshold_effective = float(restart_dominance_threshold)

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

    current_fe = int(getattr(fobj, "nfev", pop))
    detector: StagnationDetector | None = None
    if enable_stagnation:
        detector = StagnationDetector(max_fes=effective_max_fes, min_sfes_ratio=min_sfes_ratio)
        detector.initialize(best_fitness=target_fitness, fe=current_fe, iteration=0)

    _log(
        f"[{run_metadata['function_name']}] inicio | pop={pop} dim={dim} max_iter={max_iter} "
        f"lime={'si' if enable_lime else 'no'} mode={lime_selection_mode} trigger=stagnation_start_only "
        f"stagnation={'si' if enable_stagnation else 'no'} max_fes={effective_max_fes} "
        f"warmup_fes={warmup_fes_effective} restart={'si' if restart_enabled else 'no'} "
        f"restart_pct={restart_percent_effective:.1f} cooldown_fes={restart_cooldown_fes_effective} "
        f"best_inicial={target_fitness:.6e}"
    )

    lb_vec = _bounds_vector(lower_bound, dim)
    ub_vec = _bounds_vector(upper_bound, dim)

    u = 0.05
    v = 0.05
    l = 0.05

    t = 1
    diagnosis_id = 0
    restart_events_count = 0
    last_restart_fe = -10**18
    warmup_completed = False

    agent_samples: list[dict] = []
    full_output_rows: list[dict] = []
    contribution_rows: list[dict] = []
    stagnation_history_rows: list[dict] = []
    stagnation_event_rows: list[dict] = []

    def _append_stagnation_event(
        *,
        iteration: int,
        event: str,
        fe: int,
        sfes: int,
        min_sfes: int,
        last_improvement_fe: int,
        warmup_active: bool,
        restart_attempted: bool = False,
        restart_executed: bool = False,
        restart_blocked_by_cooldown: bool = False,
        restart_source: str = "none",
        restart_feature: str = "",
        restart_feature_score: float = 0.0,
        restart_count: int = 0,
        restart_cooldown_remaining_fes: int = 0,
        lime_triggered: bool = False,
        lime_trigger_source: str = "none",
    ) -> None:
        stagnation_event_rows.append(
            {
                "run_id": run_metadata["run_id"],
                "timestamp": run_metadata["timestamp"],
                "function_name": run_metadata["function_name"],
                "dimension": run_metadata["dimension"],
                "iteration": int(iteration),
                "event": str(event),
                "fe": int(fe),
                "sfes": int(sfes),
                "min_sfes": int(min_sfes),
                "last_improvement_fe": int(last_improvement_fe),
                "warmup_active": int(warmup_active),
                "warmup_fes": int(warmup_fes_effective),
                "restart_attempted": int(restart_attempted),
                "restart_executed": int(restart_executed),
                "restart_blocked_by_cooldown": int(restart_blocked_by_cooldown),
                "restart_source": str(restart_source),
                "restart_feature": str(restart_feature),
                "restart_feature_score": float(restart_feature_score),
                "restart_count": int(restart_count),
                "restart_cooldown_fes": int(restart_cooldown_fes_effective),
                "restart_cooldown_remaining_fes": int(restart_cooldown_remaining_fes),
                "lime_triggered": int(lime_triggered),
                "lime_trigger_source": str(lime_trigger_source),
            }
        )

    while t < max_iter + 1 and (not enable_stagnation or int(getattr(fobj, "nfev", 0)) < effective_max_fes):
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

        selection_result = select_agents_for_lime(
            fitness_new=sea_horses_fitness1,
            delta_f=delta_f,
            distance_to_elite=distance_to_elite,
            rng=rng,
        )

        lime_triggered = False
        lime_trigger_source = "none"
        lime_dataset_updated = False

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
        warmup_active = current_fe < warmup_fes_effective

        lime_triggered = False
        lime_trigger_source = "none"
        lime_dataset_updated = False
        lime_rows_current_trigger: list[dict] = []

        restart_attempted = False
        restart_executed = False
        restart_blocked_by_cooldown = False
        restart_count_this_iter = 0
        restart_source = "none"
        restart_feature = ""
        restart_feature_score = 0.0

        if enable_stagnation and detector is not None:
            snapshot = detector.update(best_fitness=target_fitness, fe=current_fe, iteration=t)
            if snapshot.event != "none":
                _append_stagnation_event(
                    iteration=t,
                    event=snapshot.event,
                    fe=snapshot.fe,
                    sfes=snapshot.sfes,
                    min_sfes=snapshot.min_sfes,
                    last_improvement_fe=snapshot.last_improvement_fe,
                    warmup_active=warmup_active,
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
        else:
            snapshot = _build_no_stagnation_snapshot(fe=current_fe, best_fitness=target_fitness, iteration=t)

        if (not warmup_completed) and (not warmup_active):
            warmup_completed = True
            _append_stagnation_event(
                iteration=t,
                event="warmup_completed",
                fe=snapshot.fe,
                sfes=snapshot.sfes,
                min_sfes=snapshot.min_sfes,
                last_improvement_fe=snapshot.last_improvement_fe,
                warmup_active=False,
            )

        if enable_lime and (not warmup_active) and ((not snapshot.stagnated) or snapshot.event == "stagnation_start"):
            agent_samples.extend(current_iteration_samples)
            lime_dataset_updated = True

        if (
            enable_lime
            and (not warmup_active)
            and snapshot.event == "stagnation_start"
            and len(agent_samples) >= min_samples_before_lime
        ):
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
                lime_rows_current_trigger = lime_rows
                lime_triggered = True
                lime_trigger_source = "stagnation_start"
                for row in lime_rows:
                    row["trigger_source"] = "stagnation_start"
                    row["trigger_event"] = snapshot.event
                    row["trigger_iteration"] = t
                contribution_rows.extend(lime_rows)

        if restart_enabled and (not warmup_active) and snapshot.event == "stagnation_start":
            restart_attempted = True
            if last_restart_fe <= -10**17:
                fe_since_last_restart = restart_cooldown_fes_effective
            else:
                fe_since_last_restart = snapshot.fe - int(last_restart_fe)
            cooldown_ok = fe_since_last_restart >= restart_cooldown_fes_effective

            if not cooldown_ok:
                restart_blocked_by_cooldown = True
            else:
                dominant_feature, dominant_score, _ = _fused_lime_dominance(lime_rows_current_trigger)
                restart_feature_score = float(dominant_score)
                if dominant_feature is not None:
                    restart_feature = str(dominant_feature)

                if dominant_feature is not None and dominant_score >= restart_dominance_threshold_effective:
                    restart_source = "lime_mutator"
                else:
                    restart_source = "random_fallback"

                restart_count_target = max(1, int(math.ceil(pop * (restart_percent_effective / 100.0))))
                restart_indices = _select_restart_indices(sea_horses_fitness=sea_horses_fitness, restart_count=restart_count_target)

                if restart_indices:
                    for agent_idx in restart_indices:
                        original = sea_horses[agent_idx, :].copy()
                        if restart_source == "lime_mutator":
                            mutated = _mutate_with_feature(
                                vector=original,
                                feature=restart_feature,
                                elite=target_position,
                                lb_vec=lb_vec,
                                ub_vec=ub_vec,
                                rng=rng,
                            )
                        else:
                            mutated = _random_restart_vector(lb_vec=lb_vec, ub_vec=ub_vec, rng=rng)

                        sea_horses[agent_idx, :] = np.clip(mutated, lb_vec, ub_vec)
                        sea_horses_fitness[agent_idx] = float(fobj(sea_horses[agent_idx, :]))

                    restart_count_this_iter = len(restart_indices)
                    restart_executed = restart_count_this_iter > 0
                    if restart_executed:
                        restart_events_count += 1
                        current_fe = int(getattr(fobj, "nfev", current_fe))
                        last_restart_fe = int(current_fe)

                        best_idx = int(np.argmin(sea_horses_fitness))
                        best_val = float(sea_horses_fitness[best_idx])
                        if best_val < target_fitness:
                            target_fitness = best_val
                            target_position = sea_horses[best_idx, :].copy()
                            convergence_curve[t - 1] = target_fitness

        if last_restart_fe <= -10**17:
            restart_cooldown_remaining_fes = 0
        else:
            restart_cooldown_remaining_fes = max(0, restart_cooldown_fes_effective - (current_fe - int(last_restart_fe)))

        if restart_attempted and restart_executed:
            _append_stagnation_event(
                iteration=t,
                event="restart_executed",
                fe=current_fe,
                sfes=snapshot.sfes,
                min_sfes=snapshot.min_sfes,
                last_improvement_fe=snapshot.last_improvement_fe,
                warmup_active=warmup_active,
                restart_attempted=True,
                restart_executed=True,
                restart_source=restart_source,
                restart_feature=restart_feature,
                restart_feature_score=restart_feature_score,
                restart_count=restart_count_this_iter,
                restart_cooldown_remaining_fes=restart_cooldown_remaining_fes,
                lime_triggered=lime_triggered,
                lime_trigger_source=lime_trigger_source,
            )
        elif restart_attempted and restart_blocked_by_cooldown:
            _append_stagnation_event(
                iteration=t,
                event="restart_blocked_cooldown",
                fe=snapshot.fe,
                sfes=snapshot.sfes,
                min_sfes=snapshot.min_sfes,
                last_improvement_fe=snapshot.last_improvement_fe,
                warmup_active=warmup_active,
                restart_attempted=True,
                restart_blocked_by_cooldown=True,
                restart_source="none",
                restart_feature="",
                restart_feature_score=0.0,
                restart_count=0,
                restart_cooldown_remaining_fes=restart_cooldown_remaining_fes,
                lime_triggered=lime_triggered,
                lime_trigger_source=lime_trigger_source,
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
                "selected_agents": len(selection_result.selected_unique),
                "elite_selected": len(selection_result.category_indices.get("elite_high_impact", [])),
                "diverse_selected": len(selection_result.category_indices.get("diverse", [])),
                "outliers_selected": len(selection_result.category_indices.get("outliers", [])),
                "random_selected": len(selection_result.category_indices.get("random", [])),
                "lime_selection_mode": lime_selection_mode,
                "lime_triggered": int(lime_triggered),
                "lime_trigger_source": lime_trigger_source,
                "diagnosis_id": diagnosis_id if lime_triggered else 0,
                "lime_dataset_updated": int(lime_dataset_updated),
                "lime_buffer_size": len(agent_samples),
                "warmup_active": int(warmup_active),
                "warmup_fes": int(warmup_fes_effective),
                "warmup_progress": float(min(1.0, current_fe / float(max(1, warmup_fes_effective)))),
                "improved_agents": improved_count,
                "fe": int(current_fe),
                "max_fes": effective_max_fes if enable_stagnation else 0,
                "last_improvement_fe": snapshot.last_improvement_fe,
                "sfes": snapshot.sfes,
                "min_sfes": snapshot.min_sfes,
                "stagnated": int(snapshot.stagnated),
                "event": snapshot.event,
                "restart_enabled": int(restart_enabled),
                "restart_attempted": int(restart_attempted),
                "restart_executed": int(restart_executed),
                "restart_blocked_by_cooldown": int(restart_blocked_by_cooldown),
                "restart_count": int(restart_count_this_iter),
                "restart_percent": float(restart_percent_effective),
                "restart_source": restart_source,
                "restart_feature": restart_feature,
                "restart_feature_score": float(restart_feature_score),
                "restart_dominance_threshold": float(restart_dominance_threshold_effective),
                "restart_cooldown_fes": int(restart_cooldown_fes_effective),
                "restart_cooldown_remaining_fes": int(restart_cooldown_remaining_fes),
                "lime_enabled": int(enable_lime),
                "stagnation_enabled": int(enable_stagnation),
            }
        )

        should_log = (
            t == 1
            or t == max_iter
            or (t % progress_every == 0)
            or lime_triggered
            or (snapshot.event != "none")
        )
        if should_log:
            elapsed = time.perf_counter() - iteration_start
            lime_status = "si" if lime_triggered else "no"
            diagnosis_text = f" id={diagnosis_id}" if lime_triggered else ""
            restart_status = "si" if restart_executed else "no"
            _log(
                f"[{run_metadata['function_name']}] iter {t}/{max_iter} | "
                f"best={target_fitness:.6e} avg={float(np.mean(sea_horses_fitness1)):.6e} "
                f"improved={improved_count}/{pop} selected={len(selection_result.selected_unique)} "
                f"lime={lime_status}{diagnosis_text} source={lime_trigger_source} mode={lime_selection_mode} "
                f"warmup={'si' if warmup_active else 'no'} "
                f"dataset={len(agent_samples)} updated={'si' if lime_dataset_updated else 'no'} "
                f"fe={int(current_fe)}/{effective_max_fes if enable_stagnation else 0} "
                f"sfes={snapshot.sfes}/{snapshot.min_sfes} "
                f"restart={restart_status} source={restart_source} count={restart_count_this_iter} "
                f"cooldown_rem={restart_cooldown_remaining_fes} "
                f"stagnated={'si' if snapshot.stagnated else 'no'} dt={elapsed:.2f}s"
            )

        t += 1

    final_fe = int(getattr(fobj, "nfev", current_fe))
    stop_reason = "max_fes" if enable_stagnation and final_fe >= effective_max_fes else "max_iter"

    _log(
        f"[{run_metadata['function_name']}] fin | best={target_fitness:.6e} "
        f"diagnosticos={diagnosis_id} muestras={len(agent_samples)} "
        f"stagnation_events={len(stagnation_event_rows)} fe={final_fe} stop={stop_reason}"
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
        "stagnation_history_rows": stagnation_history_rows,
        "stagnation_event_rows": stagnation_event_rows,
        "stagnation_meta": {
            "max_fes": int(effective_max_fes if enable_stagnation else 0),
            "min_sfes_ratio": float(min_sfes_ratio if enable_stagnation else 0.0),
            "min_sfes": int(detector.min_sfes if enable_stagnation and detector is not None else 0),
            "warmup_fes": int(warmup_fes_effective),
            "warmup_completed": int(warmup_completed),
            "restart_enabled": int(restart_enabled),
            "restart_percent": float(restart_percent_effective),
            "restart_cooldown_fes": int(restart_cooldown_fes_effective),
            "restart_dominance_threshold": float(restart_dominance_threshold_effective),
            "restart_events": int(restart_events_count),
            "final_fe": int(final_fe),
            "stop_reason": stop_reason,
            "enabled": int(enable_stagnation),
        },
    }
