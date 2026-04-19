"""Sea-Horse Optimizer (SHO) core implementation translated from MATLAB."""

from __future__ import annotations

import numpy as np

from initialization import initialization
from levy import levy


def _bounds_vector(bounds, dim: int) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(dim, arr.item(), dtype=float)
    return arr


def SHO(pop: int, Max_iter: int, LB, UB, Dim: int, fobj):
    sea_horses = initialization(pop, Dim, UB, LB)

    sea_horses_fitness = np.zeros(pop, dtype=float)
    fitness_history = np.zeros((pop, Max_iter), dtype=float)
    population_history = np.zeros((pop, Dim, Max_iter), dtype=float)
    convergence_curve = np.zeros(Max_iter, dtype=float)
    trajectories = np.zeros((pop, Max_iter), dtype=float)

    for i in range(pop):
        sea_horses_fitness[i] = fobj(sea_horses[i, :])
        fitness_history[i, 0] = sea_horses_fitness[i]
        population_history[i, :, 0] = sea_horses[i, :]
    trajectories[:, 0] = sea_horses[:, 0]

    sorted_indexes = np.argsort(sea_horses_fitness)
    target_position = sea_horses[sorted_indexes[0], :].copy()
    target_fitness = sea_horses_fitness[sorted_indexes[0]]
    convergence_curve[0] = target_fitness

    lb_vec = _bounds_vector(LB, Dim)
    ub_vec = _bounds_vector(UB, Dim)

    t = 1
    u = 0.05
    v = 0.05
    l = 0.05

    while t < Max_iter + 1:
        beta = np.random.randn(pop, Dim)
        elite = np.tile(target_position, (pop, 1))

        # Motor behavior
        r1 = np.random.randn(pop)
        step_length = levy(pop, Dim, 1.5)
        sea_horses_new1 = np.zeros_like(sea_horses)

        for i in range(pop):
            for j in range(Dim):
                if r1[i] > 0:
                    r = np.random.rand()
                    theta = r * 2 * np.pi
                    row = u * np.exp(theta * v)
                    x = row * np.cos(theta)
                    y = row * np.sin(theta)
                    z = row * theta
                    sea_horses_new1[i, j] = sea_horses[i, j] + step_length[i, j] * (
                        (elite[i, j] - sea_horses[i, j]) * x * y * z + elite[i, j]
                    )
                else:
                    sea_horses_new1[i, j] = sea_horses[i, j] + np.random.rand() * l * beta[i, j] * (
                        sea_horses[i, j] - beta[i, j] * elite[i, j]
                    )

        sea_horses_new1 = np.clip(sea_horses_new1, lb_vec, ub_vec)

        # Predation behavior
        sea_horses_new2 = np.zeros_like(sea_horses)
        r2 = np.random.rand(pop)
        alpha = (1 - t / Max_iter) ** (2 * t / Max_iter)

        for i in range(pop):
            for j in range(Dim):
                if r2[i] >= 0.1:
                    sea_horses_new2[i, j] = alpha * (elite[i, j] - np.random.rand() * sea_horses_new1[i, j]) + (1 - alpha) * elite[i, j]
                else:
                    sea_horses_new2[i, j] = (1 - alpha) * (sea_horses_new1[i, j] - np.random.rand() * elite[i, j]) + alpha * sea_horses_new1[i, j]

        sea_horses_new2 = np.clip(sea_horses_new2, lb_vec, ub_vec)
        sea_horses_fitness1 = np.array([fobj(ind) for ind in sea_horses_new2], dtype=float)

        # Reproductive behavior
        index = np.argsort(sea_horses_fitness1)
        half = pop // 2
        sea_horses_father = sea_horses_new2[index[:half], :]
        sea_horses_mother = sea_horses_new2[index[half:pop], :]

        si = np.zeros((half, Dim), dtype=float)
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
        fitness_history[:, t - 1] = sortfitbestn
        population_history[:, :, t - 1] = sea_horses
        trajectories[:, t - 1] = sea_horses[:, 0]

        if sortfitbestn[0] < target_fitness:
            target_position = sea_horses[0, :].copy()
            target_fitness = sortfitbestn[0]

        convergence_curve[t - 1] = target_fitness
        t += 1

    return target_fitness, target_position, convergence_curve, trajectories, fitness_history, population_history
