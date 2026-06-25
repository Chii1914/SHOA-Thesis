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


def SHO(pop: int, Max_iter: int, LB, UB, Dim: int, fobj, batch_eval=None):
    sea_horses = initialization(pop, Dim, UB, LB)

    sea_horses_fitness = np.zeros(pop, dtype=float)
    fitness_history = np.zeros((pop, Max_iter), dtype=float)
    population_history = np.zeros((pop, Dim, Max_iter), dtype=float)
    convergence_curve = np.zeros(Max_iter, dtype=float)
    trajectories = np.zeros((pop, Max_iter), dtype=float)

    if batch_eval is not None:
        sea_horses_fitness = batch_eval(sea_horses)
        fobj.nfev = getattr(fobj, "nfev", 0) + pop
        fitness_history[:, 0] = sea_horses_fitness
        population_history[:, :, 0] = sea_horses
    else:
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

        # Motor behavior (vectorised — pre-generates random matrices for both branches)
        r1 = np.random.randn(pop)
        step_length = levy(pop, Dim, 1.5)
        rand_theta = np.random.rand(pop, Dim)
        theta      = rand_theta * (2.0 * np.pi)
        row        = u * np.exp(theta * v)
        levy_branch  = sea_horses + step_length * (
            (elite - sea_horses) * row * np.cos(theta) * np.sin(theta) * theta + elite
        )
        rand_brown = np.random.rand(pop, Dim)
        brown_branch = sea_horses + rand_brown * l * beta * (
            sea_horses - beta * elite
        )
        sea_horses_new1 = np.where(r1[:, None] > 0, levy_branch, brown_branch)
        sea_horses_new1 = np.clip(sea_horses_new1, lb_vec, ub_vec)

        # Predation behavior (vectorised)
        r2    = np.random.rand(pop)
        alpha = (1 - t / Max_iter) ** (2 * t / Max_iter)
        rA    = np.random.rand(pop, Dim)
        rB    = np.random.rand(pop, Dim)
        brA   = alpha * (elite - rA * sea_horses_new1) + (1 - alpha) * elite
        brB   = (1 - alpha) * (sea_horses_new1 - rB * elite) + alpha * sea_horses_new1
        sea_horses_new2 = np.where(r2[:, None] >= 0.1, brA, brB)
        sea_horses_new2 = np.clip(sea_horses_new2, lb_vec, ub_vec)
        if batch_eval is not None:
            sea_horses_fitness1 = batch_eval(sea_horses_new2)
            fobj.nfev = getattr(fobj, "nfev", 0) + pop
        else:
            sea_horses_fitness1 = np.array([fobj(ind) for ind in sea_horses_new2], dtype=float)

        # Reproductive behavior
        index = np.argsort(sea_horses_fitness1)
        half = pop // 2
        sea_horses_father = sea_horses_new2[index[:half], :]
        sea_horses_mother = sea_horses_new2[index[half:pop], :]

        r3 = np.random.rand(half, 1)
        si = r3 * sea_horses_father + (1 - r3) * sea_horses_mother

        sea_horses_offspring = np.clip(si, lb_vec, ub_vec)
        if batch_eval is not None:
            sea_horses_fitness2 = batch_eval(sea_horses_offspring)
            fobj.nfev = getattr(fobj, "nfev", 0) + half
        else:
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
