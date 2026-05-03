"""Main entry point for running the Sea-Horse Optimizer benchmark demo."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from BenchmarkFunctions import BenchmarkFunctions
from SHO import SHO


def main() -> None:
    popsize = 30  # Number of search agents
    max_iter = 500  # Maximum iteration
    f_name = "F1"  # Name of the benchmark function (F1 to F23)

    lb, ub, dim, fobj = BenchmarkFunctions(f_name)

    start = time.time()
    objective_fitness, objective_position, convergence_curve, trajectories, fitness_history, population_history = SHO(
        popsize, max_iter, lb, ub, dim, fobj
    )
    elapsed = time.time() - start

    plt.semilogy(np.arange(1, max_iter + 1), convergence_curve, color="r", linewidth=2.5)
    plt.title("Convergence curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best score obtained so far")
    plt.show()

    print(f"The running time is: {elapsed}")
    print(f"The best solution obtained by SHO is : {objective_fitness}")
    print(f"The best optimal sea horse of the objective function found by SHO is : {objective_position}")

    # Keep variables available for parity with MATLAB main script behavior.
    _ = trajectories, fitness_history, population_history


if __name__ == "__main__":
    main()
