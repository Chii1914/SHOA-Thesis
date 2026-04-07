"""Main entry point for running SHO + LIME on a selected Opfunu CEC function."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from SHO_LIME_Controller import SHOXAIConfig, SHO_with_lime_controller, build_opfunu_cec_objective


def main() -> None:
    # Fine-tuned single instance: keep high sensitivity while improving late-stage escape quality.
    popsize = 30  # Slightly higher diversity without losing plateau-prone behavior
    max_iter = 500  # Extra budget to exploit after rescues
    f_name = "F42022"  # CEC class name in Opfunu (e.g. F12022, F22022, ... , F122022)
    dim_req = 10  # Desired dimensionality for the chosen CEC function


    lb, ub, dim, objective, problem = build_opfunu_cec_objective(f_name, dim_req)

    config = SHOXAIConfig(
        pop_size=popsize,
        max_iter=max_iter,
        window_size=15,
        epsilon_stagnation=2e-2,
        cooldown_iters=14,
        lime_num_samples=1600,
        importance_threshold=0.05,
        delta_tolerance=2e-5,
        fidelity_threshold=0.15,
        rescue_mode="levy_teleport",
        rescue_levy_scale=0.14,
        seed=7,
    )

    start = time.time()
    result = SHO_with_lime_controller(objective, lb, ub, dim, config)
    elapsed = time.time() - start

    plt.semilogy(np.arange(1, max_iter + 1), result.convergence_curve, color="r", linewidth=2.5)
    plt.title("Convergence curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best score obtained so far")
    plt.show()

    positive_diag_iters = [
        diag["iteration"] for diag in result.diagnostics_log if diag.get("status") == "POSITIVE_STAGNATION"
    ]

    print(f"The running time is: {elapsed}")
    print(f"Selected CEC function: {f_name}")
    print(f"CEC display name: {problem.name}")
    print(f"Dimension used: {dim}")
    print(f"The best solution obtained by SHO+LIME is : {result.best_fitness}")
    print(f"The best optimal sea horse of the objective function found by SHO+LIME is : {result.best_position}")
    print(f"Diagnostics invoked: {len(result.diagnostics_log)}")
    print(f"Diagnostics invocation iterations: {result.diagnostics_invocation_iterations}")
    print(f"Rescues applied: {result.rescue_count}")
    print(f"Positive diagnosis iterations: {positive_diag_iters}")


if __name__ == "__main__":
    main()
