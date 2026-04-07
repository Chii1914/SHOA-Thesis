"""Main entry point for SHO + LIME stagnation controller with Opfunu CEC."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from SHO_LIME_Controller import SHOXAIConfig, SHO_with_lime_controller, build_opfunu_cec_objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHO with LIME-based stagnation control on Opfunu CEC")
    parser.add_argument("--function", default="F12022", help="Opfunu class name, e.g. F12022")
    parser.add_argument("--ndim", type=int, default=10, help="Problem dimension")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--window-size", type=int, default=15, help="Sliding window size")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Trigger threshold for std(window)")
    parser.add_argument("--cooldown", type=int, default=20, help="Cooldown iterations after rescue")
    parser.add_argument("--lime-samples", type=int, default=1500, help="LIME perturbation samples")
    parser.add_argument("--importance-threshold", type=float, default=0.15, help="Weight threshold for stochastic features")
    parser.add_argument("--delta-tolerance", type=float, default=1e-8, help="Low expected improvement threshold")
    parser.add_argument("--fidelity-threshold", type=float, default=0.4, help="Minimum local fidelity for positive diagnosis")
    parser.add_argument("--rescue-mode", choices=["levy_teleport", "leader_repulsion"], default="levy_teleport")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Disable convergence plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lb, ub, dim, objective, problem = build_opfunu_cec_objective(args.function, args.ndim)

    cfg = SHOXAIConfig(
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        window_size=args.window_size,
        epsilon_stagnation=args.epsilon,
        cooldown_iters=args.cooldown,
        lime_num_samples=args.lime_samples,
        importance_threshold=args.importance_threshold,
        delta_tolerance=args.delta_tolerance,
        fidelity_threshold=args.fidelity_threshold,
        rescue_mode=args.rescue_mode,
        seed=args.seed,
    )

    start = time.time()
    result = SHO_with_lime_controller(objective, lb, ub, dim, cfg)
    elapsed = time.time() - start

    positive_iters = [
        diag["iteration"] for diag in result.diagnostics_log if diag.get("status") == "POSITIVE_STAGNATION"
    ]

    print(f"Function: {args.function}")
    print(f"Problem name: {problem.name}")
    print(f"Dimension used: {dim}")
    print(f"Runtime (s): {elapsed:.4f}")
    print(f"Best fitness: {result.best_fitness}")
    print(f"Best position: {result.best_position}")
    print(f"Diagnostics invoked: {len(result.diagnostics_log)}")
    print(f"Rescues applied: {result.rescue_count}")
    print(f"Positive diagnosis iterations: {positive_iters}")

    if result.diagnostics_log:
        print("Last diagnosis:")
        print(result.diagnostics_log[-1])

    if not args.no_plot:
        x_axis = np.arange(1, cfg.max_iter + 1)
        plt.figure(figsize=(10, 6))
        plt.semilogy(x_axis, result.convergence_curve, color="tab:blue", linewidth=2.0, label="Best-so-far")

        for idx, iteration in enumerate(positive_iters):
            label = "Rescue trigger" if idx == 0 else None
            plt.axvline(iteration, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.7, label=label)

        plt.title(f"SHO + LIME convergence ({args.function}, dim={dim})")
        plt.xlabel("Iteration")
        plt.ylabel("Best score obtained so far")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
