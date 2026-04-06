import argparse
import csv
import importlib.util
import io
import random
import statistics
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path


def load_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el spec del modulo: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pso(pso_module, seed: int, max_iter: int, population: int, verbose_solver: bool):
    random.seed(seed)
    solver = pso_module.PSO()
    solver.max_iter = max_iter
    solver.n_particles = population

    if verbose_solver:
        solver.solve()
    else:
        with redirect_stdout(io.StringIO()):
            solver.solve()

    return float(solver.g.fitness_p_best()), list(solver.g.p_best)


def run_shoa(shoa_module, seed: int, max_iter: int, population: int, verbose_solver: bool):
    solver = shoa_module.SHOA_TMLAP()
    solver.seed = seed
    solver.max_iter = max_iter
    solver.n_agents = population

    if verbose_solver:
        solver.solve()
    else:
        with redirect_stdout(io.StringIO()):
            solver.solve()

    return float(solver.best_fitness), list(solver.best_assignment)


def compute_stats(values):
    if not values:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan"), "stdev": float("nan")}

    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compara PSO vs SHOA en TMLAP con mismas semillas")
    parser.add_argument("--runs", type=int, default=30, help="Numero de corridas (default: 30)")
    parser.add_argument("--seed-start", type=int, default=1, help="Semilla inicial")
    parser.add_argument("--max-iter", type=int, default=25, help="Iteraciones por corrida para ambos solvers")
    parser.add_argument("--population", type=int, default=10, help="Tamano de poblacion para ambos solvers")
    parser.add_argument("--show-each", action="store_true", help="Mostrar resultado por semilla")
    parser.add_argument("--verbose-solvers", action="store_true", help="No suprimir trazas internas de PSO/SHOA")
    parser.add_argument(
        "--output-dir",
        default="SHOA-TMLAP/results",
        help="Carpeta donde se guardan CSV y resumen",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs debe ser mayor que 0")

    project_root = Path(__file__).resolve().parent.parent
    pso_path = project_root / "PSO-TMLAP" / "PSO-TMLAP.py"
    shoa_path = project_root / "SHOA-TMLAP" / "SHOA-TMLAP.py"

    pso_module = load_module("pso_tmlap_module", pso_path)
    shoa_module = load_module("shoa_tmlap_module", shoa_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"compare_runs{args.runs}_iter{args.max_iter}_pop{args.population}_{timestamp}"
    out_dir = project_root / args.output_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pso_values = []
    shoa_values = []
    pso_wins = 0
    shoa_wins = 0
    ties = 0

    for run_id in range(1, args.runs + 1):
        seed = args.seed_start + run_id - 1

        pso_fit, pso_assign = run_pso(
            pso_module,
            seed=seed,
            max_iter=args.max_iter,
            population=args.population,
            verbose_solver=args.verbose_solvers,
        )
        shoa_fit, shoa_assign = run_shoa(
            shoa_module,
            seed=seed,
            max_iter=args.max_iter,
            population=args.population,
            verbose_solver=args.verbose_solvers,
        )

        if shoa_fit < pso_fit - 1e-12:
            winner = "SHOA"
            shoa_wins += 1
        elif pso_fit < shoa_fit - 1e-12:
            winner = "PSO"
            pso_wins += 1
        else:
            winner = "TIE"
            ties += 1

        row = {
            "run_id": run_id,
            "seed": seed,
            "pso_fitness": pso_fit,
            "shoa_fitness": shoa_fit,
            "shoa_minus_pso": shoa_fit - pso_fit,
            "winner": winner,
            "pso_assignment": " ".join(str(v) for v in pso_assign),
            "shoa_assignment": " ".join(str(v) for v in shoa_assign),
        }
        rows.append(row)
        pso_values.append(pso_fit)
        shoa_values.append(shoa_fit)

        if args.show_each:
            print(
                f"run={run_id:03d} seed={seed} "
                f"PSO={pso_fit:.4f} SHOA={shoa_fit:.4f} winner={winner}"
            )

    pso_stats = compute_stats(pso_values)
    shoa_stats = compute_stats(shoa_values)

    csv_path = out_dir / "per_seed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "seed",
                "pso_fitness",
                "shoa_fitness",
                "shoa_minus_pso",
                "winner",
                "pso_assignment",
                "shoa_assignment",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Runs: {args.runs}\n")
        f.write(f"Seed range: {args.seed_start}..{args.seed_start + args.runs - 1}\n")
        f.write(f"Max iterations: {args.max_iter}\n")
        f.write(f"Population: {args.population}\n\n")

        f.write("PSO stats:\n")
        f.write(f"  min: {pso_stats['min']:.6f}\n")
        f.write(f"  max: {pso_stats['max']:.6f}\n")
        f.write(f"  mean: {pso_stats['mean']:.6f}\n")
        f.write(f"  median: {pso_stats['median']:.6f}\n")
        f.write(f"  stdev: {pso_stats['stdev']:.6f}\n\n")

        f.write("SHOA stats:\n")
        f.write(f"  min: {shoa_stats['min']:.6f}\n")
        f.write(f"  max: {shoa_stats['max']:.6f}\n")
        f.write(f"  mean: {shoa_stats['mean']:.6f}\n")
        f.write(f"  median: {shoa_stats['median']:.6f}\n")
        f.write(f"  stdev: {shoa_stats['stdev']:.6f}\n\n")

        f.write("Wins:\n")
        f.write(f"  PSO: {pso_wins}\n")
        f.write(f"  SHOA: {shoa_wins}\n")
        f.write(f"  Ties: {ties}\n")

    print("Comparacion completada.")
    print(f"Resultados por semilla: {csv_path}")
    print(f"Resumen: {summary_path}")
    print(
        f"Wins -> PSO: {pso_wins}, SHOA: {shoa_wins}, Ties: {ties} | "
        f"mean(PSO)={pso_stats['mean']:.4f}, mean(SHOA)={shoa_stats['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
