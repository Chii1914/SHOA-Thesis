from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner local para benchmark TMLAP con SHOA+LIME")
    parser.add_argument("--mode", choices=["smoke", "full", "profiles"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Argumentos extra para el comando seleccionado")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    benchmark = root / "benchmarks" / "run_tmlap_sholime_benchmark.py"
    profiles = root / "benchmarks" / "run_tmlap_sholime_profiles_matrix.py"

    if args.mode == "smoke":
        cmd = [
            sys.executable,
            str(benchmark),
            "--instances",
            "1.instancia_simple.txt",
            "--runs",
            "2",
            "--seed-start",
            "1",
            "--pop-size",
            "20",
            "--max-iter",
            "60",
            "--lime-samples",
            "400",
            "--output-dir",
            "results/benchmark_logs",
            "--tag",
            "smoke",
        ]
    elif args.mode == "full":
        cmd = [
            sys.executable,
            str(benchmark),
            "--instances",
            "all",
            "--runs",
            "30",
            "--seed-start",
            "1",
            "--pop-size",
            "30",
            "--max-iter",
            "500",
            "--output-dir",
            "results/benchmark_logs",
            "--tag",
            "full",
        ]
    else:
        cmd = [
            sys.executable,
            str(profiles),
            "--profiles",
            "all",
        ]

    cmd.extend(args.extra)

    print("Command:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
