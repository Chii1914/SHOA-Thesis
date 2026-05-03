from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner local para benchmark CEC2022 con SHOA+LIME")
    parser.add_argument("--mode", choices=["smoke", "full", "profiles"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Argumentos extra para el comando seleccionado")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    benchmark = root / "benchmarks" / "run_cec2022_benchmark.py"
    profiles = root / "benchmarks" / "run_sholime_profiles_matrix.py"

    if args.mode == "smoke":
        cmd = [
            sys.executable,
            str(benchmark),
            "--functions",
            "F12022,F22022",
            "--dims",
            "10",
            "--runs",
            "2",
            "--seed-start",
            "1",
            "--pop-size",
            "30",
            "--max-iter",
            "40",
            "--lime-samples",
            "400",
            "--output-dir",
            "benchmark_logs",
            "--tag",
            "smoke",
        ]
    elif args.mode == "full":
        cmd = [
            sys.executable,
            str(benchmark),
            "--functions",
            "all",
            "--dims",
            "10",
            "--runs",
            "30",
            "--seed-start",
            "1",
            "--pop-size",
            "30",
            "--max-iter",
            "500",
            "--output-dir",
            "benchmark_logs",
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
