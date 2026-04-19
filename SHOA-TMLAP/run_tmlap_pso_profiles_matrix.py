"""Run a profile matrix for PSO benchmark on TMLAP instances."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ARG_ORDER = [
    "instances",
    "instance_dir",
    "runs",
    "seed_start",
    "pop_size",
    "max_iter",
    "inertia",
    "cognitive",
    "social",
    "v_max_frac",
    "output_dir",
    "tag",
]


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _append_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        cmd.append(_flag(key))
        cmd.append(str(value).lower())
        return

    if isinstance(value, (list, tuple)):
        cmd.append(_flag(key))
        cmd.append(",".join(str(item) for item in value))
        return

    cmd.append(_flag(key))
    cmd.append(str(value))


def _build_command(benchmark_script: Path, merged_args: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, str(benchmark_script)]

    for key in ARG_ORDER:
        if key in merged_args:
            _append_arg(cmd, key, merged_args[key])

    for key in sorted(merged_args.keys()):
        if key not in ARG_ORDER:
            _append_arg(cmd, key, merged_args[key])

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lanza benchmark PSO en TMLAP para perfiles soft/medium/hard"
    )
    parser.add_argument(
        "--config",
        default="pso_tmlap_profiles_config.json",
        help="Ruta del archivo de configuracion JSON",
    )
    parser.add_argument(
        "--benchmark-script",
        default="run_tmlap_pso_benchmark.py",
        help="Ruta del benchmark principal",
    )
    parser.add_argument(
        "--profiles",
        default="all",
        help="Perfiles separados por coma o 'all' (ej: soft,hard)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime comandos, no ejecuta",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = (script_dir / args.config).resolve()
    benchmark_script = (script_dir / args.benchmark_script).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"No existe config: {config_path}")
    if not benchmark_script.exists():
        raise FileNotFoundError(f"No existe benchmark script: {benchmark_script}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    base_args: dict[str, Any] = dict(config.get("base_args", {}))
    profiles_cfg: dict[str, dict[str, Any]] = dict(config.get("profiles", {}))

    if not profiles_cfg:
        raise ValueError("El config debe incluir al menos un perfil en 'profiles'")

    all_profiles = list(profiles_cfg.keys())
    selected_profiles = all_profiles if args.profiles == "all" else _parse_csv(args.profiles)

    invalid_profiles = [name for name in selected_profiles if name not in profiles_cfg]
    if invalid_profiles:
        allowed = ", ".join(all_profiles)
        bad = ", ".join(invalid_profiles)
        raise ValueError(f"Perfiles invalidos: {bad}. Disponibles: {allowed}")

    jobs: list[tuple[str, dict[str, Any]]] = []
    base_tag = str(base_args.get("tag", "")).strip()

    for profile_name in selected_profiles:
        merged = {**base_args, **dict(profiles_cfg[profile_name])}
        merged["tag"] = f"{base_tag}_{profile_name}" if base_tag else profile_name
        jobs.append((profile_name, merged))

    print(f"Config: {config_path}")
    print(f"Benchmark script: {benchmark_script}")
    print(f"Total jobs: {len(jobs)}")

    for idx, (profile_name, merged_args) in enumerate(jobs, start=1):
        cmd = _build_command(benchmark_script, merged_args)
        print(f"\n[{idx}/{len(jobs)}] profile={profile_name}")
        print(shlex.join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)

    print("\nProfile matrix finished.")


if __name__ == "__main__":
    main()
