"""Organize TMLAP benchmark outputs and run analyses by folder/subfolder.

This script discovers benchmark run folders recursively, groups them by algorithm/profile,
creates an organized directory structure, and launches per-group analyses with
analyze_tmlap_benchmark_logs.py.

Organized structure:
- <organized_root>/<algorithm>/<profile>/<run_name>  (symlink or copy)

Analysis outputs:
- <analysis_root>/<algorithm>/<profile>/analysis_<timestamp>/...
- <analysis_root>/<algorithm>/_all_profiles/analysis_<timestamp>/...
- <analysis_root>/all_algorithms/analysis_<timestamp>/...
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    run_dir: Path
    algorithm: str
    profile: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ordena outputs TMLAP y ejecuta analisis por carpeta/subcarpeta"
    )
    parser.add_argument(
        "--benchmark-root",
        default="results/benchmark_logs",
        help="Carpeta raiz donde buscar corridas (busqueda recursiva)",
    )
    parser.add_argument(
        "--organized-root",
        default="results/organized_outputs",
        help="Carpeta donde se crean enlaces/copias organizadas por algoritmo/perfil",
    )
    parser.add_argument(
        "--analysis-root",
        default="results/analysis_by_group",
        help="Carpeta raiz de analisis por grupo",
    )
    parser.add_argument(
        "--analyzer-script",
        default="analyze_tmlap_benchmark_logs.py",
        help="Script analizador a ejecutar",
    )
    parser.add_argument(
        "--organize-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Modo para ordenar corridas: symlink (default) o copy",
    )
    parser.add_argument(
        "--no-show-fliers",
        action="store_true",
        help="Pasa la bandera --no-show-fliers al analizador",
    )
    parser.add_argument(
        "--skip-organize",
        action="store_true",
        help="No reordenar corridas; solo ejecutar analisis sobre organized-root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime acciones/comandos sin ejecutarlos",
    )
    return parser.parse_args()


def _resolve_path(script_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (script_dir / path).resolve()
    return path


def _infer_profile(run_name: str) -> str:
    lname = run_name.lower()
    for profile in ("soft", "medium", "hard"):
        if lname.endswith(f"_{profile}"):
            return profile
    return "unknown"


def _infer_algorithm(run_name: str) -> str:
    lname = run_name.lower()
    if lname.startswith("tmlap_sholime_"):
        return "sholime"
    if lname.startswith("tmlap_pso_"):
        return "pso"
    if lname.startswith("tmlap_shoa_puro_"):
        return "shoa_puro"
    return "unknown"


def _is_valid_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "full_output.csv").exists() and (path / "runs_raw.csv").exists()


def _discover_runs_recursively(benchmark_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    seen: set[Path] = set()

    for full_output in benchmark_root.rglob("full_output.csv"):
        run_dir = full_output.parent
        if run_dir in seen:
            continue
        if not _is_valid_run_dir(run_dir):
            continue

        # Avoid re-processing analysis outputs that may contain copied CSV files.
        parts = {part.lower() for part in run_dir.parts}
        if any(part.startswith("analysis_") for part in parts):
            continue

        seen.add(run_dir)
        run_name = run_dir.name
        runs.append(
            RunInfo(
                run_name=run_name,
                run_dir=run_dir,
                algorithm=_infer_algorithm(run_name),
                profile=_infer_profile(run_name),
            )
        )

    runs.sort(key=lambda item: (item.algorithm, item.profile, item.run_name))
    return runs


def _ensure_link_or_copy(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and mode == "symlink":
            target = dst.resolve(strict=False)
            if target == src.resolve():
                return
        if dst.is_dir() and mode == "copy":
            # Existing copied run; keep as-is to avoid expensive recopies.
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)

    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def _write_manifest(manifest_path: Path, rows: list[dict], dry_run: bool) -> None:
    if dry_run:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_name", "algorithm", "profile", "source_run_dir", "organized_run_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _collect_runs_from_leaf(leaf_dir: Path) -> list[Path]:
    if not leaf_dir.exists() or not leaf_dir.is_dir():
        return []

    run_dirs = []
    for candidate in sorted(leaf_dir.iterdir()):
        if _is_valid_run_dir(candidate):
            run_dirs.append(candidate)
    return run_dirs


def _run_analysis(
    analyzer_script: Path,
    benchmark_root: Path,
    output_dir: Path,
    no_show_fliers: bool,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        str(analyzer_script),
        "--benchmark-root",
        str(benchmark_root),
        "--runs",
        "all",
        "--output-dir",
        str(output_dir),
    ]
    if no_show_fliers:
        cmd.append("--no-show-fliers")

    print(" ".join(cmd))
    if dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def _prepare_stage_dir(stage_dir: Path, run_dirs: list[Path], dry_run: bool) -> None:
    if dry_run:
        return

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        link_path = stage_dir / run_dir.name
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(run_dir, target_is_directory=True)


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    benchmark_root = _resolve_path(script_dir, args.benchmark_root)
    organized_root = _resolve_path(script_dir, args.organized_root)
    analysis_root = _resolve_path(script_dir, args.analysis_root)
    analyzer_script = _resolve_path(script_dir, args.analyzer_script)

    if not analyzer_script.exists():
        raise FileNotFoundError(f"No existe analyzer script: {analyzer_script}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    discovered: list[RunInfo] = []

    if not args.skip_organize:
        if not benchmark_root.exists() or not benchmark_root.is_dir():
            raise FileNotFoundError(f"No existe benchmark root: {benchmark_root}")

        discovered = _discover_runs_recursively(benchmark_root)
        if not discovered:
            raise ValueError(f"No se encontraron corridas validas en: {benchmark_root}")

        print(f"Corridas descubiertas: {len(discovered)}")

        manifest_rows: list[dict] = []
        for run in discovered:
            dst = organized_root / run.algorithm / run.profile / run.run_name
            print(f"Organizando: {run.run_name} -> {dst}")
            _ensure_link_or_copy(run.run_dir, dst, mode=args.organize_mode, dry_run=args.dry_run)
            manifest_rows.append(
                {
                    "run_name": run.run_name,
                    "algorithm": run.algorithm,
                    "profile": run.profile,
                    "source_run_dir": str(run.run_dir),
                    "organized_run_dir": str(dst),
                }
            )

        _write_manifest(organized_root / "run_manifest.csv", manifest_rows, dry_run=args.dry_run)

    if not organized_root.exists() or not organized_root.is_dir():
        if args.dry_run and discovered:
            print("\nDry-run sin organized_root fisico: simulando analisis por grupos descubiertos.")

            by_algorithm_profile: dict[tuple[str, str], list[RunInfo]] = {}
            for run in discovered:
                key = (run.algorithm, run.profile)
                by_algorithm_profile.setdefault(key, []).append(run)

            for (algorithm, profile), runs in sorted(by_algorithm_profile.items()):
                simulated_profile_dir = organized_root / algorithm / profile
                output_dir = analysis_root / algorithm / profile / f"analysis_{timestamp}"
                print(f"\nAnalizando subcarpeta: {algorithm}/{profile} ({len(runs)} corridas)")
                _run_analysis(
                    analyzer_script=analyzer_script,
                    benchmark_root=simulated_profile_dir,
                    output_dir=output_dir,
                    no_show_fliers=args.no_show_fliers,
                    dry_run=True,
                )

            algorithms = sorted({run.algorithm for run in discovered})
            for algorithm in algorithms:
                runs = [run for run in discovered if run.algorithm == algorithm]
                simulated_stage = analysis_root / "_staging" / algorithm
                output_dir = analysis_root / algorithm / "_all_profiles" / f"analysis_{timestamp}"
                print(f"\nAnalizando carpeta: {algorithm} ({len(runs)} corridas)")
                _run_analysis(
                    analyzer_script=analyzer_script,
                    benchmark_root=simulated_stage,
                    output_dir=output_dir,
                    no_show_fliers=args.no_show_fliers,
                    dry_run=True,
                )

            simulated_global_stage = analysis_root / "_staging" / "all_algorithms"
            output_dir = analysis_root / "all_algorithms" / f"analysis_{timestamp}"
            print(f"\nAnalizando carpeta global all_algorithms ({len(discovered)} corridas)")
            _run_analysis(
                analyzer_script=analyzer_script,
                benchmark_root=simulated_global_stage,
                output_dir=output_dir,
                no_show_fliers=args.no_show_fliers,
                dry_run=True,
            )

            print("\nProceso completado (dry-run).")
            print(f"Organized root: {organized_root}")
            print(f"Analysis root: {analysis_root}")
            return

        raise FileNotFoundError(f"No existe organized root: {organized_root}")

    stage_root = analysis_root / "_staging"
    analysis_jobs: list[dict] = []

    algorithm_dirs = [path for path in sorted(organized_root.iterdir()) if path.is_dir()]
    for algorithm_dir in algorithm_dirs:
        algorithm = algorithm_dir.name

        profile_dirs = [path for path in sorted(algorithm_dir.iterdir()) if path.is_dir()]
        for profile_dir in profile_dirs:
            profile = profile_dir.name
            run_dirs = _collect_runs_from_leaf(profile_dir)
            if not run_dirs:
                continue

            output_dir = analysis_root / algorithm / profile / f"analysis_{timestamp}"
            print(f"\nAnalizando subcarpeta: {algorithm}/{profile} ({len(run_dirs)} corridas)")
            _run_analysis(
                analyzer_script=analyzer_script,
                benchmark_root=profile_dir,
                output_dir=output_dir,
                no_show_fliers=args.no_show_fliers,
                dry_run=args.dry_run,
            )
            analysis_jobs.append(
                {
                    "scope": "profile",
                    "algorithm": algorithm,
                    "profile": profile,
                    "runs": len(run_dirs),
                    "analysis_dir": str(output_dir),
                }
            )

        # Algorithm-level analysis across all profiles.
        all_run_dirs: list[Path] = []
        for profile_dir in profile_dirs:
            all_run_dirs.extend(_collect_runs_from_leaf(profile_dir))
        if all_run_dirs:
            stage_dir = stage_root / algorithm
            _prepare_stage_dir(stage_dir, all_run_dirs, dry_run=args.dry_run)

            output_dir = analysis_root / algorithm / "_all_profiles" / f"analysis_{timestamp}"
            print(f"\nAnalizando carpeta: {algorithm} ({len(all_run_dirs)} corridas)")
            _run_analysis(
                analyzer_script=analyzer_script,
                benchmark_root=stage_dir,
                output_dir=output_dir,
                no_show_fliers=args.no_show_fliers,
                dry_run=args.dry_run,
            )
            analysis_jobs.append(
                {
                    "scope": "algorithm",
                    "algorithm": algorithm,
                    "profile": "_all_profiles",
                    "runs": len(all_run_dirs),
                    "analysis_dir": str(output_dir),
                }
            )

    # Global analysis across all algorithms.
    global_runs: list[Path] = []
    for algorithm_dir in algorithm_dirs:
        for profile_dir in sorted([p for p in algorithm_dir.iterdir() if p.is_dir()]):
            global_runs.extend(_collect_runs_from_leaf(profile_dir))

    if global_runs:
        stage_dir = stage_root / "all_algorithms"
        _prepare_stage_dir(stage_dir, global_runs, dry_run=args.dry_run)

        output_dir = analysis_root / "all_algorithms" / f"analysis_{timestamp}"
        print(f"\nAnalizando carpeta global all_algorithms ({len(global_runs)} corridas)")
        _run_analysis(
            analyzer_script=analyzer_script,
            benchmark_root=stage_dir,
            output_dir=output_dir,
            no_show_fliers=args.no_show_fliers,
            dry_run=args.dry_run,
        )
        analysis_jobs.append(
            {
                "scope": "global",
                "algorithm": "all_algorithms",
                "profile": "all",
                "runs": len(global_runs),
                "analysis_dir": str(output_dir),
            }
        )

    jobs_csv = analysis_root / f"analysis_jobs_{timestamp}.csv"
    if not args.dry_run:
        jobs_csv.parent.mkdir(parents=True, exist_ok=True)
        with jobs_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["scope", "algorithm", "profile", "runs", "analysis_dir"],
            )
            writer.writeheader()
            writer.writerows(analysis_jobs)

    print("\nProceso completado.")
    print(f"Organized root: {organized_root}")
    print(f"Analysis root: {analysis_root}")
    if not args.dry_run:
        print(f"Analysis jobs CSV: {jobs_csv}")


if __name__ == "__main__":
    main()
