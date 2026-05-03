"""Genera y guarda graficos de contribucion LIME desde lime_contributions.csv.

Uso rapido:
python plot_lime_contributions.py --csv-path benchmark_logs/.../lime_contributions.csv --iteration 492 --function F12022 --run-id 3 --seed 3 --show

Si una iteracion aparece en multiples corridas, agrega filtros (function/run-id/seed)
o usa --all-matches para exportar todas las coincidencias.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "No se pudo importar pandas. Instala dependencias con: pip install pandas matplotlib numpy"
    ) from exc

FEATURE_COLUMNS = [
    ("weight_r1", "r1"),
    ("weight_mag_browniano", "Mag. Browniano"),
    ("weight_mag_levy", "Mag. Levy"),
    ("weight_r2", "r2"),
    ("weight_mag_predacion", "Mag. Predacion"),
]

REQUIRED_COLUMNS = {
    "function",
    "dimension",
    "run_id",
    "seed",
    "diagnosis_id",
    "diagnosis_iteration",
    "diagnosis_status",
    "pred_delta",
    "fidelity",
    *[column for column, _ in FEATURE_COLUMNS],
}


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "item"


def _normalize_percent(weights_raw: list[float]) -> tuple[np.ndarray, float]:
    raw = np.asarray(weights_raw, dtype=float)
    total_abs = float(np.sum(np.abs(raw)))
    if total_abs <= 1e-15:
        return np.zeros_like(raw), total_abs
    return (raw / total_abs) * 100.0, total_abs


def _build_output_name(row: dict) -> str:
    base = (
        f"{row['function']}_d{int(row['dimension'])}_run{int(row['run_id'])}_"
        f"seed{int(row['seed'])}_it{int(row['diagnosis_iteration'])}_"
        f"diag{int(row['diagnosis_id'])}_{row['diagnosis_status']}"
    )
    return f"{_slugify(base)}.png"


def _apply_filters(
    df: pd.DataFrame,
    iteration: int | None,
    function: str | None,
    dimension: int | None,
    run_id: int | None,
    seed: int | None,
    diagnosis_id: int | None,
    status: str | None,
) -> pd.DataFrame:
    filtered = df.copy()

    if iteration is not None:
        filtered = filtered[filtered["diagnosis_iteration"] == iteration]
    if function is not None:
        filtered = filtered[filtered["function"] == function]
    if dimension is not None:
        filtered = filtered[filtered["dimension"] == dimension]
    if run_id is not None:
        filtered = filtered[filtered["run_id"] == run_id]
    if seed is not None:
        filtered = filtered[filtered["seed"] == seed]
    if diagnosis_id is not None:
        filtered = filtered[filtered["diagnosis_id"] == diagnosis_id]
    if status is not None:
        filtered = filtered[filtered["diagnosis_status"].astype(str) == status]

    return filtered


def _plot_single_row(
    row: dict,
    output_dir: Path,
    dpi: int,
    fig_width: float,
    fig_height: float,
    show_plot: bool,
    save_plot: bool,
) -> Path | None:
    features = [label for _, label in FEATURE_COLUMNS]
    weights_raw = [float(row[column]) for column, _ in FEATURE_COLUMNS]

    weights_norm, total_abs = _normalize_percent(weights_raw)

    colors = []
    for value in weights_norm:
        if value > 0:
            colors.append("#2ca02c")
        elif value < 0:
            colors.append("#d62728")
        else:
            colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.barh(features, weights_norm, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=1.2)

    for bar, value in zip(bars, weights_norm):
        y_center = bar.get_y() + bar.get_height() / 2.0
        offset = 0.75 if value >= 0 else -0.75
        align = "left" if value >= 0 else "right"
        ax.text(value + offset, y_center, f"{value:.1f}%", va="center", ha=align, fontsize=9)

    status = str(row["diagnosis_status"])
    fidelity = float(row["fidelity"])
    pred_delta = float(row["pred_delta"])

    title_line1 = f"Explicacion LIME - Iteracion {int(row['diagnosis_iteration'])}"
    title_line2 = f"Status: {status} | Fidelidad (R2): {fidelity:.3f} | Pred Delta: {pred_delta:.3e}"
    title_line3 = (
        f"Func: {row['function']} | d={int(row['dimension'])} | run={int(row['run_id'])} | "
        f"seed={int(row['seed'])} | diag={int(row['diagnosis_id'])}"
    )
    ax.set_title(f"{title_line1}\n{title_line2}\n{title_line3}")

    if total_abs <= 1e-15:
        ax.text(
            0.5,
            -0.14,
            "Advertencia: suma absoluta de pesos = 0, se grafican ceros.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Contribucion al Delta Fitness (%)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    output_path = None
    if save_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _build_output_name(row)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)

    return output_path


def plot_lime_contribution(
    csv_path: str | Path,
    target_iteration: int | None = None,
    output_dir: str | Path = "lime_plots",
    function: str | None = None,
    dimension: int | None = None,
    run_id: int | None = None,
    seed: int | None = None,
    diagnosis_id: int | None = None,
    status: str | None = None,
    all_matches: bool = False,
    show_plot: bool = False,
    save_plot: bool = True,
    dpi: int = 220,
    fig_width: float = 8.0,
    fig_height: float = 5.0,
) -> list[Path]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing_sorted}")

    filtered = _apply_filters(
        df=df,
        iteration=target_iteration,
        function=function,
        dimension=dimension,
        run_id=run_id,
        seed=seed,
        diagnosis_id=diagnosis_id,
        status=status,
    )

    if filtered.empty:
        raise ValueError("No hay filas que cumplan los filtros solicitados.")

    filtered = filtered.sort_values(
        by=["function", "dimension", "run_id", "seed", "diagnosis_iteration", "diagnosis_id"]
    )

    if len(filtered) > 1 and not all_matches:
        sample_cols = ["function", "dimension", "run_id", "seed", "diagnosis_iteration", "diagnosis_id"]
        sample = filtered[sample_cols].head(10).to_string(index=False)
        raise ValueError(
            "El filtro devuelve multiples filas. Agrega filtros mas especificos "
            "(--function/--run-id/--seed/--diagnosis-id) o usa --all-matches.\n\n"
            f"Primeras coincidencias:\n{sample}"
        )

    rows = filtered.to_dict(orient="records")
    if not all_matches:
        rows = [rows[0]]

    saved_paths: list[Path] = []
    output_dir = Path(output_dir)

    for row in rows:
        out = _plot_single_row(
            row=row,
            output_dir=output_dir,
            dpi=dpi,
            fig_width=fig_width,
            fig_height=fig_height,
            show_plot=show_plot,
            save_plot=save_plot,
        )
        if out is not None:
            saved_paths.append(out)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica y guarda contribuciones de LIME desde lime_contributions.csv"
    )
    parser.add_argument("--csv-path", required=True, help="Ruta al archivo lime_contributions.csv")
    parser.add_argument("--iteration", type=int, default=None, help="diagnosis_iteration objetivo")
    parser.add_argument("--function", default=None, help="Filtro por funcion (ej: F12022)")
    parser.add_argument("--dimension", type=int, default=None, help="Filtro por dimension")
    parser.add_argument("--run-id", type=int, default=None, help="Filtro por run_id")
    parser.add_argument("--seed", type=int, default=None, help="Filtro por seed")
    parser.add_argument("--diagnosis-id", type=int, default=None, help="Filtro por diagnosis_id")
    parser.add_argument("--status", default=None, help="Filtro por diagnosis_status")
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="Exporta todas las filas que cumplan los filtros",
    )
    parser.add_argument(
        "--output-dir",
        default="lime_plots",
        help="Directorio donde se guardaran los PNG",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Resolucion del PNG")
    parser.add_argument("--fig-width", type=float, default=8.0, help="Ancho de figura")
    parser.add_argument("--fig-height", type=float, default=5.0, help="Alto de figura")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra cada grafico en pantalla ademas de guardarlo",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guarda PNG; solo muestra en pantalla",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    saved = plot_lime_contribution(
        csv_path=args.csv_path,
        target_iteration=args.iteration,
        output_dir=args.output_dir,
        function=args.function,
        dimension=args.dimension,
        run_id=args.run_id,
        seed=args.seed,
        diagnosis_id=args.diagnosis_id,
        status=args.status,
        all_matches=args.all_matches,
        show_plot=args.show,
        save_plot=not args.no_save,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

    if args.no_save:
        print("Graficos generados en pantalla (sin guardar archivos).")
    else:
        print(f"Graficos guardados: {len(saved)}")
        for path in saved:
            print(path)


if __name__ == "__main__":
    main()
