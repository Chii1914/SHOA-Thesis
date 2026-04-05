from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from utils_cec2022 import build_function_instance, draw_function, get_cec2022_classes


def main():
    parser = argparse.ArgumentParser(
        description="Genera graficos 2D y 3D para todas las funciones CEC 2022 de Opfunu"
    )
    parser.add_argument("--points", type=int, default=120, help="Puntos por eje para graficos")
    parser.add_argument(
        "--output",
        default="opfunu-examples/cec2022/outputs/all",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--dims",
        default="10,20,2,30,50",
        help="Lista de dimensiones candidatas separadas por coma",
    )
    args = parser.parse_args()

    preferred_dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    classes = get_cec2022_classes()
    print(f"Total funciones CEC 2022 encontradas: {len(classes)}")

    summary_rows = []

    for idx, (name, cls) in enumerate(classes.items(), start=1):
        print(f"[{idx}/{len(classes)}] Procesando {name}...")
        try:
            func, used_dim = build_function_instance(cls, preferred_dims)
            sample = func.create_solution()
            sample_value = func.evaluate(sample)

            func_dir = out_root / name
            func_dir.mkdir(parents=True, exist_ok=True)

            output_2d, output_3d = draw_function(
                func,
                output_base=func_dir / name,
                n_points=args.points,
            )

            summary_rows.append(
                {
                    "class_name": name,
                    "display_name": func.name,
                    "used_dim": func.ndim if used_dim is None else used_dim,
                    "lb_first": float(func.lb[0]),
                    "ub_first": float(func.ub[0]),
                    "f_global": float(func.f_global),
                    "sample_eval": float(sample_value),
                    "plot_2d": str(output_2d),
                    "plot_3d": str(output_3d),
                }
            )
        except Exception as exc:
            print(f"Error en {name}: {exc}")
            summary_rows.append(
                {
                    "class_name": name,
                    "display_name": "ERROR",
                    "used_dim": "",
                    "lb_first": "",
                    "ub_first": "",
                    "f_global": "",
                    "sample_eval": "",
                    "plot_2d": "",
                    "plot_3d": "",
                }
            )

    summary_path = out_root / "cec2022_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "display_name",
                "used_dim",
                "lb_first",
                "ub_first",
                "f_global",
                "sample_eval",
                "plot_2d",
                "plot_3d",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Resumen guardado en: {summary_path}")
    print(f"Galeria completa en: {out_root}")


if __name__ == "__main__":
    main()
