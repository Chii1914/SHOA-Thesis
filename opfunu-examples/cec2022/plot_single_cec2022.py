from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from utils_cec2022 import build_function_instance, draw_function, get_cec2022_classes


def main():
    parser = argparse.ArgumentParser(
        description="Ejemplo individual para graficar una funcion CEC 2022 de Opfunu"
    )
    parser.add_argument("--function", default="F12022", help="Nombre de la clase (ejemplo: F12022)")
    parser.add_argument("--ndim", type=int, default=None, help="Dimension a usar")
    parser.add_argument("--points", type=int, default=120, help="Puntos por eje para graficos")
    parser.add_argument(
        "--output",
        default="opfunu-examples/cec2022/outputs/single",
        help="Directorio de salida para imagenes",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = get_cec2022_classes()
    if args.function not in classes:
        all_names = ", ".join(classes)
        raise SystemExit(f"Clase no valida: {args.function}. Disponibles: {all_names}")

    preferred_dims = [args.ndim] if args.ndim is not None else [10, 20, 2, 30, 50]
    func, selected_dim = build_function_instance(classes[args.function], preferred_dims)

    sample = func.create_solution()
    sample_value = func.evaluate(sample)

    print(f"Funcion: {func.__class__.__name__}")
    print(f"Dimension usada: {func.ndim}")
    if selected_dim is not None:
        print(f"Dimension seleccionada automaticamente: {selected_dim}")
    print(f"Bounds (primera dimension): [{func.lb[0]}, {func.ub[0]}]")
    print(f"f_global reportado: {func.f_global}")
    print(f"Evaluacion de muestra: {sample_value}")

    output_base = out_dir / func.__class__.__name__
    output_2d, output_3d = draw_function(func, output_base=output_base, n_points=args.points)

    print(f"Grafico 2D guardado en: {output_2d}")
    print(f"Grafico 3D guardado en: {output_3d}")


if __name__ == "__main__":
    main()
