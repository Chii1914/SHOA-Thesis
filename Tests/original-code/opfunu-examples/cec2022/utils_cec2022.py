from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def import_cec2022_module():
    try:
        from opfunu.benchmark import Benchmark
        from opfunu.cec_based import cec2022
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise SystemExit(
                "No se pudo importar pkg_resources. Ejecuta: pip install setuptools==75.8.0"
            ) from exc
        raise
    return Benchmark, cec2022


def get_cec2022_classes():
    benchmark_base, cec2022 = import_cec2022_module()
    classes = {}
    for name, obj in inspect.getmembers(cec2022, inspect.isclass):
        if name.startswith("F") and name.endswith("2022") and issubclass(obj, benchmark_base):
            classes[name] = obj
    return dict(sorted(classes.items()))


def build_function_instance(cls, preferred_dims):
    errors = []
    for candidate in preferred_dims:
        try:
            return cls(ndim=candidate), candidate
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            errors.append(f"ndim={candidate}: {exc}")

    try:
        return cls(), None
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        errors.append(f"default: {exc}")

    details = " | ".join(errors)
    raise RuntimeError(f"No se pudo instanciar {cls.__name__}. Detalles: {details}")


def _normalize_bounds(func):
    lb = np.asarray(func.lb, dtype=float).reshape(-1)
    ub = np.asarray(func.ub, dtype=float).reshape(-1)

    if lb.size == 1:
        lb = np.full(func.ndim, lb.item(), dtype=float)
    if ub.size == 1:
        ub = np.full(func.ndim, ub.item(), dtype=float)

    if lb.size != func.ndim or ub.size != func.ndim:
        raise ValueError(
            f"Bounds incompatibles en {func.__class__.__name__}: "
            f"len(lb)={lb.size}, len(ub)={ub.size}, ndim={func.ndim}"
        )

    return lb, ub


def evaluate_grid(func, n_points=120):
    if func.ndim < 2:
        raise ValueError(f"{func.__class__.__name__} requiere al menos 2 dimensiones para dibujar")

    lb, ub = _normalize_bounds(func)

    x_axis = np.linspace(lb[0], ub[0], n_points)
    y_axis = np.linspace(lb[1], ub[1], n_points)
    X, Y = np.meshgrid(x_axis, y_axis)

    base = (lb + ub) / 2.0
    Z = np.full_like(X, np.nan, dtype=float)

    for i in range(n_points):
        for j in range(n_points):
            candidate = base.copy()
            candidate[0] = X[i, j]
            candidate[1] = Y[i, j]
            try:
                value = float(func.evaluate(candidate))
            except KeyboardInterrupt:
                raise
            except BaseException:
                value = np.nan
            Z[i, j] = value

    return X, Y, Z


def _prepare_z_for_plot(Z):
    z_plot = np.array(Z, dtype=float, copy=True)
    finite = z_plot[np.isfinite(z_plot)]
    if finite.size == 0:
        return z_plot

    lo, hi = np.percentile(finite, [2.0, 98.0])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        z_plot = np.clip(z_plot, lo, hi)
    return z_plot


def save_2d_plot(X, Y, Z, title, output_png):
    z_plot = _prepare_z_for_plot(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, z_plot, levels=35, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def save_3d_plot(X, Y, Z, title, output_png):
    z_plot = _prepare_z_for_plot(Z)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X, Y, z_plot, cmap="viridis", linewidth=0, antialiased=True)
    fig.colorbar(surface, ax=ax, shrink=0.6, aspect=14)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def draw_function(func, output_base: Path, n_points=120):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    X, Y, Z = evaluate_grid(func, n_points=n_points)

    output_2d = output_base.with_name(f"{output_base.name}_2d.png")
    output_3d = output_base.with_name(f"{output_base.name}_3d.png")

    title = f"{func.__class__.__name__} | ndim={func.ndim}"
    save_2d_plot(X, Y, Z, title, output_2d)
    save_3d_plot(X, Y, Z, title, output_3d)

    return output_2d, output_3d
