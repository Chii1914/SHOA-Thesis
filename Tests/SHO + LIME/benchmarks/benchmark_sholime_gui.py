"""Interactive GUI viewer for SHO+LIME benchmark outputs.

This tool visualizes:
1) convergence trace (best fitness by iteration), and
2) LIME contribution bars for the selected iteration.

It supports full execution playback with:
- previous/next iteration
- jump -10 / +10 iterations
- play/pause
- previous/next case (function-dim-run-seed)
- sliders for case and iteration

Example:
python benchmark_sholime_gui.py --run-dir benchmark_logs/cec2022_sholime_d10_r30_20260406_161523
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider

FEATURE_COLUMNS = [
    ("weight_r1", "r1"),
    ("weight_mag_browniano", "Mag. Browniano"),
    ("weight_mag_levy", "Mag. Levy"),
    ("weight_r2", "r2"),
    ("weight_mag_predacion", "Mag. Predacion"),
]

CASE_COLUMNS = ["function", "dimension", "run_id", "seed"]
REQUIRED_FULL_COLUMNS = ["function", "dimension", "run_id", "seed", "iteration", "best_fitness"]


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _normalize_percent(weights_raw: list[float]) -> tuple[np.ndarray, float]:
    raw = np.asarray(weights_raw, dtype=float)
    total_abs = float(np.sum(np.abs(raw)))
    if total_abs <= 1e-15:
        return np.zeros_like(raw), total_abs
    return (raw / total_abs) * 100.0, total_abs


def _find_latest_run_dir(benchmark_root: Path) -> Path:
    candidates = [
        path
        for path in benchmark_root.iterdir()
        if path.is_dir() and (path / "full_output.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No se encontro ningun run con full_output.csv en: {benchmark_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_csv_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir: Path

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.full_output_csv:
        run_dir = Path(args.full_output_csv).resolve().parent
    else:
        run_dir = _find_latest_run_dir(Path(args.benchmark_root))

    full_output_path = Path(args.full_output_csv) if args.full_output_csv else run_dir / "full_output.csv"
    lime_path = Path(args.lime_csv) if args.lime_csv else run_dir / "lime_contributions.csv"

    if not full_output_path.exists():
        raise FileNotFoundError(f"No existe full_output.csv: {full_output_path}")

    return run_dir, full_output_path, lime_path


def _prepare_full_dataframe(full_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_FULL_COLUMNS if column not in full_df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Faltan columnas requeridas en full_output.csv: {missing_text}")

    df = full_df.copy()
    df["function"] = df["function"].astype(str)

    for column in ["dimension", "run_id", "seed", "iteration"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["best_fitness"] = pd.to_numeric(df["best_fitness"], errors="coerce")

    df = df.dropna(subset=["dimension", "run_id", "seed", "iteration", "best_fitness"])

    for column in ["dimension", "run_id", "seed", "iteration"]:
        df[column] = df[column].astype(int)

    defaults = {
        "diagnostics_invoked": False,
        "diagnosis_status": "NONE",
        "diagnosis_pred_delta": np.nan,
        "diagnosis_fidelity": np.nan,
        "rescue_applied": False,
        "diagnosis_id": 0,
        "strong_stochastic_importance": False,
        "low_expected_improvement": False,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    for weight_column, _ in FEATURE_COLUMNS:
        if weight_column not in df.columns:
            df[weight_column] = np.nan
        abs_column = f"abs_{weight_column}"
        if abs_column not in df.columns:
            df[abs_column] = np.nan

    # Precompute booleans once to avoid per-frame casting during playback.
    df["diagnostics_invoked_bool"] = df["diagnostics_invoked"].map(_as_bool)
    df["rescue_applied_bool"] = df["rescue_applied"].map(_as_bool)

    return df.sort_values(by=CASE_COLUMNS + ["iteration"]).reset_index(drop=True)


def _prepare_lime_dataframe(lime_df: pd.DataFrame) -> pd.DataFrame:
    if lime_df.empty:
        return lime_df

    required = ["function", "dimension", "run_id", "seed", "diagnosis_iteration"]
    missing = [column for column in required if column not in lime_df.columns]
    if missing:
        return pd.DataFrame(columns=lime_df.columns)

    df = lime_df.copy()
    for column in ["dimension", "run_id", "seed", "diagnosis_iteration"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["dimension", "run_id", "seed", "diagnosis_iteration"])

    for column in ["dimension", "run_id", "seed", "diagnosis_iteration"]:
        df[column] = df[column].astype(int)

    if "diagnosis_id" in df.columns:
        df["diagnosis_id"] = pd.to_numeric(df["diagnosis_id"], errors="coerce").fillna(0).astype(int)
    else:
        df["diagnosis_id"] = 0

    for feature_column, _ in FEATURE_COLUMNS:
        if feature_column not in df.columns:
            df[feature_column] = np.nan

    if "diagnosis_status" not in df.columns:
        df["diagnosis_status"] = "UNKNOWN"
    if "pred_delta" not in df.columns:
        df["pred_delta"] = np.nan
    if "fidelity" not in df.columns:
        df["fidelity"] = np.nan

    return df.sort_values(by=CASE_COLUMNS + ["diagnosis_iteration", "diagnosis_id"]).reset_index(drop=True)


def _case_key(case_row: dict) -> tuple[str, int, int, int]:
    return (
        str(case_row["function"]),
        int(case_row["dimension"]),
        int(case_row["run_id"]),
        int(case_row["seed"]),
    )


class SHOBenchmarkViewer:
    def __init__(
        self,
        run_dir: Path,
        full_df: pd.DataFrame,
        lime_df: pd.DataFrame,
        fps: float,
        start_case_idx: int,
        start_iteration: int,
        play_update_every: int,
        play_step: int,
    ) -> None:
        self.run_dir = run_dir
        self.full_df = full_df
        self.lime_df = lime_df
        self.fps = max(float(fps), 0.5)
        self.play_update_every = max(1, int(play_update_every))
        self.play_step = max(1, int(play_step))

        self.cases = self._build_cases()
        if not self.cases:
            raise ValueError("No hay casos en full_output.csv para visualizar.")

        self.case_data = self._build_case_data()
        self.case_plot_cache = self._build_case_plot_cache()
        self.lime_lookup = self._build_lime_lookup()

        self.global_max_iteration = int(self.full_df["iteration"].max())

        self.current_case_idx = int(np.clip(start_case_idx, 0, len(self.cases) - 1))
        self.current_iteration = max(1, int(start_iteration))

        self.playing = False
        self._play_tick = 0
        self._ignore_slider_events = False
        self._render_in_progress = False

        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.9, 1.1], wspace=0.25)
        self.ax_conv = self.fig.add_subplot(gs[0, 0])
        self.ax_lime = self.fig.add_subplot(gs[0, 1])

        self._active_case_key = None
        self._conv_full_line = None
        self._conv_upto_line = None
        self._conv_current_marker = None

        self._lime_bars = None
        self._lime_value_texts: list = []
        self._lime_no_diag_text = None
        self._lime_initialized = False

        self.info_text = self.fig.text(0.02, 0.97, "", va="top", ha="left", fontsize=10)

        self._build_controls()
        self._configure_timer()
        self._connect_keyboard()
        self._render(force_slider_sync=True)

    def _build_cases(self) -> list[dict]:
        unique_cases = (
            self.full_df[CASE_COLUMNS]
            .drop_duplicates()
            .sort_values(by=CASE_COLUMNS)
            .to_dict(orient="records")
        )
        return unique_cases

    def _build_case_data(self) -> dict[tuple[str, int, int, int], pd.DataFrame]:
        case_data: dict[tuple[str, int, int, int], pd.DataFrame] = {}
        grouped = self.full_df.groupby(CASE_COLUMNS, sort=False)
        for key_raw, case_df in grouped:
            key = (str(key_raw[0]), int(key_raw[1]), int(key_raw[2]), int(key_raw[3]))
            case_data[key] = case_df.sort_values("iteration").reset_index(drop=True)
        return case_data

    def _build_case_plot_cache(self) -> dict[tuple[str, int, int, int], dict]:
        cache: dict[tuple[str, int, int, int], dict] = {}
        for key, case_df in self.case_data.items():
            x_full = case_df["iteration"].to_numpy(dtype=int)
            y_full = case_df["best_fitness"].to_numpy(dtype=float)
            iter_to_idx = {int(iteration): idx for idx, iteration in enumerate(x_full)}

            diag_rows = case_df.loc[case_df["diagnostics_invoked_bool"]]
            rescue_rows = case_df.loc[case_df["rescue_applied_bool"]]

            cache[key] = {
                "x_full": x_full,
                "y_full": y_full,
                "iter_to_idx": iter_to_idx,
                "diag_x": diag_rows["iteration"].to_numpy(dtype=int),
                "diag_y": diag_rows["best_fitness"].to_numpy(dtype=float),
                "rescue_x": rescue_rows["iteration"].to_numpy(dtype=int),
                "rescue_y": rescue_rows["best_fitness"].to_numpy(dtype=float),
            }
        return cache

    def _build_lime_lookup(self) -> dict[tuple[tuple[str, int, int, int], int], dict]:
        lookup: dict[tuple[tuple[str, int, int, int], int], dict] = {}
        if self.lime_df.empty:
            return lookup

        for _, row in self.lime_df.iterrows():
            key = (str(row["function"]), int(row["dimension"]), int(row["run_id"]), int(row["seed"]))
            iter_idx = int(row["diagnosis_iteration"])
            lookup[(key, iter_idx)] = row.to_dict()
        return lookup

    def _build_controls(self) -> None:
        self.fig.subplots_adjust(bottom=0.20, top=0.88)

        self.ax_btn_prev_case = self.fig.add_axes([0.02, 0.08, 0.07, 0.05])
        self.ax_btn_next_case = self.fig.add_axes([0.10, 0.08, 0.07, 0.05])
        self.ax_btn_prev_iter = self.fig.add_axes([0.22, 0.08, 0.07, 0.05])
        self.ax_btn_back10 = self.fig.add_axes([0.30, 0.08, 0.07, 0.05])
        self.ax_btn_play = self.fig.add_axes([0.38, 0.08, 0.07, 0.05])
        self.ax_btn_fwd10 = self.fig.add_axes([0.46, 0.08, 0.07, 0.05])
        self.ax_btn_next_iter = self.fig.add_axes([0.54, 0.08, 0.07, 0.05])

        self.ax_case_slider = self.fig.add_axes([0.66, 0.095, 0.31, 0.03])
        self.ax_iter_slider = self.fig.add_axes([0.66, 0.045, 0.31, 0.03])

        self.btn_prev_case = Button(self.ax_btn_prev_case, "Case -")
        self.btn_next_case = Button(self.ax_btn_next_case, "Case +")
        self.btn_prev_iter = Button(self.ax_btn_prev_iter, "Iter -")
        self.btn_back10 = Button(self.ax_btn_back10, "-10")
        self.btn_play = Button(self.ax_btn_play, "Play")
        self.btn_fwd10 = Button(self.ax_btn_fwd10, "+10")
        self.btn_next_iter = Button(self.ax_btn_next_iter, "Iter +")

        case_slider_max = max(1, len(self.cases) - 1)

        self.case_slider = Slider(
            self.ax_case_slider,
            "Case",
            0,
            case_slider_max,
            valinit=self.current_case_idx,
            valstep=1,
        )
        self.iter_slider = Slider(
            self.ax_iter_slider,
            "Iter",
            1,
            max(1, self.global_max_iteration),
            valinit=self.current_iteration,
            valstep=1,
        )

        self.btn_prev_case.on_clicked(lambda _event: self._step_case(-1))
        self.btn_next_case.on_clicked(lambda _event: self._step_case(+1))
        self.btn_prev_iter.on_clicked(lambda _event: self._step_iteration(-1))
        self.btn_back10.on_clicked(lambda _event: self._step_iteration(-10))
        self.btn_play.on_clicked(lambda _event: self._toggle_play())
        self.btn_fwd10.on_clicked(lambda _event: self._step_iteration(+10))
        self.btn_next_iter.on_clicked(lambda _event: self._step_iteration(+1))

        self.case_slider.on_changed(self._on_case_slider)
        self.iter_slider.on_changed(self._on_iter_slider)

        if len(self.cases) == 1:
            self.case_slider.set_active(False)

    def _configure_timer(self) -> None:
        interval_ms = int(round(1000.0 / self.fps))
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        # Single-shot + manual restart avoids callback pileups on slower backends.
        if hasattr(self.timer, "single_shot"):
            self.timer.single_shot = True
        self.timer.add_callback(self._on_timer)

    def _connect_keyboard(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _current_case(self) -> dict:
        return self.cases[self.current_case_idx]

    def _current_case_key(self) -> tuple[str, int, int, int]:
        return _case_key(self._current_case())

    def _current_case_df(self) -> pd.DataFrame:
        return self.case_data[self._current_case_key()]

    def _clamp_iteration_to_case(self, iteration: int) -> int:
        case_df = self._current_case_df()
        min_iter = int(case_df["iteration"].min())
        max_iter = int(case_df["iteration"].max())
        return int(np.clip(iteration, min_iter, max_iter))

    def _set_case(self, new_case_idx: int) -> None:
        self.current_case_idx = int(np.clip(new_case_idx, 0, len(self.cases) - 1))
        self.current_iteration = self._clamp_iteration_to_case(1)
        self._play_tick = 0
        self._render(force_slider_sync=True)

    def _set_iteration(
        self,
        new_iteration: int,
        sync_slider: bool = True,
        lightweight: bool = False,
    ) -> None:
        self.current_iteration = self._clamp_iteration_to_case(int(new_iteration))
        self._render(force_slider_sync=sync_slider, lightweight=lightweight)

    def _step_case(self, delta: int) -> None:
        self._set_case(self.current_case_idx + int(delta))

    def _step_iteration(self, delta: int) -> None:
        self._set_iteration(self.current_iteration + int(delta))

    def _toggle_play(self) -> None:
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.btn_play.label.set_text("Play")
            self._play_tick = 0
            self._render(force_slider_sync=True, lightweight=False)
            return

        self.playing = True
        self._play_tick = 0
        self.timer.start()
        self.btn_play.label.set_text("Pause")

    def _on_timer(self) -> None:
        if (not self.playing) or self._render_in_progress:
            return

        case_df = self._current_case_df()
        max_iter = int(case_df["iteration"].max())

        if self.current_iteration >= max_iter:
            self.playing = False
            self.timer.stop()
            self.btn_play.label.set_text("Play")
            self._sync_sliders()
            self._play_tick = 0
            return

        # During playback, avoid slider sync every frame to prevent UI stalls.
        self._play_tick += 1
        next_iteration = min(self.current_iteration + self.play_step, max_iter)
        self._set_iteration(next_iteration, sync_slider=False, lightweight=True)

        if self.playing:
            self.timer.start()

    def _on_case_slider(self, slider_value: float) -> None:
        if self._ignore_slider_events:
            return
        self._set_case(int(round(slider_value)))

    def _on_iter_slider(self, slider_value: float) -> None:
        if self._ignore_slider_events:
            return
        self._set_iteration(int(round(slider_value)))

    def _on_key_press(self, event) -> None:
        if event.key == "left":
            self._step_iteration(-1)
        elif event.key == "right":
            self._step_iteration(+1)
        elif event.key == "down":
            self._step_case(-1)
        elif event.key == "up":
            self._step_case(+1)
        elif event.key == " ":
            self._toggle_play()
        elif event.key == "pageup":
            self._step_iteration(+10)
        elif event.key == "pagedown":
            self._step_iteration(-10)

    def _sync_sliders(self) -> None:
        self._ignore_slider_events = True
        if int(round(self.case_slider.val)) != self.current_case_idx:
            self.case_slider.set_val(self.current_case_idx)
        if int(round(self.iter_slider.val)) != self.current_iteration:
            self.iter_slider.set_val(self.current_iteration)
        self._ignore_slider_events = False

    def _fetch_current_iteration_row(self) -> pd.Series:
        case_key = self._current_case_key()
        case_df = self.case_data[case_key]
        cache = self.case_plot_cache[case_key]

        idx = cache["iter_to_idx"].get(self.current_iteration)
        if idx is None:
            x_full = cache["x_full"]
            idx = int(np.searchsorted(x_full, self.current_iteration, side="right") - 1)
            idx = int(np.clip(idx, 0, len(x_full) - 1))
            self.current_iteration = int(x_full[idx])

        return case_df.iloc[int(idx)]

    def _ensure_convergence_artists(self, case_key: tuple[str, int, int, int]) -> None:
        if self._active_case_key == case_key and self._conv_full_line is not None:
            return

        cache = self.case_plot_cache[case_key]
        x_full = cache["x_full"]
        y_full = cache["y_full"]

        self.ax_conv.clear()
        self._conv_full_line, = self.ax_conv.plot(
            x_full,
            y_full,
            color="#9ecae1",
            linewidth=1.8,
            label="Convergencia completa",
        )
        self._conv_upto_line, = self.ax_conv.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=2.4,
            label="Hasta iteracion actual",
        )
        self._conv_current_marker, = self.ax_conv.plot(
            [],
            [],
            marker="o",
            linestyle="None",
            color="#d62728",
            markeredgecolor="black",
            markersize=7,
            label="Iteracion actual",
        )

        handles = [self._conv_full_line, self._conv_upto_line, self._conv_current_marker]
        labels = ["Convergencia completa", "Hasta iteracion actual", "Iteracion actual"]

        diag_x = cache["diag_x"]
        diag_y = cache["diag_y"]
        if diag_x.size > 0:
            diag_scatter = self.ax_conv.scatter(
                diag_x,
                diag_y,
                marker="^",
                s=42,
                color="#ff7f0e",
                edgecolor="black",
                alpha=0.85,
                label="LIME diagnosis",
            )
            handles.append(diag_scatter)
            labels.append("LIME diagnosis")

        rescue_x = cache["rescue_x"]
        rescue_y = cache["rescue_y"]
        if rescue_x.size > 0:
            rescue_scatter = self.ax_conv.scatter(
                rescue_x,
                rescue_y,
                marker="*",
                s=95,
                color="#2ca02c",
                edgecolor="black",
                alpha=0.9,
                label="Rescue aplicado",
            )
            handles.append(rescue_scatter)
            labels.append("Rescue aplicado")

        self.ax_conv.set_xlabel("Iteracion")
        self.ax_conv.set_ylabel("Best fitness")
        self.ax_conv.grid(alpha=0.25)
        self.ax_conv.legend(handles=handles, labels=labels, loc="best")

        if x_full.size > 0:
            self.ax_conv.set_xlim(int(x_full[0]), int(x_full[-1]))

        finite_y = y_full[np.isfinite(y_full)]
        if finite_y.size > 0:
            y_min = float(np.min(finite_y))
            y_max = float(np.max(finite_y))
            pad = float(max((y_max - y_min) * 0.05, 1e-12))
            if np.isclose(y_min, y_max):
                pad = max(abs(y_max) * 0.05, 1e-9)
            self.ax_conv.set_ylim(y_min - pad, y_max + pad)

        self._active_case_key = case_key

    def _init_lime_artists(self) -> None:
        if self._lime_initialized:
            return

        labels = [label for _, label in FEATURE_COLUMNS]
        zero_widths = np.zeros(len(labels), dtype=float)

        self.ax_lime.clear()
        self._lime_bars = self.ax_lime.barh(labels, zero_widths, color="#7f7f7f", edgecolor="black")
        self.ax_lime.axvline(0.0, color="black", linewidth=1.1)

        self._lime_value_texts = []
        for bar in self._lime_bars:
            y_center = bar.get_y() + bar.get_height() / 2.0
            text_artist = self.ax_lime.text(0.9, y_center, "0.0%", va="center", ha="left", fontsize=9)
            self._lime_value_texts.append(text_artist)

        self.ax_lime.set_xlabel("Contribucion al Delta Fitness (%)")
        self.ax_lime.grid(axis="x", alpha=0.25)
        self.ax_lime.set_xlim(-110.0, 110.0)

        self._lime_no_diag_text = self.ax_lime.text(
            0.5,
            -0.17,
            "No hay diagnostico LIME en esta iteracion.",
            transform=self.ax_lime.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            visible=False,
        )
        self._lime_initialized = True

    def _extract_lime_payload(self, iteration_row: pd.Series) -> dict:
        status = str(iteration_row.get("diagnosis_status", "NONE"))
        pred_delta = float(iteration_row.get("diagnosis_pred_delta", np.nan))
        fidelity = float(iteration_row.get("diagnosis_fidelity", np.nan))
        diagnosis_id = int(iteration_row.get("diagnosis_id", 0))

        weights = [float(iteration_row.get(column, np.nan)) for column, _ in FEATURE_COLUMNS]
        has_weights_in_full_output = not np.all(np.isnan(weights))

        if has_weights_in_full_output:
            return {
                "status": status,
                "pred_delta": pred_delta,
                "fidelity": fidelity,
                "diagnosis_id": diagnosis_id,
                "weights": weights,
                "diagnostics_invoked": _as_bool(iteration_row.get("diagnostics_invoked", False)),
            }

        lookup_key = (self._current_case_key(), int(self.current_iteration))
        lime_row = self.lime_lookup.get(lookup_key)
        if lime_row is not None:
            return {
                "status": str(lime_row.get("diagnosis_status", status)),
                "pred_delta": float(lime_row.get("pred_delta", pred_delta)),
                "fidelity": float(lime_row.get("fidelity", fidelity)),
                "diagnosis_id": int(lime_row.get("diagnosis_id", diagnosis_id)),
                "weights": [float(lime_row.get(column, np.nan)) for column, _ in FEATURE_COLUMNS],
                "diagnostics_invoked": True,
            }

        return {
            "status": status,
            "pred_delta": pred_delta,
            "fidelity": fidelity,
            "diagnosis_id": diagnosis_id,
            "weights": [np.nan for _ in FEATURE_COLUMNS],
            "diagnostics_invoked": _as_bool(iteration_row.get("diagnostics_invoked", False)),
        }

    def _render_convergence(self, case_df: pd.DataFrame, lightweight: bool = False) -> float:
        case_key = self._current_case_key()
        cache = self.case_plot_cache[case_key]
        self._ensure_convergence_artists(case_key)

        current_row = self._fetch_current_iteration_row()
        current_best = float(current_row["best_fitness"])

        x_full = cache["x_full"]
        y_full = cache["y_full"]
        idx = cache["iter_to_idx"].get(self.current_iteration)
        if idx is None:
            idx = int(np.searchsorted(x_full, self.current_iteration, side="right") - 1)
            idx = int(np.clip(idx, 0, len(x_full) - 1))

        if (not lightweight) or (self._play_tick % self.play_update_every == 0):
            self._conv_upto_line.set_data(x_full[: idx + 1], y_full[: idx + 1])
        self._conv_current_marker.set_data([self.current_iteration], [current_best])

        return current_best

    def _render_lime(self, iteration_row: pd.Series) -> dict:
        self._init_lime_artists()
        payload = self._extract_lime_payload(iteration_row)

        weights = payload["weights"]
        weights_arr = np.asarray(weights, dtype=float)

        has_lime_here = not np.all(np.isnan(weights_arr))
        if has_lime_here:
            weights_norm, _ = _normalize_percent(weights_arr.tolist())
        else:
            weights_norm = np.zeros(len(FEATURE_COLUMNS), dtype=float)

        colors = []
        for value in weights_norm:
            if value > 0:
                colors.append("#2ca02c")
            elif value < 0:
                colors.append("#d62728")
            else:
                colors.append("#7f7f7f")

        for bar, text_artist, value, color in zip(
            self._lime_bars,
            self._lime_value_texts,
            weights_norm,
            colors,
        ):
            bar.set_width(float(value))
            bar.set_facecolor(color)
            y_center = bar.get_y() + bar.get_height() / 2.0
            offset = 0.9 if value >= 0 else -0.9
            align = "left" if value >= 0 else "right"
            text_artist.set_position((float(value + offset), float(y_center)))
            text_artist.set_ha(align)
            text_artist.set_text(f"{value:.1f}%")

        abs_max = float(np.max(np.abs(weights_norm))) if weights_norm.size > 0 else 0.0
        x_lim = max(10.0, min(110.0, abs_max * 1.35))
        self.ax_lime.set_xlim(-x_lim, x_lim)

        if has_lime_here:
            self.ax_lime.set_title(f"Contribucion LIME (iteracion {self.current_iteration})")
            self._lime_no_diag_text.set_visible(False)
        else:
            self.ax_lime.set_title(f"Contribucion LIME (iteracion {self.current_iteration})")
            self._lime_no_diag_text.set_visible(True)

        return payload

    def _render(self, force_slider_sync: bool = False, lightweight: bool = False) -> None:
        if self._render_in_progress:
            return

        self._render_in_progress = True

        case = self._current_case()
        case_df = self._current_case_df()

        try:
            self.current_iteration = self._clamp_iteration_to_case(self.current_iteration)
            iteration_row = self._fetch_current_iteration_row()
            current_best = self._render_convergence(case_df, lightweight=lightweight)

            should_update_heavy = (
                (not lightweight)
                or (self._play_tick % self.play_update_every == 0)
                or bool(iteration_row.get("diagnostics_invoked_bool", False))
                or bool(iteration_row.get("rescue_applied_bool", False))
            )

            if should_update_heavy:
                lime_payload = self._render_lime(iteration_row)
            else:
                lime_payload = {
                    "status": str(iteration_row.get("diagnosis_status", "NONE")),
                    "pred_delta": float(iteration_row.get("diagnosis_pred_delta", np.nan)),
                    "fidelity": float(iteration_row.get("diagnosis_fidelity", np.nan)),
                    "diagnosis_id": int(iteration_row.get("diagnosis_id", 0)),
                }

            max_iter = int(case_df["iteration"].max())
            status = str(lime_payload["status"])
            pred_delta = float(lime_payload["pred_delta"])
            fidelity = float(lime_payload["fidelity"])
            diagnosis_id = int(lime_payload["diagnosis_id"])

            pred_text = f"{pred_delta:.3e}" if np.isfinite(pred_delta) else "nan"
            fidelity_text = f"{fidelity:.3f}" if np.isfinite(fidelity) else "nan"

            self.fig.suptitle(
                (
                    f"SHO+LIME Benchmark Playback | Case {self.current_case_idx + 1}/{len(self.cases)}"
                    f" | iter {self.current_iteration}/{max_iter}"
                ),
                fontsize=13,
            )

            if should_update_heavy or (not self.playing):
                self.info_text.set_text(
                    (
                        f"Run folder: {self.run_dir}\n"
                        f"function={case['function']} | dim={case['dimension']} | run={case['run_id']} | seed={case['seed']}\n"
                        f"best_fitness={current_best:.6e} | diagnosis_status={status} | diagnosis_id={diagnosis_id} "
                        f"| pred_delta={pred_text} | fidelity={fidelity_text}"
                    )
                )

            if force_slider_sync:
                self._sync_sliders()

            self.fig.canvas.draw_idle()
            if self.playing:
                try:
                    self.fig.canvas.flush_events()
                except Exception:
                    pass
        finally:
            self._render_in_progress = False



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interfaz grafica para reproducir benchmark SHO+LIME (convergencia + contribucion LIME)"
        )
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Carpeta de un benchmark run (debe contener full_output.csv). "
            "Si se omite, se usa el ultimo run en benchmark_logs."
        ),
    )
    parser.add_argument("--full-output-csv", default=None, help="Ruta explicita a full_output.csv")
    parser.add_argument("--lime-csv", default=None, help="Ruta explicita a lime_contributions.csv")
    parser.add_argument(
        "--benchmark-root",
        default="benchmark_logs",
        help="Raiz donde buscar el ultimo run cuando no se pasa --run-dir",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="Velocidad de reproduccion (iteraciones/segundo)")
    parser.add_argument(
        "--play-update-every",
        type=int,
        default=10,
        help="En play, actualiza elementos pesados cada N ticks (default: 10)",
    )
    parser.add_argument(
        "--play-step",
        type=int,
        default=2,
        help="En play, avanza N iteraciones por tick (default: 2)",
    )
    parser.add_argument("--start-case-index", type=int, default=0, help="Indice inicial de caso (base 0)")
    parser.add_argument("--start-iteration", type=int, default=1, help="Iteracion inicial")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, full_output_path, lime_path = _resolve_csv_paths(args)

    full_df = _prepare_full_dataframe(pd.read_csv(full_output_path))

    if lime_path.exists():
        lime_df = _prepare_lime_dataframe(pd.read_csv(lime_path))
    else:
        lime_df = pd.DataFrame()

    print(f"Run folder: {run_dir}")
    print(f"Full output: {full_output_path}")
    if lime_path.exists():
        print(f"LIME contributions: {lime_path}")
    else:
        print("LIME contributions: no encontrado (se mostrara solo convergencia).")
    print("Controles: left/right iter, up/down case, space play/pause, pageup/pagedown +/-10")

    SHOBenchmarkViewer(
        run_dir=run_dir,
        full_df=full_df,
        lime_df=lime_df,
        fps=args.fps,
        start_case_idx=args.start_case_index,
        start_iteration=args.start_iteration,
        play_update_every=args.play_update_every,
        play_step=args.play_step,
    )
    plt.show()


if __name__ == "__main__":
    main()
