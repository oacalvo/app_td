#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom as ndimage_zoom

try:
    from .core import (
        CUBE_AXIS_ORDERS,
        MAX_PANELS,
        Cut,
        FeatureAxis,
        TDPanel,
        angle_difference_deg,
        bilinear_sample,
        clamp_int,
        clamp_point,
        clamp_value,
        compute_td,
        cube_axis_order_display_label,
        cube_axis_order_numeric_label,
        cut_cache_key,
        cut_center,
        cut_directed_angle_deg,
        cut_display_angle_deg,
        cut_length,
        distance,
        distance_point_to_segment,
        display_ticks,
        feature_axis_sample_positions,
        frame_limits,
        line_geometry,
        line_geometry_from_points,
        load_cube,
        make_default_panels,
        map_value_to_display,
        min_segment_distance,
        normalize_cube_axis_order,
        on_segment,
        orient,
        panel_title,
        polyline_arc_lengths,
        polyline_length,
        polyline_point_at_length,
        polyline_tangent_at_length,
        ray_limit,
        rotate_cut,
        rotate_point,
        segment_from_angle_length,
        segments_intersect,
        shift_cut,
        td_visual_spline,
        vector_from_vertical_angle,
        weighted_profile,
        width_offsets_and_weights,
    )
except ImportError:
    from core import (
        CUBE_AXIS_ORDERS,
        MAX_PANELS,
        Cut,
        FeatureAxis,
        TDPanel,
        angle_difference_deg,
        bilinear_sample,
        clamp_int,
        clamp_point,
        clamp_value,
        compute_td,
        cube_axis_order_display_label,
        cube_axis_order_numeric_label,
        cut_cache_key,
        cut_center,
        cut_directed_angle_deg,
        cut_display_angle_deg,
        cut_length,
        distance,
        distance_point_to_segment,
        display_ticks,
        feature_axis_sample_positions,
        frame_limits,
        line_geometry,
        line_geometry_from_points,
        load_cube,
        make_default_panels,
        map_value_to_display,
        min_segment_distance,
        normalize_cube_axis_order,
        on_segment,
        orient,
        panel_title,
        polyline_arc_lengths,
        polyline_length,
        polyline_point_at_length,
        polyline_tangent_at_length,
        ray_limit,
        rotate_cut,
        rotate_point,
        segment_from_angle_length,
        segments_intersect,
        shift_cut,
        td_visual_spline,
        vector_from_vertical_angle,
        weighted_profile,
        width_offsets_and_weights,
    )


DEFAULT_CUBE = (
    Path(__file__).resolve().parents[4]
    / "cube_core_avg_all_WOW_nodenoise_g1.0.fits"
)
MAX_FEATURE_AXIS_CUTS = 600
LAYOUT_PRESETS = {
    "1x1": (1, 1),
    "2x1": (2, 1),
    "2x2": (2, 2),
}
COLOR_CYCLE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

DEFAULT_CREST_TRACKING = {
    "cad": 1.35,
    "res": 0.03,
    "grad": 0.5,
    "min_tlen": 20,
    "max_dist_jump": 3,
    "max_time_skip": 4,
    "invert": False,
    "gauss": False,
}

DEFAULT_WAVELET_FILTER = {
    "p_min": 10.0,
    "p_max": 100.0,
    "power_ratio_thresh": 2.4,
    "segment_power_frac": 0.35,
    "min_points_segment": 25,
    "min_amp_arcsec": 0.055,
    "max_jump_pix": 1.5,
    "min_points_cut_seg": 8,
    "rms_amp_ratio_max": 0.7,
    "km_per_arcsec": 725.27,
    "density_kg_m3": float("nan"),
    "phase_speed_km_s": float("nan"),
}

SESSION_VERSION = 4
DEFAULT_AUTOSAVE_EVERY = 10
WAVELET_QA_EDGE_FRACTION = 0.1
WAVELET_QA_RESIDUAL_WARN_FRAC = 0.85
WAVELET_QA_LOW_POINTS_MARGIN = 2
MAX_WAVELET_HISTORY = 60
WAVELET_SEGMENT_TIMEOUT_S = 8.0
WAVELET_SEGMENT_POLL_S = 0.25
DEBUG_STACK_WAVELET_TRACE = False

PARAMETER_PRESETS = {
    "custom": {
        "label": "Custom",
        "crest": dict(DEFAULT_CREST_TRACKING),
        "wavelet": dict(DEFAULT_WAVELET_FILTER),
    },
    "default": {
        "label": "Default",
        "crest": dict(DEFAULT_CREST_TRACKING),
        "wavelet": dict(DEFAULT_WAVELET_FILTER),
    },
    "sst_halpha": {
        "label": "SST H-alpha",
        "crest": {
            **dict(DEFAULT_CREST_TRACKING),
            "cad": 1.00,
            "res": 0.059,
            "grad": 0.45,
            "min_tlen": 18,
            "max_dist_jump": 2,
            "max_time_skip": 3,
        },
        "wavelet": {
            **dict(DEFAULT_WAVELET_FILTER),
            "p_min": 8.0,
            "p_max": 120.0,
            "power_ratio_thresh": 2.2,
            "segment_power_frac": 0.32,
            "min_points_segment": 20,
            "min_amp_arcsec": 0.035,
            "max_jump_pix": 1.2,
            "min_points_cut_seg": 7,
            "rms_amp_ratio_max": 0.75,
            "km_per_arcsec": 725.27,
        },
    },
    "iris_sji": {
        "label": "IRIS SJI",
        "crest": {
            **dict(DEFAULT_CREST_TRACKING),
            "cad": 5.0,
            "res": 0.167,
            "grad": 0.60,
            "min_tlen": 12,
            "max_dist_jump": 3,
            "max_time_skip": 4,
        },
        "wavelet": {
            **dict(DEFAULT_WAVELET_FILTER),
            "p_min": 15.0,
            "p_max": 180.0,
            "power_ratio_thresh": 2.1,
            "segment_power_frac": 0.30,
            "min_points_segment": 12,
            "min_amp_arcsec": 0.050,
            "max_jump_pix": 1.7,
            "min_points_cut_seg": 6,
            "rms_amp_ratio_max": 0.80,
            "km_per_arcsec": 725.27,
        },
    },
    "aia_euv": {
        "label": "SDO/AIA EUV",
        "crest": {
            **dict(DEFAULT_CREST_TRACKING),
            "cad": 12.0,
            "res": 0.60,
            "grad": 0.70,
            "min_tlen": 8,
            "max_dist_jump": 4,
            "max_time_skip": 4,
        },
        "wavelet": {
            **dict(DEFAULT_WAVELET_FILTER),
            "p_min": 24.0,
            "p_max": 480.0,
            "power_ratio_thresh": 2.0,
            "segment_power_frac": 0.28,
            "min_points_segment": 8,
            "min_amp_arcsec": 0.080,
            "max_jump_pix": 2.2,
            "min_points_cut_seg": 5,
            "rms_amp_ratio_max": 0.90,
            "km_per_arcsec": 725.27,
        },
    },
}

_NUWT_API: dict[str, Any] | None = None
_NUWT_IMPORT_ERROR: str | None = None
_WAVELET_FILTER_API: dict[str, Any] | None = None
_WAVELET_FILTER_IMPORT_ERROR: str | None = None


def load_local_nuwt_api() -> tuple[dict[str, Any] | None, str | None]:
    global _NUWT_API, _NUWT_IMPORT_ERROR

    if _NUWT_API is not None:
        return _NUWT_API, None
    if _NUWT_IMPORT_ERROR is not None:
        return None, _NUWT_IMPORT_ERROR

    package_root = Path(__file__).resolve().parents[2]
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

    try:
        from nuwt import follow_threads, locate_things, patch_up_threads
    except Exception as exc:
        _NUWT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None, _NUWT_IMPORT_ERROR

    _NUWT_API = {
        "locate_things": locate_things,
        "follow_threads": follow_threads,
        "patch_up_threads": patch_up_threads,
    }
    return _NUWT_API, None


def load_local_wavelet_filter_api() -> tuple[dict[str, Any] | None, str | None]:
    global _WAVELET_FILTER_API, _WAVELET_FILTER_IMPORT_ERROR

    if _WAVELET_FILTER_API is not None:
        return _WAVELET_FILTER_API, None
    if _WAVELET_FILTER_IMPORT_ERROR is not None:
        return None, _WAVELET_FILTER_IMPORT_ERROR

    package_root = Path(__file__).resolve().parent
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

    try:
        from td_wavelet_filter import (
            DEFAULT_WAVELET_FILTER as module_defaults,
            analyze_tracked_segment_with_wavelet,
            analyze_tracked_threads_with_wavelets,
            split_thread_on_jumps,
        )
    except Exception as exc:
        _WAVELET_FILTER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None, _WAVELET_FILTER_IMPORT_ERROR

    _WAVELET_FILTER_API = {
        "defaults": module_defaults,
        "analyze_tracked_segment_with_wavelet": analyze_tracked_segment_with_wavelet,
        "analyze_tracked_threads_with_wavelets": analyze_tracked_threads_with_wavelets,
        "split_thread_on_jumps": split_thread_on_jumps,
    }
    return _WAVELET_FILTER_API, None


def _trace_stack_wavelet(message: str) -> None:
    if not DEBUG_STACK_WAVELET_TRACE:
        return
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[td-debug {timestamp}] {message}", file=sys.stderr, flush=True)


def _wavelet_segment_process_main(task_queue: Any, result_queue: Any) -> None:
    api, import_error = load_local_wavelet_filter_api()
    if api is None:
        result_queue.put(
            {
                "task_id": -1,
                "ok": False,
                "fatal": True,
                "error": str(import_error or "wavelet import failed"),
            }
        )
        return

    analyze_tracked_segment_with_wavelet = api["analyze_tracked_segment_with_wavelet"]
    while True:
        task = task_queue.get()
        if task is None:
            return

        task_id = int(task.get("task_id", -1))
        try:
            analysis = analyze_tracked_segment_with_wavelet(
                np.asarray(task.get("t_idx_seg", []), dtype=np.float64),
                np.asarray(task.get("y_idx_seg", []), dtype=np.float64),
                cadence=float(task["cad"]),
                pix_scale=float(task["res"]),
                km_per_arcsec=float(task["km_per_arcsec"]),
                p_min=float(task["p_min"]),
                p_max=float(task["p_max"]),
                power_ratio_thresh=float(task["power_ratio_thresh"]),
                segment_power_frac=float(task["segment_power_frac"]),
                min_points_segment=int(task["min_points_segment"]),
                min_amp_arcsec=float(task["min_amp_arcsec"]),
                rms_amp_ratio_max=float(task["rms_amp_ratio_max"]),
                density_kg_m3=float(task["density_kg_m3"]),
                phase_speed_km_s=float(task["phase_speed_km_s"]),
            )
            result_queue.put({"task_id": task_id, "ok": True, "analysis": analysis})
        except Exception as exc:
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": False,
                    "fatal": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive time-distance mosaic app for a 3D FITS cube."
    )
    parser.add_argument(
        "--cube",
        type=Path,
        default=DEFAULT_CUBE,
        help=f"Path to the FITS cube. Default: {DEFAULT_CUBE}",
    )
    parser.add_argument(
        "--cube-order",
        default="TYX",
        help="Input cube axis order. Use TYX/TXY/YTX/YXT/XTY/XYT or 123/132/213/231/312/321.",
    )
    return parser.parse_args()


class TDMosaicApp:
    def __init__(self, cube_path: Path, cube_axis_order: str = "TYX"):
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.Rectangle = Rectangle

        self.cube_path = cube_path.expanduser().resolve()
        self.cube_axis_order = normalize_cube_axis_order(cube_axis_order)
        self.cube, self.header = load_cube(self.cube_path, axis_order=self.cube_axis_order)
        self.nt, self.ny, self.nx = self.cube.shape

        self.root = tk.Tk()
        self.root.title("Time-Distance Mosaic")
        self.root.geometry("1700x950")

        self.layout_name = "2x2"
        self.visible_panels = 4
        self.panels = make_default_panels(self.nt)
        self.cuts: dict[int, Cut] = {}
        self.next_cut_id = 1
        self.feature_axes: dict[int, FeatureAxis] = {}
        self.next_feature_axis_id = 1
        self.active_panel_id = 1
        self.selected_cut_id: int | None = None
        self.selected_feature_axis_id: int | None = None
        self.clipboard_cut: dict[str, Any] | None = None
        self.draw_mode = False
        self.force_new_cut = False
        self.pending_point: tuple[float, float] | None = None
        self.hover_point: tuple[float, float] | None = None
        self.feature_draw_mode: str | None = None
        self.feature_pending_points: list[tuple[float, float]] = []
        self.feature_hover_point: tuple[float, float] | None = None
        self.drag_state: dict[str, Any] | None = None
        self.control_update_guard = False
        self.geometry_update_guard = False
        self.panel_selector_update_guard = False
        self.status_var = tk.StringVar(
            value="Click twice on the map to create a cut. Click a TD panel to edit it."
        )

        self.t_visual_var = tk.IntVar(value=min(100, self.nt - 1))
        self.layout_var = tk.StringVar(value=self.layout_name)
        self.panel_t_ini_var = tk.IntVar(value=0)
        self.panel_t_fin_var = tk.IntVar(value=self.nt - 1)
        self.panel_stride_var = tk.IntVar(value=1)
        self.panel_width_var = tk.IntVar(value=1)
        self.panel_weighting_var = tk.StringVar(value="uniform")
        self.selected_cut_name_var = tk.StringVar(value="No cut selected")
        self.selected_feature_axis_name_var = tk.StringVar(value="No feature axis selected")
        self.geometry_angle_var = tk.StringVar(value="")
        self.geometry_length_var = tk.StringVar(value="")
        self.geometry_anchor_var = tk.StringVar(value="center")
        self.geometry_length_mode_var = tk.StringVar(value="symmetric")
        self.geometry_x1_var = tk.StringVar(value="")
        self.geometry_y1_var = tk.StringVar(value="")
        self.geometry_x2_var = tk.StringVar(value="")
        self.geometry_y2_var = tk.StringVar(value="")
        self.dynamic_cut_enabled_var = tk.BooleanVar(value=False)
        self.dynamic_reference_frame_var = tk.StringVar(value="")
        self.dynamic_keyframe_summary_var = tk.StringVar(value="Dynamic geometry disabled.")
        self.reference_cut_var = tk.StringVar(value="")
        self.target_cut_var = tk.StringVar(value="")
        self.panel_cut_var = tk.StringVar(value="")
        self.center_distance_var = tk.StringVar(value="")
        self.measure_center_var = tk.StringVar(value="Center distance: n/a")
        self.measure_min_var = tk.StringVar(value="Min distance: n/a")
        self.measure_angle_var = tk.StringVar(value="Angle difference: n/a")
        self.feature_spacing_var = tk.StringVar(value="10.0")
        self.feature_length_var = tk.StringVar(value="40.0")
        self.feature_angle_offset_var = tk.StringVar(value="0.0")
        self.feature_create_stack_var = tk.BooleanVar(value=True)
        self.td_aspect_var = tk.StringVar(value="equal")
        self.td_render_var = tk.StringVar(value="scientific")
        self.td_zoom_var = tk.StringVar(value="1x")
        self.map_swap_xy_var = tk.BooleanVar(value=False)
        self.td_swap_axes_var = tk.BooleanVar(value=False)
        self.map_flip_x_var = tk.BooleanVar(value=False)
        self.map_flip_y_var = tk.BooleanVar(value=False)
        self.td_flip_x_var = tk.BooleanVar(value=False)
        self.td_flip_y_var = tk.BooleanVar(value=False)

        self.figure = self.Figure(figsize=(12.5, 7.5), dpi=100)
        self.canvas = None
        self.map_ax = None
        self.map_image = None
        self.panel_axes: dict[int, Any] = {}
        self.axis_to_panel_id: dict[Any, int] = {}
        self.td_windows: dict[int, dict[str, Any]] = {}
        self.cut_analysis_state: dict[int, dict[str, Any]] = {}
        self.panel_analysis_state: dict[int, dict[str, Any]] = {
            panel.panel_id: self._make_default_panel_analysis_state()
            for panel in self.panels
        }
        self.stacks: dict[int, dict[str, Any]] = {}
        self.next_stack_id = 1
        self.active_stack_id: int | None = None
        self.selected_stack_cut_id: int | None = None
        self.stack_browsers: dict[int, dict[str, Any]] = {}
        self.next_stack_browser_id = 1
        self.stack_browser_refresh_job: Any = None
        self.stack_browser_refresh_in_progress = False
        self.background_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.background_jobs: dict[str, dict[str, Any]] = {}
        self.next_background_job_id = 1
        self.metrics_window: dict[str, Any] | None = None
        self.link_groups_window: dict[str, Any] | None = None
        self.propagation_window: dict[str, Any] | None = None
        self.saved_fits_window: dict[str, Any] | None = None
        self.last_session_path: Path | None = None
        self.autosave_every = DEFAULT_AUTOSAVE_EVERY
        self.autosave_change_count = 0
        self.link_source_event_ref: dict[str, Any] | None = None
        self.next_link_group_id = 1
        self.export_dir_var = tk.StringVar(value=str(self.cube_path.parent))
        self.export_info_var = tk.StringVar(value="")
        self.autosave_path = (
            Path(__file__).resolve().parent.parent
            / f"{self.cube_path.stem}_td_session_autosave.json"
        )

        self._build_layout()
        self._build_controls()
        self._build_sidebar()
        self._build_figure()
        self._bind_shortcuts()
        self._apply_layout()
        self._sync_controls_from_active_panel()
        self.refresh_all()
        self.root.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self.root.after(120, self._poll_background_jobs)

    def run(self) -> None:
        self.root.mainloop()

    @property
    def active_panel(self) -> TDPanel:
        return self.panels[self.active_panel_id - 1]

    def _make_default_panel_analysis_state(self) -> dict[str, Any]:
        return {
            "td_params": {
                "t_ini": 0,
                "t_fin": self.nt - 1,
                "stride": 1,
                "width": 1,
                "weighting": "uniform",
            },
            "td_cache_key": None,
            "td_cache_td": None,
            "td_cache_meta": None,
            "crest_params": dict(DEFAULT_CREST_TRACKING),
            "wavelet_params": dict(DEFAULT_WAVELET_FILTER),
            "preset_name": "custom",
            "crest_tracking_result": None,
            "crest_tracking_td_key": None,
            "wavelet_filter_result": None,
            "wavelet_thread_filter_text": "",
            "wavelet_events": [],
            "wavelet_next_event_id": 1,
            "wavelet_selected_event_id": None,
            "wavelet_events_filter": "accepted",
            "wavelet_advanced_filters": {
                "qa": "all",
                "locked": "all",
                "linked": "all",
                "score_min": "",
                "period_min": "",
                "period_max": "",
                "amp_min": "",
                "amp_max": "",
                "energy_min": "",
                "energy_max": "",
            },
            "wavelet_undo_stack": [],
            "wavelet_redo_stack": [],
            "roi_enabled": False,
            "roi_t_span": "",
            "roi_d_span": "",
            "roi_center_t": None,
            "roi_center_d": None,
            "dynamic_enabled": False,
            "dynamic_reference_frame": 0,
            "dynamic_keyframes": {},
        }

    def _make_default_stack_state(
        self, stack_id: int, name: str, cut_ids: list[int] | None = None
    ) -> dict[str, Any]:
        return {
            "stack_id": int(stack_id),
            "name": str(name),
            "cut_ids": list(cut_ids or []),
            "notes": "",
            "order_mode": "manual",
        }

    def _cut_analysis(self, cut_id: int) -> dict[str, Any]:
        state = self.cut_analysis_state.get(cut_id)
        if state is None:
            state = self._make_default_panel_analysis_state()
            self.cut_analysis_state[cut_id] = state
        default_td_params = self._make_default_panel_analysis_state()["td_params"]
        state["td_params"] = {
            **dict(default_td_params),
            **dict(state.get("td_params") or {}),
        }
        state.setdefault("td_cache_key", None)
        state.setdefault("td_cache_td", None)
        state.setdefault("td_cache_meta", None)
        state["dynamic_enabled"] = bool(state.get("dynamic_enabled", False))
        state["dynamic_reference_frame"] = clamp_int(
            int(state.get("dynamic_reference_frame", 0)), 0, self.nt - 1
        )
        normalized_keyframes: dict[int, dict[str, Any]] = {}
        raw_keyframes = state.get("dynamic_keyframes") or {}
        if isinstance(raw_keyframes, dict):
            for frame_key, payload in raw_keyframes.items():
                try:
                    frame_idx = clamp_int(int(frame_key), 0, self.nt - 1)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                p0_raw = payload.get("p0") or [0.0, 0.0]
                p1_raw = payload.get("p1") or [1.0, 1.0]
                try:
                    p0 = clamp_point((float(p0_raw[0]), float(p0_raw[1])), self.nx, self.ny)
                    p1 = clamp_point((float(p1_raw[0]), float(p1_raw[1])), self.nx, self.ny)
                except Exception:
                    continue
                if distance(p0, p1) < 1.0:
                    continue
                normalized_keyframes[frame_idx] = {
                    "p0": [float(p0[0]), float(p0[1])],
                    "p1": [float(p1[0]), float(p1[1])],
                }
        state["dynamic_keyframes"] = normalized_keyframes
        return state

    def _cut_td_params(self, cut_id: int) -> dict[str, Any]:
        state = self._cut_analysis(cut_id)
        td_params = {
            **dict(self._make_default_panel_analysis_state()["td_params"]),
            **dict(state.get("td_params") or {}),
        }
        td_params["t_ini"] = clamp_int(int(td_params["t_ini"]), 0, self.nt - 1)
        td_params["t_fin"] = clamp_int(
            int(td_params["t_fin"]), int(td_params["t_ini"]), self.nt - 1
        )
        td_params["stride"] = max(int(td_params["stride"]), 1)
        td_params["width"] = max(int(td_params["width"]), 1)
        td_params["weighting"] = (
            str(td_params["weighting"])
            if str(td_params["weighting"]) in {"uniform", "gaussian"}
            else "uniform"
        )
        state["td_params"] = dict(td_params)
        return td_params

    def _sync_panels_from_cut_td_params(self, cut_id: int) -> None:
        params = self._cut_td_params(cut_id)
        for panel in self.panels:
            if panel.cut_id != cut_id:
                continue
            panel.t_ini = int(params["t_ini"])
            panel.t_fin = int(params["t_fin"])
            panel.stride = int(params["stride"])
            panel.width = int(params["width"])
            panel.weighting = str(params["weighting"])

    def _seed_cut_td_params_from_panel(self, cut_id: int, panel: TDPanel) -> None:
        state = self._cut_analysis(cut_id)
        current = dict(state.get("td_params") or {})
        default_td_params = self._make_default_panel_analysis_state()["td_params"]
        if current and current != default_td_params:
            self._sync_panels_from_cut_td_params(cut_id)
            return
        state["td_params"] = {
            "t_ini": int(panel.t_ini),
            "t_fin": int(panel.t_fin),
            "stride": int(panel.stride),
            "width": int(panel.width),
            "weighting": str(panel.weighting),
        }
        self._sync_panels_from_cut_td_params(cut_id)

    def _panel_td_params(self, panel: TDPanel) -> dict[str, Any]:
        if panel.cut_id is not None and panel.cut_id in self.cuts:
            params = self._cut_td_params(panel.cut_id)
            panel.t_ini = int(params["t_ini"])
            panel.t_fin = int(params["t_fin"])
            panel.stride = int(params["stride"])
            panel.width = int(params["width"])
            panel.weighting = str(params["weighting"])
            return params
        return {
            "t_ini": int(panel.t_ini),
            "t_fin": int(panel.t_fin),
            "stride": int(panel.stride),
            "width": int(panel.width),
            "weighting": str(panel.weighting),
        }

    def _panel_analysis(self, panel_id: int) -> dict[str, Any]:
        if 1 <= int(panel_id) <= len(self.panels):
            panel = self.panels[int(panel_id) - 1]
            if panel.cut_id is not None and panel.cut_id in self.cuts:
                return self._cut_analysis(panel.cut_id)
        state = self.panel_analysis_state.get(panel_id)
        if state is None:
            state = self._make_default_panel_analysis_state()
            self.panel_analysis_state[panel_id] = state
        return state

    def _panels_for_cut(self, cut_id: int) -> list[TDPanel]:
        return [panel for panel in self.panels if panel.cut_id == cut_id]

    def _primary_panel_for_cut(self, cut_id: int) -> TDPanel | None:
        panels = self._panels_for_cut(cut_id)
        return panels[0] if panels else None

    def _cut_analysis_snapshot(self, cut_id: int) -> dict[str, Any]:
        for panel in self._panels_for_cut(cut_id):
            if panel.panel_id in self.td_windows:
                return self._panel_analysis_snapshot(panel.panel_id)
        return self._clone_wavelet_payload(self._cut_analysis(cut_id))

    def _stack_memberships_for_cut(self, cut_id: int) -> list[dict[str, Any]]:
        memberships: list[dict[str, Any]] = []
        for stack in sorted(self.stacks.values(), key=lambda item: int(item["stack_id"])):
            cut_ids = [
                int(member_cut_id)
                for member_cut_id in (stack.get("cut_ids") or [])
                if int(member_cut_id) in self.cuts
            ]
            for index, member_cut_id in enumerate(cut_ids, start=1):
                if member_cut_id != cut_id:
                    continue
                memberships.append(
                    {
                        "stack_id": int(stack["stack_id"]),
                        "stack_name": str(stack["name"]),
                        "stack_index": int(index),
                        "stack_size": int(len(cut_ids)),
                    }
                )
        return memberships

    def _cut_dynamic_enabled(self, cut_id: int) -> bool:
        if cut_id not in self.cuts:
            return False
        return bool(self._cut_analysis(cut_id).get("dynamic_enabled", False))

    def _cut_dynamic_reference_frame(self, cut_id: int) -> int:
        if cut_id not in self.cuts:
            return 0
        return clamp_int(
            int(self._cut_analysis(cut_id).get("dynamic_reference_frame", 0)),
            0,
            self.nt - 1,
        )

    def _cut_dynamic_keyframes(self, cut_id: int) -> dict[int, dict[str, Any]]:
        if cut_id not in self.cuts:
            return {}
        return dict(self._cut_analysis(cut_id).get("dynamic_keyframes") or {})

    def _cut_geometry_signature(self, cut_id: int) -> tuple[Any, ...]:
        if cut_id not in self.cuts:
            return ("missing", int(cut_id))
        state = self._cut_analysis(cut_id)
        signature: list[Any] = [
            "dynamic",
            bool(state.get("dynamic_enabled", False)),
            int(state.get("dynamic_reference_frame", 0)),
        ]
        for frame_idx, payload in sorted((state.get("dynamic_keyframes") or {}).items()):
            p0 = payload.get("p0") or [0.0, 0.0]
            p1 = payload.get("p1") or [0.0, 0.0]
            signature.extend(
                [
                    int(frame_idx),
                    round(float(p0[0]), 3),
                    round(float(p0[1]), 3),
                    round(float(p1[0]), 3),
                    round(float(p1[1]), 3),
                ]
            )
        return tuple(signature)

    def _cut_geometry_for_frame(
        self, cut_id: int, frame_idx: int | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        cut = self.cuts.get(cut_id)
        if cut is None:
            return (0.0, 0.0), (1.0, 1.0)
        if not self._cut_dynamic_enabled(cut_id):
            return cut.p0, cut.p1

        frame_idx = (
            int(self.t_visual_var.get()) if frame_idx is None else clamp_int(int(frame_idx), 0, self.nt - 1)
        )
        anchors: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {
            int(self._cut_dynamic_reference_frame(cut_id)): (cut.p0, cut.p1)
        }
        for keyframe_idx, payload in self._cut_dynamic_keyframes(cut_id).items():
            p0_raw = payload.get("p0") or [cut.p0[0], cut.p0[1]]
            p1_raw = payload.get("p1") or [cut.p1[0], cut.p1[1]]
            anchors[int(keyframe_idx)] = (
                clamp_point((float(p0_raw[0]), float(p0_raw[1])), self.nx, self.ny),
                clamp_point((float(p1_raw[0]), float(p1_raw[1])), self.nx, self.ny),
            )
        anchor_frames = sorted(anchors.keys())
        if not anchor_frames:
            return cut.p0, cut.p1
        if frame_idx <= anchor_frames[0]:
            return anchors[anchor_frames[0]]
        if frame_idx >= anchor_frames[-1]:
            return anchors[anchor_frames[-1]]
        if frame_idx in anchors:
            return anchors[frame_idx]

        lower = max(frame for frame in anchor_frames if frame < frame_idx)
        upper = min(frame for frame in anchor_frames if frame > frame_idx)
        p0_lo, p1_lo = anchors[lower]
        p0_hi, p1_hi = anchors[upper]
        alpha = (frame_idx - lower) / max(upper - lower, 1)
        p0 = (
            float((1.0 - alpha) * p0_lo[0] + alpha * p0_hi[0]),
            float((1.0 - alpha) * p0_lo[1] + alpha * p0_hi[1]),
        )
        p1 = (
            float((1.0 - alpha) * p1_lo[0] + alpha * p1_hi[0]),
            float((1.0 - alpha) * p1_lo[1] + alpha * p1_hi[1]),
        )
        return clamp_point(p0, self.nx, self.ny), clamp_point(p1, self.nx, self.ny)

    def _cut_preview(self, cut_id: int, frame_idx: int | None = None) -> Cut | None:
        cut = self.cuts.get(cut_id)
        if cut is None:
            return None
        p0, p1 = self._cut_geometry_for_frame(cut_id, frame_idx)
        return Cut(
            cut_id=cut.cut_id,
            name=cut.name,
            color=cut.color,
            p0=p0,
            p1=p1,
            visible=cut.visible,
            locked=cut.locked,
        )

    def _dynamic_cut_geometry_samples(
        self, cut_id: int, t_ini: int, t_fin: int, stride: int
    ) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
        if cut_id not in self.cuts or not self._cut_dynamic_enabled(cut_id):
            return {}
        samples: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        for frame_idx in np.arange(t_ini, t_fin + 1, max(int(stride), 1), dtype=np.int32):
            samples[int(frame_idx)] = self._cut_geometry_for_frame(cut_id, int(frame_idx))
        return samples

    def _set_cut_dynamic_keyframe(
        self,
        cut_id: int,
        frame_idx: int,
        p0: tuple[float, float],
        p1: tuple[float, float],
        *,
        enable_dynamic: bool = True,
    ) -> bool:
        if cut_id not in self.cuts:
            return False
        p0 = clamp_point(p0, self.nx, self.ny)
        p1 = clamp_point(p1, self.nx, self.ny)
        if distance(p0, p1) < 1.0:
            return False
        cut = self.cuts.get(cut_id)
        if cut is None:
            return False
        state = self._cut_analysis(cut_id)
        frame_idx = clamp_int(int(frame_idx), 0, self.nt - 1)
        keyframes = dict(state.get("dynamic_keyframes") or {})
        if frame_idx == int(state.get("dynamic_reference_frame", 0)):
            cut.p0 = p0
            cut.p1 = p1
            keyframes.pop(frame_idx, None)
        else:
            keyframes[frame_idx] = {
                "p0": [float(p0[0]), float(p0[1])],
                "p1": [float(p1[0]), float(p1[1])],
            }
        state["dynamic_keyframes"] = keyframes
        if enable_dynamic:
            state["dynamic_enabled"] = True
        return True

    def _delete_cut_dynamic_keyframe(self, cut_id: int, frame_idx: int) -> bool:
        if cut_id not in self.cuts:
            return False
        state = self._cut_analysis(cut_id)
        keyframes = dict(state.get("dynamic_keyframes") or {})
        removed = keyframes.pop(clamp_int(int(frame_idx), 0, self.nt - 1), None)
        state["dynamic_keyframes"] = keyframes
        return removed is not None

    def _safe_float_text(self, value: Any, default: float) -> float:
        try:
            text = str(value).strip()
            return float(text) if text else float(default)
        except Exception:
            return float(default)

    def _safe_int_text(self, value: Any, default: int) -> int:
        try:
            text = str(value).strip()
            return int(float(text)) if text else int(default)
        except Exception:
            return int(default)

    def _timestamp_now(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _safe_export_slug(self, text: str) -> str:
        cleaned = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in str(text).strip()
        )
        cleaned = cleaned.strip("._")
        return cleaned or "item"

    def _resolved_export_dir(self) -> Path:
        raw = str(self.export_dir_var.get()).strip()
        target = Path(raw).expanduser() if raw else self.cube_path.parent
        resolved = target.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self.export_dir_var.set(str(resolved))
        return resolved

    def _browse_export_dir(self) -> None:
        initial_dir = str(
            Path(str(self.export_dir_var.get()).strip() or self.cube_path.parent).expanduser()
        )
        selected = self.filedialog.askdirectory(
            title="Select export folder",
            initialdir=initial_dir,
            mustexist=False,
        )
        if not selected:
            return
        self.export_dir_var.set(str(Path(selected).expanduser()))
        self._refresh_export_controls()
        self._refresh_saved_fits_browser()
        self._record_session_change()
        self._set_status(f"Export folder set to {self.export_dir_var.get()}.")

    def _refresh_export_controls(self) -> None:
        if not hasattr(self, "export_info_var"):
            return
        current_t = int(self.t_visual_var.get())
        lines = [
            f"Folder: {str(self.export_dir_var.get()).strip() or self.cube_path.parent}",
            f"Cube order: {cube_axis_order_display_label(self.cube_axis_order)}",
            f"Current map frame: t={current_t}",
        ]
        cut = self._selected_cut()
        if cut is None:
            lines.append("Selected cut: none")
        else:
            params = self._cut_td_params(cut.cut_id)
            p0_now, p1_now = self._cut_geometry_for_frame(cut.cut_id, current_t)
            lines.append(
                f"Selected cut {cut.cut_id} {cut.name} | "
                f"TD t={int(params['t_ini'])}:{int(params['t_fin'])}:{int(params['stride'])} | "
                f"coords@t={current_t}: ({p0_now[0]:.1f},{p0_now[1]:.1f}) -> "
                f"({p1_now[0]:.1f},{p1_now[1]:.1f})"
            )
        stack = self._selected_stack()
        if stack is not None:
            lines.append(
                f"Active stack {int(stack['stack_id'])}: {stack['name']} | "
                f"cuts={len(stack.get('cut_ids') or [])}"
            )
        self.export_info_var.set("\n".join(lines))

    def _point_on_segment_distance(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        dist_px: float,
    ) -> tuple[float, float]:
        total_length = distance(p0, p1)
        if total_length <= 1e-9:
            return float(p0[0]), float(p0[1])
        alpha = clamp_value(float(dist_px) / total_length, 0.0, 1.0)
        return (
            float((1.0 - alpha) * p0[0] + alpha * p1[0]),
            float((1.0 - alpha) * p0[1] + alpha * p1[1]),
        )

    def _current_map_cut_rows(self, frame_idx: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cut_id in sorted(self.cuts.keys()):
            cut = self.cuts[cut_id]
            p0, p1 = self._cut_geometry_for_frame(cut_id, frame_idx)
            memberships = self._stack_memberships_for_cut(cut_id)
            rows.append(
                {
                    "cut_id": int(cut_id),
                    "cut_name": str(cut.name),
                    "x1": float(p0[0]),
                    "y1": float(p0[1]),
                    "x2": float(p1[0]),
                    "y2": float(p1[1]),
                    "length_px": float(distance(p0, p1)),
                    "dynamic": bool(self._cut_dynamic_enabled(cut_id)),
                    "selected": bool(int(cut_id) == int(self.selected_cut_id or -1)),
                    "stack_names": ",".join(
                        str(item["stack_name"]) for item in memberships
                    ),
                }
            )
        return rows

    def _cut_geometry_rows(
        self, cut_id: int, t_indices: np.ndarray | list[int]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for t_idx in np.asarray(t_indices, dtype=np.int64):
            p0, p1 = self._cut_geometry_for_frame(cut_id, int(t_idx))
            rows.append(
                {
                    "t_index": int(t_idx),
                    "x1": float(p0[0]),
                    "y1": float(p0[1]),
                    "x2": float(p1[0]),
                    "y2": float(p1[1]),
                    "length_px": float(distance(p0, p1)),
                }
            )
        return rows

    def _event_trace_rows(
        self,
        cut_id: int,
        event: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if cut_id not in self.cuts:
            return []
        if meta is None:
            _td, meta = self._cut_td(cut_id)
        if meta is None:
            return []
        distances = np.asarray(meta.get("distances", []), dtype=np.float64)
        if distances.size == 0:
            return []
        dist_index = np.arange(distances.size, dtype=np.float64)
        analysis = event.get("analysis") or {}
        series_specs = (
            (
                "source",
                np.asarray(event.get("source_t_idx", []), dtype=np.float64),
                np.asarray(event.get("source_y_idx", []), dtype=np.float64),
            ),
            (
                "wave",
                np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64),
                np.asarray(analysis.get("wave_y_idx", []), dtype=np.float64),
            ),
        )
        rows: list[dict[str, Any]] = []
        max_dist_index = float(dist_index[-1]) if dist_index.size else 0.0
        for series_name, t_values, y_values in series_specs:
            if t_values.size == 0 or y_values.size != t_values.size:
                continue
            for point_index, (t_value, y_value) in enumerate(
                zip(t_values, y_values), start=1
            ):
                if not np.isfinite(t_value) or not np.isfinite(y_value):
                    continue
                dist_idx = clamp_value(float(y_value), 0.0, max_dist_index)
                dist_px = float(np.interp(dist_idx, dist_index, distances))
                frame_idx = clamp_int(int(round(float(t_value))), 0, self.nt - 1)
                p0, p1 = self._cut_geometry_for_frame(cut_id, frame_idx)
                map_x, map_y = self._point_on_segment_distance(p0, p1, dist_px)
                rows.append(
                    {
                        "series": series_name,
                        "point_index": int(point_index),
                        "t_idx": float(t_value),
                        "frame_idx": int(frame_idx),
                        "dist_idx": float(dist_idx),
                        "dist_px": float(dist_px),
                        "map_x": float(map_x),
                        "map_y": float(map_y),
                        "cut_x1": float(p0[0]),
                        "cut_y1": float(p0[1]),
                        "cut_x2": float(p1[0]),
                        "cut_y2": float(p1[1]),
                    }
                )
        rows.sort(key=lambda row: (str(row["series"]), float(row["t_idx"]), int(row["point_index"])))
        return rows

    def _event_summary_rows_for_cut(
        self, cut_id: int, meta: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        state = self._cut_analysis_snapshot(cut_id)
        rows: list[dict[str, Any]] = []
        for event in state.get("wavelet_events") or []:
            self._ensure_wavelet_event_fields(event)
            trace_rows = self._event_trace_rows(cut_id, event, meta=meta)
            preferred_rows = [row for row in trace_rows if row["series"] == "wave"]
            if not preferred_rows:
                preferred_rows = [row for row in trace_rows if row["series"] == "source"]
            first_row = preferred_rows[0] if preferred_rows else None
            last_row = preferred_rows[-1] if preferred_rows else None
            analysis = event.get("analysis") or {}
            rows.append(
                {
                    "event_id": int(event.get("event_id", -1)),
                    "status": self._td_window_wavelet_event_status(event),
                    "origin": str(event.get("origin", "")),
                    "class_name": str(event.get("propagation_class") or ""),
                    "group_id": str(event.get("link_group_id") or ""),
                    "thread_index": int(analysis.get("thread_index", -1)),
                    "seg_id": int(analysis.get("seg_id", -1)),
                    "wseg_id": int(analysis.get("wseg_id", -1)),
                    "period_s": float(analysis.get("peak_period_s", float("nan"))),
                    "amp_arcsec": float(analysis.get("fit_amp_arcsec", float("nan"))),
                    "duration_s": float(analysis.get("duration_s", float("nan"))),
                    "power_ratio": float(analysis.get("power_ratio", float("nan"))),
                    "confidence": float(self._wavelet_event_confidence_score(event)),
                    "qa_flags": ",".join(self._td_window_wavelet_event_qa_flags(event)),
                    "review_locked": bool(event.get("review_locked")),
                    "counted": bool(self._td_window_wavelet_event_is_counted(event)),
                    "link_count": int(self._wavelet_event_link_count(event)),
                    "review_notes": str(event.get("review_notes", "")),
                    "point_count": int(len(preferred_rows)),
                    "start_t_idx": (
                        float(first_row["t_idx"]) if first_row is not None else float("nan")
                    ),
                    "end_t_idx": (
                        float(last_row["t_idx"]) if last_row is not None else float("nan")
                    ),
                    "start_x": (
                        float(first_row["map_x"]) if first_row is not None else float("nan")
                    ),
                    "start_y": (
                        float(first_row["map_y"]) if first_row is not None else float("nan")
                    ),
                    "end_x": (
                        float(last_row["map_x"]) if last_row is not None else float("nan")
                    ),
                    "end_y": (
                        float(last_row["map_y"]) if last_row is not None else float("nan")
                    ),
                }
            )
        return rows

    def _fits_string_column(self, name: str, values: list[Any]) -> fits.Column:
        encoded_values = [
            str("" if value is None else value).encode("utf-8", errors="replace")
            for value in values
        ]
        width = max((len(item) for item in encoded_values), default=1)
        array = np.asarray(encoded_values, dtype=f"S{max(width, 1)}")
        return fits.Column(name=name, format=f"{max(width, 1)}A", array=array)

    def _saved_fits_file_items(self) -> list[dict[str, Any]]:
        export_dir = Path(str(self.export_dir_var.get()).strip() or self.cube_path.parent).expanduser()
        if not export_dir.exists() or not export_dir.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(export_dir.glob("*.fits")):
            try:
                header = fits.getheader(path, ext=0)
            except Exception:
                continue
            exptype = str(header.get("EXPTYPE", "") or "").strip() or "UNKNOWN"
            cut_id = header.get("CUTID")
            cut_name = str(header.get("CUTNAME", "") or "").strip()
            event_id = header.get("EVENTID")
            items.append(
                {
                    "path": path.resolve(),
                    "name": path.name,
                    "exptype": exptype,
                    "cut_id": (None if cut_id is None else int(cut_id)),
                    "cut_name": cut_name,
                    "event_id": (None if event_id is None else int(event_id)),
                    "t_ini": header.get("TINI"),
                    "t_fin": header.get("TFIN"),
                    "t_index": header.get("TINDEX"),
                }
            )
        type_order = {"TD_CUT": 0, "TD_TRACE": 1, "MAP_FRAME": 2}
        items.sort(
            key=lambda item: (
                type_order.get(str(item["exptype"]), 99),
                int(item["cut_id"]) if item["cut_id"] is not None else 10**9,
                int(item["event_id"]) if item["event_id"] is not None else 10**9,
                str(item["name"]).lower(),
            )
        )
        return items

    def _saved_fits_item_label(self, item: dict[str, Any]) -> str:
        exptype = str(item.get("exptype", "") or "UNKNOWN")
        name = str(item.get("name", ""))
        if exptype == "TD_CUT":
            return (
                f"TD | cut {int(item['cut_id']) if item.get('cut_id') is not None else '-'} "
                f"{str(item.get('cut_name') or '').strip() or '-'} | "
                f"t={item.get('t_ini', '-')}-{item.get('t_fin', '-')} | {name}"
            )
        if exptype == "TD_TRACE":
            return (
                f"Trace | cut {int(item['cut_id']) if item.get('cut_id') is not None else '-'} "
                f"evt {int(item['event_id']) if item.get('event_id') is not None else '-'} | {name}"
            )
        if exptype == "MAP_FRAME":
            return f"Map | t={item.get('t_index', '-')} | {name}"
        return f"{exptype} | {name}"

    def _fits_table_cell_display(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8", errors="replace").strip()
        if isinstance(value, np.ndarray):
            flat = np.ravel(value)
            preview = ",".join(self._fits_table_cell_display(item) for item in flat[:6])
            if flat.size > 6:
                preview += ",..."
            return preview
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.6g}" if np.isfinite(value) else "nan"
        return str(value)

    def _load_saved_fits_product(self, path: Path) -> dict[str, Any]:
        with fits.open(path, memmap=True) as hdul:
            primary = hdul[0]
            header = primary.header.copy()
            data = None if primary.data is None else np.asarray(primary.data, dtype=np.float32)
            tables: dict[str, dict[str, Any]] = {}
            for index, hdu in enumerate(hdul[1:], start=1):
                if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) or hdu.data is None:
                    continue
                table_name = str(hdu.name or f"HDU{index}")
                columns = [str(name) for name in (hdu.columns.names or [])]
                tables[table_name] = {
                    "name": table_name,
                    "columns": columns,
                    "data": hdu.data.copy(),
                    "header": hdu.header.copy(),
                }
        return {
            "path": path,
            "header": header,
            "data": data,
            "tables": tables,
        }

    def _saved_fits_summary_text(self, product: dict[str, Any]) -> str:
        header = product["header"]
        exptype = str(header.get("EXPTYPE", "") or "UNKNOWN")
        lines = [
            f"{product['path']}",
            f"type={exptype} | cube={str(header.get('SRCNAME', '-'))} | axis={str(header.get('AXORDER', '-'))}",
        ]
        if header.get("CUTID") is not None:
            lines.append(
                f"cut={int(header['CUTID'])} {str(header.get('CUTNAME', '') or '').strip() or '-'}"
            )
        if header.get("EVENTID") is not None:
            lines.append(f"event={int(header['EVENTID'])}")
        if header.get("TINI") is not None or header.get("TFIN") is not None:
            lines.append(
                f"td range={header.get('TINI', '-')}:{header.get('TFIN', '-')}:{header.get('STRIDE', '-')}"
            )
        if header.get("TINDEX") is not None:
            lines.append(f"map t={header.get('TINDEX')}")
        if header.get("WIDTH") is not None:
            lines.append(
                f"width={header.get('WIDTH')} {str(header.get('WEIGHT', '') or '').strip()}"
            )
        table_names = list(product.get("tables", {}).keys())
        lines.append(
            "tables=" + (", ".join(table_names) if table_names else "none")
        )
        return " | ".join(lines)

    def _saved_fits_plot_product(self, browser: dict[str, Any], product: dict[str, Any]) -> None:
        ax = browser["axis"]
        ax.clear()
        header = product["header"]
        data = product.get("data")
        exptype = str(header.get("EXPTYPE", "") or "UNKNOWN")
        if data is None or data.ndim != 2:
            ax.text(
                0.5,
                0.5,
                f"{exptype}\nNo 2D image in primary HDU.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            browser["canvas"].draw_idle()
            return

        vmin, vmax = frame_limits(data)
        if exptype == "TD_CUT":
            time_table = product.get("tables", {}).get("TIME_AXIS")
            dist_table = product.get("tables", {}).get("DIST_AXIS")
            t_values = (
                np.asarray(time_table["data"]["T_INDEX"], dtype=np.float64)
                if time_table is not None and "T_INDEX" in time_table["columns"]
                else np.arange(data.shape[0], dtype=np.float64)
            )
            d_values = (
                np.asarray(dist_table["data"]["DIST_PX"], dtype=np.float64)
                if dist_table is not None and "DIST_PX" in dist_table["columns"]
                else np.arange(data.shape[1], dtype=np.float64)
            )
            t_lo = float(t_values[0]) if t_values.size else 0.0
            t_hi = float(t_values[-1]) if t_values.size > 1 else float(t_lo + 1.0)
            d_lo = float(d_values[0]) if d_values.size else 0.0
            d_hi = float(d_values[-1]) if d_values.size > 1 else float(d_lo + 1.0)
            ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="gray",
                interpolation="nearest",
                extent=[d_lo, d_hi, t_lo, t_hi],
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("distance [pixel]")
            ax.set_ylabel("time index")
        else:
            ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="gray",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
        ax.set_title(
            f"{exptype} | {str(header.get('CUTNAME', '') or product['path'].name)}",
            fontsize=10,
        )
        browser["figure"].tight_layout()
        browser["canvas"].draw_idle()

    def _refresh_saved_fits_table(self) -> None:
        browser = self.saved_fits_window
        if browser is None:
            return
        product = browser.get("loaded_product")
        tree = browser.get("table_tree")
        summary_var = browser.get("table_summary_var")
        if tree is None or summary_var is None:
            return
        children = tree.get_children()
        if children:
            tree.delete(*children)
        if product is None:
            tree.configure(columns=())
            summary_var.set("Select a saved FITS product.")
            return
        table_name = str(browser["table_var"].get() or "")
        tables = product.get("tables", {})
        if table_name not in tables:
            tree.configure(columns=())
            summary_var.set("No table selected.")
            return
        table = tables[table_name]
        columns = [str(column) for column in table.get("columns", [])]
        tree.configure(columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=110, stretch=True)
        data = table.get("data")
        max_rows = 2000
        row_count = 0 if data is None else int(len(data))
        if data is not None:
            for row in data[:max_rows]:
                tree.insert(
                    "",
                    "end",
                    values=tuple(
                        self._fits_table_cell_display(row[column]) for column in columns
                    ),
                )
        summary_var.set(
            f"{table_name}: {row_count} row(s)"
            + (f" | showing first {max_rows}" if row_count > max_rows else "")
        )

    def _on_saved_fits_table_select(self, _event: Any = None) -> None:
        self._refresh_saved_fits_table()

    def _saved_fits_selected_item(self) -> dict[str, Any] | None:
        browser = self.saved_fits_window
        if browser is None:
            return None
        selection = browser["file_listbox"].curselection()
        items = browser.get("items", [])
        if not selection:
            return items[0] if items else None
        index = int(selection[0])
        return items[index] if 0 <= index < len(items) else None

    def _on_saved_fits_file_select(self, _event: Any = None) -> None:
        browser = self.saved_fits_window
        if browser is None:
            return
        item = self._saved_fits_selected_item()
        if item is None:
            browser["loaded_product"] = None
            browser["detail_var"].set("No saved FITS selected.")
            browser["table_selector"]["values"] = []
            browser["table_var"].set("")
            self._refresh_saved_fits_table()
            return
        try:
            product = self._load_saved_fits_product(Path(item["path"]))
        except Exception as exc:
            browser["loaded_product"] = None
            browser["detail_var"].set(
                f"Could not read {item['name']}: {type(exc).__name__}: {exc}"
            )
            browser["table_selector"]["values"] = []
            browser["table_var"].set("")
            self._refresh_saved_fits_table()
            return
        browser["loaded_product"] = product
        browser["detail_var"].set(self._saved_fits_summary_text(product))
        table_names = list(product.get("tables", {}).keys())
        browser["table_selector"]["values"] = table_names
        current_table = str(browser["table_var"].get() or "")
        if current_table not in table_names:
            browser["table_var"].set(table_names[0] if table_names else "")
        self._saved_fits_plot_product(browser, product)
        self._refresh_saved_fits_table()

    def _open_saved_fits_browser(self) -> None:
        existing = self.saved_fits_window
        if existing is not None:
            top = existing.get("top")
            if top is not None and top.winfo_exists():
                top.deiconify()
                top.lift()
                self._refresh_saved_fits_browser()
                return
            self.saved_fits_window = None

        top = self.tk.Toplevel(self.root)
        top.title("Saved FITS Browser")
        top.geometry("1540x930")
        top.rowconfigure(1, weight=1)
        top.columnconfigure(0, weight=1)

        header = self.ttk.Frame(top, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        summary_var = self.tk.StringVar(value="")
        detail_var = self.tk.StringVar(value="")
        self.ttk.Label(
            header, textvariable=summary_var, justify="left", wraplength=1500
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header, textvariable=detail_var, justify="left", wraplength=1500
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)

        files_frame = self.ttk.LabelFrame(body, text="Saved FITS", padding=6)
        files_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        files_frame.rowconfigure(0, weight=1)
        files_frame.columnconfigure(0, weight=1)
        file_listbox = self.tk.Listbox(files_frame, width=52, exportselection=False)
        file_listbox.grid(row=0, column=0, sticky="nsew")
        file_scroll = self.ttk.Scrollbar(
            files_frame, orient="vertical", command=file_listbox.yview
        )
        file_scroll.grid(row=0, column=1, sticky="ns")
        file_listbox.configure(yscrollcommand=file_scroll.set)
        file_listbox.bind("<<ListboxSelect>>", self._on_saved_fits_file_select)
        self.ttk.Button(
            files_frame,
            text="Refresh",
            command=self._refresh_saved_fits_browser,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        view_frame = self.ttk.Frame(body)
        view_frame.grid(row=0, column=1, sticky="nsew")
        view_frame.rowconfigure(0, weight=1)
        view_frame.rowconfigure(2, weight=1)
        view_frame.columnconfigure(0, weight=1)

        plot_frame = self.ttk.LabelFrame(view_frame, text="Preview", padding=6)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        fig = self.Figure(figsize=(8.6, 4.8), dpi=120)
        axis = fig.add_subplot(111)
        canvas = self.FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_header = self.ttk.Frame(view_frame, padding=(0, 8, 0, 4))
        table_header.grid(row=1, column=0, sticky="ew")
        table_header.columnconfigure(3, weight=1)
        table_var = self.tk.StringVar(value="")
        table_summary_var = self.tk.StringVar(value="")
        self.ttk.Label(table_header, text="Table").grid(row=0, column=0, sticky="w")
        table_selector = self.ttk.Combobox(
            table_header,
            textvariable=table_var,
            state="readonly",
            width=22,
        )
        table_selector.grid(row=0, column=1, sticky="w", padx=(6, 12))
        table_selector.bind("<<ComboboxSelected>>", self._on_saved_fits_table_select)
        self.ttk.Label(
            table_header, textvariable=table_summary_var, justify="left"
        ).grid(row=0, column=3, sticky="w")

        table_frame = self.ttk.LabelFrame(view_frame, text="Table Preview", padding=6)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        table_tree = self.ttk.Treeview(table_frame, show="headings", height=14)
        table_tree.grid(row=0, column=0, sticky="nsew")
        table_y = self.ttk.Scrollbar(table_frame, orient="vertical", command=table_tree.yview)
        table_y.grid(row=0, column=1, sticky="ns")
        table_x = self.ttk.Scrollbar(table_frame, orient="horizontal", command=table_tree.xview)
        table_x.grid(row=1, column=0, sticky="ew")
        table_tree.configure(yscrollcommand=table_y.set, xscrollcommand=table_x.set)

        self.saved_fits_window = {
            "top": top,
            "summary_var": summary_var,
            "detail_var": detail_var,
            "file_listbox": file_listbox,
            "items": [],
            "loaded_product": None,
            "figure": fig,
            "axis": axis,
            "canvas": canvas,
            "table_var": table_var,
            "table_selector": table_selector,
            "table_tree": table_tree,
            "table_summary_var": table_summary_var,
        }
        top.protocol("WM_DELETE_WINDOW", self._close_saved_fits_browser)
        self._refresh_saved_fits_browser()

    def _close_saved_fits_browser(self) -> None:
        existing = self.saved_fits_window
        if existing is None:
            return
        top = existing.get("top")
        if top is not None and top.winfo_exists():
            top.destroy()
        self.saved_fits_window = None

    def _refresh_saved_fits_browser(self) -> None:
        browser = self.saved_fits_window
        if browser is None:
            return
        top = browser.get("top")
        if top is None or not top.winfo_exists():
            self.saved_fits_window = None
            return
        items = self._saved_fits_file_items()
        browser["items"] = items
        listbox = browser["file_listbox"]
        listbox.delete(0, self.tk.END)
        for item in items:
            listbox.insert(self.tk.END, self._saved_fits_item_label(item))
        browser["summary_var"].set(
            f"Export folder: {str(self.export_dir_var.get()).strip() or self.cube_path.parent} | "
            f"saved FITS found: {len(items)}"
        )
        if items:
            current_selection = listbox.curselection()
            index = int(current_selection[0]) if current_selection else 0
            index = clamp_int(index, 0, len(items) - 1)
            listbox.selection_clear(0, self.tk.END)
            listbox.selection_set(index)
            listbox.activate(index)
        else:
            browser["loaded_product"] = None
            browser["detail_var"].set("No saved FITS found in the export folder.")
            browser["table_selector"]["values"] = []
            browser["table_var"].set("")
            self._refresh_saved_fits_table()
            axis = browser["axis"]
            axis.clear()
            axis.text(
                0.5,
                0.5,
                "No saved FITS found.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            browser["canvas"].draw_idle()
            return
        self._on_saved_fits_file_select()

    def _choose_cube_dialog(self) -> tuple[Path, str] | None:
        top = self.tk.Toplevel(self.root)
        top.title("Open Input Cube")
        top.geometry("720x220")
        top.transient(self.root)
        top.grab_set()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        frame = self.ttk.Frame(top, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        path_var = self.tk.StringVar(value=str(self.cube_path))
        order_var = self.tk.StringVar(value=cube_axis_order_display_label(self.cube_axis_order))
        info_var = self.tk.StringVar(
            value=(
                "Choose the raw input cube and how its file axes map to T,Y,X. "
                "Examples: 123|TYX, 213|YTX, 312|XTY."
            )
        )
        result: dict[str, Any] = {}

        def _browse() -> None:
            selected = self.filedialog.askopenfilename(
                title="Open FITS cube",
                initialdir=str(Path(path_var.get()).expanduser().parent),
                filetypes=[("FITS", "*.fits *.fit *.fts"), ("All files", "*.*")],
            )
            if selected:
                path_var.set(selected)

        def _accept() -> None:
            cube_text = str(path_var.get()).strip()
            if not cube_text:
                self.messagebox.showerror("Open Input Cube", "Choose a FITS cube first.")
                return
            cube_path = Path(cube_text).expanduser()
            if not cube_path.exists():
                self.messagebox.showerror("Open Input Cube", "The selected cube does not exist.")
                return
            try:
                order = normalize_cube_axis_order(str(order_var.get()).split("|", 1)[0].strip())
            except Exception as exc:
                self.messagebox.showerror("Open Input Cube", str(exc))
                return
            result["value"] = (cube_path.resolve(), order)
            top.destroy()

        self.ttk.Label(frame, text="Cube").grid(row=0, column=0, sticky="w")
        self.ttk.Entry(frame, textvariable=path_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 8)
        )
        self.ttk.Button(frame, text="Browse", command=_browse).grid(row=0, column=2, sticky="ew")

        self.ttk.Label(frame, text="Axis order").grid(row=1, column=0, sticky="w", pady=(12, 0))
        order_values = [cube_axis_order_display_label(order) for order in CUBE_AXIS_ORDERS]
        order_box = self.ttk.Combobox(
            frame,
            textvariable=order_var,
            values=order_values,
            state="readonly",
            width=24,
        )
        order_box.grid(row=1, column=1, sticky="w", padx=(8, 8), pady=(12, 0))
        self.ttk.Label(
            frame,
            textvariable=info_var,
            justify="left",
            wraplength=660,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))

        buttons = self.ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=3, sticky="e", pady=(16, 0))
        self.ttk.Button(buttons, text="Cancel", command=top.destroy).grid(row=0, column=0, padx=(0, 8))
        self.ttk.Button(buttons, text="Open Cube", command=_accept).grid(row=0, column=1)

        top.bind("<Return>", lambda _event: _accept())
        top.bind("<Escape>", lambda _event: top.destroy())
        top.wait_window()
        return result.get("value")

    def _choose_input_cube_and_restart(self) -> None:
        selection = self._choose_cube_dialog()
        if selection is None:
            return
        cube_path, cube_axis_order = selection
        if cube_path == self.cube_path and cube_axis_order == self.cube_axis_order:
            self._set_status("Cube and axis order are unchanged.")
            return
        script_path = Path(__file__).resolve()
        argv = [
            sys.executable,
            str(script_path),
            "--cube",
            str(cube_path),
            "--cube-order",
            str(cube_axis_order),
        ]
        if self.root.winfo_exists():
            self.root.destroy()
        os.execv(sys.executable, argv)

    def _write_current_map_fits(self, save_path: Path, frame_idx: int) -> None:
        frame_idx = clamp_int(int(frame_idx), 0, self.nt - 1)
        frame = np.asarray(self.cube[frame_idx], dtype=np.float32)
        cut_rows = self._current_map_cut_rows(frame_idx)
        header = fits.Header()
        header["EXPTYPE"] = ("MAP_FRAME", "Current frame of the source cube")
        header["SRCNAME"] = (self.cube_path.name, "Source cube filename")
        header["AXORDER"] = (self.cube_axis_order, "Input cube axis order")
        header["TINDEX"] = (int(frame_idx), "Time index in the cube")
        header["NXCUBE"] = (int(self.nx), "Cube size X")
        header["NYCUBE"] = (int(self.ny), "Cube size Y")
        header["MAPSXY"] = (bool(self.map_swap_xy_var.get()), "Map XY swapped in GUI")
        header["MAPFLPX"] = (bool(self.map_flip_x_var.get()), "Map X flipped in GUI")
        header["MAPFLPY"] = (bool(self.map_flip_y_var.get()), "Map Y flipped in GUI")
        header["SELCUT"] = (int(self.selected_cut_id or -1), "Selected cut id")
        hdus: list[Any] = [fits.PrimaryHDU(data=frame, header=header)]
        if cut_rows:
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(
                            name="CUT_ID",
                            format="K",
                            array=np.asarray([row["cut_id"] for row in cut_rows], dtype=np.int64),
                        ),
                        self._fits_string_column(
                            "CUT_NAME", [row["cut_name"] for row in cut_rows]
                        ),
                        fits.Column(
                            name="X1",
                            format="D",
                            array=np.asarray([row["x1"] for row in cut_rows], dtype=np.float64),
                        ),
                        fits.Column(
                            name="Y1",
                            format="D",
                            array=np.asarray([row["y1"] for row in cut_rows], dtype=np.float64),
                        ),
                        fits.Column(
                            name="X2",
                            format="D",
                            array=np.asarray([row["x2"] for row in cut_rows], dtype=np.float64),
                        ),
                        fits.Column(
                            name="Y2",
                            format="D",
                            array=np.asarray([row["y2"] for row in cut_rows], dtype=np.float64),
                        ),
                        fits.Column(
                            name="LENGTHPX",
                            format="D",
                            array=np.asarray(
                                [row["length_px"] for row in cut_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="DYNAMIC",
                            format="L",
                            array=np.asarray(
                                [row["dynamic"] for row in cut_rows], dtype=bool
                            ),
                        ),
                        fits.Column(
                            name="SELECTED",
                            format="L",
                            array=np.asarray(
                                [row["selected"] for row in cut_rows], dtype=bool
                            ),
                        ),
                        self._fits_string_column(
                            "STACKS", [row["stack_names"] for row in cut_rows]
                        ),
                    ],
                    name="CUTS_AT_T",
                )
            )
        fits.HDUList(hdus).writeto(save_path, overwrite=True)

    def _draw_export_map_axis(
        self,
        ax: Any,
        frame_idx: int,
        *,
        cut_ids: list[int] | None = None,
        focus_cut_id: int | None = None,
        trace_rows: list[dict[str, Any]] | None = None,
        title: str | None = None,
    ) -> None:
        frame_idx = clamp_int(int(frame_idx), 0, self.nt - 1)
        frame = np.asarray(self.cube[frame_idx], dtype=np.float32)
        if self.map_swap_xy_var.get():
            frame = frame.T
        vmin, vmax = frame_limits(frame)
        ax.clear()
        ax.imshow(
            frame,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ordered_cut_ids = (
            [int(cut_id) for cut_id in cut_ids if int(cut_id) in self.cuts]
            if cut_ids is not None
            else sorted(self.cuts.keys())
        )
        for cut_id in ordered_cut_ids:
            cut = self.cuts.get(int(cut_id))
            if cut is None or not cut.visible:
                continue
            p0_now, p1_now = self._cut_geometry_for_frame(int(cut_id), frame_idx)
            p0_disp = self._map_data_to_display(p0_now[0], p0_now[1])
            p1_disp = self._map_data_to_display(p1_now[0], p1_now[1])
            is_focus = int(cut_id) == int(focus_cut_id or -1)
            ax.plot(
                [p0_disp[0], p1_disp[0]],
                [p0_disp[1], p1_disp[1]],
                color=cut.color,
                linewidth=(3.0 if is_focus else 1.7),
                alpha=(1.0 if is_focus else 0.65),
                linestyle="--" if self._cut_dynamic_enabled(int(cut_id)) else "-",
            )
            ax.scatter(
                [p0_disp[0], p1_disp[0]],
                [p0_disp[1], p1_disp[1]],
                color=cut.color,
                s=(34 if is_focus else 20),
                alpha=(1.0 if is_focus else 0.8),
                zorder=3,
            )
            mid_x = 0.5 * (p0_disp[0] + p1_disp[0])
            mid_y = 0.5 * (p0_disp[1] + p1_disp[1])
            label = f"{cut.name}"
            if is_focus:
                label += " [focus]"
            ax.text(
                mid_x,
                mid_y,
                label,
                color=cut.color,
                fontsize=8,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
            )

        if trace_rows:
            series_styles = {
                "source": {"color": "deepskyblue", "marker": "o", "linewidth": 1.6},
                "wave": {"color": "gold", "marker": "o", "linewidth": 2.4},
            }
            for series_name, style in series_styles.items():
                rows = [row for row in trace_rows if str(row.get("series")) == series_name]
                if not rows:
                    continue
                rows.sort(key=lambda row: (float(row["t_idx"]), int(row["point_index"])))
                xs: list[float] = []
                ys: list[float] = []
                for row in rows:
                    x_disp, y_disp = self._map_data_to_display(
                        float(row["map_x"]),
                        float(row["map_y"]),
                    )
                    xs.append(float(x_disp))
                    ys.append(float(y_disp))
                ax.plot(
                    xs,
                    ys,
                    color=str(style["color"]),
                    linewidth=float(style["linewidth"]),
                    alpha=0.9,
                )
                ax.scatter(
                    xs,
                    ys,
                    color=str(style["color"]),
                    s=18,
                    marker=str(style["marker"]),
                    alpha=0.95,
                    zorder=4,
                )
                ax.scatter(
                    [xs[0]],
                    [ys[0]],
                    color=str(style["color"]),
                    s=52,
                    marker="s",
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )

        disp_nx, disp_ny = self._map_display_shape()
        if self.map_flip_x_var.get():
            ax.set_xlim(disp_nx - 0.5, -0.5)
        else:
            ax.set_xlim(-0.5, disp_nx - 0.5)
        if self.map_flip_y_var.get():
            ax.set_ylim(disp_ny - 0.5, -0.5)
        else:
            ax.set_ylim(-0.5, disp_ny - 0.5)
        if self.map_swap_xy_var.get():
            ax.set_xlabel("y [pixel]")
            ax.set_ylabel("x [pixel]")
        else:
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
        ax.set_title(title or f"Map at t={frame_idx}", fontsize=10)

    def _save_export_figure_png(self, fig: Any, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
                category=UserWarning,
            )
            fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.clear()

    def _write_current_map_png(self, save_path: Path, frame_idx: int) -> None:
        fig = self.Figure(figsize=(7.4, 6.8), dpi=160)
        ax = fig.add_subplot(111)
        self._draw_export_map_axis(
            ax,
            frame_idx,
            cut_ids=sorted(self.cuts.keys()),
            focus_cut_id=self.selected_cut_id,
            title=f"Map overview at t={int(frame_idx)}",
        )
        self._save_export_figure_png(fig, save_path)

    def _write_cut_quicklook_png(
        self,
        cut_id: int,
        save_path: Path,
        *,
        map_cut_ids: list[int] | None = None,
        selected_event_id: int | None = None,
    ) -> None:
        if cut_id not in self.cuts:
            raise ValueError(f"Unknown cut id {cut_id}.")
        frame_idx = int(self.t_visual_var.get())
        trace_rows: list[dict[str, Any]] = []
        if selected_event_id is not None:
            event = self._wavelet_event_ref_by_cut(cut_id, int(selected_event_id))
            if event is not None:
                td, meta = self._cut_td(cut_id)
                if td is not None and meta is not None and "error" not in meta:
                    trace_rows = self._event_trace_rows(cut_id, event, meta=meta)

        fig = self.Figure(figsize=(12.8, 5.8), dpi=160)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.24)
        map_ax = fig.add_subplot(grid[0, 0])
        td_ax = fig.add_subplot(grid[0, 1])

        context_cut_ids = (
            [int(current_cut_id) for current_cut_id in map_cut_ids if int(current_cut_id) in self.cuts]
            if map_cut_ids is not None
            else [int(cut_id)]
        )
        if int(cut_id) not in context_cut_ids:
            context_cut_ids.append(int(cut_id))

        if selected_event_id is not None:
            map_title = (
                f"Map trace overview | cut {int(cut_id)} | event {int(selected_event_id)} | "
                f"t={frame_idx}"
            )
        elif len(context_cut_ids) > 1:
            map_title = f"Stack/cuts overview at t={frame_idx}"
        else:
            map_title = f"Cut overview at t={frame_idx}"
        self._draw_export_map_axis(
            map_ax,
            frame_idx,
            cut_ids=context_cut_ids,
            focus_cut_id=cut_id,
            trace_rows=trace_rows,
            title=map_title,
        )
        self._draw_cut_td_axis(
            td_ax,
            cut_id,
            use_zoom=False,
            title_fontsize=10.0,
            selected_event_id=selected_event_id,
            title_prefix=("Trace: " if selected_event_id is not None else ""),
        )
        self._save_export_figure_png(fig, save_path)

    def _write_cut_td_fits(self, cut_id: int, save_path: Path) -> None:
        cut = self.cuts.get(cut_id)
        if cut is None:
            raise ValueError(f"Unknown cut id {cut_id}.")
        td, meta = self._cut_td(cut_id)
        if td is None or meta is None:
            raise ValueError(f"TD map unavailable for cut {cut.name}.")
        if "error" in meta:
            raise ValueError(str(meta["error"]))

        params = self._cut_td_params(cut_id)
        current_t = int(self.t_visual_var.get())
        p0_now, p1_now = self._cut_geometry_for_frame(cut_id, current_t)
        geometry_rows = self._cut_geometry_rows(cut_id, meta["t_indices"])
        event_rows = self._event_summary_rows_for_cut(cut_id, meta=meta)
        trace_rows: list[dict[str, Any]] = []
        for event in self._cut_analysis_snapshot(cut_id).get("wavelet_events") or []:
            trace_rows.extend(self._event_trace_rows(cut_id, event, meta=meta))

        header = fits.Header()
        header["EXPTYPE"] = ("TD_CUT", "Time-distance map for one cut")
        header["SRCNAME"] = (self.cube_path.name, "Source cube filename")
        header["AXORDER"] = (self.cube_axis_order, "Input cube axis order")
        header["CUTID"] = (int(cut_id), "Cut identifier")
        header["CUTNAME"] = (str(cut.name), "Cut label")
        header["TINI"] = (int(params["t_ini"]), "Initial time index")
        header["TFIN"] = (int(params["t_fin"]), "Final time index")
        header["STRIDE"] = (int(params["stride"]), "Time stride")
        header["WIDTH"] = (int(params["width"]), "Cut width")
        header["WEIGHT"] = (str(params["weighting"]), "Width weighting mode")
        header["CURT"] = (int(current_t), "GUI current time index")
        header["X1_CUR"] = (float(p0_now[0]), "Cut x1 at current t")
        header["Y1_CUR"] = (float(p0_now[1]), "Cut y1 at current t")
        header["X2_CUR"] = (float(p1_now[0]), "Cut x2 at current t")
        header["Y2_CUR"] = (float(p1_now[1]), "Cut y2 at current t")
        header["DYNGEOM"] = (
            bool(self._cut_dynamic_enabled(cut_id)),
            "Cut uses time-varying geometry",
        )
        header["NTRACE"] = (int(len(event_rows)), "Event/trace count")

        hdus: list[Any] = [fits.PrimaryHDU(data=np.asarray(td, dtype=np.float32), header=header)]
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="T_INDEX",
                        format="K",
                        array=np.asarray(meta["t_indices"], dtype=np.int64),
                    )
                ],
                name="TIME_AXIS",
            )
        )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="DIST_PX",
                        format="D",
                        array=np.asarray(meta["distances"], dtype=np.float64),
                    )
                ],
                name="DIST_AXIS",
            )
        )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="T_INDEX",
                        format="K",
                        array=np.asarray(
                            [row["t_index"] for row in geometry_rows], dtype=np.int64
                        ),
                    ),
                    fits.Column(
                        name="X1",
                        format="D",
                        array=np.asarray([row["x1"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="Y1",
                        format="D",
                        array=np.asarray([row["y1"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="X2",
                        format="D",
                        array=np.asarray([row["x2"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="Y2",
                        format="D",
                        array=np.asarray([row["y2"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="LENGTHPX",
                        format="D",
                        array=np.asarray(
                            [row["length_px"] for row in geometry_rows], dtype=np.float64
                        ),
                    ),
                ],
                name="CUT_GEOM",
            )
        )
        if event_rows:
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(
                            name="EVENT_ID",
                            format="K",
                            array=np.asarray(
                                [row["event_id"] for row in event_rows], dtype=np.int64
                            ),
                        ),
                        self._fits_string_column(
                            "STATUS", [row["status"] for row in event_rows]
                        ),
                        self._fits_string_column(
                            "ORIGIN", [row["origin"] for row in event_rows]
                        ),
                        self._fits_string_column(
                            "CLASS", [row["class_name"] for row in event_rows]
                        ),
                        self._fits_string_column(
                            "GROUP_ID", [row["group_id"] for row in event_rows]
                        ),
                        fits.Column(
                            name="THREAD",
                            format="K",
                            array=np.asarray(
                                [row["thread_index"] for row in event_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="SEG_ID",
                            format="K",
                            array=np.asarray(
                                [row["seg_id"] for row in event_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="WSEG_ID",
                            format="K",
                            array=np.asarray(
                                [row["wseg_id"] for row in event_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="PERIOD_S",
                            format="D",
                            array=np.asarray(
                                [row["period_s"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="AMP_ARCS",
                            format="D",
                            array=np.asarray(
                                [row["amp_arcsec"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="DUR_S",
                            format="D",
                            array=np.asarray(
                                [row["duration_s"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="POWRATIO",
                            format="D",
                            array=np.asarray(
                                [row["power_ratio"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="CONF",
                            format="D",
                            array=np.asarray(
                                [row["confidence"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        self._fits_string_column(
                            "QA_FLAGS", [row["qa_flags"] for row in event_rows]
                        ),
                        fits.Column(
                            name="LOCKED",
                            format="L",
                            array=np.asarray(
                                [row["review_locked"] for row in event_rows], dtype=bool
                            ),
                        ),
                        fits.Column(
                            name="COUNTED",
                            format="L",
                            array=np.asarray(
                                [row["counted"] for row in event_rows], dtype=bool
                            ),
                        ),
                        fits.Column(
                            name="LINK_CNT",
                            format="K",
                            array=np.asarray(
                                [row["link_count"] for row in event_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="POINTS",
                            format="K",
                            array=np.asarray(
                                [row["point_count"] for row in event_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="TSTART",
                            format="D",
                            array=np.asarray(
                                [row["start_t_idx"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="TEND",
                            format="D",
                            array=np.asarray(
                                [row["end_t_idx"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="XSTART",
                            format="D",
                            array=np.asarray(
                                [row["start_x"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="YSTART",
                            format="D",
                            array=np.asarray(
                                [row["start_y"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="XEND",
                            format="D",
                            array=np.asarray(
                                [row["end_x"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="YEND",
                            format="D",
                            array=np.asarray(
                                [row["end_y"] for row in event_rows], dtype=np.float64
                            ),
                        ),
                        self._fits_string_column(
                            "NOTES", [row["review_notes"] for row in event_rows]
                        ),
                    ],
                    name="WAVE_EVT",
                )
            )
        if trace_rows:
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        self._fits_string_column(
                            "SERIES", [row["series"] for row in trace_rows]
                        ),
                        fits.Column(
                            name="POINT_ID",
                            format="K",
                            array=np.asarray(
                                [row["point_index"] for row in trace_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="T_IDX",
                            format="D",
                            array=np.asarray([row["t_idx"] for row in trace_rows], dtype=np.float64),
                        ),
                        fits.Column(
                            name="FRAMEIDX",
                            format="K",
                            array=np.asarray(
                                [row["frame_idx"] for row in trace_rows], dtype=np.int64
                            ),
                        ),
                        fits.Column(
                            name="DISTIDX",
                            format="D",
                            array=np.asarray(
                                [row["dist_idx"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="DIST_PX",
                            format="D",
                            array=np.asarray(
                                [row["dist_px"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="MAP_X",
                            format="D",
                            array=np.asarray(
                                [row["map_x"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="MAP_Y",
                            format="D",
                            array=np.asarray(
                                [row["map_y"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="CUT_X1",
                            format="D",
                            array=np.asarray(
                                [row["cut_x1"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="CUT_Y1",
                            format="D",
                            array=np.asarray(
                                [row["cut_y1"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="CUT_X2",
                            format="D",
                            array=np.asarray(
                                [row["cut_x2"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                        fits.Column(
                            name="CUT_Y2",
                            format="D",
                            array=np.asarray(
                                [row["cut_y2"] for row in trace_rows], dtype=np.float64
                            ),
                        ),
                    ],
                    name="TRACE_PTS",
                )
            )
        fits.HDUList(hdus).writeto(save_path, overwrite=True)

    def _write_trace_event_fits(
        self,
        cut_id: int,
        event: dict[str, Any],
        save_path: Path,
    ) -> bool:
        cut = self.cuts.get(cut_id)
        if cut is None:
            return False
        td, meta = self._cut_td(cut_id)
        if td is None or meta is None or "error" in meta:
            return False
        trace_rows = self._event_trace_rows(cut_id, event, meta=meta)
        if not trace_rows:
            return False
        summary_rows = [
            row
            for row in self._event_summary_rows_for_cut(cut_id, meta=meta)
            if int(row["event_id"]) == int(event.get("event_id", -1))
        ]
        geometry_rows = self._cut_geometry_rows(
            cut_id,
            sorted({int(row["frame_idx"]) for row in trace_rows}),
        )
        header = fits.Header()
        header["EXPTYPE"] = ("TD_TRACE", "One wave/oscillation trace")
        header["SRCNAME"] = (self.cube_path.name, "Source cube filename")
        header["AXORDER"] = (self.cube_axis_order, "Input cube axis order")
        header["CUTID"] = (int(cut_id), "Cut identifier")
        header["CUTNAME"] = (str(cut.name), "Cut label")
        header["EVENTID"] = (int(event.get("event_id", -1)), "Event identifier")
        header["CURT"] = (int(self.t_visual_var.get()), "GUI current time index")
        header["NPOINTS"] = (int(len(trace_rows)), "Trace point count")

        hdus: list[Any] = [fits.PrimaryHDU(header=header)]
        if summary_rows:
            summary = summary_rows[0]
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(
                            name="EVENT_ID",
                            format="K",
                            array=np.asarray([summary["event_id"]], dtype=np.int64),
                        ),
                        self._fits_string_column("STATUS", [summary["status"]]),
                        self._fits_string_column("ORIGIN", [summary["origin"]]),
                        self._fits_string_column("CLASS", [summary["class_name"]]),
                        self._fits_string_column("GROUP_ID", [summary["group_id"]]),
                        fits.Column(
                            name="THREAD",
                            format="K",
                            array=np.asarray([summary["thread_index"]], dtype=np.int64),
                        ),
                        fits.Column(
                            name="PERIOD_S",
                            format="D",
                            array=np.asarray([summary["period_s"]], dtype=np.float64),
                        ),
                        fits.Column(
                            name="AMP_ARCS",
                            format="D",
                            array=np.asarray([summary["amp_arcsec"]], dtype=np.float64),
                        ),
                        fits.Column(
                            name="TSTART",
                            format="D",
                            array=np.asarray([summary["start_t_idx"]], dtype=np.float64),
                        ),
                        fits.Column(
                            name="TEND",
                            format="D",
                            array=np.asarray([summary["end_t_idx"]], dtype=np.float64),
                        ),
                        self._fits_string_column("QA_FLAGS", [summary["qa_flags"]]),
                        self._fits_string_column("NOTES", [summary["review_notes"]]),
                    ],
                    name="EVENT_SUM",
                )
            )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="T_INDEX",
                        format="K",
                        array=np.asarray(meta["t_indices"], dtype=np.int64),
                    )
                ],
                name="TIME_AXIS",
            )
        )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="DIST_PX",
                        format="D",
                        array=np.asarray(meta["distances"], dtype=np.float64),
                    )
                ],
                name="DIST_AXIS",
            )
        )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="T_INDEX",
                        format="K",
                        array=np.asarray(
                            [row["t_index"] for row in geometry_rows], dtype=np.int64
                        ),
                    ),
                    fits.Column(
                        name="X1",
                        format="D",
                        array=np.asarray([row["x1"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="Y1",
                        format="D",
                        array=np.asarray([row["y1"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="X2",
                        format="D",
                        array=np.asarray([row["x2"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="Y2",
                        format="D",
                        array=np.asarray([row["y2"] for row in geometry_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="LENGTHPX",
                        format="D",
                        array=np.asarray(
                            [row["length_px"] for row in geometry_rows], dtype=np.float64
                        ),
                    ),
                ],
                name="CUT_GEOM",
            )
        )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    self._fits_string_column("SERIES", [row["series"] for row in trace_rows]),
                    fits.Column(
                        name="POINT_ID",
                        format="K",
                        array=np.asarray(
                            [row["point_index"] for row in trace_rows], dtype=np.int64
                        ),
                    ),
                    fits.Column(
                        name="T_IDX",
                        format="D",
                        array=np.asarray([row["t_idx"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="FRAMEIDX",
                        format="K",
                        array=np.asarray(
                            [row["frame_idx"] for row in trace_rows], dtype=np.int64
                        ),
                    ),
                    fits.Column(
                        name="DISTIDX",
                        format="D",
                        array=np.asarray([row["dist_idx"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="DIST_PX",
                        format="D",
                        array=np.asarray([row["dist_px"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="MAP_X",
                        format="D",
                        array=np.asarray([row["map_x"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="MAP_Y",
                        format="D",
                        array=np.asarray([row["map_y"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="CUT_X1",
                        format="D",
                        array=np.asarray([row["cut_x1"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="CUT_Y1",
                        format="D",
                        array=np.asarray([row["cut_y1"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="CUT_X2",
                        format="D",
                        array=np.asarray([row["cut_x2"] for row in trace_rows], dtype=np.float64),
                    ),
                    fits.Column(
                        name="CUT_Y2",
                        format="D",
                        array=np.asarray([row["cut_y2"] for row in trace_rows], dtype=np.float64),
                    ),
                ],
                name="TRACE_PTS",
            )
        )
        fits.HDUList(hdus).writeto(save_path, overwrite=True)
        return True

    def _export_current_map_fits(self) -> None:
        try:
            export_dir = self._resolved_export_dir()
            frame_idx = int(self.t_visual_var.get())
            save_path = export_dir / f"{self.cube_path.stem}_t{frame_idx:04d}_map.fits"
            png_path = save_path.with_suffix(".png")
            self._write_current_map_fits(save_path, frame_idx)
            self._write_current_map_png(png_path, frame_idx)
        except Exception as exc:
            self._set_status(f"Map FITS export failed: {type(exc).__name__}: {exc}")
            return
        self._refresh_saved_fits_browser()
        self._set_status(f"Saved current map FITS to {save_path} and PNG to {png_path}.")

    def _export_cut_ids_fits(self, cut_ids: list[int], label: str) -> None:
        valid_cut_ids = [int(cut_id) for cut_id in cut_ids if int(cut_id) in self.cuts]
        if not valid_cut_ids:
            self._set_status(f"No valid cuts available for {label}.")
            return
        self._sync_all_panel_analysis_state_from_windows()
        try:
            export_dir = self._resolved_export_dir()
            written = 0
            png_written = 0
            for cut_id in valid_cut_ids:
                cut = self.cuts[cut_id]
                save_path = export_dir / (
                    f"{self.cube_path.stem}_cut{cut_id:03d}_{self._safe_export_slug(cut.name)}_td.fits"
                )
                png_path = save_path.with_suffix(".png")
                self._write_cut_td_fits(cut_id, save_path)
                self._write_cut_quicklook_png(
                    cut_id,
                    png_path,
                    map_cut_ids=valid_cut_ids,
                )
                written += 1
                png_written += 1
        except Exception as exc:
            self._set_status(f"Cut FITS export failed: {type(exc).__name__}: {exc}")
            return
        self._refresh_saved_fits_browser()
        self._set_status(
            f"Saved {written} cut FITS file(s) and {png_written} PNG quicklook(s) "
            f"for {label} in {export_dir}."
        )

    def _export_trace_ids_fits(self, cut_ids: list[int], label: str) -> None:
        valid_cut_ids = [int(cut_id) for cut_id in cut_ids if int(cut_id) in self.cuts]
        if not valid_cut_ids:
            self._set_status(f"No valid cuts available for {label}.")
            return
        self._sync_all_panel_analysis_state_from_windows()
        try:
            export_dir = self._resolved_export_dir()
            written = 0
            png_written = 0
            for cut_id in valid_cut_ids:
                cut = self.cuts[cut_id]
                state = self._cut_analysis_snapshot(cut_id)
                for event in state.get("wavelet_events") or []:
                    event_id = int(event.get("event_id", -1))
                    save_path = export_dir / (
                        f"{self.cube_path.stem}_cut{cut_id:03d}_{self._safe_export_slug(cut.name)}"
                        f"_evt{event_id:04d}_trace.fits"
                    )
                    png_path = save_path.with_suffix(".png")
                    if self._write_trace_event_fits(cut_id, event, save_path):
                        self._write_cut_quicklook_png(
                            cut_id,
                            png_path,
                            map_cut_ids=valid_cut_ids,
                            selected_event_id=event_id,
                        )
                        written += 1
                        png_written += 1
            if written <= 0:
                self._set_status(f"No traces/waves available for {label}.")
                return
        except Exception as exc:
            self._set_status(f"Trace FITS export failed: {type(exc).__name__}: {exc}")
            return
        self._refresh_saved_fits_browser()
        self._set_status(
            f"Saved {written} trace FITS file(s) and {png_written} PNG quicklook(s) "
            f"for {label} in {export_dir}."
        )

    def _export_selected_cut_fits(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("Select a cut first.")
            return
        self._export_cut_ids_fits([int(cut.cut_id)], f"selected cut {cut.name}")

    def _export_stack_cut_fits(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        self._export_cut_ids_fits(
            [int(cut_id) for cut_id in stack.get("cut_ids") or []],
            f"stack {stack['name']}",
        )

    def _export_all_cut_fits(self) -> None:
        self._export_cut_ids_fits(sorted(self.cuts.keys()), "all cuts")

    def _export_selected_cut_trace_fits(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("Select a cut first.")
            return
        self._export_trace_ids_fits([int(cut.cut_id)], f"selected cut {cut.name}")

    def _export_stack_trace_fits(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        self._export_trace_ids_fits(
            [int(cut_id) for cut_id in stack.get("cut_ids") or []],
            f"stack {stack['name']}",
        )

    def _export_all_trace_fits(self) -> None:
        self._export_trace_ids_fits(sorted(self.cuts.keys()), "all cuts")

    def _next_link_group_label(self) -> str:
        label = f"LG{self.next_link_group_id:04d}"
        self.next_link_group_id += 1
        return label

    def _format_wavelet_thread_filter_text(self, thread_indices: list[int]) -> str:
        normalized = sorted({int(idx) for idx in thread_indices if int(idx) >= 0})
        return ",".join(str(idx + 1) for idx in normalized)

    def _parse_wavelet_thread_filter_text(
        self,
        text: str,
        *,
        max_threads: int | None = None,
    ) -> tuple[list[int] | None, str | None]:
        raw = str(text).strip()
        if not raw or raw.lower() == "all":
            return None, None

        indices: set[int] = set()
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                try:
                    start_value = int(float(start_text.strip()))
                    end_value = int(float(end_text.strip()))
                except Exception:
                    return None, f"Invalid wavelet thread range '{token}'."
                if start_value <= 0 or end_value <= 0:
                    return None, "Wavelet thread numbers start at 1."
                if end_value < start_value:
                    start_value, end_value = end_value, start_value
                for value in range(start_value, end_value + 1):
                    indices.add(value - 1)
                continue
            try:
                value = int(float(token))
            except Exception:
                return None, f"Invalid wavelet thread '{token}'."
            if value <= 0:
                return None, "Wavelet thread numbers start at 1."
            indices.add(value - 1)

        if not indices:
            return None, "Wavelet thread filter is empty."
        if max_threads is not None:
            invalid = [idx + 1 for idx in sorted(indices) if idx >= int(max_threads)]
            if invalid:
                if max_threads <= 0:
                    return None, "There are no tracked threads available yet."
                return (
                    None,
                    "Wavelet thread filter asks for unavailable thread(s): "
                    + ",".join(str(value) for value in invalid)
                    + f" (available 1-{int(max_threads)}).",
                )
        return sorted(indices), None

    def _wavelet_event_thread_index(self, event: dict[str, Any]) -> int | None:
        analysis = event.get("analysis") or {}
        try:
            thread_index = int(analysis.get("thread_index", -1))
        except Exception:
            return None
        return thread_index if thread_index >= 0 else None

    def _set_wavelet_thread_filter_for_cut(self, cut_id: int, filter_text: str) -> None:
        if cut_id not in self.cuts:
            return
        text = str(filter_text).strip()
        state = self._cut_analysis(cut_id)
        state["wavelet_thread_filter_text"] = text
        for panel_id, panel in enumerate(self.panels, start=1):
            if panel.cut_id != cut_id:
                continue
            existing = self.td_windows.get(panel_id)
            if existing is not None:
                existing["wavelet_thread_filter_var"].set(text)

    def _apply_wavelet_thread_filter_to_cuts(
        self,
        cut_ids: list[int],
        thread_indices: list[int],
    ) -> tuple[list[int], list[int]]:
        filter_text = self._format_wavelet_thread_filter_text(thread_indices)
        applied_cut_ids: list[int] = []
        skipped_cut_ids: list[int] = []
        seen_cut_ids: set[int] = set()
        for cut_id in cut_ids:
            current_cut_id = int(cut_id)
            if current_cut_id in seen_cut_ids or current_cut_id not in self.cuts:
                continue
            seen_cut_ids.add(current_cut_id)
            state = self._cut_analysis(current_cut_id)
            tracking_result = state.get("crest_tracking_result")
            if tracking_result is not None:
                threads = tracking_result.get("threads") or []
                if not threads or any(int(idx) >= len(threads) for idx in thread_indices):
                    skipped_cut_ids.append(current_cut_id)
                    continue
            self._set_wavelet_thread_filter_for_cut(current_cut_id, filter_text)
            applied_cut_ids.append(current_cut_id)
        if applied_cut_ids:
            self._record_session_change()
        return applied_cut_ids, skipped_cut_ids

    def _resolve_td_window_target_stack_id(self, panel_id: int) -> int | None:
        panel = self._td_window_panel(panel_id)
        if panel is None or panel.cut_id is None or panel.cut_id not in self.cuts:
            return None
        existing = self.td_windows.get(panel_id)
        source_stack_id = None if existing is None else existing.get("source_stack_id")
        if source_stack_id is not None and int(source_stack_id) in self.stacks:
            stack = self.stacks[int(source_stack_id)]
            stack_cut_ids = {
                int(cut_id)
                for cut_id in (stack.get("cut_ids") or [])
                if int(cut_id) in self.cuts
            }
            if int(panel.cut_id) in stack_cut_ids:
                return int(source_stack_id)
        if self.active_stack_id is not None and int(self.active_stack_id) in self.stacks:
            stack = self.stacks[int(self.active_stack_id)]
            stack_cut_ids = {
                int(cut_id)
                for cut_id in (stack.get("cut_ids") or [])
                if int(cut_id) in self.cuts
            }
            if int(panel.cut_id) in stack_cut_ids:
                return int(self.active_stack_id)
        memberships = self._stack_memberships_for_cut(int(panel.cut_id))
        if len(memberships) == 1:
            return int(memberships[0]["stack_id"])
        return None

    def _wavelet_params_match_preset(self, current: dict[str, Any], target: dict[str, Any]) -> bool:
        for key, target_value in target.items():
            current_value = current.get(key)
            if isinstance(target_value, bool):
                if bool(current_value) != bool(target_value):
                    return False
                continue
            if isinstance(target_value, int) and not isinstance(target_value, bool):
                try:
                    if int(current_value) != int(target_value):
                        return False
                except Exception:
                    return False
                continue
            try:
                current_float = float(current_value)
                target_float = float(target_value)
            except Exception:
                return False
            if np.isnan(current_float) and np.isnan(target_float):
                continue
            if not math.isclose(
                current_float, target_float, rel_tol=1e-6, abs_tol=1e-6
            ):
                return False
        return True

    def _matching_parameter_preset_name(
        self, crest_params: dict[str, Any], wavelet_params: dict[str, Any]
    ) -> str:
        for preset_name, preset in PARAMETER_PRESETS.items():
            if preset_name == "custom":
                continue
            if self._wavelet_params_match_preset(
                crest_params, dict(DEFAULT_CREST_TRACKING) | dict(preset.get("crest") or {})
            ) and self._wavelet_params_match_preset(
                wavelet_params, dict(DEFAULT_WAVELET_FILTER) | dict(preset.get("wavelet") or {})
            ):
                return preset_name
        return "custom"

    def _current_parameter_payloads(
        self, panel_id: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            state = self._panel_analysis(panel_id)
            crest_params = dict(DEFAULT_CREST_TRACKING)
            crest_params.update(state.get("crest_params") or {})
            wavelet_params = dict(DEFAULT_WAVELET_FILTER)
            wavelet_params.update(state.get("wavelet_params") or {})
            return crest_params, wavelet_params

        density_text = str(existing["wavelet_density_var"].get()).strip()
        phase_speed_text = str(existing["wavelet_phase_speed_var"].get()).strip()
        crest_params = {
            "cad": self._safe_float_text(
                existing["crest_cad_var"].get(), DEFAULT_CREST_TRACKING["cad"]
            ),
            "res": self._safe_float_text(
                existing["crest_res_var"].get(), DEFAULT_CREST_TRACKING["res"]
            ),
            "grad": self._safe_float_text(
                existing["crest_grad_var"].get(), DEFAULT_CREST_TRACKING["grad"]
            ),
            "min_tlen": self._safe_int_text(
                existing["crest_min_tlen_var"].get(), DEFAULT_CREST_TRACKING["min_tlen"]
            ),
            "max_dist_jump": self._safe_int_text(
                existing["crest_max_dist_jump_var"].get(),
                DEFAULT_CREST_TRACKING["max_dist_jump"],
            ),
            "max_time_skip": self._safe_int_text(
                existing["crest_max_time_skip_var"].get(),
                DEFAULT_CREST_TRACKING["max_time_skip"],
            ),
            "invert": bool(existing["crest_invert_var"].get()),
            "gauss": bool(existing["crest_gauss_var"].get()),
        }
        wavelet_params = {
            "p_min": self._safe_float_text(
                existing["wavelet_p_min_var"].get(), DEFAULT_WAVELET_FILTER["p_min"]
            ),
            "p_max": self._safe_float_text(
                existing["wavelet_p_max_var"].get(), DEFAULT_WAVELET_FILTER["p_max"]
            ),
            "power_ratio_thresh": self._safe_float_text(
                existing["wavelet_power_ratio_var"].get(),
                DEFAULT_WAVELET_FILTER["power_ratio_thresh"],
            ),
            "segment_power_frac": self._safe_float_text(
                existing["wavelet_segment_frac_var"].get(),
                DEFAULT_WAVELET_FILTER["segment_power_frac"],
            ),
            "min_points_segment": self._safe_int_text(
                existing["wavelet_min_points_seg_var"].get(),
                DEFAULT_WAVELET_FILTER["min_points_segment"],
            ),
            "min_amp_arcsec": self._safe_float_text(
                existing["wavelet_min_amp_var"].get(),
                DEFAULT_WAVELET_FILTER["min_amp_arcsec"],
            ),
            "max_jump_pix": self._safe_float_text(
                existing["wavelet_max_jump_var"].get(),
                DEFAULT_WAVELET_FILTER["max_jump_pix"],
            ),
            "min_points_cut_seg": self._safe_int_text(
                existing["wavelet_min_points_cut_var"].get(),
                DEFAULT_WAVELET_FILTER["min_points_cut_seg"],
            ),
            "rms_amp_ratio_max": self._safe_float_text(
                existing["wavelet_rms_amp_ratio_var"].get(),
                DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"],
            ),
            "km_per_arcsec": self._safe_float_text(
                existing["wavelet_km_per_arcsec_var"].get(),
                DEFAULT_WAVELET_FILTER["km_per_arcsec"],
            ),
            "density_kg_m3": self._safe_float_text(density_text, float("nan")),
            "phase_speed_km_s": self._safe_float_text(phase_speed_text, float("nan")),
        }
        return crest_params, wavelet_params

    def _update_td_window_preset_from_values(self, panel_id: int) -> str:
        crest_params, wavelet_params = self._current_parameter_payloads(panel_id)
        preset_name = self._matching_parameter_preset_name(crest_params, wavelet_params)
        state = self._panel_analysis(panel_id)
        state["preset_name"] = preset_name
        state["crest_params"] = dict(crest_params)
        state["wavelet_params"] = dict(wavelet_params)
        existing = self.td_windows.get(panel_id)
        if existing is not None and existing.get("preset_var") is not None:
            existing["preset_var"].set(preset_name)
        return preset_name

    def _apply_td_window_parameter_preset(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        preset_name = str(existing["preset_var"].get() or "custom")
        preset = PARAMETER_PRESETS.get(preset_name)
        if preset is None:
            self._set_status(f"Unknown parameter preset '{preset_name}'.")
            return

        crest_params = dict(DEFAULT_CREST_TRACKING)
        crest_params.update(preset.get("crest") or {})
        wavelet_params = dict(DEFAULT_WAVELET_FILTER)
        wavelet_params.update(preset.get("wavelet") or {})

        existing["crest_cad_var"].set(f"{float(crest_params['cad']):.2f}")
        existing["crest_res_var"].set(f"{float(crest_params['res']):.2f}")
        existing["crest_grad_var"].set(f"{float(crest_params['grad']):.2f}")
        existing["crest_min_tlen_var"].set(str(int(crest_params["min_tlen"])))
        existing["crest_max_dist_jump_var"].set(str(int(crest_params["max_dist_jump"])))
        existing["crest_max_time_skip_var"].set(str(int(crest_params["max_time_skip"])))
        existing["crest_invert_var"].set(bool(crest_params["invert"]))
        existing["crest_gauss_var"].set(bool(crest_params["gauss"]))

        existing["wavelet_p_min_var"].set(f"{float(wavelet_params['p_min']):.2f}")
        existing["wavelet_p_max_var"].set(f"{float(wavelet_params['p_max']):.2f}")
        existing["wavelet_power_ratio_var"].set(
            f"{float(wavelet_params['power_ratio_thresh']):.2f}"
        )
        existing["wavelet_segment_frac_var"].set(
            f"{float(wavelet_params['segment_power_frac']):.2f}"
        )
        existing["wavelet_min_points_seg_var"].set(
            str(int(wavelet_params["min_points_segment"]))
        )
        existing["wavelet_min_amp_var"].set(
            f"{float(wavelet_params['min_amp_arcsec']):.3f}"
        )
        existing["wavelet_max_jump_var"].set(
            f"{float(wavelet_params['max_jump_pix']):.2f}"
        )
        existing["wavelet_min_points_cut_var"].set(
            str(int(wavelet_params["min_points_cut_seg"]))
        )
        existing["wavelet_rms_amp_ratio_var"].set(
            f"{float(wavelet_params['rms_amp_ratio_max']):.2f}"
        )
        existing["wavelet_km_per_arcsec_var"].set(
            f"{float(wavelet_params['km_per_arcsec']):.2f}"
        )
        existing["wavelet_density_var"].set(
            ""
            if not np.isfinite(float(wavelet_params["density_kg_m3"]))
            else f"{float(wavelet_params['density_kg_m3']):.3e}"
        )
        existing["wavelet_phase_speed_var"].set(
            ""
            if not np.isfinite(float(wavelet_params["phase_speed_km_s"]))
            else f"{float(wavelet_params['phase_speed_km_s']):.2f}"
        )

        state = self._panel_analysis(panel_id)
        state["preset_name"] = preset_name
        state["crest_params"] = dict(crest_params)
        state["wavelet_params"] = dict(wavelet_params)
        self._record_session_change()
        self._set_status(f"Applied preset '{preset_name}' to {panel.name}.")

    def _apply_td_window_advanced_filters(self, panel_id: int) -> None:
        if panel_id not in self.td_windows:
            return
        self._sync_panel_analysis_state_from_window(panel_id)
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=False)

    def _clear_td_window_advanced_filters(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        existing["wavelet_filter_qa_var"].set("all")
        existing["wavelet_filter_locked_var"].set("all")
        existing["wavelet_filter_linked_var"].set("all")
        existing["wavelet_filter_score_min_var"].set("")
        existing["wavelet_filter_period_min_var"].set("")
        existing["wavelet_filter_period_max_var"].set("")
        existing["wavelet_filter_amp_min_var"].set("")
        existing["wavelet_filter_amp_max_var"].set("")
        existing["wavelet_filter_energy_min_var"].set("")
        existing["wavelet_filter_energy_max_var"].set("")
        self._apply_td_window_advanced_filters(panel_id)

    def _refresh_all_open_td_window_wavelet_views(
        self, *, redraw_td: bool = True
    ) -> None:
        for open_panel_id in list(self.td_windows.keys()):
            self._refresh_td_window_wavelet_views(open_panel_id, redraw_td=redraw_td)

    def _sync_next_link_group_id_from_state(self) -> None:
        next_link_group_id = 1
        for panel in self.panels:
            for event in self._panel_wavelet_events_snapshot(panel.panel_id):
                group_id = str(event.get("link_group_id") or "")
                if len(group_id) <= 2 or not group_id.startswith("LG"):
                    continue
                suffix = group_id[2:]
                if suffix.isdigit():
                    next_link_group_id = max(next_link_group_id, int(suffix) + 1)
        self.next_link_group_id = next_link_group_id

    def _panel_analysis_snapshot(self, panel_id: int) -> dict[str, Any]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return self._clone_wavelet_payload(self._panel_analysis(panel_id))

        current_state = self._panel_analysis(panel_id)
        snapshot = self._make_default_panel_analysis_state()
        snapshot["dynamic_enabled"] = bool(current_state.get("dynamic_enabled", False))
        snapshot["dynamic_reference_frame"] = int(
            current_state.get("dynamic_reference_frame", 0)
        )
        snapshot["dynamic_keyframes"] = self._clone_wavelet_payload(
            current_state.get("dynamic_keyframes") or {}
        )
        snapshot["td_params"] = {
            "t_ini": self._safe_int_text(existing["panel_t_ini_var"].get(), 0),
            "t_fin": self._safe_int_text(existing["panel_t_fin_var"].get(), self.nt - 1),
            "stride": self._safe_int_text(existing["panel_stride_var"].get(), 1),
            "width": self._safe_int_text(existing["panel_width_var"].get(), 1),
            "weighting": str(existing["panel_weighting_var"].get() or "uniform"),
        }
        snapshot["crest_params"] = {
            "cad": self._safe_float_text(existing["crest_cad_var"].get(), DEFAULT_CREST_TRACKING["cad"]),
            "res": self._safe_float_text(existing["crest_res_var"].get(), DEFAULT_CREST_TRACKING["res"]),
            "grad": self._safe_float_text(existing["crest_grad_var"].get(), DEFAULT_CREST_TRACKING["grad"]),
            "min_tlen": self._safe_int_text(existing["crest_min_tlen_var"].get(), DEFAULT_CREST_TRACKING["min_tlen"]),
            "max_dist_jump": self._safe_int_text(existing["crest_max_dist_jump_var"].get(), DEFAULT_CREST_TRACKING["max_dist_jump"]),
            "max_time_skip": self._safe_int_text(existing["crest_max_time_skip_var"].get(), DEFAULT_CREST_TRACKING["max_time_skip"]),
            "invert": bool(existing["crest_invert_var"].get()),
            "gauss": bool(existing["crest_gauss_var"].get()),
        }
        density_text = str(existing["wavelet_density_var"].get()).strip()
        phase_speed_text = str(existing["wavelet_phase_speed_var"].get()).strip()
        snapshot["wavelet_params"] = {
            "p_min": self._safe_float_text(existing["wavelet_p_min_var"].get(), DEFAULT_WAVELET_FILTER["p_min"]),
            "p_max": self._safe_float_text(existing["wavelet_p_max_var"].get(), DEFAULT_WAVELET_FILTER["p_max"]),
            "power_ratio_thresh": self._safe_float_text(existing["wavelet_power_ratio_var"].get(), DEFAULT_WAVELET_FILTER["power_ratio_thresh"]),
            "segment_power_frac": self._safe_float_text(existing["wavelet_segment_frac_var"].get(), DEFAULT_WAVELET_FILTER["segment_power_frac"]),
            "min_points_segment": self._safe_int_text(existing["wavelet_min_points_seg_var"].get(), DEFAULT_WAVELET_FILTER["min_points_segment"]),
            "min_amp_arcsec": self._safe_float_text(existing["wavelet_min_amp_var"].get(), DEFAULT_WAVELET_FILTER["min_amp_arcsec"]),
            "max_jump_pix": self._safe_float_text(existing["wavelet_max_jump_var"].get(), DEFAULT_WAVELET_FILTER["max_jump_pix"]),
            "min_points_cut_seg": self._safe_int_text(existing["wavelet_min_points_cut_var"].get(), DEFAULT_WAVELET_FILTER["min_points_cut_seg"]),
            "rms_amp_ratio_max": self._safe_float_text(existing["wavelet_rms_amp_ratio_var"].get(), DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"]),
            "km_per_arcsec": self._safe_float_text(existing["wavelet_km_per_arcsec_var"].get(), DEFAULT_WAVELET_FILTER["km_per_arcsec"]),
            "density_kg_m3": self._safe_float_text(density_text, float("nan")),
            "phase_speed_km_s": self._safe_float_text(phase_speed_text, float("nan")),
        }
        snapshot["crest_tracking_result"] = self._clone_wavelet_payload(
            existing.get("crest_tracking_result")
        )
        snapshot["crest_tracking_td_key"] = self._clone_wavelet_payload(
            existing.get("crest_tracking_td_key")
        )
        snapshot["wavelet_filter_result"] = self._clone_wavelet_payload(
            existing.get("wavelet_filter_result")
        )
        snapshot["wavelet_thread_filter_text"] = str(
            existing["wavelet_thread_filter_var"].get()
        ).strip()
        snapshot["wavelet_events"] = self._clone_wavelet_payload(
            existing.get("wavelet_events") or []
        )
        snapshot["wavelet_next_event_id"] = int(existing.get("wavelet_next_event_id", 1))
        snapshot["wavelet_selected_event_id"] = existing.get("wavelet_selected_event_id")
        snapshot["wavelet_events_filter"] = str(
            existing["wavelet_events_filter_var"].get() or "accepted"
        )
        snapshot["wavelet_advanced_filters"] = {
            "qa": str(existing["wavelet_filter_qa_var"].get() or "all"),
            "locked": str(existing["wavelet_filter_locked_var"].get() or "all"),
            "linked": str(existing["wavelet_filter_linked_var"].get() or "all"),
            "score_min": str(existing["wavelet_filter_score_min_var"].get()),
            "period_min": str(existing["wavelet_filter_period_min_var"].get()),
            "period_max": str(existing["wavelet_filter_period_max_var"].get()),
            "amp_min": str(existing["wavelet_filter_amp_min_var"].get()),
            "amp_max": str(existing["wavelet_filter_amp_max_var"].get()),
            "energy_min": str(existing["wavelet_filter_energy_min_var"].get()),
            "energy_max": str(existing["wavelet_filter_energy_max_var"].get()),
        }
        snapshot["wavelet_undo_stack"] = self._clone_wavelet_payload(
            existing.get("wavelet_undo_stack") or []
        )
        snapshot["wavelet_redo_stack"] = self._clone_wavelet_payload(
            existing.get("wavelet_redo_stack") or []
        )
        snapshot["preset_name"] = str(
            existing.get("preset_var").get() if existing.get("preset_var") is not None else "custom"
        )
        snapshot["roi_enabled"] = bool(existing["roi_enabled_var"].get())
        snapshot["roi_t_span"] = str(existing["roi_t_span_var"].get())
        snapshot["roi_d_span"] = str(existing["roi_d_span_var"].get())
        snapshot["roi_center_t"] = existing.get("roi_center_t")
        snapshot["roi_center_d"] = existing.get("roi_center_d")
        return snapshot

    def _sync_panel_analysis_state_from_window(self, panel_id: int) -> None:
        if panel_id in self.td_windows:
            snapshot = self._panel_analysis_snapshot(panel_id)
            self.panel_analysis_state[panel_id] = self._clone_wavelet_payload(snapshot)
            if 1 <= int(panel_id) <= len(self.panels):
                panel = self.panels[int(panel_id) - 1]
                if panel.cut_id is not None and panel.cut_id in self.cuts:
                    self.cut_analysis_state[panel.cut_id] = self._clone_wavelet_payload(
                        snapshot
                    )
                    self._sync_panels_from_cut_td_params(panel.cut_id)

    def _sync_all_panel_analysis_state_from_windows(self) -> None:
        for panel_id in list(self.td_windows.keys()):
            self._sync_panel_analysis_state_from_window(panel_id)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        return value

    def _close_all_td_windows(self) -> None:
        for panel_id in list(self.td_windows.keys()):
            self._close_td_window(panel_id)

    def _build_session_payload(self) -> dict[str, Any]:
        self._sync_all_panel_analysis_state_from_windows()
        return {
            "session_version": SESSION_VERSION,
            "cube_path": str(self.cube_path),
            "cube_axis_order": str(self.cube_axis_order),
            "layout": self.layout_var.get(),
            "t_visual": int(self.t_visual_var.get()),
            "active_panel_id": int(self.active_panel_id),
            "selected_cut_id": self.selected_cut_id,
            "next_cut_id": int(self.next_cut_id),
            "selected_feature_axis_id": self.selected_feature_axis_id,
            "next_feature_axis_id": int(self.next_feature_axis_id),
            "autosave_every": int(self.autosave_every),
            "axis_flips": {
                "td_aspect": str(self.td_aspect_var.get()),
                "td_zoom": str(self.td_zoom_var.get()),
                "map_swap_xy": bool(self.map_swap_xy_var.get()),
                "td_swap_axes": bool(self.td_swap_axes_var.get()),
                "map_x": bool(self.map_flip_x_var.get()),
                "map_y": bool(self.map_flip_y_var.get()),
                "td_x": bool(self.td_flip_x_var.get()),
                "td_y": bool(self.td_flip_y_var.get()),
            },
            "cuts": [
                {
                    "cut_id": cut.cut_id,
                    "name": cut.name,
                    "color": cut.color,
                    "p0": [round(cut.p0[0], 4), round(cut.p0[1], 4)],
                    "p1": [round(cut.p1[0], 4), round(cut.p1[1], 4)],
                    "visible": cut.visible,
                    "locked": cut.locked,
                }
                for cut in self.cuts.values()
            ],
            "feature_axes": [
                {
                    "axis_id": axis.axis_id,
                    "name": axis.name,
                    "color": axis.color,
                    "mode": axis.mode,
                    "visible": axis.visible,
                    "points": [
                        [round(point[0], 4), round(point[1], 4)] for point in axis.points
                    ],
                }
                for axis in self.feature_axes.values()
            ],
            "panels": [
                {
                    "panel_id": panel.panel_id,
                    "name": panel.name,
                    "cut_id": panel.cut_id,
                    "t_ini": panel.t_ini,
                    "t_fin": panel.t_fin,
                    "stride": panel.stride,
                    "width": panel.width,
                    "weighting": panel.weighting,
                }
                for panel in self.panels
            ],
            "next_stack_id": int(self.next_stack_id),
            "active_stack_id": self.active_stack_id,
            "stacks": self._json_safe(list(self.stacks.values())),
            "cut_analysis_state": self._json_safe(self.cut_analysis_state),
            "panel_analysis_state": self._json_safe(self.panel_analysis_state),
            "export_settings": {
                "dir": str(self.export_dir_var.get() or ""),
            },
        }

    def _write_session_file(self, path: Path, *, autosave: bool = False) -> None:
        payload = self._build_session_payload()
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, indent=2)
        if autosave:
            self.autosave_change_count = 0
        else:
            self.last_session_path = path
            self.autosave_change_count = 0

    def _record_session_change(self) -> None:
        _trace_stack_wavelet("record_session_change start")
        self.autosave_change_count += 1
        if self.autosave_change_count >= max(int(self.autosave_every), 1):
            try:
                self._write_session_file(self.autosave_path, autosave=True)
                self._set_status(f"Auto-saved session to {self.autosave_path}")
            except Exception as exc:
                self._set_status(
                    f"Auto-save failed: {type(exc).__name__}: {exc}"
                )
        self._refresh_metrics_window()
        self._refresh_link_groups_window()
        self._refresh_propagation_window()
        self._schedule_stack_browser_refresh()
        _trace_stack_wavelet("record_session_change end")

    def _load_session(self) -> None:
        session_path = self.filedialog.askopenfilename(
            title="Load session",
            initialdir=str(Path(__file__).resolve().parent.parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not session_path:
            return
        self._load_session_from_path(Path(session_path))

    def _load_session_from_path(self, session_path: Path) -> None:
        session_path = session_path.expanduser().resolve()
        with open(session_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        session_cube = payload.get("cube_path")
        if session_cube and Path(session_cube).expanduser().resolve() != self.cube_path:
            self.messagebox.showerror(
                "Load session",
                "This session was saved for a different cube. Open the matching cube first.",
            )
            return
        session_axis_order = str(payload.get("cube_axis_order", self.cube_axis_order))
        if normalize_cube_axis_order(session_axis_order) != self.cube_axis_order:
            self.messagebox.showerror(
                "Load session",
                "This session was saved for a different cube axis order. Open the cube with the same axis permutation first.",
            )
            return

        self._cancel_all_background_jobs()
        self._close_all_td_windows()
        self._close_link_groups_window()
        self._close_propagation_window()
        self._close_all_stack_browsers()

        self.layout_var.set(
            payload.get("layout")
            if payload.get("layout") in LAYOUT_PRESETS
            else self.layout_var.get()
        )
        self.t_visual_var.set(
            clamp_int(int(payload.get("t_visual", self.t_visual_var.get())), 0, self.nt - 1)
        )
        axis_flips = payload.get("axis_flips") or {}
        self.td_aspect_var.set(str(axis_flips.get("td_aspect", self.td_aspect_var.get())))
        self.td_zoom_var.set(str(axis_flips.get("td_zoom", self.td_zoom_var.get())))
        self.map_swap_xy_var.set(bool(axis_flips.get("map_swap_xy", self.map_swap_xy_var.get())))
        self.td_swap_axes_var.set(bool(axis_flips.get("td_swap_axes", self.td_swap_axes_var.get())))
        self.map_flip_x_var.set(bool(axis_flips.get("map_x", self.map_flip_x_var.get())))
        self.map_flip_y_var.set(bool(axis_flips.get("map_y", self.map_flip_y_var.get())))
        self.td_flip_x_var.set(bool(axis_flips.get("td_x", self.td_flip_x_var.get())))
        self.td_flip_y_var.set(bool(axis_flips.get("td_y", self.td_flip_y_var.get())))
        self.autosave_every = max(int(payload.get("autosave_every", DEFAULT_AUTOSAVE_EVERY)), 1)

        self.cuts = {}
        self.next_cut_id = 1
        self.feature_axes = {}
        self.next_feature_axis_id = 1
        self.selected_feature_axis_id = None
        self.feature_draw_mode = None
        self.feature_pending_points = []
        self.feature_hover_point = None
        for cut_payload in payload.get("cuts", []):
            cut = Cut(
                cut_id=int(cut_payload["cut_id"]),
                name=str(cut_payload.get("name", f"Cut {cut_payload['cut_id']}")),
                color=str(cut_payload.get("color", COLOR_CYCLE[(int(cut_payload["cut_id"]) - 1) % len(COLOR_CYCLE)])),
                p0=(
                    float((cut_payload.get("p0") or [0.0, 0.0])[0]),
                    float((cut_payload.get("p0") or [0.0, 0.0])[1]),
                ),
                p1=(
                    float((cut_payload.get("p1") or [1.0, 1.0])[0]),
                    float((cut_payload.get("p1") or [1.0, 1.0])[1]),
                ),
                visible=bool(cut_payload.get("visible", True)),
                locked=bool(cut_payload.get("locked", False)),
            )
            self.cuts[cut.cut_id] = cut
            self.next_cut_id = max(self.next_cut_id, cut.cut_id + 1)
        self.next_cut_id = max(int(payload.get("next_cut_id", self.next_cut_id)), self.next_cut_id)
        for axis_payload in payload.get("feature_axes", []):
            if not isinstance(axis_payload, dict):
                continue
            raw_points = axis_payload.get("points") or []
            normalized_points: list[tuple[float, float]] = []
            for point in raw_points:
                try:
                    normalized_points.append(
                        clamp_point((float(point[0]), float(point[1])), self.nx, self.ny)
                    )
                except Exception:
                    continue
            if len(normalized_points) < 2 or polyline_length(normalized_points) < 1.0:
                continue
            axis = FeatureAxis(
                axis_id=int(axis_payload.get("axis_id", self.next_feature_axis_id)),
                name=str(axis_payload.get("name", f"Axis {axis_payload.get('axis_id', self.next_feature_axis_id)}")),
                color=str(
                    axis_payload.get(
                        "color",
                        COLOR_CYCLE[
                            (int(axis_payload.get("axis_id", self.next_feature_axis_id)) - 1)
                            % len(COLOR_CYCLE)
                        ],
                    )
                ),
                points=normalized_points,
                mode="line" if str(axis_payload.get("mode", "curve")) == "line" else "curve",
                visible=bool(axis_payload.get("visible", True)),
            )
            self.feature_axes[axis.axis_id] = axis
            self.next_feature_axis_id = max(self.next_feature_axis_id, axis.axis_id + 1)
        self.next_feature_axis_id = max(
            int(payload.get("next_feature_axis_id", self.next_feature_axis_id)),
            self.next_feature_axis_id,
        )
        selected_feature_axis_id = payload.get("selected_feature_axis_id")
        try:
            selected_feature_axis_id_int = (
                None if selected_feature_axis_id is None else int(selected_feature_axis_id)
            )
        except Exception:
            selected_feature_axis_id_int = None
        if (
            selected_feature_axis_id_int is not None
            and selected_feature_axis_id_int in self.feature_axes
        ):
            self.selected_feature_axis_id = selected_feature_axis_id_int
        elif self.feature_axes:
            self.selected_feature_axis_id = min(self.feature_axes.keys())

        self.panels = make_default_panels(self.nt)
        panel_payload_by_id = {
            int(item.get("panel_id", -1)): item for item in payload.get("panels", [])
        }
        for panel in self.panels:
            panel_payload = panel_payload_by_id.get(panel.panel_id)
            if panel_payload is None:
                continue
            panel.name = str(panel_payload.get("name", panel.name))
            cut_id = panel_payload.get("cut_id")
            panel.cut_id = int(cut_id) if cut_id is not None and int(cut_id) in self.cuts else None
            panel.t_ini = clamp_int(int(panel_payload.get("t_ini", panel.t_ini)), 0, self.nt - 1)
            panel.t_fin = clamp_int(int(panel_payload.get("t_fin", panel.t_fin)), panel.t_ini, self.nt - 1)
            panel.stride = max(int(panel_payload.get("stride", panel.stride)), 1)
            panel.width = max(int(panel_payload.get("width", panel.width)), 1)
            panel.weighting = str(panel_payload.get("weighting", panel.weighting))
            panel.cache_key = None
            panel.cache_td = None
            panel.cache_meta = None

        loaded_panel_state = payload.get("panel_analysis_state") or {}
        loaded_cut_state = payload.get("cut_analysis_state") or {}
        self.cut_analysis_state = {}
        for cut_id_text, loaded_state in loaded_cut_state.items():
            try:
                cut_id = int(cut_id_text)
            except Exception:
                continue
            if cut_id not in self.cuts or not isinstance(loaded_state, dict):
                continue
            state = self._make_default_panel_analysis_state()
            state.update(self._clone_wavelet_payload(loaded_state))
            state["td_params"] = {
                **dict(self._make_default_panel_analysis_state()["td_params"]),
                **dict(loaded_state.get("td_params") or {}),
            }
            state["crest_params"] = {
                **dict(DEFAULT_CREST_TRACKING),
                **dict(loaded_state.get("crest_params") or {}),
            }
            state["wavelet_params"] = {
                **dict(DEFAULT_WAVELET_FILTER),
                **dict(loaded_state.get("wavelet_params") or {}),
            }
            state["wavelet_advanced_filters"] = {
                **dict(self._make_default_panel_analysis_state()["wavelet_advanced_filters"]),
                **dict(loaded_state.get("wavelet_advanced_filters") or {}),
            }
            for event in state.get("wavelet_events") or []:
                self._ensure_wavelet_event_fields(event)
                self._wavelet_event_confidence_details(event)
            if not state.get("preset_name"):
                state["preset_name"] = self._matching_parameter_preset_name(
                    dict(state["crest_params"]), dict(state["wavelet_params"])
                )
            self.cut_analysis_state[cut_id] = state

        self.panel_analysis_state = {}
        for panel in self.panels:
            state = self._make_default_panel_analysis_state()
            loaded_state = loaded_panel_state.get(str(panel.panel_id)) or loaded_panel_state.get(panel.panel_id)
            if isinstance(loaded_state, dict):
                state.update(self._clone_wavelet_payload(loaded_state))
                state["crest_params"] = {
                    **dict(DEFAULT_CREST_TRACKING),
                    **dict(loaded_state.get("crest_params") or {}),
                }
                state["wavelet_params"] = {
                    **dict(DEFAULT_WAVELET_FILTER),
                    **dict(loaded_state.get("wavelet_params") or {}),
                }
                state["wavelet_advanced_filters"] = {
                    **dict(self._make_default_panel_analysis_state()["wavelet_advanced_filters"]),
                    **dict(loaded_state.get("wavelet_advanced_filters") or {}),
                }
                for event in state.get("wavelet_events") or []:
                    self._ensure_wavelet_event_fields(event)
                    self._wavelet_event_confidence_details(event)
                if not state.get("preset_name"):
                    state["preset_name"] = self._matching_parameter_preset_name(
                        dict(state["crest_params"]), dict(state["wavelet_params"])
                    )
            self.panel_analysis_state[panel.panel_id] = state
            if panel.cut_id is not None and panel.cut_id in self.cuts:
                if panel.cut_id not in self.cut_analysis_state and isinstance(loaded_state, dict):
                    migrated = self._clone_wavelet_payload(state)
                    migrated["td_params"] = {
                        "t_ini": int(panel.t_ini),
                        "t_fin": int(panel.t_fin),
                        "stride": int(panel.stride),
                        "width": int(panel.width),
                        "weighting": str(panel.weighting),
                    }
                    self.cut_analysis_state[panel.cut_id] = migrated
                elif panel.cut_id in self.cut_analysis_state:
                    self._sync_panels_from_cut_td_params(panel.cut_id)

        self.stacks = {}
        for stack_payload in payload.get("stacks", []):
            if not isinstance(stack_payload, dict):
                continue
            stack_id = int(stack_payload.get("stack_id", self.next_stack_id))
            cut_ids = [
                int(cut_id)
                for cut_id in (stack_payload.get("cut_ids") or [])
                if int(cut_id) in self.cuts
            ]
            stack_state = self._make_default_stack_state(
                stack_id,
                str(stack_payload.get("name", f"Stack {stack_id}")),
                cut_ids,
            )
            stack_state["notes"] = str(stack_payload.get("notes", ""))
            stack_state["order_mode"] = str(stack_payload.get("order_mode", "manual"))
            self.stacks[stack_id] = stack_state
        self.next_stack_id = max(
            int(payload.get("next_stack_id", 1)),
            max(self.stacks.keys(), default=0) + 1,
        )
        self.active_stack_id = payload.get("active_stack_id")
        if self.active_stack_id is not None:
            try:
                self.active_stack_id = int(self.active_stack_id)
            except Exception:
                self.active_stack_id = None
        if self.active_stack_id not in self.stacks:
            self.active_stack_id = next(iter(self.stacks.keys()), None)
        self.selected_stack_cut_id = None

        self.active_panel_id = clamp_int(
            int(payload.get("active_panel_id", self.active_panel_id)),
            1,
            len(self.panels),
        )
        selected_cut_id = payload.get("selected_cut_id")
        self.selected_cut_id = (
            int(selected_cut_id) if selected_cut_id is not None and int(selected_cut_id) in self.cuts else None
        )
        export_settings = payload.get("export_settings") or {}
        self.export_dir_var.set(str(export_settings.get("dir") or self.cube_path.parent))

        self._apply_layout()
        self._sync_controls_from_active_panel()
        self.autosave_change_count = 0
        self.last_session_path = session_path
        self._sync_next_link_group_id_from_state()
        self.refresh_all()
        self._set_status(f"Session loaded from {session_path}")

    def _create_scrolled_frame(
        self,
        parent: Any,
        *,
        padding: int | tuple[int, ...] = 0,
        height: int | None = None,
    ) -> tuple[Any, Any, Any]:
        outer = self.ttk.Frame(parent)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = self.tk.Canvas(
            outer,
            highlightthickness=0,
            borderwidth=0,
            background=self.root.cget("background"),
        )
        if height is not None:
            canvas.configure(height=height)
        scrollbar = self.ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = self.ttk.Frame(canvas, padding=padding)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _sync_scrollregion(_event: Any) -> None:
            bbox = canvas.bbox("all")
            if bbox is not None:
                canvas.configure(scrollregion=bbox)

        def _sync_width(event: Any) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)
        return outer, content, canvas

    def _new_background_job(
        self, *, kind: str, panel_id: int | None = None, panel_name: str = ""
    ) -> tuple[str, dict[str, Any]]:
        job_id = f"{kind}-{self.next_background_job_id}"
        self.next_background_job_id += 1
        job = {
            "job_id": job_id,
            "kind": kind,
            "panel_id": panel_id,
            "panel_name": panel_name,
            "cancel_event": threading.Event(),
            "thread": None,
            "stage": "queued",
            "current": 0,
            "total": 1,
            "message": "",
        }
        self.background_jobs[job_id] = job
        return job_id, job

    def _background_wavelet_job_for_panel(self, panel_id: int) -> tuple[str | None, dict[str, Any] | None]:
        for job_id, job in self.background_jobs.items():
            if job.get("kind") == "wavelet" and int(job.get("panel_id", -1)) == int(panel_id):
                return job_id, job
        return None, None

    def _cancel_all_background_jobs(self) -> None:
        for job in self.background_jobs.values():
            cancel_event = job.get("cancel_event")
            if cancel_event is not None:
                cancel_event.set()
        self.background_jobs.clear()

    def _poll_background_jobs(self) -> None:
        try:
            while True:
                message = self.background_queue.get_nowait()
                self._handle_background_message(message)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(120, self._poll_background_jobs)

    def _handle_background_message(self, message: dict[str, Any]) -> None:
        job_id = str(message.get("job_id", ""))
        job = self.background_jobs.get(job_id)
        if job is None:
            return

        message_type = str(message.get("type", ""))
        if message_type == "progress":
            job["stage"] = message.get("stage", job.get("stage", "running"))
            job["current"] = int(message.get("current", job.get("current", 0)))
            job["total"] = max(int(message.get("total", job.get("total", 1))), 1)
            job["message"] = str(message.get("message", ""))
            if job.get("kind") == "wavelet":
                self._update_wavelet_job_widgets(int(job.get("panel_id", 0)))
                if job["message"]:
                    self._set_status(job["message"])
            elif job.get("kind") == "batch":
                self._set_status(job["message"])
            return

        if message_type == "done":
            if job.get("kind") == "wavelet":
                panel_id = int(job.get("panel_id", 0) or 0)
                _trace_stack_wavelet(
                    f"background done wavelet panel={panel_id} panel_name={job.get('panel_name', '')}"
                )
                self._apply_background_wavelet_results(
                    panel_id,
                    message.get("segments") or [],
                    dict(message.get("params") or {}),
                    str(message.get("panel_name", job.get("panel_name", ""))),
                    message.get("warnings") or [],
                )
                self.background_jobs.pop(job_id, None)
                if panel_id:
                    self._update_wavelet_job_widgets(panel_id)
            elif job.get("kind") == "batch":
                self._apply_background_batch_results(
                    message.get("results") or {},
                    str(message.get("summary", "")),
                )
                self.background_jobs.pop(job_id, None)
            return

        if message_type == "cancelled":
            panel_id = int(job.get("panel_id", 0) or 0)
            panel_name = str(job.get("panel_name", ""))
            self.background_jobs.pop(job_id, None)
            if job.get("kind") == "wavelet" and panel_id:
                self._update_wavelet_job_widgets(panel_id)
                self._set_status(f"Wavelet background job cancelled for {panel_name}.")
            elif job.get("kind") == "batch":
                self._set_status("Batch pipeline cancelled.")
            return

        if message_type == "error":
            panel_id = int(job.get("panel_id", 0) or 0)
            panel_name = str(job.get("panel_name", ""))
            error_text = str(message.get("error", "Background job failed."))
            self.background_jobs.pop(job_id, None)
            if job.get("kind") == "wavelet" and panel_id:
                existing = self.td_windows.get(panel_id)
                if existing is not None:
                    existing["wavelet_summary_var"].set(f"Wavelet filter failed: {error_text}")
                    existing["wavelet_physics_var"].set("")
                    self._update_wavelet_job_widgets(panel_id)
                    self._refresh_td_window(panel_id)
                self._set_status(f"Wavelet filter failed for {panel_name}.")
            elif job.get("kind") == "batch":
                self._set_status(f"Batch pipeline failed: {error_text}")

    def _update_wavelet_job_widgets(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        _job_id, job = self._background_wavelet_job_for_panel(panel_id)
        run_button = existing.get("wavelet_run_button")
        cancel_button = existing.get("wavelet_cancel_button")
        progress = existing.get("wavelet_progressbar")
        progress_var = existing.get("wavelet_progress_var")
        if job is None:
            if run_button is not None:
                run_button.configure(state="normal")
            if cancel_button is not None:
                cancel_button.configure(state="disabled")
            if progress is not None:
                progress.configure(mode="determinate", maximum=1.0, value=0.0)
            if progress_var is not None:
                progress_var.set("Idle.")
            return

        if run_button is not None:
            run_button.configure(state="disabled")
        if cancel_button is not None:
            cancel_button.configure(state="normal")
        current = float(job.get("current", 0))
        total = max(float(job.get("total", 1)), 1.0)
        if progress is not None:
            progress.configure(mode="determinate", maximum=total, value=min(current, total))
        if progress_var is not None:
            progress_var.set(str(job.get("message", "Running wavelet filter...")))

    def _wavelet_worker(
        self,
        job_id: str,
        panel_id: int,
        panel_name: str,
        thread_entries: list[dict[str, Any]],
        t_indices: np.ndarray,
        params: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        api, import_error = load_local_wavelet_filter_api()
        if api is None:
            self.background_queue.put(
                {
                    "type": "error",
                    "job_id": job_id,
                    "error": f"{import_error}",
                }
            )
            return

        split_thread_on_jumps = api["split_thread_on_jumps"]
        t_indices = np.asarray(t_indices, dtype=np.float64)
        prepared_segments: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        ctx = mp.get_context("spawn")
        segment_process: Any = None
        task_queue: Any = None
        result_queue: Any = None

        def _stop_segment_worker() -> None:
            nonlocal segment_process, task_queue, result_queue
            if task_queue is not None:
                try:
                    task_queue.put_nowait(None)
                except Exception:
                    pass
            if segment_process is not None:
                try:
                    segment_process.join(timeout=0.2)
                except Exception:
                    pass
                if segment_process.is_alive():
                    try:
                        segment_process.terminate()
                    except Exception:
                        pass
                    try:
                        segment_process.join(timeout=0.5)
                    except Exception:
                        pass
                try:
                    segment_process.close()
                except Exception:
                    pass
            for queue_obj in (task_queue, result_queue):
                if queue_obj is not None:
                    try:
                        queue_obj.close()
                    except Exception:
                        pass
            segment_process = None
            task_queue = None
            result_queue = None

        def _start_segment_worker() -> None:
            nonlocal segment_process, task_queue, result_queue
            _stop_segment_worker()
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            segment_process = ctx.Process(
                target=_wavelet_segment_process_main,
                args=(task_queue, result_queue),
                daemon=True,
            )
            segment_process.start()

        try:
            total_threads = max(len(thread_entries), 1)
            for current_index, entry in enumerate(thread_entries, start=1):
                if cancel_event.is_set():
                    self.background_queue.put({"type": "cancelled", "job_id": job_id})
                    return
                thread_index = int(entry.get("thread_index", current_index - 1))
                thread = entry.get("thread") or {}
                pos = np.asarray(thread.get("pos", []), dtype=np.float64)
                if pos.size != t_indices.size:
                    continue
                valid = np.isfinite(pos) & (pos >= 0.0)
                if np.count_nonzero(valid) < int(params["min_points_cut_seg"]):
                    continue
                t_valid = t_indices[valid]
                y_valid = pos[valid]
                split_segments = split_thread_on_jumps(
                    t_valid,
                    y_valid,
                    max_jump_pix=float(params["max_jump_pix"]),
                    min_points=int(params["min_points_cut_seg"]),
                )
                for seg_id, (t_idx_seg, y_idx_seg) in enumerate(split_segments):
                    prepared_segments.append(
                        (
                            int(thread_index),
                            int(seg_id),
                            np.asarray(t_idx_seg, dtype=np.float64),
                            np.asarray(y_idx_seg, dtype=np.float64),
                        )
                    )
                self.background_queue.put(
                    {
                        "type": "progress",
                        "job_id": job_id,
                        "stage": "prepare",
                        "current": current_index,
                        "total": total_threads,
                        "message": (
                            f"{panel_name}: preparing wavelet segments "
                            f"{current_index}/{total_threads}"
                        ),
                    }
                )

            results: list[dict[str, Any]] = []
            warning_messages: list[str] = []
            total_segments = max(len(prepared_segments), 1)
            self.background_queue.put(
                {
                    "type": "progress",
                    "job_id": job_id,
                    "stage": "analyze",
                    "current": 0,
                    "total": total_segments,
                    "message": (
                        f"{panel_name}: analyzing {len(prepared_segments)} wavelet segment(s)"
                    ),
                }
            )
            if prepared_segments:
                _start_segment_worker()
            for index, (thread_index, seg_id, t_idx_seg, y_idx_seg) in enumerate(
                prepared_segments,
                start=1,
            ):
                if cancel_event.is_set():
                    self.background_queue.put({"type": "cancelled", "job_id": job_id})
                    return
                if segment_process is None or not segment_process.is_alive():
                    _start_segment_worker()
                self.background_queue.put(
                    {
                        "type": "progress",
                        "job_id": job_id,
                        "stage": "analyze",
                        "current": index - 1,
                        "total": total_segments,
                        "message": (
                            f"{panel_name}: analyzing segment {index}/{total_segments} "
                            f"(n={int(len(t_idx_seg))})"
                        ),
                    }
                )
                task_queue.put(
                    {
                        "task_id": int(index),
                        "t_idx_seg": np.asarray(t_idx_seg, dtype=np.float64),
                        "y_idx_seg": np.asarray(y_idx_seg, dtype=np.float64),
                        "cad": float(params["cad"]),
                        "res": float(params["res"]),
                        "km_per_arcsec": float(params["km_per_arcsec"]),
                        "p_min": float(params["p_min"]),
                        "p_max": float(params["p_max"]),
                        "power_ratio_thresh": float(params["power_ratio_thresh"]),
                        "segment_power_frac": float(params["segment_power_frac"]),
                        "min_points_segment": int(params["min_points_segment"]),
                        "min_amp_arcsec": float(params["min_amp_arcsec"]),
                        "rms_amp_ratio_max": float(params["rms_amp_ratio_max"]),
                        "density_kg_m3": float(params["density_kg_m3"]),
                        "phase_speed_km_s": float(params["phase_speed_km_s"]),
                    }
                )
                analysis: dict[str, Any] | None = None
                task_error: str | None = None
                deadline = time.monotonic() + WAVELET_SEGMENT_TIMEOUT_S
                while analysis is None and task_error is None:
                    if cancel_event.is_set():
                        self.background_queue.put({"type": "cancelled", "job_id": job_id})
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        task_error = (
                            f"timeout after {WAVELET_SEGMENT_TIMEOUT_S:.1f}s"
                        )
                        break
                    try:
                        result = result_queue.get(timeout=min(WAVELET_SEGMENT_POLL_S, remaining))
                    except queue.Empty:
                        if segment_process is not None and not segment_process.is_alive():
                            task_error = "worker exited unexpectedly"
                            break
                        continue
                    if bool(result.get("fatal")):
                        raise RuntimeError(
                            str(result.get("error", "wavelet segment worker failed"))
                        )
                    if int(result.get("task_id", -1)) != int(index):
                        continue
                    if not bool(result.get("ok")):
                        task_error = str(
                            result.get("error", "wavelet segment analysis failed")
                        )
                        break
                    analysis = dict(result.get("analysis") or {})

                if task_error is not None:
                    warning_messages.append(
                        f"segment {index}/{total_segments} "
                        f"(thread {thread_index}, seg {seg_id}, n={int(len(t_idx_seg))}): {task_error}"
                    )
                    self.background_queue.put(
                        {
                            "type": "progress",
                            "job_id": job_id,
                            "stage": "analyze",
                            "current": index,
                            "total": total_segments,
                            "message": (
                                f"{panel_name}: skipped segment {index}/{total_segments} "
                                f"({task_error})"
                            ),
                        }
                    )
                    _start_segment_worker()
                    continue

                for candidate in analysis.get("candidates", []):
                    results.append(
                        {
                            "thread_index": int(thread_index),
                            "seg_id": int(seg_id),
                            **candidate,
                            "source_t_idx": np.asarray(t_idx_seg, dtype=np.float64),
                            "source_y_idx": np.asarray(y_idx_seg, dtype=np.float64),
                        }
                    )
                self.background_queue.put(
                    {
                        "type": "progress",
                        "job_id": job_id,
                        "stage": "analyze",
                        "current": index,
                        "total": total_segments,
                        "message": (
                            f"{panel_name}: completed segment {index}/{total_segments} "
                            f"(n={int(len(t_idx_seg))})"
                        ),
                    }
                )

            self.background_queue.put(
                {
                    "type": "done",
                    "job_id": job_id,
                    "segments": results,
                    "params": dict(params),
                    "panel_name": panel_name,
                    "warnings": warning_messages,
                }
            )
        except Exception as exc:
            self.background_queue.put(
                {
                    "type": "error",
                    "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            _stop_segment_worker()

    def _cancel_td_window_wavelet_job(self, panel_id: int) -> None:
        _job_id, job = self._background_wavelet_job_for_panel(panel_id)
        if job is None:
            self._set_status(f"No running wavelet job for panel {panel_id}.")
            return
        cancel_event = job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        self._set_status(
            f"Cancelling wavelet background job for {job.get('panel_name', f'P{panel_id}')}"
        )

    def _preserved_locked_wavelet_events(self, panel_id: int) -> list[dict[str, Any]]:
        preserved: list[dict[str, Any]] = []
        for event in self._panel_wavelet_events_snapshot(panel_id):
            self._ensure_wavelet_event_fields(event)
            if bool(event.get("review_locked")):
                preserved.append(self._clone_wavelet_payload(event))
        preserved.sort(key=lambda item: int(item.get("event_id", -1)))
        return preserved

    def _apply_background_wavelet_results(
        self,
        panel_id: int,
        segments: list[dict[str, Any]],
        params: dict[str, Any],
        panel_name: str,
        warnings: list[str] | None = None,
    ) -> None:
        _trace_stack_wavelet(
            f"apply_background_wavelet_results start panel={panel_id} segments={len(segments)} warnings={len(warnings or [])}"
        )
        existing = self.td_windows.get(panel_id)
        warning_messages = [str(item) for item in (warnings or []) if str(item)]
        preserved_locked_events = self._preserved_locked_wavelet_events(panel_id)
        filtered_segments = [
            segment
            for segment in segments
            if not any(
                self._wavelet_segment_matches_locked_event(segment, locked_event)
                for locked_event in preserved_locked_events
            )
        ]
        _trace_stack_wavelet(
            f"apply_background_wavelet_results filtered panel={panel_id} filtered_segments={len(filtered_segments)} preserved_locked={len(preserved_locked_events)}"
        )
        accepted_count = sum(1 for segment in filtered_segments if segment.get("accepted"))
        with_segment_count = sum(
            1 for segment in filtered_segments if segment.get("has_segment")
        )
        used_ids = {int(event.get("event_id", -1)) for event in preserved_locked_events}
        next_event_id = max(used_ids, default=0) + 1
        events = self._clone_wavelet_payload(preserved_locked_events)
        for segment in filtered_segments:
            new_event = self._make_td_window_wavelet_event(next_event_id, segment, params)
            events.append(new_event)
            next_event_id += 1
        _trace_stack_wavelet(
            f"apply_background_wavelet_results events_built panel={panel_id} events={len(events)}"
        )
        best_segment = self._best_wavelet_segment(filtered_segments)
        best_event_id = None
        if best_segment is not None:
            for event in events:
                analysis = event.get("analysis") or {}
                if (
                    int(analysis.get("thread_index", -999)) == int(best_segment.get("thread_index", -1))
                    and int(analysis.get("seg_id", -999)) == int(best_segment.get("seg_id", -1))
                    and int(analysis.get("wseg_id", -999)) == int(best_segment.get("wseg_id", -1))
                ):
                    best_event_id = int(event["event_id"])
                    break

        state = self._panel_analysis(panel_id)
        _trace_stack_wavelet(f"apply_background_wavelet_results state_assign panel={panel_id}")
        state["wavelet_filter_result"] = self._lightweight_wavelet_filter_result(
            segments=filtered_segments,
            params=params,
            best_segment=best_segment,
            preserved_locked_event_ids=[
                int(event.get("event_id", -1)) for event in preserved_locked_events
            ],
            warnings=warning_messages,
        )
        state["wavelet_events"] = self._clone_wavelet_payload(events)
        state["wavelet_next_event_id"] = next_event_id
        state["wavelet_selected_event_id"] = best_event_id or (
            int(events[0]["event_id"]) if events else None
        )
        _trace_stack_wavelet(f"apply_background_wavelet_results state_assigned panel={panel_id}")

        if existing is not None:
            _trace_stack_wavelet(f"apply_background_wavelet_results existing_assign panel={panel_id}")
            existing["wavelet_filter_result"] = dict(state["wavelet_filter_result"])
            existing["wavelet_events"] = self._clone_wavelet_payload(events)
            existing["wavelet_next_event_id"] = next_event_id
            existing["wavelet_selected_event_id"] = state["wavelet_selected_event_id"]
            existing["wavelet_summary_var"].set(
                f"Wavelet accepted {accepted_count}/{len(filtered_segments)} segment(s). "
                f"Selected by wavelet: {with_segment_count}. "
                f"Locked preserved: {len(preserved_locked_events)}."
                + (f" Warnings: {len(warning_messages)}." if warning_messages else "")
            )
            self._update_wavelet_job_widgets(panel_id)
            _trace_stack_wavelet(f"apply_background_wavelet_results before_refresh_td_wavelet_views panel={panel_id}")
            self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
            _trace_stack_wavelet(f"apply_background_wavelet_results after_refresh_td_wavelet_views panel={panel_id}")

        _trace_stack_wavelet(f"apply_background_wavelet_results before_record_session_change panel={panel_id}")
        self._record_session_change()
        _trace_stack_wavelet(f"apply_background_wavelet_results end panel={panel_id}")
        self._set_status(
            f"Wavelet filter completed for {panel_name}: {accepted_count} accepted segment(s)."
            + (f" First warning: {warning_messages[0]}" if warning_messages else "")
        )

    def _run_batch_pipeline(self) -> None:
        for job in self.background_jobs.values():
            if job.get("kind") in {"batch", "wavelet"}:
                self._set_status(
                    "Wait for the current background wavelet/batch job to finish first."
                )
                return

        self._sync_all_panel_analysis_state_from_windows()
        panel_specs: list[dict[str, Any]] = []
        for panel in self.panels:
            if panel.cut_id is None or panel.cut_id not in self.cuts:
                continue
            cut = self.cuts[panel.cut_id]
            state = self._panel_analysis_snapshot(panel.panel_id)
            crest_params = dict(DEFAULT_CREST_TRACKING)
            crest_params.update(state.get("crest_params") or {})
            wavelet_params = dict(DEFAULT_WAVELET_FILTER)
            wavelet_params.update(state.get("wavelet_params") or {})
            panel_specs.append(
                {
                    "panel_id": panel.panel_id,
                    "panel_name": panel.name,
                    "cut_id": cut.cut_id,
                    "cut_name": cut.name,
                    "cut_p0": [float(cut.p0[0]), float(cut.p0[1])],
                    "cut_p1": [float(cut.p1[0]), float(cut.p1[1])],
                    "t_ini": int(panel.t_ini),
                    "t_fin": int(panel.t_fin),
                    "stride": int(panel.stride),
                    "width": int(panel.width),
                    "weighting": str(panel.weighting),
                    "crest_params": crest_params,
                    "wavelet_params": wavelet_params,
                }
            )

        if not panel_specs:
            self._set_status("Assign at least one cut to a panel before running batch mode.")
            return

        job_id, job = self._new_background_job(kind="batch", panel_name="batch")
        job["message"] = f"Queued batch pipeline for {len(panel_specs)} panel(s)."
        thread = threading.Thread(
            target=self._batch_pipeline_worker,
            args=(job_id, panel_specs, job["cancel_event"]),
            daemon=True,
        )
        job["thread"] = thread
        thread.start()
        self._set_status(f"Running batch pipeline for {len(panel_specs)} panel(s)...")

    def _batch_pipeline_worker(
        self,
        job_id: str,
        panel_specs: list[dict[str, Any]],
        cancel_event: threading.Event,
    ) -> None:
        nuwt_api, nuwt_error = load_local_nuwt_api()
        if nuwt_api is None:
            self.background_queue.put(
                {"type": "error", "job_id": job_id, "error": str(nuwt_error)}
            )
            return

        wavelet_api, wavelet_error = load_local_wavelet_filter_api()
        if wavelet_api is None:
            self.background_queue.put(
                {"type": "error", "job_id": job_id, "error": str(wavelet_error)}
            )
            return

        results: dict[str, Any] = {}
        total = max(len(panel_specs), 1)
        try:
            for index, spec in enumerate(panel_specs, start=1):
                if cancel_event.is_set():
                    self.background_queue.put({"type": "cancelled", "job_id": job_id})
                    return
                cut = Cut(
                    cut_id=int(spec["cut_id"]),
                    name=str(spec["cut_name"]),
                    color="tab:blue",
                    p0=(float(spec["cut_p0"][0]), float(spec["cut_p0"][1])),
                    p1=(float(spec["cut_p1"][0]), float(spec["cut_p1"][1])),
                )
                td, meta = compute_td(
                    self.cube,
                    cut,
                    int(spec["t_ini"]),
                    int(spec["t_fin"]),
                    int(spec["stride"]),
                    int(spec["width"]),
                    str(spec["weighting"]),
                )
                td_nuwt = np.asarray(td, dtype=np.float64).T
                finite = np.isfinite(td_nuwt)
                if not np.any(finite):
                    raise ValueError(f"{spec['panel_name']} has no finite TD values.")
                fill_value = float(np.nanmin(td_nuwt[finite]))
                if not np.all(finite):
                    td_nuwt = np.where(finite, td_nuwt, fill_value)

                crest_params = dict(spec["crest_params"])
                located = nuwt_api["locate_things"](
                    td_nuwt,
                    invert=bool(crest_params["invert"]),
                    grad=float(crest_params["grad"]),
                    res=float(crest_params["res"]),
                    cad=float(crest_params["cad"]),
                    nearest_pixel=not bool(crest_params["gauss"]),
                )
                threads, _ = nuwt_api["follow_threads"](
                    located,
                    min_tlen=int(crest_params["min_tlen"]),
                    max_dist_jump=int(crest_params["max_dist_jump"]),
                    max_time_skip=int(crest_params["max_time_skip"]),
                )
                threads = nuwt_api["patch_up_threads"](
                    threads, fit_flag=0, simp_fill=False, debug=False
                )
                wavelet_params = dict(spec["wavelet_params"])
                segments = wavelet_api["analyze_tracked_threads_with_wavelets"](
                    threads,
                    np.asarray(meta["t_indices"], dtype=np.float64),
                    cadence=float(crest_params["cad"]),
                    pix_scale=float(crest_params["res"]),
                    km_per_arcsec=float(wavelet_params["km_per_arcsec"]),
                    p_min=float(wavelet_params["p_min"]),
                    p_max=float(wavelet_params["p_max"]),
                    power_ratio_thresh=float(wavelet_params["power_ratio_thresh"]),
                    segment_power_frac=float(wavelet_params["segment_power_frac"]),
                    min_points_segment=int(wavelet_params["min_points_segment"]),
                    min_amp_arcsec=float(wavelet_params["min_amp_arcsec"]),
                    max_jump_pix=float(wavelet_params["max_jump_pix"]),
                    min_points_cut_seg=int(wavelet_params["min_points_cut_seg"]),
                    rms_amp_ratio_max=float(wavelet_params["rms_amp_ratio_max"]),
                    density_kg_m3=float(wavelet_params["density_kg_m3"]),
                    phase_speed_km_s=float(wavelet_params["phase_speed_km_s"]),
                )
                located_count = int(np.count_nonzero(np.asarray(located["errs"]) > 0))
                thread_count = len(threads)
                longest = max((int(th.get("length", 0)) for th in threads), default=0)
                results[str(spec["panel_id"])] = {
                    "crest_tracking_result": {
                        "located": located,
                        "threads": threads,
                        "params": crest_params,
                    },
                    "crest_tracking_td_key": (
                        int(spec["cut_id"]),
                        round(float(spec["cut_p0"][0]), 3),
                        round(float(spec["cut_p0"][1]), 3),
                        round(float(spec["cut_p1"][0]), 3),
                        round(float(spec["cut_p1"][1]), 3),
                        int(spec["t_ini"]),
                        int(spec["t_fin"]),
                        int(spec["stride"]),
                        int(spec["width"]),
                        str(spec["weighting"]),
                    ),
                    "crest_summary": (
                        f"Located {located_count} crest bins. "
                        f"Threads: {thread_count}. Longest: {longest}."
                    ),
                    "wavelet_segments": segments,
                    "wavelet_params": {
                        "cad": float(crest_params["cad"]),
                        "res": float(crest_params["res"]),
                        **wavelet_params,
                    },
                }
                self.background_queue.put(
                    {
                        "type": "progress",
                        "job_id": job_id,
                        "stage": "batch",
                        "current": index,
                        "total": total,
                        "message": f"Batch pipeline: {spec['panel_name']} ({index}/{total})",
                    }
                )
        except Exception as exc:
            self.background_queue.put(
                {
                    "type": "error",
                    "job_id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return

        self.background_queue.put(
            {
                "type": "done",
                "job_id": job_id,
                "results": results,
                "summary": f"Batch pipeline completed for {len(panel_specs)} panel(s).",
            }
        )

    def _apply_background_batch_results(
        self, results: dict[str, Any], summary: str
    ) -> None:
        for panel_id_text, payload in results.items():
            panel_id = int(panel_id_text)
            state = self._panel_analysis(panel_id)
            state["crest_tracking_result"] = self._clone_wavelet_payload(
                payload.get("crest_tracking_result")
            )
            state["crest_tracking_td_key"] = self._clone_wavelet_payload(
                payload.get("crest_tracking_td_key")
            )
            params = dict(payload.get("wavelet_params") or {})
            segments = payload.get("wavelet_segments") or []
            preserved_locked_events = self._preserved_locked_wavelet_events(panel_id)
            filtered_segments = [
                segment
                for segment in segments
                if not any(
                    self._wavelet_segment_matches_locked_event(segment, locked_event)
                    for locked_event in preserved_locked_events
                )
            ]
            used_ids = {
                int(event.get("event_id", -1)) for event in preserved_locked_events
            }
            next_event_id = max(used_ids, default=0) + 1
            events = self._clone_wavelet_payload(preserved_locked_events)
            for segment in filtered_segments:
                events.append(
                    self._make_td_window_wavelet_event(next_event_id, segment, params)
                )
                next_event_id += 1
            best_segment = self._best_wavelet_segment(filtered_segments)
            best_event_id = None
            if best_segment is not None:
                for event in events:
                    analysis = event.get("analysis") or {}
                    if (
                        int(analysis.get("thread_index", -999)) == int(best_segment.get("thread_index", -1))
                        and int(analysis.get("seg_id", -999)) == int(best_segment.get("seg_id", -1))
                        and int(analysis.get("wseg_id", -999)) == int(best_segment.get("wseg_id", -1))
                    ):
                        best_event_id = int(event["event_id"])
                        break
            state["wavelet_filter_result"] = self._lightweight_wavelet_filter_result(
                segments=filtered_segments,
                params=params,
                best_segment=best_segment,
                preserved_locked_event_ids=[
                    int(event.get("event_id", -1)) for event in preserved_locked_events
                ],
                warnings=[],
            )
            state["wavelet_events"] = self._clone_wavelet_payload(events)
            state["wavelet_next_event_id"] = next_event_id
            state["wavelet_selected_event_id"] = best_event_id or (
                int(events[0]["event_id"]) if events else None
            )

            existing = self.td_windows.get(panel_id)
            if existing is not None:
                existing["crest_tracking_result"] = self._clone_wavelet_payload(
                    state["crest_tracking_result"]
                )
                existing["crest_tracking_td_key"] = self._clone_wavelet_payload(
                    state["crest_tracking_td_key"]
                )
                existing["wavelet_filter_result"] = dict(state["wavelet_filter_result"])
                existing["wavelet_events"] = self._clone_wavelet_payload(events)
                existing["wavelet_next_event_id"] = next_event_id
                existing["wavelet_selected_event_id"] = state["wavelet_selected_event_id"]
                existing["crest_summary_var"].set(
                    str(payload.get("crest_summary", "Crest tracking completed."))
                )
                accepted_count = sum(
                    1 for segment in filtered_segments if segment.get("accepted")
                )
                with_segment_count = sum(
                    1 for segment in filtered_segments if segment.get("has_segment")
                )
                existing["wavelet_summary_var"].set(
                    f"Wavelet accepted {accepted_count}/{len(filtered_segments)} segment(s). "
                    f"Selected by wavelet: {with_segment_count}. "
                    f"Locked preserved: {len(preserved_locked_events)}."
                )
                self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)

        self._record_session_change()
        self.refresh_td_views()
        self._set_status(summary or "Batch pipeline completed.")

    def _td_window_wavelet_event_final_mode(self, event: dict[str, Any]) -> str:
        origin = str(event.get("origin", "") or "")
        if event.get("split_children_ids") or origin == "manual-split":
            return "split"
        if origin == "manual-trim":
            return "trim"
        if event.get("manual_decision") is not None or event.get("customized"):
            return "manual"
        return "auto"

    def _td_window_wavelet_event_qa_flags(self, event: dict[str, Any]) -> list[str]:
        analysis = event.get("analysis") or {}
        params = dict(event.get("current_params") or event.get("base_params") or {})
        flags: list[str] = []
        point_count = int(np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64).size)
        min_points = max(
            min(
                int(params.get("min_points_segment", 3)),
                int(params.get("min_points_cut_seg", 3)),
            ),
            3,
        )
        if point_count <= min_points + WAVELET_QA_LOW_POINTS_MARGIN:
            flags.append("few_points")
        peak_period = float(analysis.get("peak_period_s", float("nan")))
        p_min = float(params.get("p_min", float("nan")))
        p_max = float(params.get("p_max", float("nan")))
        if (
            np.isfinite(peak_period)
            and np.isfinite(p_min)
            and np.isfinite(p_max)
            and p_max > p_min
        ):
            edge_margin = max((p_max - p_min) * WAVELET_QA_EDGE_FRACTION, 1e-6)
            if (peak_period - p_min) <= edge_margin or (p_max - peak_period) <= edge_margin:
                flags.append("period_edge")
        residual = float(
            analysis.get(
                "fit_rms_over_amp",
                analysis.get("rms_amp_ratio", float("nan")),
            )
        )
        rms_limit = float(params.get("rms_amp_ratio_max", float("nan")))
        if (
            np.isfinite(residual)
            and np.isfinite(rms_limit)
            and residual >= rms_limit * WAVELET_QA_RESIDUAL_WARN_FRAC
        ):
            flags.append("high_residual")
        return flags

    def _curated_group_key(
        self,
        cut_id: int | None,
        event_id: int | None,
        link_group_id: str | None,
    ) -> str:
        group_id = str(link_group_id or "").strip()
        if group_id:
            return group_id
        return f"CUT{int(cut_id or -1):04d}:EVENT{int(event_id or -1):04d}"

    def _curated_row_rank_value(self, value: Any) -> float:
        try:
            number = float(value)
        except Exception:
            return float("-inf")
        return number if np.isfinite(number) else float("-inf")

    def _collect_curated_event_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cut_id in sorted(self.cuts.keys()):
            cut = self.cuts.get(cut_id)
            if cut is None:
                continue
            state = self._cut_analysis_snapshot(cut_id)
            crest_params = dict(DEFAULT_CREST_TRACKING)
            crest_params.update(state.get("crest_params") or {})
            wavelet_params = dict(DEFAULT_WAVELET_FILTER)
            wavelet_params.update(state.get("wavelet_params") or {})
            panels = self._panels_for_cut(cut_id)
            primary_panel = panels[0] if panels else None
            memberships = self._stack_memberships_for_cut(cut_id)
            dynamic_enabled = self._cut_dynamic_enabled(cut_id)
            dynamic_reference_frame = self._cut_dynamic_reference_frame(cut_id)
            dynamic_keyframes = self._cut_dynamic_keyframes(cut_id)
            for event in state.get("wavelet_events") or []:
                self._ensure_wavelet_event_fields(event)
                analysis = event.get("analysis") or {}
                qa_flags = self._td_window_wavelet_event_qa_flags(event)
                confidence_score, confidence_label = self._wavelet_event_confidence_details(
                    event
                )
                link_refs = self._wavelet_event_link_refs(event)
                rows.append(
                    {
                        "panel_id": int(primary_panel.panel_id) if primary_panel is not None else None,
                        "panel_name": primary_panel.name if primary_panel is not None else "",
                        "primary_panel_id": int(primary_panel.panel_id) if primary_panel is not None else None,
                        "primary_panel_name": primary_panel.name if primary_panel is not None else "",
                        "panel_ids": [int(panel.panel_id) for panel in panels],
                        "panel_names": [str(panel.name) for panel in panels],
                        "panel_count": int(len(panels)),
                        "cut_id": int(cut.cut_id),
                        "cut_name": cut.name,
                        "event_id": int(event.get("event_id", -1)),
                        "parent_event_id": (
                            int(event.get("parent_event_id"))
                            if event.get("parent_event_id") is not None
                            else None
                        ),
                        "final_mode": self._td_window_wavelet_event_final_mode(event),
                        "status": self._td_window_wavelet_event_status(event),
                        "counted": bool(self._td_window_wavelet_event_is_counted(event)),
                        "origin": str(event.get("origin", "")),
                        "manual_decision": event.get("manual_decision"),
                        "reason": self._td_window_wavelet_event_reason(event),
                        "qa_flags": qa_flags,
                        "qa_flags_text": ",".join(qa_flags),
                        "confidence_score": confidence_score,
                        "confidence_label": confidence_label,
                        "review_locked": bool(event.get("review_locked")),
                        "review_notes": str(event.get("review_notes", "")),
                        "history_count": len(event.get("history") or []),
                        "history": self._clone_wavelet_payload(event.get("history") or []),
                        "link_group_id": event.get("link_group_id"),
                        "propagation_class": str(event.get("propagation_class") or ""),
                        "link_count": self._wavelet_event_link_count(event),
                        "link_refs": self._clone_wavelet_payload(link_refs),
                        "stack_ids": [int(item["stack_id"]) for item in memberships],
                        "stack_names": [str(item["stack_name"]) for item in memberships],
                        "stack_positions": [
                            f"{int(item['stack_index'])}/{int(item['stack_size'])}"
                            for item in memberships
                        ],
                        "stack_refs": self._clone_wavelet_payload(memberships),
                        "dynamic_enabled": bool(dynamic_enabled),
                        "dynamic_reference_frame": int(dynamic_reference_frame),
                        "dynamic_keyframe_count": int(len(dynamic_keyframes)),
                        "dynamic_keyframe_frames": [int(frame_idx) for frame_idx in sorted(dynamic_keyframes.keys())],
                        "preset_name": str(state.get("preset_name", "custom")),
                        "thread_index": int(analysis.get("thread_index", -1)),
                        "seg_id": int(analysis.get("seg_id", -1)),
                        "wseg_id": int(analysis.get("wseg_id", -1)),
                        "peak_period_s": float(analysis.get("peak_period_s", float("nan"))),
                        "freq_mhz": float(analysis.get("freq_mhz", float("nan"))),
                        "fit_amp_arcsec": float(analysis.get("fit_amp_arcsec", float("nan"))),
                        "fit_amp_km": float(analysis.get("fit_amp_km", float("nan"))),
                        "velocity_amp_km_s": float(analysis.get("velocity_amp_km_s", float("nan"))),
                        "accel_amp_km_s2": float(analysis.get("accel_amp_km_s2", float("nan"))),
                        "specific_energy_j_kg": float(analysis.get("specific_energy_j_kg", float("nan"))),
                        "energy_flux_w_m2": float(analysis.get("energy_flux_w_m2", float("nan"))),
                        "duration_s": float(analysis.get("duration_s", float("nan"))),
                        "power_ratio": float(analysis.get("power_ratio", float("nan"))),
                        "fit_rms_over_amp": float(
                            analysis.get(
                                "fit_rms_over_amp",
                                analysis.get("rms_amp_ratio", float("nan")),
                            )
                        ),
                        "point_count": int(np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64).size),
                        "source_start_frame": (
                            float(np.asarray(event.get("source_t_idx", []), dtype=np.float64)[0])
                            if np.asarray(event.get("source_t_idx", []), dtype=np.float64).size
                            else float("nan")
                        ),
                        "source_end_frame": (
                            float(np.asarray(event.get("source_t_idx", []), dtype=np.float64)[-1])
                            if np.asarray(event.get("source_t_idx", []), dtype=np.float64).size
                            else float("nan")
                        ),
                        "wave_start_frame": (
                            float(np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64)[0])
                            if np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64).size
                            else float("nan")
                        ),
                        "wave_end_frame": (
                            float(np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64)[-1])
                            if np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64).size
                            else float("nan")
                        ),
                        "cad": float(crest_params["cad"]),
                        "res": float(crest_params["res"]),
                        "grad": float(crest_params["grad"]),
                        "min_tlen": int(crest_params["min_tlen"]),
                        "max_dist_jump": int(crest_params["max_dist_jump"]),
                        "max_time_skip": int(crest_params["max_time_skip"]),
                        "invert": bool(crest_params["invert"]),
                        "gauss": bool(crest_params["gauss"]),
                        "p_min": float(wavelet_params["p_min"]),
                        "p_max": float(wavelet_params["p_max"]),
                        "power_ratio_thresh": float(wavelet_params["power_ratio_thresh"]),
                        "segment_power_frac": float(wavelet_params["segment_power_frac"]),
                        "min_points_segment": int(wavelet_params["min_points_segment"]),
                        "min_amp_arcsec": float(wavelet_params["min_amp_arcsec"]),
                        "max_jump_pix": float(wavelet_params["max_jump_pix"]),
                        "min_points_cut_seg": int(wavelet_params["min_points_cut_seg"]),
                        "rms_amp_ratio_max": float(wavelet_params["rms_amp_ratio_max"]),
                        "km_per_arcsec": float(wavelet_params["km_per_arcsec"]),
                        "density_kg_m3": float(wavelet_params["density_kg_m3"]),
                        "phase_speed_km_s": float(wavelet_params["phase_speed_km_s"]),
                    }
                )
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            group_key = self._curated_group_key(
                row.get("cut_id"),
                row.get("event_id"),
                row.get("link_group_id"),
            )
            groups.setdefault(group_key, []).append(row)
        for group_key, members in groups.items():
            representative_amp = max(
                members,
                key=lambda row: (
                    self._curated_row_rank_value(row.get("fit_amp_arcsec")),
                    self._curated_row_rank_value(row.get("confidence_score")),
                    self._curated_row_rank_value(row.get("duration_s")),
                ),
            )
            representative_conf = max(
                members,
                key=lambda row: (
                    self._curated_row_rank_value(row.get("confidence_score")),
                    self._curated_row_rank_value(row.get("fit_amp_arcsec")),
                    self._curated_row_rank_value(row.get("duration_s")),
                ),
            )
            class_labels = sorted(
                {
                    str(member.get("propagation_class") or "").strip()
                    for member in members
                    if str(member.get("propagation_class") or "").strip()
                }
            )
            resolved_class = (
                class_labels[0]
                if len(class_labels) == 1
                else ",".join(class_labels)
                if class_labels
                else ("local" if len(members) == 1 else "same-wave")
            )
            for row in members:
                row["group_key"] = group_key
                row["group_member_count"] = int(len(members))
                row["is_group_representative_amp"] = bool(row is representative_amp)
                row["is_group_representative_conf"] = bool(row is representative_conf)
                row["resolved_propagation_class"] = resolved_class
        return rows

    def _export_curated_results(self) -> None:
        rows = self._collect_curated_event_rows()
        default_name = self.cube_path.stem + "_wavelet_curated.csv"
        save_path = self.filedialog.asksaveasfilename(
            title="Export curated results",
            initialdir=str(Path(__file__).resolve().parent.parent),
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not save_path:
            return
        save_path_obj = Path(save_path).expanduser().resolve()
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if save_path_obj.suffix.lower() == ".json":
            payload = {
                "cube_path": str(self.cube_path),
                "rows": self._json_safe(rows),
                "summary": {
                    "event_count": len(rows),
                    "counted_event_count": sum(1 for row in rows if row["counted"]),
                },
            }
            with open(save_path_obj, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        else:
            fieldnames = list(rows[0].keys()) if rows else [
                "panel_id",
                "panel_name",
                "cut_id",
                "cut_name",
                "event_id",
                "final_mode",
                "status",
                "counted",
                "qa_flags_text",
            ]
            with open(save_path_obj, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    out_row = dict(row)
                    for key, value in list(out_row.items()):
                        if isinstance(value, list) and key == "qa_flags":
                            out_row[key] = ",".join(str(item) for item in value)
                        elif isinstance(value, (list, dict)):
                            out_row[key] = json.dumps(self._json_safe(value), ensure_ascii=False)
                    writer.writerow(out_row)
        self._set_status(f"Curated results exported to {save_path_obj}")

    def _export_curated_report(self) -> None:
        rows = self._collect_curated_event_rows()
        default_name = self.cube_path.stem + "_wavelet_report.pdf"
        save_path = self.filedialog.asksaveasfilename(
            title="Export curated report",
            initialdir=str(Path(__file__).resolve().parent.parent),
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"), ("All files", "*.*")],
        )
        if not save_path:
            return

        save_path_obj = Path(save_path).expanduser().resolve()
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        counted_rows = [row for row in rows if row["counted"]]
        flagged_rows = [row for row in rows if row["qa_flags"]]
        linked_rows = [row for row in rows if int(row.get("link_count", 0)) > 0]
        locked_rows = [row for row in rows if bool(row.get("review_locked"))]
        fig = self.Figure(figsize=(11.0, 8.5), dpi=160)
        grid = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 1.0], hspace=0.38, wspace=0.25)
        text_ax = fig.add_subplot(grid[0, :])
        axes = (
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[2, 1]),
        )

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in counted_rows:
            key = (str(row["panel_name"]), str(row["cut_name"]))
            grouped.setdefault(key, []).append(row)

        text_ax.axis("off")
        summary_lines = [
            f"Cube: {self.cube_path.name}",
            f"Panels with cuts: {sum(1 for panel in self.panels if panel.cut_id is not None)}",
            f"Events total: {len(rows)} | counted: {len(counted_rows)} | flagged: {len(flagged_rows)} | "
            f"locked: {len(locked_rows)} | linked: {len(linked_rows)}",
            f"Generated: {self._timestamp_now()}",
            "",
        ]
        if grouped:
            for (panel_name, cut_name), items in sorted(grouped.items()):
                periods = [
                    float(item["peak_period_s"])
                    for item in items
                    if np.isfinite(float(item["peak_period_s"]))
                ]
                amps = [
                    float(item["fit_amp_arcsec"])
                    for item in items
                    if np.isfinite(float(item["fit_amp_arcsec"]))
                ]
                vels = [
                    float(item["velocity_amp_km_s"])
                    for item in items
                    if np.isfinite(float(item["velocity_amp_km_s"]))
                ]
                energies = [
                    float(item["specific_energy_j_kg"])
                    for item in items
                    if np.isfinite(float(item["specific_energy_j_kg"]))
                ]
                summary_lines.append(
                    f"{panel_name}/{cut_name or '-'}: n={len(items)} | "
                    f"Pmed={np.nanmedian(periods) if periods else float('nan'):.2f}s | "
                    f"Amed={np.nanmedian(amps) if amps else float('nan'):.3f}'' | "
                    f"vmed={np.nanmedian(vels) if vels else float('nan'):.2f} km/s | "
                    f"Emed={np.nanmedian(energies) if energies else float('nan'):.3e} J/kg"
                )
        else:
            summary_lines.append("No counted curated events available.")
        text_ax.text(
            0.0,
            1.0,
            "\n".join(summary_lines[:18]),
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            transform=text_ax.transAxes,
        )

        metric_specs = [
            ("peak_period_s", "Period [s]"),
            ("fit_amp_arcsec", "Amplitude ['']"),
            ("velocity_amp_km_s", "Velocity [km/s]"),
            ("specific_energy_j_kg", "Energy/m [J/kg]"),
        ]
        for axis, (key, title) in zip(axes, metric_specs):
            values = [
                float(row[key]) for row in counted_rows if np.isfinite(float(row[key]))
            ]
            if values:
                axis.hist(values, bins=min(max(len(values), 5), 20), color="tab:blue", alpha=0.82)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No counted events.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(title, fontsize=10)
            axis.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(save_path_obj, bbox_inches="tight")
        self._set_status(f"Curated report exported to {save_path_obj}")

    def _collect_propagation_group_rows(
        self, observation_rows: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        rows = observation_rows if observation_rows is not None else self._collect_curated_event_rows()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            group_key = self._curated_group_key(
                row.get("cut_id"),
                row.get("event_id"),
                row.get("link_group_id"),
            )
            grouped.setdefault(group_key, []).append(row)

        group_rows: list[dict[str, Any]] = []
        for group_key, members in sorted(grouped.items()):
            representative_amp = max(
                members,
                key=lambda row: (
                    self._curated_row_rank_value(row.get("fit_amp_arcsec")),
                    self._curated_row_rank_value(row.get("confidence_score")),
                    self._curated_row_rank_value(row.get("duration_s")),
                ),
            )
            representative_conf = max(
                members,
                key=lambda row: (
                    self._curated_row_rank_value(row.get("confidence_score")),
                    self._curated_row_rank_value(row.get("fit_amp_arcsec")),
                    self._curated_row_rank_value(row.get("duration_s")),
                ),
            )
            unique_classes = sorted(
                {
                    str(member.get("propagation_class") or "").strip()
                    for member in members
                    if str(member.get("propagation_class") or "").strip()
                }
            )
            resolved_class = (
                unique_classes[0]
                if len(unique_classes) == 1
                else ",".join(unique_classes)
                if unique_classes
                else ("local" if len(members) == 1 else "same-wave")
            )
            cut_ids = sorted({int(member["cut_id"]) for member in members if member.get("cut_id") is not None})
            stack_ids = sorted(
                {
                    int(stack_id)
                    for member in members
                    for stack_id in (member.get("stack_ids") or [])
                }
            )
            periods = [
                float(member["peak_period_s"])
                for member in members
                if np.isfinite(float(member.get("peak_period_s", float("nan"))))
            ]
            velocities = [
                float(member["velocity_amp_km_s"])
                for member in members
                if np.isfinite(float(member.get("velocity_amp_km_s", float("nan"))))
            ]
            energies = [
                float(member["specific_energy_j_kg"])
                for member in members
                if np.isfinite(float(member.get("specific_energy_j_kg", float("nan"))))
            ]
            amps = [
                float(member["fit_amp_arcsec"])
                for member in members
                if np.isfinite(float(member.get("fit_amp_arcsec", float("nan"))))
            ]
            confidences = [
                float(member["confidence_score"])
                for member in members
                if np.isfinite(float(member.get("confidence_score", float("nan"))))
            ]
            group_rows.append(
                {
                    "group_key": group_key,
                    "link_group_id": representative_amp.get("link_group_id"),
                    "synthetic_group": not bool(str(representative_amp.get("link_group_id") or "").strip()),
                    "classification": resolved_class,
                    "class_labels": unique_classes,
                    "member_count": int(len(members)),
                    "cut_count": int(len(cut_ids)),
                    "counted_count": int(sum(1 for member in members if member.get("counted"))),
                    "locked_count": int(sum(1 for member in members if member.get("review_locked"))),
                    "qa_flagged_count": int(sum(1 for member in members if member.get("qa_flags"))),
                    "cut_ids": cut_ids,
                    "cut_names": sorted({str(member["cut_name"]) for member in members}),
                    "stack_ids": stack_ids,
                    "stack_names": sorted(
                        {
                            str(stack_name)
                            for member in members
                            for stack_name in (member.get("stack_names") or [])
                        }
                    ),
                    "representative_cut_id": representative_amp.get("cut_id"),
                    "representative_cut_name": representative_amp.get("cut_name"),
                    "representative_event_id": representative_amp.get("event_id"),
                    "representative_amp_arcsec": representative_amp.get("fit_amp_arcsec"),
                    "best_conf_cut_id": representative_conf.get("cut_id"),
                    "best_conf_cut_name": representative_conf.get("cut_name"),
                    "best_conf_event_id": representative_conf.get("event_id"),
                    "best_confidence_score": representative_conf.get("confidence_score"),
                    "max_amp_arcsec": float(np.nanmax(amps)) if amps else float("nan"),
                    "median_period_s": float(np.nanmedian(periods)) if periods else float("nan"),
                    "median_velocity_km_s": float(np.nanmedian(velocities)) if velocities else float("nan"),
                    "median_specific_energy_j_kg": (
                        float(np.nanmedian(energies)) if energies else float("nan")
                    ),
                    "mean_confidence_score": (
                        float(np.nanmean(confidences)) if confidences else float("nan")
                    ),
                    "members": self._clone_wavelet_payload(members),
                }
            )
        return group_rows

    def _export_propagation_tables(self) -> None:
        observation_rows = self._collect_curated_event_rows()
        group_rows = self._collect_propagation_group_rows(observation_rows)
        default_name = self.cube_path.stem + "_propagation_tables.json"
        save_path = self.filedialog.asksaveasfilename(
            title="Export propagation tables",
            initialdir=str(Path(__file__).resolve().parent.parent),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not save_path:
            return

        save_path_obj = Path(save_path).expanduser().resolve()
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if save_path_obj.suffix.lower() == ".json":
            payload = {
                "cube_path": str(self.cube_path),
                "observations": self._json_safe(observation_rows),
                "groups": self._json_safe(group_rows),
                "summary": {
                    "observation_count": len(observation_rows),
                    "group_count": len(group_rows),
                    "linked_group_count": sum(
                        1 for row in group_rows if not bool(row.get("synthetic_group"))
                    ),
                },
            }
            with open(save_path_obj, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._set_status(f"Propagation tables exported to {save_path_obj}")
            return

        observations_path = save_path_obj
        groups_path = save_path_obj.with_name(save_path_obj.stem + "_groups.csv")

        observation_fields = list(observation_rows[0].keys()) if observation_rows else [
            "cut_id",
            "cut_name",
            "event_id",
            "group_key",
            "classification",
        ]
        with open(observations_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=observation_fields)
            writer.writeheader()
            for row in observation_rows:
                out_row = dict(row)
                for key, value in list(out_row.items()):
                    if isinstance(value, (list, dict)):
                        out_row[key] = json.dumps(self._json_safe(value), ensure_ascii=False)
                writer.writerow(out_row)

        group_fields = list(group_rows[0].keys()) if group_rows else [
            "group_key",
            "classification",
            "member_count",
            "cut_count",
        ]
        with open(groups_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=group_fields)
            writer.writeheader()
            for row in group_rows:
                out_row = dict(row)
                for key, value in list(out_row.items()):
                    if isinstance(value, (list, dict)):
                        out_row[key] = json.dumps(self._json_safe(value), ensure_ascii=False)
                writer.writerow(out_row)

        self._set_status(
            f"Propagation tables exported to {observations_path} and {groups_path}"
        )

    def _open_metrics_window(self) -> None:
        existing = self.metrics_window
        if existing is not None:
            top = existing.get("top")
            if top is not None and top.winfo_exists():
                top.deiconify()
                top.lift()
                self._refresh_metrics_window()
                return
            self.metrics_window = None

        top = self.tk.Toplevel(self.root)
        top.title("Wavelet Metrics")
        top.geometry("1160x840")
        top.rowconfigure(1, weight=1)
        top.columnconfigure(0, weight=1)
        header = self.ttk.Frame(top, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        summary_var = self.tk.StringVar(value="")
        self.ttk.Label(
            header, textvariable=summary_var, justify="left", wraplength=1100
        ).grid(row=0, column=0, sticky="w")

        body = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        fig = self.Figure(figsize=(9.2, 6.6), dpi=120)
        axes = tuple(np.ravel(fig.subplots(2, 2)))
        canvas = self.FigureCanvasTkAgg(fig, master=body)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.metrics_window = {
            "top": top,
            "summary_var": summary_var,
            "figure": fig,
            "axes": axes,
            "canvas": canvas,
        }
        top.protocol("WM_DELETE_WINDOW", self._close_metrics_window)
        self._refresh_metrics_window()

    def _close_metrics_window(self) -> None:
        existing = self.metrics_window
        if existing is None:
            return
        top = existing.get("top")
        if top is not None and top.winfo_exists():
            top.destroy()
        self.metrics_window = None

    def _refresh_metrics_window(self) -> None:
        existing = self.metrics_window
        if existing is None:
            return
        top = existing.get("top")
        if top is None or not top.winfo_exists():
            self.metrics_window = None
            return
        rows = [row for row in self._collect_curated_event_rows() if row["counted"]]
        summary_var = existing["summary_var"]
        axes = existing["axes"]
        fig = existing["figure"]
        canvas = existing["canvas"]
        for axis in axes:
            axis.clear()

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (str(row["panel_name"]), str(row["cut_name"]))
            grouped.setdefault(key, []).append(row)

        lines = [f"Counted events: {len(rows)} | panel/cut groups: {len(grouped)}"]
        for (panel_name, cut_name), items in sorted(grouped.items()):
            periods = [float(item["peak_period_s"]) for item in items if np.isfinite(float(item["peak_period_s"]))]
            amps = [float(item["fit_amp_arcsec"]) for item in items if np.isfinite(float(item["fit_amp_arcsec"]))]
            vels = [float(item["velocity_amp_km_s"]) for item in items if np.isfinite(float(item["velocity_amp_km_s"]))]
            energies = [float(item["specific_energy_j_kg"]) for item in items if np.isfinite(float(item["specific_energy_j_kg"]))]
            lines.append(
                f"{panel_name}/{cut_name}: n={len(items)} | "
                f"Pmed={np.nanmedian(periods) if periods else float('nan'):.2f}s | "
                f"Amed={np.nanmedian(amps) if amps else float('nan'):.3f}'' | "
                f"vmed={np.nanmedian(vels) if vels else float('nan'):.2f} km/s | "
                f"Emed={np.nanmedian(energies) if energies else float('nan'):.3e} J/kg"
            )
        summary_var.set("\n".join(lines))

        metric_specs = [
            ("peak_period_s", "Period [s]"),
            ("fit_amp_arcsec", "Amplitude ['']"),
            ("velocity_amp_km_s", "Velocity [km/s]"),
            ("specific_energy_j_kg", "Energy/m [J/kg]"),
        ]
        for axis, (key, title) in zip(axes, metric_specs):
            values = [
                float(row[key]) for row in rows if np.isfinite(float(row[key]))
            ]
            if values:
                axis.hist(values, bins=min(max(len(values), 5), 18), color="tab:blue", alpha=0.8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No counted events.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(title, fontsize=10)
            axis.set_ylabel("count")
        fig.tight_layout()
        canvas.draw_idle()

    def _open_propagation_window(self) -> None:
        existing = self.propagation_window
        if existing is not None:
            top = existing.get("top")
            if top is not None and top.winfo_exists():
                top.deiconify()
                top.lift()
                self._refresh_propagation_window()
                return
            self.propagation_window = None

        top = self.tk.Toplevel(self.root)
        top.title("Propagation Groups")
        top.geometry("1520x860")
        top.rowconfigure(1, weight=1)
        top.columnconfigure(0, weight=1)

        header = self.ttk.Frame(top, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        summary_var = self.tk.StringVar(value="")
        detail_var = self.tk.StringVar(value="")
        self.ttk.Label(
            header, textvariable=summary_var, justify="left", wraplength=1480
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header, textvariable=detail_var, justify="left", wraplength=1480
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        groups_frame = self.ttk.LabelFrame(body, text="Groups", padding=6)
        groups_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        groups_frame.rowconfigure(0, weight=1)
        groups_frame.columnconfigure(0, weight=1)
        groups_tree = self.ttk.Treeview(
            groups_frame,
            columns=("group", "class", "members", "cuts", "rep", "best", "amp", "period", "conf"),
            show="headings",
            height=18,
        )
        groups_tree.grid(row=0, column=0, sticky="nsew")
        groups_y = self.ttk.Scrollbar(groups_frame, orient="vertical", command=groups_tree.yview)
        groups_y.grid(row=0, column=1, sticky="ns")
        groups_x = self.ttk.Scrollbar(groups_frame, orient="horizontal", command=groups_tree.xview)
        groups_x.grid(row=1, column=0, sticky="ew")
        groups_tree.configure(yscrollcommand=groups_y.set, xscrollcommand=groups_x.set)
        for column, label, width in (
            ("group", "Group", 140),
            ("class", "Class", 110),
            ("members", "Members", 70),
            ("cuts", "Cuts", 55),
            ("rep", "Rep Amp", 95),
            ("best", "Best Conf", 95),
            ("amp", "Max A ['']", 85),
            ("period", "Pmed [s]", 85),
            ("conf", "Mean Conf", 85),
        ):
            groups_tree.heading(column, text=label)
            groups_tree.column(column, width=width, stretch=(column == "group"))

        members_frame = self.ttk.LabelFrame(body, text="Observations", padding=6)
        members_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        members_frame.rowconfigure(0, weight=1)
        members_frame.columnconfigure(0, weight=1)
        members_tree = self.ttk.Treeview(
            members_frame,
            columns=("cut", "event", "status", "class", "period", "amp", "conf", "stack", "note"),
            show="headings",
            height=18,
        )
        members_tree.grid(row=0, column=0, sticky="nsew")
        members_y = self.ttk.Scrollbar(members_frame, orient="vertical", command=members_tree.yview)
        members_y.grid(row=0, column=1, sticky="ns")
        members_x = self.ttk.Scrollbar(members_frame, orient="horizontal", command=members_tree.xview)
        members_x.grid(row=1, column=0, sticky="ew")
        members_tree.configure(yscrollcommand=members_y.set, xscrollcommand=members_x.set)
        for column, label, width in (
            ("cut", "Cut", 120),
            ("event", "Event", 55),
            ("status", "Status", 115),
            ("class", "Class", 95),
            ("period", "P [s]", 70),
            ("amp", "A ['']", 70),
            ("conf", "Conf", 60),
            ("stack", "Stack", 120),
            ("note", "Note", 220),
        ):
            members_tree.heading(column, text=label)
            members_tree.column(column, width=width, stretch=(column in {"cut", "stack", "note"}))

        actions = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        actions.grid(row=2, column=0, sticky="ew")
        for idx in range(5):
            actions.columnconfigure(idx, weight=1)
        self.ttk.Button(
            actions,
            text="Open Representative",
            command=self._focus_selected_propagation_representative,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            actions,
            text="Open Member",
            command=self._focus_selected_propagation_member,
        ).grid(row=0, column=1, sticky="ew", padx=4)
        self.ttk.Button(
            actions,
            text="Export Tables",
            command=self._export_propagation_tables,
        ).grid(row=0, column=2, sticky="ew", padx=4)
        self.ttk.Button(
            actions,
            text="Refresh",
            command=self._refresh_propagation_window,
        ).grid(row=0, column=3, sticky="ew", padx=4)
        self.ttk.Button(
            actions,
            text="Close",
            command=self._close_propagation_window,
        ).grid(row=0, column=4, sticky="ew", padx=(4, 0))

        self.propagation_window = {
            "top": top,
            "summary_var": summary_var,
            "detail_var": detail_var,
            "groups_tree": groups_tree,
            "members_tree": members_tree,
            "group_lookup": {},
            "member_lookup": {},
            "selected_group_key": None,
            "selected_member_ref": None,
            "tree_updating": False,
        }
        groups_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._on_propagation_group_select()
        )
        members_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._on_propagation_member_select()
        )
        top.protocol("WM_DELETE_WINDOW", self._close_propagation_window)
        self._refresh_propagation_window()

    def _close_propagation_window(self) -> None:
        existing = self.propagation_window
        if existing is None:
            return
        top = existing.get("top")
        if top is not None and top.winfo_exists():
            top.destroy()
        self.propagation_window = None

    def _refresh_propagation_window(self) -> None:
        existing = self.propagation_window
        if existing is None:
            return
        top = existing.get("top")
        if top is None or not top.winfo_exists():
            self.propagation_window = None
            return

        observation_rows = self._collect_curated_event_rows()
        group_rows = self._collect_propagation_group_rows(observation_rows)
        existing["group_lookup"] = {
            str(group["group_key"]): group for group in group_rows
        }
        groups_tree = existing["groups_tree"]
        selected_group_key = (
            existing.get("selected_group_key")
            if existing.get("selected_group_key") in existing["group_lookup"]
            else (group_rows[0]["group_key"] if group_rows else None)
        )

        existing["tree_updating"] = True
        try:
            children = groups_tree.get_children()
            if children:
                groups_tree.delete(*children)
            for group in group_rows:
                group_key = str(group["group_key"])
                groups_tree.insert(
                    "",
                    "end",
                    iid=f"grp-{group_key}",
                    values=(
                        group_key,
                        str(group["classification"]),
                        str(int(group["member_count"])),
                        str(int(group["cut_count"])),
                        (
                            f"{int(group['representative_cut_id'])}:{int(group['representative_event_id'])}"
                            if group.get("representative_cut_id") is not None
                            else "-"
                        ),
                        (
                            f"{int(group['best_conf_cut_id'])}:{int(group['best_conf_event_id'])}"
                            if group.get("best_conf_cut_id") is not None
                            else "-"
                        ),
                        (
                            f"{float(group['max_amp_arcsec']):.3f}"
                            if np.isfinite(float(group["max_amp_arcsec"]))
                            else "-"
                        ),
                        (
                            f"{float(group['median_period_s']):.2f}"
                            if np.isfinite(float(group["median_period_s"]))
                            else "-"
                        ),
                        (
                            f"{float(group['mean_confidence_score']):.1f}"
                            if np.isfinite(float(group["mean_confidence_score"]))
                            else "-"
                        ),
                    ),
                )
            existing["selected_group_key"] = selected_group_key
            if selected_group_key is not None:
                target_iid = f"grp-{selected_group_key}"
                groups_tree.selection_set(target_iid)
                groups_tree.focus(target_iid)
            else:
                groups_tree.selection_remove(groups_tree.selection())
        finally:
            existing["tree_updating"] = False

        linked_count = sum(1 for group in group_rows if not bool(group.get("synthetic_group")))
        existing["summary_var"].set(
            f"Propagation groups: {len(group_rows)} | linked groups: {linked_count} | observations: {len(observation_rows)}"
        )
        self._refresh_propagation_members()

    def _refresh_propagation_members(self) -> None:
        existing = self.propagation_window
        if existing is None:
            return
        members_tree = existing["members_tree"]
        group = existing.get("group_lookup", {}).get(existing.get("selected_group_key"))
        members = [] if group is None else list(group.get("members", []))
        existing["member_lookup"] = {
            (int(member["cut_id"]), int(member["event_id"])): member for member in members
        }
        selected_member_ref = (
            existing.get("selected_member_ref")
            if existing.get("selected_member_ref") in existing["member_lookup"]
            else (
                (int(members[0]["cut_id"]), int(members[0]["event_id"]))
                if members
                else None
            )
        )
        existing["tree_updating"] = True
        try:
            children = members_tree.get_children()
            if children:
                members_tree.delete(*children)
            for member in members:
                cut_id = int(member["cut_id"])
                event_id = int(member["event_id"])
                iid = f"mem-{cut_id}-{event_id}"
                members_tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=(
                        f"{cut_id:02d} {member['cut_name']}",
                        str(event_id),
                        str(member["status"]),
                        str(member.get("resolved_propagation_class") or member.get("propagation_class") or "-"),
                        (
                            f"{float(member['peak_period_s']):.2f}"
                            if np.isfinite(float(member["peak_period_s"]))
                            else "-"
                        ),
                        (
                            f"{float(member['fit_amp_arcsec']):.3f}"
                            if np.isfinite(float(member["fit_amp_arcsec"]))
                            else "-"
                        ),
                        (
                            f"{float(member['confidence_score']):.1f}"
                            if np.isfinite(float(member["confidence_score"]))
                            else "-"
                        ),
                        ",".join(member.get("stack_positions") or []) or "-",
                        str(member.get("review_notes") or "-"),
                    ),
                )
            existing["selected_member_ref"] = selected_member_ref
            if selected_member_ref is not None:
                target_iid = f"mem-{selected_member_ref[0]}-{selected_member_ref[1]}"
                members_tree.selection_set(target_iid)
                members_tree.focus(target_iid)
            else:
                members_tree.selection_remove(members_tree.selection())
        finally:
            existing["tree_updating"] = False

        if group is None:
            existing["detail_var"].set("Select a propagation group to inspect its observations.")
            return
        existing["detail_var"].set(
            f"Group {group['group_key']} | class={group['classification']} | cuts={','.join(str(cut_id) for cut_id in group['cut_ids'])} | "
            f"rep={group['representative_cut_id']}:{group['representative_event_id']} | "
            f"best conf={group['best_conf_cut_id']}:{group['best_conf_event_id']}"
        )

    def _on_propagation_group_select(self) -> None:
        existing = self.propagation_window
        if existing is None or existing.get("tree_updating"):
            return
        selection = existing["groups_tree"].selection()
        if not selection:
            existing["selected_group_key"] = None
        else:
            iid = str(selection[0])
            existing["selected_group_key"] = iid[4:] if iid.startswith("grp-") else None
        self._refresh_propagation_members()

    def _on_propagation_member_select(self) -> None:
        existing = self.propagation_window
        if existing is None or existing.get("tree_updating"):
            return
        selection = existing["members_tree"].selection()
        if not selection:
            existing["selected_member_ref"] = None
            return
        iid = str(selection[0])
        if iid.startswith("mem-"):
            try:
                _prefix, cut_id_text, event_id_text = iid.split("-", 2)
                existing["selected_member_ref"] = (
                    int(cut_id_text),
                    int(event_id_text),
                )
            except Exception:
                existing["selected_member_ref"] = None

    def _selected_propagation_member(self) -> dict[str, Any] | None:
        existing = self.propagation_window
        if existing is None:
            return None
        member_ref = existing.get("selected_member_ref")
        if member_ref is None:
            return None
        return existing.get("member_lookup", {}).get(member_ref)

    def _focus_wavelet_event_for_cut(self, cut_id: int, event_id: int) -> None:
        if cut_id not in self.cuts:
            self._set_status("Invalid cut target.")
            return
        if self._wavelet_event_ref_by_cut(cut_id, event_id) is None:
            self._set_status(f"Cut {cut_id} event {event_id} is no longer available.")
            return
        self.selected_cut_id = cut_id
        self._open_cut_in_td_window(cut_id)
        panel = self._primary_panel_for_cut(cut_id)
        if panel is None:
            return
        existing = self.td_windows.get(panel.panel_id)
        if existing is None:
            return
        existing["wavelet_selected_event_id"] = int(event_id)
        self._refresh_td_window_wavelet_views(panel.panel_id, redraw_td=True)
        self._set_status(f"Focused cut {cut_id} event {event_id}.")

    def _focus_selected_propagation_member(self) -> None:
        member = self._selected_propagation_member()
        if member is None:
            self._set_status("Select an observation first.")
            return
        self._focus_wavelet_event_for_cut(int(member["cut_id"]), int(member["event_id"]))

    def _focus_selected_propagation_representative(self) -> None:
        existing = self.propagation_window
        if existing is None:
            return
        group = existing.get("group_lookup", {}).get(existing.get("selected_group_key"))
        if group is None:
            self._set_status("Select a propagation group first.")
            return
        cut_id = group.get("representative_cut_id")
        event_id = group.get("representative_event_id")
        if cut_id is None or event_id is None:
            self._set_status("This group has no representative event.")
            return
        self._focus_wavelet_event_for_cut(int(cut_id), int(event_id))

    def _collect_wavelet_link_groups(self) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for row in self._collect_curated_event_rows():
            group_id = str(row.get("link_group_id") or "").strip()
            if not group_id:
                continue
            member = {
                "panel_id": row.get("primary_panel_id"),
                "panel_name": row.get("primary_panel_name") or "-",
                "cut_id": row.get("cut_id"),
                "cut_name": row.get("cut_name") or "",
                "event_id": row.get("event_id"),
                "status": row.get("status"),
                "counted": bool(row.get("counted")),
                "confidence_score": row.get("confidence_score"),
                "review_locked": bool(row.get("review_locked")),
                "review_notes": str(row.get("review_notes", "")),
                "qa_flags_text": str(row.get("qa_flags_text", "")),
            }
            group = groups.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "members": [],
                },
            )
            group["members"].append(member)

        output: list[dict[str, Any]] = []
        for group_id, group in sorted(groups.items()):
            members = sorted(
                group["members"],
                key=lambda item: (
                    int(item["cut_id"]) if item.get("cut_id") is not None else -1,
                    int(item["event_id"]),
                ),
            )
            scores = [
                float(item["confidence_score"])
                for item in members
                if np.isfinite(float(item["confidence_score"]))
            ]
            cut_names = sorted({str(item["cut_name"] or "-") for item in members})
            statuses = sorted({str(item["status"]) for item in members})
            output.append(
                {
                    "group_id": group_id,
                    "members": members,
                    "event_count": len(members),
                    "panel_count": len(
                        {
                            int(item["panel_id"])
                            for item in members
                            if item.get("panel_id") is not None
                        }
                    ),
                    "cut_count": len(
                        {
                            int(item["cut_id"])
                            for item in members
                            if item.get("cut_id") is not None
                        }
                    ),
                    "counted_count": sum(1 for item in members if item["counted"]),
                    "locked_count": sum(1 for item in members if item["review_locked"]),
                    "mean_confidence": (
                        float(np.nanmean(scores)) if scores else float("nan")
                    ),
                    "cuts_text": ",".join(cut_names),
                    "status_text": ",".join(statuses),
                }
            )
        return output

    def _open_link_groups_window(
        self,
        *,
        initial_group_id: str | None = None,
        initial_member_ref: tuple[int, int] | None = None,
    ) -> None:
        existing = self.link_groups_window
        if existing is not None:
            top = existing.get("top")
            if top is not None and top.winfo_exists():
                if initial_group_id is not None:
                    existing["pending_group_id"] = str(initial_group_id)
                if initial_member_ref is not None:
                    existing["pending_member_ref"] = (
                        int(initial_member_ref[0]),
                        int(initial_member_ref[1]),
                    )
                top.deiconify()
                top.lift()
                self._refresh_link_groups_window()
                return
            self.link_groups_window = None

        top = self.tk.Toplevel(self.root)
        top.title("Linked Wavelet Groups")
        top.geometry("1420x780")
        top.rowconfigure(1, weight=1)
        top.columnconfigure(0, weight=1)

        header = self.ttk.Frame(top, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        summary_var = self.tk.StringVar(value="")
        detail_var = self.tk.StringVar(value="")
        self.ttk.Label(
            header, textvariable=summary_var, justify="left", wraplength=1380
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header, textvariable=detail_var, justify="left", wraplength=1380
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        groups_frame = self.ttk.LabelFrame(body, text="Groups", padding=6)
        groups_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        groups_frame.rowconfigure(0, weight=1)
        groups_frame.columnconfigure(0, weight=1)
        groups_tree = self.ttk.Treeview(
            groups_frame,
            columns=("group", "members", "panels", "counted", "locked", "conf", "status"),
            show="headings",
            height=16,
        )
        groups_tree.grid(row=0, column=0, sticky="nsew")
        groups_y = self.ttk.Scrollbar(groups_frame, orient="vertical", command=groups_tree.yview)
        groups_y.grid(row=0, column=1, sticky="ns")
        groups_x = self.ttk.Scrollbar(groups_frame, orient="horizontal", command=groups_tree.xview)
        groups_x.grid(row=1, column=0, sticky="ew")
        groups_tree.configure(yscrollcommand=groups_y.set, xscrollcommand=groups_x.set)
        for column, label, width in (
            ("group", "Group", 90),
            ("members", "Events", 60),
            ("panels", "Cuts", 60),
            ("counted", "Counted", 70),
            ("locked", "Locked", 65),
            ("conf", "Mean Conf", 85),
            ("status", "Statuses", 220),
        ):
            groups_tree.heading(column, text=label)
            groups_tree.column(column, width=width, stretch=(column == "status"))

        members_frame = self.ttk.LabelFrame(body, text="Members", padding=6)
        members_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        members_frame.rowconfigure(0, weight=1)
        members_frame.columnconfigure(0, weight=1)
        members_tree = self.ttk.Treeview(
            members_frame,
            columns=("panel", "cut", "event", "status", "conf", "lock", "qa", "note"),
            show="headings",
            height=16,
        )
        members_tree.grid(row=0, column=0, sticky="nsew")
        members_y = self.ttk.Scrollbar(
            members_frame, orient="vertical", command=members_tree.yview
        )
        members_y.grid(row=0, column=1, sticky="ns")
        members_x = self.ttk.Scrollbar(
            members_frame, orient="horizontal", command=members_tree.xview
        )
        members_x.grid(row=1, column=0, sticky="ew")
        members_tree.configure(yscrollcommand=members_y.set, xscrollcommand=members_x.set)
        for column, label, width in (
            ("panel", "Panel", 90),
            ("cut", "Cut", 110),
            ("event", "Event", 55),
            ("status", "Status", 120),
            ("conf", "Conf", 60),
            ("lock", "Lock", 55),
            ("qa", "QA", 120),
            ("note", "Note", 220),
        ):
            members_tree.heading(column, text=label)
            members_tree.column(column, width=width, stretch=(column in {"note", "qa"}))

        actions = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        actions.grid(row=2, column=0, sticky="ew")
        for idx in range(4):
            actions.columnconfigure(idx, weight=1)
        self.ttk.Button(
            actions,
            text="Open Member",
            command=self._focus_selected_link_group_member,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            actions,
            text="Sync Group From Member",
            command=self._sync_selected_link_group_member,
        ).grid(row=0, column=1, sticky="ew", padx=4)
        self.ttk.Button(
            actions,
            text="Refresh",
            command=self._refresh_link_groups_window,
        ).grid(row=0, column=2, sticky="ew", padx=4)
        self.ttk.Button(
            actions,
            text="Close",
            command=self._close_link_groups_window,
        ).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        self.link_groups_window = {
            "top": top,
            "summary_var": summary_var,
            "detail_var": detail_var,
            "groups_tree": groups_tree,
            "members_tree": members_tree,
            "group_lookup": {},
            "member_lookup": {},
            "selected_group_id": None,
            "selected_member_ref": None,
            "pending_group_id": initial_group_id,
            "pending_member_ref": initial_member_ref,
            "tree_updating": False,
        }
        groups_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._on_link_groups_group_select()
        )
        members_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._on_link_groups_member_select()
        )
        top.protocol("WM_DELETE_WINDOW", self._close_link_groups_window)
        self._refresh_link_groups_window()

    def _close_link_groups_window(self) -> None:
        existing = self.link_groups_window
        if existing is None:
            return
        top = existing.get("top")
        if top is not None and top.winfo_exists():
            top.destroy()
        self.link_groups_window = None

    def _refresh_link_groups_window(self) -> None:
        existing = self.link_groups_window
        if existing is None:
            return
        top = existing.get("top")
        if top is None or not top.winfo_exists():
            self.link_groups_window = None
            return

        groups = self._collect_wavelet_link_groups()
        existing["group_lookup"] = {group["group_id"]: group for group in groups}
        groups_tree = existing["groups_tree"]
        selected_group_id = (
            existing.get("pending_group_id")
            or existing.get("selected_group_id")
            or (groups[0]["group_id"] if groups else None)
        )
        existing["tree_updating"] = True
        try:
            children = groups_tree.get_children()
            if children:
                groups_tree.delete(*children)
            for group in groups:
                group_id = str(group["group_id"])
                groups_tree.insert(
                    "",
                    "end",
                    iid=f"grp-{group_id}",
                    values=(
                        group_id,
                        str(int(group["event_count"])),
                        str(int(group.get("cut_count", group["panel_count"]))),
                        str(int(group["counted_count"])),
                        str(int(group["locked_count"])),
                        (
                            f"{float(group['mean_confidence']):.1f}"
                            if np.isfinite(float(group["mean_confidence"]))
                            else "-"
                        ),
                        str(group["status_text"]),
                    ),
                )
            if selected_group_id not in existing["group_lookup"]:
                selected_group_id = groups[0]["group_id"] if groups else None
            existing["selected_group_id"] = selected_group_id
            if selected_group_id is not None:
                target_iid = f"grp-{selected_group_id}"
                groups_tree.selection_set(target_iid)
                groups_tree.focus(target_iid)
            else:
                groups_tree.selection_remove(groups_tree.selection())
        finally:
            existing["tree_updating"] = False

        linked_events = sum(int(group["event_count"]) for group in groups)
        linked_cuts = len(
            {
                int(member["cut_id"])
                for group in groups
                for member in group.get("members", [])
                if member.get("cut_id") is not None
            }
        )
        existing["summary_var"].set(
            f"Link groups: {len(groups)} | linked events: {linked_events} | linked cuts: {linked_cuts}"
        )
        existing["pending_group_id"] = None
        self._refresh_link_group_members()

    def _refresh_link_group_members(self) -> None:
        existing = self.link_groups_window
        if existing is None:
            return
        members_tree = existing["members_tree"]
        selected_group_id = existing.get("selected_group_id")
        group = existing.get("group_lookup", {}).get(selected_group_id)
        members = [] if group is None else list(group.get("members", []))
        existing["member_lookup"] = {
            (
                int(member["cut_id"])
                if member.get("cut_id") is not None
                else int(member["panel_id"]),
                int(member["event_id"]),
            ): member
            for member in members
        }
        selected_member_ref = (
            existing.get("pending_member_ref")
            or existing.get("selected_member_ref")
            or (
                (
                    int(members[0]["cut_id"])
                    if members[0].get("cut_id") is not None
                    else int(members[0]["panel_id"]),
                    int(members[0]["event_id"]),
                )
                if members
                else None
            )
        )
        existing["tree_updating"] = True
        try:
            children = members_tree.get_children()
            if children:
                members_tree.delete(*children)
            for member in members:
                member_ref = (
                    int(member["cut_id"])
                    if member.get("cut_id") is not None
                    else int(member["panel_id"]),
                    int(member["event_id"]),
                )
                iid = f"mem-{member_ref[0]}-{member_ref[1]}"
                members_tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=(
                        str(member["panel_name"]),
                        str(member["cut_name"] or "-"),
                        str(int(member["event_id"])),
                        str(member["status"]),
                        f"{float(member['confidence_score']):.1f}",
                        "yes" if bool(member["review_locked"]) else "no",
                        str(member["qa_flags_text"] or "-"),
                        str(member["review_notes"] or "-"),
                    ),
                )
            if selected_member_ref not in existing["member_lookup"]:
                selected_member_ref = (
                    (
                        int(members[0]["cut_id"])
                        if members[0].get("cut_id") is not None
                        else int(members[0]["panel_id"]),
                        int(members[0]["event_id"]),
                    )
                    if members
                    else None
                )
            existing["selected_member_ref"] = selected_member_ref
            if selected_member_ref is not None:
                target_iid = f"mem-{selected_member_ref[0]}-{selected_member_ref[1]}"
                members_tree.selection_set(target_iid)
                members_tree.focus(target_iid)
            else:
                members_tree.selection_remove(members_tree.selection())
        finally:
            existing["tree_updating"] = False

        if group is None:
            existing["detail_var"].set("Select a linked group to inspect its members.")
        else:
            existing["detail_var"].set(
                f"Group {group['group_id']} | cuts={group['cuts_text']} | "
                f"events={group['event_count']} | cuts={group.get('cut_count', 0)} | counted={group['counted_count']} | "
                f"locked={group['locked_count']} | mean conf="
                + (
                    f"{float(group['mean_confidence']):.1f}"
                    if np.isfinite(float(group["mean_confidence"]))
                    else "-"
                )
            )
        existing["pending_member_ref"] = None

    def _on_link_groups_group_select(self) -> None:
        existing = self.link_groups_window
        if existing is None or existing.get("tree_updating"):
            return
        tree = existing["groups_tree"]
        selection = tree.selection()
        if not selection:
            existing["selected_group_id"] = None
        else:
            iid = str(selection[0])
            existing["selected_group_id"] = iid[4:] if iid.startswith("grp-") else None
        self._refresh_link_group_members()

    def _on_link_groups_member_select(self) -> None:
        existing = self.link_groups_window
        if existing is None or existing.get("tree_updating"):
            return
        tree = existing["members_tree"]
        selection = tree.selection()
        if not selection:
            existing["selected_member_ref"] = None
            return
        iid = str(selection[0])
        if iid.startswith("mem-"):
            try:
                _prefix, panel_id_text, event_id_text = iid.split("-", 2)
                existing["selected_member_ref"] = (
                    int(panel_id_text),
                    int(event_id_text),
                )
            except Exception:
                existing["selected_member_ref"] = None

    def _selected_link_group_member(self) -> dict[str, Any] | None:
        existing = self.link_groups_window
        if existing is None:
            return None
        member_ref = existing.get("selected_member_ref")
        if member_ref is None:
            return None
        return existing.get("member_lookup", {}).get(member_ref)

    def _focus_wavelet_event(self, panel_id: int, event_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        if panel is None:
            self._set_status("Invalid linked event target.")
            return
        if self._wavelet_event_ref(panel_id, event_id) is None:
            self._set_status(
                f"{panel.name} event {event_id} is no longer available."
            )
            return
        self.active_panel_id = panel_id
        self._sync_controls_from_active_panel()
        self._refresh_panel_list()
        self._open_td_window(panel_id)
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        existing["wavelet_selected_event_id"] = int(event_id)
        top = existing.get("top")
        if top is not None and top.winfo_exists():
            top.deiconify()
            top.lift()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Focused {panel.name} event {event_id}.")

    def _focus_selected_link_group_member(self) -> None:
        member = self._selected_link_group_member()
        if member is None:
            self._set_status("Select a linked event member first.")
            return
        if member.get("cut_id") is not None:
            self._focus_wavelet_event_for_cut(
                int(member["cut_id"]), int(member["event_id"])
            )
            return
        if member.get("panel_id") is not None:
            self._focus_wavelet_event(int(member["panel_id"]), int(member["event_id"]))
            return
        self._set_status("The selected linked event member is no longer available.")

    def _sync_selected_link_group_member(self) -> None:
        member = self._selected_link_group_member()
        if member is None:
            self._set_status("Select a linked event member first.")
            return
        if member.get("cut_id") is not None:
            self._sync_linked_wavelet_group_from_cut_event(
                int(member["cut_id"]), int(member["event_id"])
            )
            return
        if member.get("panel_id") is not None:
            self._sync_linked_wavelet_group_from_event(
                int(member["panel_id"]), int(member["event_id"])
            )
            return
        self._set_status("The selected linked event member is no longer available.")

    def _close_all_stack_browsers(self) -> None:
        if self.stack_browser_refresh_job is not None and self.root.winfo_exists():
            try:
                self.root.after_cancel(self.stack_browser_refresh_job)
            except Exception:
                pass
            self.stack_browser_refresh_job = None
        for browser_id in list(self.stack_browsers.keys()):
            self._close_stack_browser(browser_id)

    def _schedule_stack_browser_refresh(self, delay_ms: int = 80) -> None:
        if self.stack_browser_refresh_in_progress:
            _trace_stack_wavelet("schedule_stack_browser_refresh skipped=in_progress")
            return
        if self.stack_browser_refresh_job is not None:
            _trace_stack_wavelet("schedule_stack_browser_refresh skipped=already_queued")
            return
        if not self.root.winfo_exists():
            _trace_stack_wavelet("schedule_stack_browser_refresh skipped=no_root")
            return

        def _run() -> None:
            self.stack_browser_refresh_job = None
            _trace_stack_wavelet("schedule_stack_browser_refresh run")
            self._refresh_all_stack_browsers()

        _trace_stack_wavelet(f"schedule_stack_browser_refresh queued delay_ms={int(delay_ms)}")
        self.stack_browser_refresh_job = self.root.after(int(delay_ms), _run)

    def _refresh_all_stack_browsers(self) -> None:
        if self.stack_browser_refresh_in_progress:
            _trace_stack_wavelet("refresh_all_stack_browsers skipped=in_progress")
            return
        _trace_stack_wavelet(f"refresh_all_stack_browsers start count={len(self.stack_browsers)}")
        self.stack_browser_refresh_in_progress = True
        try:
            for browser_id in list(self.stack_browsers.keys()):
                self._refresh_stack_browser(browser_id)
        finally:
            self.stack_browser_refresh_in_progress = False
            _trace_stack_wavelet("refresh_all_stack_browsers end")

    def _cut_td(self, cut_id: int) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        if cut_id not in self.cuts:
            return None, None
        cut = self.cuts[cut_id]
        state = self._cut_analysis(cut_id)
        params = self._cut_td_params(cut_id)
        key = cut_cache_key(cut, params) + self._cut_geometry_signature(cut_id)
        if (
            state.get("td_cache_key") == key
            and state.get("td_cache_td") is not None
            and state.get("td_cache_meta") is not None
        ):
            return state["td_cache_td"], state["td_cache_meta"]
        try:
            td, meta = compute_td(
                self.cube,
                cut,
                int(params["t_ini"]),
                int(params["t_fin"]),
                int(params["stride"]),
                int(params["width"]),
                str(params["weighting"]),
                dynamic_geometry=self._dynamic_cut_geometry_samples(
                    cut_id,
                    int(params["t_ini"]),
                    int(params["t_fin"]),
                    int(params["stride"]),
                ),
            )
        except ValueError as exc:
            state["td_cache_key"] = key
            state["td_cache_td"] = None
            state["td_cache_meta"] = {"error": str(exc)}
            return None, state["td_cache_meta"]
        state["td_cache_key"] = key
        state["td_cache_td"] = td
        state["td_cache_meta"] = meta
        return td, meta

    def _draw_cut_analysis_overlay(
        self, ax: Any, cut_id: int, meta: dict[str, Any] | None, *, selected_event_id: int | None = None
    ) -> None:
        if meta is None:
            return
        state = self._cut_analysis(cut_id)
        tracking_result = state.get("crest_tracking_result")
        if tracking_result:
            threads = tracking_result.get("threads") or []
            distances = np.asarray(meta["distances"], dtype=np.float64)
            t_indices = np.asarray(meta["t_indices"], dtype=np.float64)
            if distances.size and t_indices.size:
                dist_index = np.arange(distances.size, dtype=np.float64)
                dist_hi = dist_index[-1]
                for idx, thread in enumerate(threads):
                    pos = np.asarray(thread.get("pos", []), dtype=np.float64)
                    if pos.size != t_indices.size:
                        continue
                    mask = np.isfinite(pos) & (pos >= 0.0) & (pos <= dist_hi + 1e-9)
                    if np.count_nonzero(mask) < 2:
                        continue
                    dist_vals = np.interp(pos[mask], dist_index, distances)
                    time_vals = t_indices[mask]
                    color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                    if self.td_swap_axes_var.get():
                        ax.plot(time_vals, dist_vals, color=color, linewidth=1.3, alpha=0.8)
                    else:
                        ax.plot(dist_vals, time_vals, color=color, linewidth=1.3, alpha=0.8)
        events = state.get("wavelet_events") or []
        if not events:
            return
        distances = np.asarray(meta["distances"], dtype=np.float64)
        if distances.size == 0:
            return
        dist_index = np.arange(distances.size, dtype=np.float64)
        dist_hi = dist_index[-1]
        for event in events:
            self._ensure_wavelet_event_fields(event)
            analysis = event.get("analysis") or {}
            wave_t_idx = np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64)
            wave_y_idx = np.asarray(analysis.get("wave_y_idx", []), dtype=np.float64)
            if wave_t_idx.size < 2 or wave_y_idx.size != wave_t_idx.size:
                continue
            mask = np.isfinite(wave_y_idx) & (wave_y_idx >= 0.0) & (wave_y_idx <= dist_hi + 1e-9)
            if np.count_nonzero(mask) < 2:
                continue
            dist_vals = np.interp(wave_y_idx[mask], dist_index, distances)
            time_vals = wave_t_idx[mask]
            status = self._td_window_wavelet_event_status(event)
            color = "lime"
            if status == "manual accepted":
                color = "deepskyblue"
            elif status == "custom accepted":
                color = "cyan"
            elif status not in {"auto accepted", "custom accepted", "manual accepted"}:
                color = "darkorange"
            linewidth = 2.2 if self._td_window_wavelet_event_is_counted(event) else 1.4
            alpha = 0.9 if self._td_window_wavelet_event_is_counted(event) else 0.5
            if int(event.get("event_id", -1)) == int(selected_event_id or -1):
                color = "gold"
                linewidth = 3.0
                alpha = 1.0
            if self.td_swap_axes_var.get():
                ax.plot(time_vals, dist_vals, color=color, linewidth=linewidth, alpha=alpha)
            else:
                ax.plot(dist_vals, time_vals, color=color, linewidth=linewidth, alpha=alpha)

    def _draw_cut_td_axis(
        self,
        ax: Any,
        cut_id: int | None,
        *,
        use_zoom: bool = True,
        title_fontsize: float = 9.0,
        selected_event_id: int | None = None,
        title_prefix: str = "",
    ) -> None:
        current_t = int(self.t_visual_var.get())
        td_aspect = "equal" if self.td_aspect_var.get() == "equal" else "auto"
        td_zoom = float(str(self.td_zoom_var.get()).rstrip("x") or "1")
        ax.clear()
        if cut_id is None or cut_id not in self.cuts:
            ax.text(0.5, 0.5, "No cut.", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        cut = self.cuts[cut_id]
        params = self._cut_td_params(cut_id)
        td, meta = self._cut_td(cut_id)
        if td is None or meta is None:
            error_text = "invalid TD"
            if meta is not None and "error" in meta:
                error_text = str(meta["error"])
            ax.text(
                0.5,
                0.5,
                f"{cut.name}\n{error_text}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title_prefix}{cut.name}".strip(), fontsize=title_fontsize)
            return

        distances = meta["distances"]
        t_indices = meta["t_indices"]
        if len(t_indices) == 1:
            t_min = float(t_indices[0])
            t_max = float(t_indices[0]) + 1.0
        else:
            t_min = float(t_indices[0])
            t_max = float(t_indices[-1])

        vmin, vmax = frame_limits(td)
        dist_max = float(distances[-1])
        time_center = (
            float(current_t)
            if int(params["t_ini"]) <= current_t <= int(params["t_fin"])
            else 0.5 * (t_min + t_max)
        )
        dist_center = 0.5 * dist_max

        if self.td_swap_axes_var.get():
            ax.imshow(
                td.T,
                origin="lower",
                aspect=td_aspect,
                cmap="gray",
                extent=[t_min, t_max, 0.0, dist_max],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if int(params["t_ini"]) <= current_t <= int(params["t_fin"]):
                ax.axvline(current_t, color=cut.color, linestyle="--", linewidth=1.0)
            if use_zoom:
                x0, x1 = self._zoom_limits(t_min, t_max, time_center, td_zoom)
                y0, y1 = self._zoom_limits(0.0, dist_max, dist_center, td_zoom)
            else:
                x0, x1 = t_min, t_max
                y0, y1 = 0.0, dist_max
            ax.set_xlim((x1, x0) if self.td_flip_x_var.get() else (x0, x1))
            ax.set_ylim((y1, y0) if self.td_flip_y_var.get() else (y0, y1))
            ax.set_xlabel("time index")
            ax.set_ylabel("distance [pixel]")
        else:
            ax.imshow(
                td,
                origin="lower",
                aspect=td_aspect,
                cmap="gray",
                extent=[0.0, dist_max, t_min, t_max],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if int(params["t_ini"]) <= current_t <= int(params["t_fin"]):
                ax.axhline(current_t, color=cut.color, linestyle="--", linewidth=1.0)
            if use_zoom:
                x0, x1 = self._zoom_limits(0.0, dist_max, dist_center, td_zoom)
                y0, y1 = self._zoom_limits(t_min, t_max, time_center, td_zoom)
            else:
                x0, x1 = 0.0, dist_max
                y0, y1 = t_min, t_max
            ax.set_xlim((x1, x0) if self.td_flip_x_var.get() else (x0, x1))
            ax.set_ylim((y1, y0) if self.td_flip_y_var.get() else (y0, y1))
            ax.set_xlabel("distance [pixel]")
            ax.set_ylabel("time index")

        self._draw_cut_analysis_overlay(ax, cut_id, meta, selected_event_id=selected_event_id)
        dynamic_note = " | dynamic" if self._cut_dynamic_enabled(cut_id) else ""
        ax.set_title(
            f"{title_prefix}{cut.name} | t={int(params['t_ini'])}:{int(params['t_fin'])}:{int(params['stride'])} | "
            f"w={int(params['width'])} {params['weighting']}{dynamic_note}",
            fontsize=title_fontsize,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor(cut.color)

    def _preferred_td_window_panel_for_cut(
        self, cut_id: int, *, prefer_hidden_panel: bool = False
    ) -> int:
        existing_panel_id = next(
            (panel.panel_id for panel in self.panels if panel.cut_id == cut_id),
            None,
        )
        if existing_panel_id is not None:
            return int(existing_panel_id)

        if prefer_hidden_panel:
            hidden_candidates = [
                panel.panel_id
                for panel in self.panels
                if panel.panel_id > int(self.visible_panels)
            ]
            if hidden_candidates:
                return int(hidden_candidates[0])

            detached_candidates = [
                panel.panel_id
                for panel in self.panels
                if panel.panel_id != int(self.active_panel_id)
                and panel.panel_id not in self.td_windows
            ]
            if detached_candidates:
                return int(detached_candidates[0])

        return int(self.active_panel_id)

    def _open_cut_in_td_window(
        self,
        cut_id: int,
        *,
        prefer_hidden_panel: bool = False,
        source_stack_id: int | None = None,
    ) -> None:
        if cut_id not in self.cuts:
            self._set_status("Invalid cut selection.")
            return
        target_panel_id = self._preferred_td_window_panel_for_cut(
            cut_id,
            prefer_hidden_panel=prefer_hidden_panel,
        )
        panel = self.panels[target_panel_id - 1]
        if panel.cut_id != cut_id:
            panel.cut_id = cut_id
            if not prefer_hidden_panel:
                self._seed_cut_td_params_from_panel(
                    cut_id,
                    self.panels[self.active_panel_id - 1],
                )
        self.selected_cut_id = cut_id
        if not prefer_hidden_panel or target_panel_id <= int(self.visible_panels):
            self.active_panel_id = target_panel_id
            self._refresh_measurement_selectors()
            self._refresh_measurements()
            self._refresh_panel_cut_selector()
            self._refresh_panel_list()
            self._refresh_cut_list()
            self._refresh_stack_list()
            self._refresh_stack_member_list()
            self._sync_controls_from_active_panel()
            self._draw_map()
            self._draw_td_panels()
            self.canvas.draw_idle()
        self._open_td_window(target_panel_id, source_stack_id=source_stack_id)

    def _wavelet_edit_state_snapshot_for_cut(self, cut_id: int) -> dict[str, Any]:
        state = self._cut_analysis(cut_id)
        return {
            "events": self._clone_wavelet_payload(state.get("wavelet_events") or []),
            "selected_event_id": state.get("wavelet_selected_event_id"),
            "next_event_id": int(state.get("wavelet_next_event_id", 1)),
        }

    def _push_wavelet_undo_state_for_cut(self, cut_id: int, label: str) -> None:
        state = self._cut_analysis(cut_id)
        undo_stack = state.setdefault("wavelet_undo_stack", [])
        snapshot = self._wavelet_edit_state_snapshot_for_cut(cut_id)
        snapshot["label"] = str(label)
        undo_stack.append(self._clone_wavelet_payload(snapshot))
        if len(undo_stack) > MAX_WAVELET_HISTORY:
            del undo_stack[: len(undo_stack) - MAX_WAVELET_HISTORY]
        state["wavelet_redo_stack"] = []
        for panel_id, panel in enumerate(self.panels, start=1):
            if panel.cut_id != cut_id:
                continue
            existing = self.td_windows.get(panel_id)
            if existing is None:
                continue
            existing["wavelet_undo_stack"] = self._clone_wavelet_payload(undo_stack)
            existing["wavelet_redo_stack"] = []

    def _sync_open_td_windows_from_cut(self, cut_id: int) -> None:
        state = self._cut_analysis(cut_id)
        for panel_id, panel in enumerate(self.panels, start=1):
            if panel.cut_id != cut_id:
                continue
            existing = self.td_windows.get(panel_id)
            if existing is None:
                continue
            existing["crest_tracking_result"] = self._clone_wavelet_payload(
                state.get("crest_tracking_result")
            )
            existing["crest_tracking_td_key"] = self._clone_wavelet_payload(
                state.get("crest_tracking_td_key")
            )
            existing["wavelet_filter_result"] = self._clone_wavelet_payload(
                state.get("wavelet_filter_result")
            )
            existing["wavelet_thread_filter_var"].set(
                str(state.get("wavelet_thread_filter_text", "") or "")
            )
            existing["wavelet_events"] = self._clone_wavelet_payload(
                state.get("wavelet_events") or []
            )
            existing["wavelet_next_event_id"] = int(
                state.get("wavelet_next_event_id", 1)
            )
            existing["wavelet_selected_event_id"] = state.get("wavelet_selected_event_id")
            existing["wavelet_undo_stack"] = self._clone_wavelet_payload(
                state.get("wavelet_undo_stack") or []
            )
            existing["wavelet_redo_stack"] = self._clone_wavelet_payload(
                state.get("wavelet_redo_stack") or []
            )
            self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)

    def _link_wavelet_events_by_cut_refs(
        self,
        source_cut_id: int,
        source_event_id: int,
        target_cut_id: int,
        target_event_id: int,
    ) -> None:
        source_event = self._wavelet_event_ref_by_cut(source_cut_id, source_event_id)
        target_event = self._wavelet_event_ref_by_cut(target_cut_id, target_event_id)
        if source_event is None or target_event is None:
            self._set_status("Select valid source and target events first.")
            return
        if source_cut_id == target_cut_id and source_event_id == target_event_id:
            self._set_status("Select different events to link.")
            return
        self._push_wavelet_undo_state_for_cut(source_cut_id, "link event")
        if target_cut_id != source_cut_id:
            self._push_wavelet_undo_state_for_cut(target_cut_id, "link event")
        group_id = source_event.get("link_group_id") or target_event.get("link_group_id")
        if not group_id:
            group_id = self._next_link_group_label()
        source_event["link_group_id"] = group_id
        target_event["link_group_id"] = group_id
        self._append_wavelet_event_history(
            source_event,
            "link",
            details=f"group={group_id} target=cut{target_cut_id}:{target_event_id}",
        )
        self._append_wavelet_event_history(
            target_event,
            "link",
            details=f"group={group_id} source=cut{source_cut_id}:{source_event_id}",
        )
        self._record_session_change()
        self._sync_open_td_windows_from_cut(source_cut_id)
        if target_cut_id != source_cut_id:
            self._sync_open_td_windows_from_cut(target_cut_id)
        self._refresh_all_stack_browsers()
        self._set_status(f"Linked events into {group_id}.")

    def _set_wavelet_event_propagation_class(
        self, cut_id: int, event_id: int, class_name: str
    ) -> None:
        target_event = self._wavelet_event_ref_by_cut(cut_id, event_id)
        if target_event is None:
            self._set_status("Select a valid event first.")
            return
        group_id = str(target_event.get("link_group_id") or "")
        refs: list[tuple[int, dict[str, Any]]] = []
        for current_cut_id in self.cut_analysis_state:
            for event in self._cut_wavelet_events_snapshot(current_cut_id):
                if group_id and str(event.get("link_group_id") or "") == group_id:
                    refs.append((current_cut_id, event))
                elif not group_id and int(event.get("event_id", -1)) == int(event_id) and int(current_cut_id) == int(cut_id):
                    refs.append((current_cut_id, event))
        if not refs:
            refs = [(cut_id, target_event)]
        touched_cut_ids = sorted({int(item[0]) for item in refs})
        for current_cut_id in touched_cut_ids:
            self._push_wavelet_undo_state_for_cut(current_cut_id, f"set propagation class {class_name}")
        for current_cut_id, event in refs:
            event["propagation_class"] = str(class_name)
            self._append_wavelet_event_history(
                event,
                "propagation-class",
                details=f"class={class_name}",
            )
        self._record_session_change()
        for current_cut_id in touched_cut_ids:
            self._sync_open_td_windows_from_cut(current_cut_id)
        self._refresh_all_stack_browsers()

    def _stack_browser_event_row(self, event: dict[str, Any]) -> tuple[str, ...]:
        analysis = event.get("analysis") or {}
        return (
            str(int(event.get("event_id", -1))),
            str(event.get("link_group_id") or "-"),
            str(event.get("propagation_class") or "-"),
            self._td_window_wavelet_event_status(event),
            f"{float(analysis.get('peak_period_s', float('nan'))):.2f}",
            f"{float(analysis.get('fit_amp_arcsec', float('nan'))):.3f}",
            f"{self._wavelet_event_confidence_score(event):.1f}",
            ",".join(self._td_window_wavelet_event_qa_flags(event)) or "-",
        )

    def _open_selected_stack_browser(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        self._open_stack_browser(int(stack["stack_id"]))

    def _open_selected_cut_browser(self) -> None:
        cut = self.cuts.get(self.selected_cut_id) if self.selected_cut_id is not None else None
        if cut is None:
            self._set_status("Select a cut first.")
            return
        self._open_stack_browser(
            None,
            cut_ids_override=[int(cut.cut_id)],
            title_name=f"Detached {cut.name}",
        )

    def _open_stack_browser(
        self,
        stack_id: int | None,
        *,
        cut_ids_override: list[int] | None = None,
        title_name: str | None = None,
    ) -> None:
        stack = None if stack_id is None else self.stacks.get(stack_id)
        cut_ids = [
            int(cut_id) for cut_id in (cut_ids_override or []) if int(cut_id) in self.cuts
        ]
        if stack is None and stack_id is not None and not cut_ids:
            self._set_status("Invalid stack.")
            return
        if not cut_ids and stack is not None:
            cut_ids = [
                int(cut_id)
                for cut_id in (stack.get("cut_ids") or [])
                if int(cut_id) in self.cuts
            ]
        browser_name = str(
            title_name or (stack["name"] if stack is not None else "TD Browser")
        )
        browser_id = int(self.next_stack_browser_id)
        self.next_stack_browser_id += 1
        top = self.tk.Toplevel(self.root)
        top.title(f"TD Stack Browser - {browser_name}")
        top.geometry("1580x940")
        top.rowconfigure(1, weight=1)
        top.rowconfigure(2, weight=1)
        top.columnconfigure(0, weight=1)

        header = self.ttk.Frame(top, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        for idx in range(9):
            header.columnconfigure(idx, weight=(1 if idx in {0, 1} else 0))
        summary_var = self.tk.StringVar(value="")
        detail_var = self.tk.StringVar(value="")
        index_var = self.tk.IntVar(value=1)
        self.ttk.Label(header, textvariable=summary_var, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        self.ttk.Button(
            header,
            text="Up",
            command=lambda bid=browser_id: self._step_stack_browser_index(bid, -1),
        ).grid(row=0, column=3, sticky="ew", padx=(8, 4))
        self.ttk.Button(
            header,
            text="Down",
            command=lambda bid=browser_id: self._step_stack_browser_index(bid, 1),
        ).grid(row=0, column=4, sticky="ew", padx=4)
        self.ttk.Label(header, text="Index").grid(row=0, column=5, sticky="e", padx=(8, 4))
        index_spin = self.ttk.Spinbox(
            header,
            from_=1,
            to=max(len(cut_ids), 1),
            textvariable=index_var,
            width=6,
            command=lambda bid=browser_id: self._apply_stack_browser_index_var(bid),
        )
        index_spin.grid(row=0, column=6, sticky="ew")
        index_spin.bind(
            "<Return>",
            lambda _event, bid=browser_id: self._apply_stack_browser_index_var(bid),
        )
        index_spin.bind(
            "<FocusOut>",
            lambda _event, bid=browser_id: self._apply_stack_browser_index_var(bid),
        )
        self.ttk.Button(
            header,
            text="Open Full TD Editor",
            command=lambda bid=browser_id: self._open_stack_browser_current_cut_editor(bid),
        ).grid(row=0, column=7, sticky="ew", padx=(8, 4))
        self.ttk.Button(
            header,
            text="Close",
            command=lambda bid=browser_id: self._close_stack_browser(bid),
        ).grid(row=0, column=8, sticky="ew", padx=(4, 0))
        self.ttk.Label(header, text="stack_visual").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        index_scale = self.tk.Scale(
            header,
            from_=1,
            to=max(len(cut_ids), 1),
            orient="horizontal",
            variable=index_var,
            showvalue=True,
            command=lambda _value, bid=browser_id: self._apply_stack_browser_index_var(bid),
            length=760,
        )
        index_scale.grid(row=1, column=1, columnspan=8, sticky="ew", pady=(4, 0))
        self.ttk.Label(header, textvariable=detail_var, justify="left", wraplength=1520).grid(
            row=2, column=0, columnspan=9, sticky="w", pady=(4, 0)
        )

        plots_frame = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        plots_frame.grid(row=1, column=0, sticky="nsew")
        plots_frame.rowconfigure(0, weight=1)
        plots_frame.columnconfigure(0, weight=1)
        fig = self.Figure(figsize=(12.2, 6.2), dpi=120)
        axes = (fig.add_subplot(111),)
        canvas = self.FigureCanvasTkAgg(fig, master=plots_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tables_frame = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        tables_frame.grid(row=2, column=0, sticky="nsew")
        tables_frame.rowconfigure(0, weight=1)
        tables_frame.columnconfigure(0, weight=1)
        tables_frame.columnconfigure(1, weight=1)

        current_label = self.ttk.LabelFrame(tables_frame, text="Cut Events", padding=6)
        current_label.grid(row=0, column=0, sticky="nsew")
        current_label.rowconfigure(0, weight=1)
        current_label.columnconfigure(0, weight=1)

        columns = ("id", "group", "class", "status", "period", "amp", "conf", "qa")
        current_tree = self.ttk.Treeview(current_label, columns=columns, show="headings", height=10)
        current_tree.grid(row=0, column=0, sticky="nsew")
        current_scrollbar = self.ttk.Scrollbar(
            current_label, orient="vertical", command=current_tree.yview
        )
        current_scrollbar.grid(row=0, column=1, sticky="ns")
        current_tree.configure(yscrollcommand=current_scrollbar.set)
        for column, label, width in (
            ("id", "ID", 50),
            ("group", "Group", 90),
            ("class", "Class", 90),
            ("status", "Status", 115),
            ("period", "P [s]", 65),
            ("amp", "A ['']", 65),
            ("conf", "Conf", 58),
            ("qa", "QA", 120),
        ):
            current_tree.heading(column, text=label)
            current_tree.column(column, width=width, stretch=(column in {"group", "qa"}))
        current_tree.bind(
            "<<TreeviewSelect>>", lambda _event, bid=browser_id: self._on_stack_browser_tree_select(bid, "current")
        )

        trace_label = self.ttk.LabelFrame(tables_frame, text="Wave / Trace Points", padding=6)
        trace_label.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        trace_label.rowconfigure(1, weight=1)
        trace_label.columnconfigure(0, weight=1)
        trace_summary_var = self.tk.StringVar(value="")
        self.ttk.Label(
            trace_label, textvariable=trace_summary_var, justify="left", wraplength=720
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        trace_columns = ("series", "pt", "t", "frame", "dist_idx", "dist_px", "x", "y")
        trace_tree = self.ttk.Treeview(
            trace_label, columns=trace_columns, show="headings", height=10
        )
        trace_tree.grid(row=1, column=0, sticky="nsew")
        trace_scrollbar = self.ttk.Scrollbar(
            trace_label, orient="vertical", command=trace_tree.yview
        )
        trace_scrollbar.grid(row=1, column=1, sticky="ns")
        trace_tree.configure(yscrollcommand=trace_scrollbar.set)
        for column, label, width in (
            ("series", "Series", 70),
            ("pt", "Pt", 48),
            ("t", "t_idx", 78),
            ("frame", "Frame", 68),
            ("dist_idx", "dist idx", 78),
            ("dist_px", "dist px", 78),
            ("x", "map x", 78),
            ("y", "map y", 78),
        ):
            trace_tree.heading(column, text=label)
            trace_tree.column(column, width=width, stretch=True)

        controls = self.ttk.Frame(tables_frame)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for idx in range(7):
            controls.columnconfigure(idx, weight=1)
        self.ttk.Button(
            controls,
            text="Show Links",
            command=self._open_link_groups_window,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            controls,
            text="Thread->Stack",
            command=lambda bid=browser_id: self._apply_stack_browser_selected_thread_to_stack(bid),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        self.ttk.Button(
            controls,
            text="Same",
            command=lambda bid=browser_id: self._set_stack_browser_event_class(bid, "same-wave"),
        ).grid(row=0, column=2, sticky="ew", padx=4)
        self.ttk.Button(
            controls,
            text="Propagating",
            command=lambda bid=browser_id: self._set_stack_browser_event_class(bid, "propagating"),
        ).grid(row=0, column=3, sticky="ew", padx=4)
        self.ttk.Button(
            controls,
            text="Additional",
            command=lambda bid=browser_id: self._set_stack_browser_event_class(bid, "additional"),
        ).grid(row=0, column=4, sticky="ew", padx=4)
        self.ttk.Button(
            controls,
            text="Local",
            command=lambda bid=browser_id: self._set_stack_browser_event_class(bid, "local"),
        ).grid(row=0, column=5, sticky="ew", padx=4)
        self.ttk.Button(
            controls,
            text="Refresh",
            command=lambda bid=browser_id: self._refresh_stack_browser(bid),
        ).grid(row=0, column=6, sticky="ew", padx=(4, 0))

        self.stack_browsers[browser_id] = {
            "browser_id": browser_id,
            "top": top,
            "stack_id": None if stack_id is None else int(stack_id),
            "cut_ids_override": cut_ids if cut_ids_override is not None else None,
            "title_name": browser_name,
            "current_index": 0,
            "summary_var": summary_var,
            "detail_var": detail_var,
            "index_var": index_var,
            "index_spin": index_spin,
            "index_scale": index_scale,
            "figure": fig,
            "axes": axes,
            "canvas": canvas,
            "current_tree": current_tree,
            "trace_tree": trace_tree,
            "trace_summary_var": trace_summary_var,
            "current_selected_event_id": None,
            "tree_updating": False,
            "control_updating": False,
        }
        top.bind(
            "<Up>",
            lambda _event, bid=browser_id: self._step_stack_browser_index(bid, -1),
        )
        top.bind(
            "<Down>",
            lambda _event, bid=browser_id: self._step_stack_browser_index(bid, 1),
        )
        top.protocol("WM_DELETE_WINDOW", lambda bid=browser_id: self._close_stack_browser(bid))
        self._refresh_stack_browser(browser_id)

    def _close_stack_browser(self, browser_id: int) -> None:
        browser = self.stack_browsers.pop(browser_id, None)
        if browser is None:
            return
        top = browser.get("top")
        if top is not None and top.winfo_exists():
            top.destroy()

    def _stack_browser_cut_ids(self, browser_id: int) -> list[int]:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return []
        cut_ids_override = browser.get("cut_ids_override")
        if cut_ids_override is not None:
            return [
                int(cut_id) for cut_id in cut_ids_override if int(cut_id) in self.cuts
            ]
        stack_id = browser.get("stack_id")
        if stack_id is None:
            return []
        stack = self.stacks.get(int(stack_id))
        if stack is None:
            return []
        cut_ids = [int(cut_id) for cut_id in stack.get("cut_ids") or [] if int(cut_id) in self.cuts]
        stack["cut_ids"] = cut_ids
        return cut_ids

    def _step_stack_browser_index(self, browser_id: int, step: int) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        cut_ids = self._stack_browser_cut_ids(browser_id)
        if not cut_ids:
            return
        browser["current_index"] = clamp_int(
            int(browser.get("current_index", 0)) + int(step),
            0,
            len(cut_ids) - 1,
        )
        self._refresh_stack_browser(browser_id)

    def _apply_stack_browser_index_var(self, browser_id: int) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        if browser.get("control_updating"):
            return
        cut_ids = self._stack_browser_cut_ids(browser_id)
        if not cut_ids:
            return
        try:
            requested_index = int(browser["index_var"].get()) - 1
        except Exception:
            requested_index = int(browser.get("current_index", 0))
        browser["current_index"] = clamp_int(
            requested_index,
            0,
            len(cut_ids) - 1,
        )
        self._refresh_stack_browser(browser_id)

    def _refresh_stack_browser(self, browser_id: int) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        _trace_stack_wavelet(f"refresh_stack_browser start browser={browser_id}")
        top = browser.get("top")
        if top is None or not top.winfo_exists():
            self.stack_browsers.pop(browser_id, None)
            return
        stack_id = browser.get("stack_id")
        stack = None if stack_id is None else self.stacks.get(int(stack_id))
        if stack_id is not None and stack is None:
            browser["summary_var"].set("Stack no longer exists.")
            return
        browser_name = str(
            browser.get("title_name")
            or (stack["name"] if stack is not None else "TD Browser")
        )
        cut_ids = self._stack_browser_cut_ids(browser_id)
        if not cut_ids:
            browser["summary_var"].set(f"{browser_name} | no cuts")
            browser["detail_var"].set("This stack has no valid cuts.")
            browser["control_updating"] = True
            try:
                browser["index_var"].set(1)
                browser["index_spin"].configure(to=1)
                browser["index_scale"].configure(to=1)
            finally:
                browser["control_updating"] = False
            self._refresh_stack_browser_tree(
                browser["current_tree"],
                None,
                browser,
                "current",
                None,
            )
            self._refresh_stack_browser_trace_tree(browser_id, None, None)
            for ax in browser["axes"]:
                ax.clear()
                ax.text(0.5, 0.5, "No cuts in stack.", ha="center", va="center", transform=ax.transAxes)
            browser["canvas"].draw_idle()
            return
        browser["current_index"] = clamp_int(int(browser.get("current_index", 0)), 0, len(cut_ids) - 1)
        current_index = int(browser["current_index"])
        current_cut_id = int(cut_ids[current_index])
        browser["control_updating"] = True
        try:
            browser["index_var"].set(current_index + 1)
            browser["index_spin"].configure(to=max(len(cut_ids), 1))
            browser["index_scale"].configure(to=max(len(cut_ids), 1))
        finally:
            browser["control_updating"] = False
        browser["summary_var"].set(
            f"{browser_name} | TD {current_index + 1}/{len(cut_ids)} | cut {current_cut_id}: {self.cuts[current_cut_id].name}"
        )
        current_cut = self.cuts[current_cut_id]
        state = self._cut_analysis(current_cut_id)
        params = self._cut_td_params(current_cut_id)
        p0_now, p1_now = self._cut_geometry_for_frame(current_cut_id, int(self.t_visual_var.get()))
        browser["detail_var"].set(
            f"coords@t={int(self.t_visual_var.get())}: "
            f"({p0_now[0]:.1f},{p0_now[1]:.1f}) -> ({p1_now[0]:.1f},{p1_now[1]:.1f}) | "
            f"TD={int(params['t_ini'])}:{int(params['t_fin'])}:{int(params['stride'])} | "
            f"width={int(params['width'])} {params['weighting']} | "
            f"events={len(state.get('wavelet_events') or [])} | cut={current_cut.name}"
        )

        self._refresh_stack_browser_tree(
            browser["current_tree"],
            current_cut_id,
            browser,
            "current",
            state.get("wavelet_selected_event_id"),
        )
        self._refresh_stack_browser_trace_tree(
            browser_id,
            current_cut_id,
            browser.get("current_selected_event_id"),
        )
        self._draw_cut_td_axis(
            browser["axes"][0],
            current_cut_id,
            use_zoom=True,
            title_fontsize=10.0,
            selected_event_id=browser.get("current_selected_event_id"),
            title_prefix="Stack: ",
        )
        browser["figure"].tight_layout()
        browser["canvas"].draw_idle()
        _trace_stack_wavelet(
            f"refresh_stack_browser end browser={browser_id} cut={current_cut_id} index={current_index + 1}/{len(cut_ids)}"
        )

    def _refresh_stack_browser_tree(
        self,
        tree: Any,
        cut_id: int | None,
        browser: dict[str, Any],
        key: str,
        preferred_selected_event_id: int | None,
    ) -> None:
        browser["tree_updating"] = True
        try:
            children = tree.get_children()
            if children:
                tree.delete(*children)
            if cut_id is None or cut_id not in self.cuts:
                browser[f"{key}_selected_event_id"] = None
                return
            events = self._cut_wavelet_events_snapshot(cut_id)
            visible_ids: list[int] = []
            for event in events:
                visible_ids.append(int(event.get("event_id", -1)))
                tree.insert(
                    "",
                    "end",
                    iid=f"{key}-{int(event.get('event_id', -1))}",
                    values=self._stack_browser_event_row(event),
                )
            selected_event_id = preferred_selected_event_id
            if selected_event_id not in visible_ids:
                selected_event_id = visible_ids[0] if visible_ids else None
            browser[f"{key}_selected_event_id"] = selected_event_id
            if selected_event_id is not None:
                target_iid = f"{key}-{int(selected_event_id)}"
                tree.selection_set(target_iid)
                tree.focus(target_iid)
        finally:
            browser["tree_updating"] = False

    def _refresh_stack_browser_trace_tree(
        self,
        browser_id: int,
        cut_id: int | None,
        event_id: int | None,
    ) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        trace_tree = browser.get("trace_tree")
        summary_var = browser.get("trace_summary_var")
        if trace_tree is None or summary_var is None:
            return
        children = trace_tree.get_children()
        if children:
            trace_tree.delete(*children)
        if cut_id is None or cut_id not in self.cuts or event_id is None:
            summary_var.set("Select an event to inspect its wave/source trace.")
            return
        td, meta = self._cut_td(cut_id)
        if td is None or meta is None or "error" in meta:
            summary_var.set("Trace points unavailable because the TD map is invalid.")
            return
        event = self._wavelet_event_ref_by_cut(cut_id, int(event_id))
        if event is None:
            summary_var.set("The selected event is no longer available.")
            return
        rows = self._event_trace_rows(cut_id, event, meta=meta)
        if not rows:
            summary_var.set("The selected event has no saved wave/source points.")
            return
        for row in rows:
            trace_tree.insert(
                "",
                "end",
                values=(
                    str(row["series"]),
                    str(int(row["point_index"])),
                    f"{float(row['t_idx']):.2f}",
                    str(int(row["frame_idx"])),
                    f"{float(row['dist_idx']):.2f}",
                    f"{float(row['dist_px']):.2f}",
                    f"{float(row['map_x']):.2f}",
                    f"{float(row['map_y']):.2f}",
                ),
            )
        wave_count = sum(1 for row in rows if row["series"] == "wave")
        source_count = sum(1 for row in rows if row["series"] == "source")
        summary_var.set(
            f"Event {int(event_id)} | wave pts={wave_count} | source pts={source_count} | "
            "table includes t_idx, distance, and XY on the map."
        )

    def _on_stack_browser_tree_select(self, browser_id: int, key: str) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None or browser.get("tree_updating"):
            return
        tree = browser.get("current_tree") if key == "current" else browser.get("compare_tree")
        if tree is None:
            return
        previous_event_id = browser.get(f"{key}_selected_event_id")
        selection = tree.selection()
        if not selection:
            browser[f"{key}_selected_event_id"] = None
        else:
            iid = str(selection[0])
            try:
                browser[f"{key}_selected_event_id"] = int(iid.split("-", 1)[1])
            except Exception:
                browser[f"{key}_selected_event_id"] = None
        if browser.get(f"{key}_selected_event_id") == previous_event_id:
            return
        self._refresh_stack_browser(browser_id)

    def _open_stack_browser_current_cut_editor(self, browser_id: int) -> None:
        cut_ids = self._stack_browser_cut_ids(browser_id)
        browser = self.stack_browsers.get(browser_id)
        if browser is None or not cut_ids:
            return
        current_cut_id = int(cut_ids[int(browser.get("current_index", 0))])
        stack_id = browser.get("stack_id")
        self._open_cut_in_td_window(
            current_cut_id,
            prefer_hidden_panel=True,
            source_stack_id=(None if stack_id is None else int(stack_id)),
        )

    def _apply_stack_browser_selected_thread_to_stack(self, browser_id: int) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        stack_id = browser.get("stack_id")
        if stack_id is None or int(stack_id) not in self.stacks:
            self._set_status("This browser is detached; it has no stack target.")
            return
        cut_ids = self._stack_browser_cut_ids(browser_id)
        if not cut_ids:
            self._set_status("This stack has no valid cuts.")
            return
        current_cut_id = int(cut_ids[int(browser.get("current_index", 0))])
        current_event_id = browser.get("current_selected_event_id")
        if current_event_id is None:
            self._set_status("Select a current event first.")
            return
        event = self._wavelet_event_ref_by_cut(current_cut_id, int(current_event_id))
        if event is None:
            self._set_status("The selected event is no longer available.")
            return
        thread_index = self._wavelet_event_thread_index(event)
        if thread_index is None:
            self._set_status("The selected event has no valid thread index.")
            return
        applied_cut_ids, skipped_cut_ids = self._apply_wavelet_thread_filter_to_cuts(
            cut_ids,
            [thread_index],
        )
        stack_name = str(self.stacks[int(stack_id)]["name"])
        filter_text = self._format_wavelet_thread_filter_text([thread_index])
        if not applied_cut_ids and skipped_cut_ids:
            self._set_status(
                f"Thread filter {filter_text} could not be applied in {stack_name}; "
                "some cuts do not have that tracked thread."
            )
            return
        message = (
            f"Applied wavelet thread filter {filter_text} to {len(applied_cut_ids)} cut(s) in {stack_name}."
        )
        if skipped_cut_ids:
            message += f" Skipped {len(skipped_cut_ids)} cut(s) without that thread."
        self._set_status(message)

    def _link_selected_stack_browser_events(self, browser_id: int) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        cut_ids = self._stack_browser_cut_ids(browser_id)
        if not cut_ids:
            return
        current_cut_id = int(cut_ids[int(browser.get("current_index", 0))])
        compare_cut_id = browser.get("compare_cut_id")
        if compare_cut_id is None:
            self._set_status("No comparison cut available in this direction.")
            return
        current_event_id = browser.get("current_selected_event_id")
        compare_event_id = browser.get("compare_selected_event_id")
        if current_event_id is None or compare_event_id is None:
            self._set_status("Select one event in each table first.")
            return
        self._link_wavelet_events_by_cut_refs(
            current_cut_id,
            int(current_event_id),
            int(compare_cut_id),
            int(compare_event_id),
        )

    def _set_stack_browser_event_class(self, browser_id: int, class_name: str) -> None:
        browser = self.stack_browsers.get(browser_id)
        if browser is None:
            return
        cut_ids = self._stack_browser_cut_ids(browser_id)
        if not cut_ids:
            return
        current_cut_id = int(cut_ids[int(browser.get("current_index", 0))])
        current_event_id = browser.get("current_selected_event_id")
        if current_event_id is None:
            self._set_status("Select a current event first.")
            return
        self._set_wavelet_event_propagation_class(
            current_cut_id, int(current_event_id), class_name
        )

    def _build_layout(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        self.top_frame = self.ttk.Frame(self.root, padding=8)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.top_frame.columnconfigure(0, weight=1)

        self.figure_frame = self.ttk.Frame(self.root, padding=(8, 0, 8, 8))
        self.figure_frame.grid(row=1, column=0, sticky="nsew")
        self.figure_frame.rowconfigure(0, weight=1)
        self.figure_frame.columnconfigure(0, weight=1)

        self.sidebar = self.ttk.Frame(self.root, padding=(0, 0, 8, 8), width=360)
        self.sidebar.grid(row=1, column=1, sticky="nsew")
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar.columnconfigure(0, weight=1)

    def _build_controls(self) -> None:
        primary_controls = self.ttk.Frame(self.top_frame)
        primary_controls.grid(row=0, column=0, sticky="ew")
        primary_controls.columnconfigure(1, weight=1)

        self.ttk.Label(primary_controls, text="t_visual").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )

        self.t_visual_scale = self.tk.Scale(
            primary_controls,
            from_=0,
            to=self.nt - 1,
            orient="horizontal",
            variable=self.t_visual_var,
            showvalue=True,
            command=self._on_t_visual_change,
            length=600,
        )
        self.t_visual_scale.grid(row=0, column=1, sticky="ew")

        self.ttk.Label(primary_controls, text="Mosaic").grid(
            row=0, column=2, sticky="e", padx=(12, 4)
        )
        layout_box = self.ttk.Combobox(
            primary_controls,
            textvariable=self.layout_var,
            values=list(LAYOUT_PRESETS.keys()),
            state="readonly",
            width=6,
        )
        layout_box.grid(row=0, column=3, sticky="e")
        layout_box.bind("<<ComboboxSelected>>", self._on_layout_change)

        secondary_controls = self.ttk.Frame(self.top_frame)
        secondary_controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        secondary_controls.columnconfigure(0, weight=1)

        actions_frame = self.ttk.Frame(secondary_controls)
        actions_frame.grid(row=0, column=0, sticky="ew")
        for idx in range(4):
            actions_frame.columnconfigure(idx, weight=1)

        action_buttons = [
            ("Save Session", self._save_session),
            ("Load Session", self._load_session),
            ("Export Curated", self._export_curated_results),
            ("Metrics", self._open_metrics_window),
            ("Export Report", self._export_curated_report),
            ("Link Groups", self._open_link_groups_window),
            ("Propagation", self._open_propagation_window),
            ("Batch Pipeline", self._run_batch_pipeline),
            ("Saved FITS", self._open_saved_fits_browser),
            ("Open Cube", self._choose_input_cube_and_restart),
        ]
        for idx, (label, command) in enumerate(action_buttons):
            row = idx // 4
            column = idx % 4
            self.ttk.Button(actions_frame, text=label, command=command).grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 6, 0),
                pady=(0 if row == 0 else 6, 0),
            )

        view_frame = self.ttk.Frame(secondary_controls)
        view_frame.grid(row=0, column=1, sticky="e", padx=(16, 0))

        self.ttk.Label(view_frame, text="TD aspect").grid(
            row=0, column=0, sticky="e", padx=(0, 4)
        )
        aspect_box = self.ttk.Combobox(
            view_frame,
            textvariable=self.td_aspect_var,
            values=["equal", "auto"],
            state="readonly",
            width=7,
        )
        aspect_box.grid(row=0, column=1, sticky="e")
        aspect_box.bind("<<ComboboxSelected>>", self._on_axis_flip_event)

        self.ttk.Label(view_frame, text="TD zoom").grid(
            row=0, column=2, sticky="e", padx=(12, 4)
        )
        zoom_box = self.ttk.Combobox(
            view_frame,
            textvariable=self.td_zoom_var,
            values=["1x", "2x", "4x", "8x"],
            state="readonly",
            width=5,
        )
        zoom_box.grid(row=0, column=3, sticky="e")
        zoom_box.bind("<<ComboboxSelected>>", self._on_axis_flip_event)

        flip_frame = self.ttk.Frame(view_frame)
        flip_frame.grid(row=1, column=0, columnspan=4, sticky="e", pady=(6, 0))

        self.ttk.Checkbutton(
            flip_frame,
            text="Swap map XY",
            variable=self.map_swap_xy_var,
            command=self._on_axis_flip_change,
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Checkbutton(
            flip_frame,
            text="Swap TD axes",
            variable=self.td_swap_axes_var,
            command=self._on_axis_flip_change,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.ttk.Checkbutton(
            flip_frame,
            text="Flip map X",
            variable=self.map_flip_x_var,
            command=self._on_axis_flip_change,
        ).grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.ttk.Checkbutton(
            flip_frame,
            text="Flip map Y",
            variable=self.map_flip_y_var,
            command=self._on_axis_flip_change,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.ttk.Checkbutton(
            flip_frame,
            text="Flip TD X",
            variable=self.td_flip_x_var,
            command=self._on_axis_flip_change,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        self.ttk.Checkbutton(
            flip_frame,
            text="Flip TD Y",
            variable=self.td_flip_y_var,
            command=self._on_axis_flip_change,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(4, 0))

        self.ttk.Label(
            self.top_frame, textvariable=self.status_var
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def _build_sidebar(self) -> None:
        self.sidebar_notebook = self.ttk.Notebook(self.sidebar)
        self.sidebar_notebook.grid(row=0, column=0, sticky="nsew")

        panel_tab_outer, self.sidebar_panel_tab, _ = self._create_scrolled_frame(
            self.sidebar_notebook, padding=4
        )
        cuts_tab_outer, self.sidebar_cuts_tab, _ = self._create_scrolled_frame(
            self.sidebar_notebook, padding=4
        )
        geometry_tab_outer, self.sidebar_geometry_tab, _ = self._create_scrolled_frame(
            self.sidebar_notebook, padding=4
        )
        measure_tab_outer, self.sidebar_measure_tab, _ = self._create_scrolled_frame(
            self.sidebar_notebook, padding=4
        )
        stacks_tab_outer, self.sidebar_stacks_tab, _ = self._create_scrolled_frame(
            self.sidebar_notebook, padding=4
        )
        export_tab_outer, self.sidebar_export_tab, _ = self._create_scrolled_frame(
            self.sidebar_notebook, padding=4
        )

        for tab in (
            self.sidebar_panel_tab,
            self.sidebar_cuts_tab,
            self.sidebar_geometry_tab,
            self.sidebar_measure_tab,
            self.sidebar_stacks_tab,
            self.sidebar_export_tab,
        ):
            tab.columnconfigure(0, weight=1)

        self.sidebar_notebook.add(panel_tab_outer, text="TD")
        self.sidebar_notebook.add(cuts_tab_outer, text="Cuts")
        self.sidebar_notebook.add(geometry_tab_outer, text="Geometry")
        self.sidebar_notebook.add(measure_tab_outer, text="Measure")
        self.sidebar_notebook.add(stacks_tab_outer, text="Stacks")
        self.sidebar_notebook.add(export_tab_outer, text="Export")

        panel_frame = self.ttk.LabelFrame(
            self.sidebar_panel_tab, text="Panel Settings", padding=8
        )
        panel_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        panel_frame.columnconfigure(0, weight=1)

        self.ttk.Label(panel_frame, text="t_ini").grid(row=0, column=0, sticky="w")
        self.t_ini_scale = self.tk.Scale(
            panel_frame,
            from_=0,
            to=self.nt - 1,
            orient="horizontal",
            variable=self.panel_t_ini_var,
            showvalue=True,
            command=self._on_t_ini_change,
            length=260,
        )
        self.t_ini_scale.grid(row=1, column=0, sticky="ew")

        self.ttk.Label(panel_frame, text="t_fin").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.t_fin_scale = self.tk.Scale(
            panel_frame,
            from_=0,
            to=self.nt - 1,
            orient="horizontal",
            variable=self.panel_t_fin_var,
            showvalue=True,
            command=self._on_t_fin_change,
            length=260,
        )
        self.t_fin_scale.grid(row=3, column=0, sticky="ew")

        stride_row = self.ttk.Frame(panel_frame)
        stride_row.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        stride_row.columnconfigure(1, weight=1)
        self.ttk.Label(stride_row, text="stride").grid(row=0, column=0, sticky="w")
        self.stride_spin = self.ttk.Spinbox(
            stride_row,
            from_=1,
            to=max(self.nt, 10),
            textvariable=self.panel_stride_var,
            width=8,
            command=self._on_panel_param_commit,
        )
        self.stride_spin.grid(row=0, column=1, sticky="e")

        width_row = self.ttk.Frame(panel_frame)
        width_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        width_row.columnconfigure(1, weight=1)
        self.ttk.Label(width_row, text="width").grid(row=0, column=0, sticky="w")
        self.width_box = self.ttk.Combobox(
            width_row,
            textvariable=self.panel_width_var,
            values=[1, 3, 5, 7, 9],
            state="readonly",
            width=8,
        )
        self.width_box.grid(row=0, column=1, sticky="e")
        self.width_box.bind("<<ComboboxSelected>>", self._on_panel_param_event)

        weighting_row = self.ttk.Frame(panel_frame)
        weighting_row.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        weighting_row.columnconfigure(1, weight=1)
        self.ttk.Label(weighting_row, text="weighting").grid(row=0, column=0, sticky="w")
        self.weighting_box = self.ttk.Combobox(
            weighting_row,
            textvariable=self.panel_weighting_var,
            values=["uniform", "gaussian"],
            state="readonly",
            width=10,
        )
        self.weighting_box.grid(row=0, column=1, sticky="e")
        self.weighting_box.bind("<<ComboboxSelected>>", self._on_panel_param_event)

        panels_box = self.ttk.LabelFrame(self.sidebar_panel_tab, text="Panels", padding=8)
        panels_box.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        panels_box.columnconfigure(0, weight=1)
        panels_box.rowconfigure(0, weight=1)

        self.panel_listbox = self.tk.Listbox(panels_box, height=6, exportselection=False)
        self.panel_listbox.grid(row=0, column=0, sticky="ew")
        self.panel_listbox.bind("<<ListboxSelect>>", self._on_panel_list_select)

        panel_assign_frame = self.ttk.Frame(panels_box)
        panel_assign_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        panel_assign_frame.columnconfigure(1, weight=1)
        self.ttk.Label(panel_assign_frame, text="Panel cut").grid(
            row=0, column=0, sticky="w"
        )
        self.panel_cut_box = self.ttk.Combobox(
            panel_assign_frame,
            textvariable=self.panel_cut_var,
            state="readonly",
            width=18,
        )
        self.panel_cut_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.panel_cut_box.bind("<<ComboboxSelected>>", self._on_panel_cut_selection)

        panel_assign_buttons = self.ttk.Frame(panels_box)
        panel_assign_buttons.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        panel_assign_buttons.columnconfigure(0, weight=1)
        self.ttk.Button(
            panel_assign_buttons,
            text="Clear panel cut",
            command=self._clear_active_panel_cut,
        ).grid(row=0, column=0, sticky="ew")

        panel_button_row = self.ttk.Frame(panels_box)
        panel_button_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        panel_button_row.columnconfigure(0, weight=1)
        panel_button_row.columnconfigure(1, weight=1)
        self.ttk.Button(
            panel_button_row, text="Open TD Window", command=self._open_td_window
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            panel_button_row, text="Close TD Window", command=self._close_active_td_window
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        cuts_box = self.ttk.LabelFrame(self.sidebar_cuts_tab, text="Cuts", padding=8)
        cuts_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        cuts_box.columnconfigure(0, weight=1)
        cuts_box.rowconfigure(0, weight=1)

        self.cut_listbox = self.tk.Listbox(cuts_box, height=8, exportselection=False)
        self.cut_listbox.grid(row=0, column=0, sticky="ew")
        self.cut_listbox.bind("<<ListboxSelect>>", self._on_cut_list_select)

        button_frame = self.ttk.LabelFrame(
            self.sidebar_cuts_tab, text="Cut Actions", padding=8
        )
        button_frame.grid(row=1, column=0, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.ttk.Button(
            button_frame, text="Add Cut", command=self._start_add_cut
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            button_frame, text="Draw/Replace Cut", command=self._start_draw_cut
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            button_frame, text="Delete Cut", command=self._delete_selected_cut
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.ttk.Button(
            button_frame, text="Copy Cut", command=self._copy_selected_cut
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        self.ttk.Button(
            button_frame, text="Paste Cut", command=self._paste_cut
        ).grid(row=2, column=1, sticky="ew", pady=(6, 0), padx=(4, 0))
        self.ttk.Button(
            button_frame, text="Rotate -5°", command=lambda: self._rotate_selected_cut(-5.0)
        ).grid(row=3, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        self.ttk.Button(
            button_frame, text="Rotate +5°", command=lambda: self._rotate_selected_cut(5.0)
        ).grid(row=3, column=1, sticky="ew", pady=(6, 0), padx=(4, 0))
        self.ttk.Button(
            button_frame, text="Open Cut Browser", command=self._open_selected_cut_browser
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        feature_axis_box = self.ttk.LabelFrame(
            self.sidebar_cuts_tab, text="Feature Axis / Auto Cuts", padding=8
        )
        feature_axis_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        feature_axis_box.columnconfigure(0, weight=1)
        feature_axis_box.columnconfigure(1, weight=1)

        self.ttk.Label(
            feature_axis_box,
            textvariable=self.selected_feature_axis_name_var,
            justify="left",
            wraplength=320,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.feature_axis_listbox = self.tk.Listbox(
            feature_axis_box, height=6, exportselection=False
        )
        self.feature_axis_listbox.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        self.feature_axis_listbox.bind(
            "<<ListboxSelect>>", self._on_feature_axis_list_select
        )

        feature_button_row = self.ttk.Frame(feature_axis_box)
        feature_button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        feature_button_row.columnconfigure(0, weight=1)
        feature_button_row.columnconfigure(1, weight=1)
        self.ttk.Button(
            feature_button_row,
            text="Draw Line",
            command=lambda: self._start_draw_feature_axis("line"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            feature_button_row,
            text="Draw Curve",
            command=lambda: self._start_draw_feature_axis("curve"),
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            feature_button_row,
            text="Finish Curve",
            command=self._finish_pending_feature_axis,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))
        self.ttk.Button(
            feature_button_row,
            text="Delete Axis",
            command=self._delete_selected_feature_axis,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))

        feature_params = self.ttk.Frame(feature_axis_box)
        feature_params.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        feature_params.columnconfigure(1, weight=1)

        self.ttk.Label(feature_params, text="Spacing").grid(row=0, column=0, sticky="w")
        self.feature_spacing_entry = self.ttk.Entry(
            feature_params, textvariable=self.feature_spacing_var, width=10
        )
        self.feature_spacing_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.ttk.Label(feature_params, text="Length").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.feature_length_entry = self.ttk.Entry(
            feature_params, textvariable=self.feature_length_var, width=10
        )
        self.feature_length_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        self.ttk.Label(feature_params, text="Angle").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.feature_angle_entry = self.ttk.Entry(
            feature_params, textvariable=self.feature_angle_offset_var, width=10
        )
        self.feature_angle_entry.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        self.ttk.Label(
            feature_params,
            text="Angle is relative to the local perpendicular. 0° = automatic normal.",
            justify="left",
            wraplength=320,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.ttk.Checkbutton(
            feature_params,
            text="Create stack from generated cuts",
            variable=self.feature_create_stack_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.ttk.Button(
            feature_params,
            text="Generate Cuts",
            command=self._generate_cuts_from_selected_feature_axis,
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        geometry_frame = self.ttk.LabelFrame(
            self.sidebar_geometry_tab, text="Geometry", padding=8
        )
        geometry_frame.grid(row=0, column=0, sticky="ew")
        geometry_frame.columnconfigure(1, weight=1)

        self.ttk.Label(
            geometry_frame, textvariable=self.selected_cut_name_var
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        self.ttk.Label(geometry_frame, text="Anchor").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        anchor_box = self.ttk.Combobox(
            geometry_frame,
            textvariable=self.geometry_anchor_var,
            values=["center", "fix p0", "fix p1"],
            state="readonly",
            width=10,
        )
        anchor_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        self.ttk.Label(geometry_frame, text="Length mode").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        length_mode_box = self.ttk.Combobox(
            geometry_frame,
            textvariable=self.geometry_length_mode_var,
            values=["symmetric", "fix p0", "fix p1"],
            state="readonly",
            width=10,
        )
        length_mode_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        self.ttk.Label(geometry_frame, text="Angle").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        angle_entry = self.ttk.Entry(
            geometry_frame, textvariable=self.geometry_angle_var, width=10
        )
        self.geometry_angle_entry = angle_entry
        angle_entry.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.ttk.Button(
            geometry_frame, text="Set", command=self._set_selected_cut_angle
        ).grid(row=3, column=2, sticky="ew", padx=(6, 0), pady=(8, 0))

        angle_buttons = self.ttk.Frame(geometry_frame)
        angle_buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for idx, delta in enumerate([-10, -5, -1, 1, 5, 10]):
            text = f"{delta:+d}"
            self.ttk.Button(
                angle_buttons,
                text=text,
                command=lambda d=delta: self._adjust_selected_cut_angle(d),
            ).grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))
            angle_buttons.columnconfigure(idx, weight=1)

        self.ttk.Label(geometry_frame, text="Length").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        length_entry = self.ttk.Entry(
            geometry_frame, textvariable=self.geometry_length_var, width=10
        )
        self.geometry_length_entry = length_entry
        length_entry.grid(row=5, column=1, sticky="ew", pady=(8, 0))
        self.ttk.Button(
            geometry_frame, text="Set", command=self._set_selected_cut_length
        ).grid(row=5, column=2, sticky="ew", padx=(6, 0), pady=(8, 0))

        length_buttons = self.ttk.Frame(geometry_frame)
        length_buttons.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for idx, delta in enumerate([-20, -10, -5, 5, 10, 20]):
            text = f"{delta:+d}"
            self.ttk.Button(
                length_buttons,
                text=text,
                command=lambda d=delta: self._adjust_selected_cut_length(d),
            ).grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))
            length_buttons.columnconfigure(idx, weight=1)

        coords_frame = self.ttk.Frame(geometry_frame)
        coords_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for idx in range(4):
            coords_frame.columnconfigure(idx, weight=1)

        self.ttk.Label(coords_frame, text="x1").grid(row=0, column=0, sticky="w")
        self.ttk.Label(coords_frame, text="y1").grid(row=0, column=1, sticky="w")
        self.ttk.Label(coords_frame, text="x2").grid(row=0, column=2, sticky="w")
        self.ttk.Label(coords_frame, text="y2").grid(row=0, column=3, sticky="w")

        coord_entries = [
            self.ttk.Entry(coords_frame, textvariable=self.geometry_x1_var, width=8),
            self.ttk.Entry(coords_frame, textvariable=self.geometry_y1_var, width=8),
            self.ttk.Entry(coords_frame, textvariable=self.geometry_x2_var, width=8),
            self.ttk.Entry(coords_frame, textvariable=self.geometry_y2_var, width=8),
        ]
        self.geometry_coord_entries = coord_entries
        for idx, entry in enumerate(coord_entries):
            entry.grid(row=1, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))

        self.ttk.Button(
            geometry_frame, text="Apply coords", command=self._apply_selected_cut_coords
        ).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        dynamic_frame = self.ttk.LabelFrame(
            self.sidebar_geometry_tab, text="Dynamic Cut", padding=8
        )
        dynamic_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        dynamic_frame.columnconfigure(0, weight=1)
        dynamic_frame.columnconfigure(1, weight=1)
        dynamic_frame.columnconfigure(2, weight=1)

        self.ttk.Checkbutton(
            dynamic_frame,
            text="Enable time-varying geometry",
            variable=self.dynamic_cut_enabled_var,
            command=self._toggle_selected_cut_dynamic,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self.ttk.Label(
            dynamic_frame,
            textvariable=self.dynamic_keyframe_summary_var,
            justify="left",
            wraplength=320,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.ttk.Label(dynamic_frame, text="Ref frame").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.ttk.Label(
            dynamic_frame, textvariable=self.dynamic_reference_frame_var
        ).grid(row=2, column=1, sticky="w", pady=(6, 0))
        self.ttk.Button(
            dynamic_frame,
            text="Set Ref = t",
            command=self._set_selected_cut_dynamic_reference,
        ).grid(row=2, column=2, sticky="ew", padx=(6, 0), pady=(6, 0))

        self.dynamic_keyframe_listbox = self.tk.Listbox(
            dynamic_frame, height=6, exportselection=False
        )
        self.dynamic_keyframe_listbox.grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        self.dynamic_keyframe_listbox.bind(
            "<<ListboxSelect>>", self._on_dynamic_keyframe_select
        )

        self.ttk.Button(
            dynamic_frame,
            text="Capture @ t",
            command=self._capture_selected_cut_dynamic_keyframe,
        ).grid(row=4, column=0, sticky="ew", pady=(6, 0), padx=(0, 4))
        self.ttk.Button(
            dynamic_frame,
            text="Delete @ t",
            command=self._delete_selected_cut_dynamic_keyframe,
        ).grid(row=4, column=1, sticky="ew", pady=(6, 0), padx=4)
        self.ttk.Button(
            dynamic_frame,
            text="Clear Keys",
            command=self._clear_selected_cut_dynamic_keyframes,
        ).grid(row=4, column=2, sticky="ew", pady=(6, 0), padx=(4, 0))

        measure_frame = self.ttk.LabelFrame(
            self.sidebar_measure_tab, text="Measurements / Relative Control", padding=8
        )
        measure_frame.grid(row=0, column=0, sticky="ew")
        measure_frame.columnconfigure(1, weight=1)

        self.ttk.Label(measure_frame, text="Reference").grid(row=0, column=0, sticky="w")
        self.reference_cut_box = self.ttk.Combobox(
            measure_frame,
            textvariable=self.reference_cut_var,
            state="readonly",
            width=20,
        )
        self.reference_cut_box.grid(row=0, column=1, sticky="ew")
        self.reference_cut_box.bind("<<ComboboxSelected>>", self._on_measurement_selection)

        self.ttk.Label(measure_frame, text="Target").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.target_cut_box = self.ttk.Combobox(
            measure_frame,
            textvariable=self.target_cut_var,
            state="readonly",
            width=20,
        )
        self.target_cut_box.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.target_cut_box.bind("<<ComboboxSelected>>", self._on_measurement_selection)

        self.ttk.Label(
            measure_frame, text="Reference stays fixed. Target changes."
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.ttk.Label(
            measure_frame, textvariable=self.measure_center_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.ttk.Label(
            measure_frame, textvariable=self.measure_min_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.ttk.Label(
            measure_frame, textvariable=self.measure_angle_var
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

        rel_button_frame = self.ttk.Frame(measure_frame)
        rel_button_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        rel_button_frame.columnconfigure(0, weight=1)
        rel_button_frame.columnconfigure(1, weight=1)
        self.ttk.Button(
            rel_button_frame, text="Copy angle", command=self._copy_reference_angle
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            rel_button_frame, text="Copy length", command=self._copy_reference_length
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            rel_button_frame, text="Match center", command=self._match_centers
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))

        orient_button_frame = self.ttk.Frame(measure_frame)
        orient_button_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        orient_button_frame.columnconfigure(0, weight=1)
        orient_button_frame.columnconfigure(1, weight=1)
        self.ttk.Button(
            orient_button_frame, text="Parallel", command=self._make_parallel
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            orient_button_frame, text="Perpendicular", command=self._make_perpendicular
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        vertex_button_frame = self.ttk.Frame(measure_frame)
        vertex_button_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        vertex_button_frame.columnconfigure(0, weight=1)
        vertex_button_frame.columnconfigure(1, weight=1)
        self.ttk.Button(
            vertex_button_frame, text="Match p0", command=lambda: self._match_vertices("p0")
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            vertex_button_frame, text="Match p1", command=lambda: self._match_vertices("p1")
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            vertex_button_frame,
            text="Match both verts",
            command=self._copy_reference_geometry,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        center_dist_frame = self.ttk.Frame(measure_frame)
        center_dist_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        center_dist_frame.columnconfigure(1, weight=1)
        self.ttk.Label(center_dist_frame, text="Center distance").grid(
            row=0, column=0, sticky="w"
        )
        center_distance_entry = self.ttk.Entry(
            center_dist_frame, textvariable=self.center_distance_var, width=10
        )
        self.center_distance_entry = center_distance_entry
        center_distance_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.ttk.Button(
            center_dist_frame,
            text="Set",
            command=self._set_center_distance_between_cuts,
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        for widget in (self.stride_spin,):
            widget.bind("<Return>", self._on_panel_param_event)
            widget.bind("<FocusOut>", self._on_panel_param_event)
        for widget in (
            angle_entry,
            length_entry,
            center_distance_entry,
            *coord_entries,
        ):
            widget.bind("<Return>", self._on_geometry_entry_event)

        stacks_box = self.ttk.LabelFrame(
            self.sidebar_stacks_tab, text="Stacks", padding=8
        )
        stacks_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        stacks_box.columnconfigure(0, weight=1)
        stacks_box.rowconfigure(0, weight=1)

        self.stack_listbox = self.tk.Listbox(stacks_box, height=6, exportselection=False)
        self.stack_listbox.grid(row=0, column=0, sticky="ew")
        self.stack_listbox.bind("<<ListboxSelect>>", self._on_stack_list_select)

        stack_button_row = self.ttk.Frame(stacks_box)
        stack_button_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        stack_button_row.columnconfigure(0, weight=1)
        stack_button_row.columnconfigure(1, weight=1)
        self.ttk.Button(
            stack_button_row, text="New Empty", command=self._new_empty_stack
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            stack_button_row, text="From All Cuts", command=self._new_stack_from_all_cuts
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            stack_button_row, text="Rename", command=self._rename_selected_stack
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))
        self.ttk.Button(
            stack_button_row, text="Delete", command=self._delete_selected_stack
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))

        members_box = self.ttk.LabelFrame(
            self.sidebar_stacks_tab, text="Stack Members", padding=8
        )
        members_box.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        members_box.columnconfigure(0, weight=1)
        members_box.rowconfigure(0, weight=1)
        self.stack_member_listbox = self.tk.Listbox(
            members_box, height=10, exportselection=False
        )
        self.stack_member_listbox.grid(row=0, column=0, sticky="ew")
        self.stack_member_listbox.bind(
            "<<ListboxSelect>>", self._on_stack_member_select
        )

        member_button_row = self.ttk.Frame(members_box)
        member_button_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        member_button_row.columnconfigure(0, weight=1)
        member_button_row.columnconfigure(1, weight=1)
        self.ttk.Button(
            member_button_row, text="Add Selected Cut", command=self._add_selected_cut_to_stack
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            member_button_row, text="Add All Cuts", command=self._add_all_cuts_to_stack
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            member_button_row, text="Remove", command=self._remove_selected_cut_from_stack
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))
        self.ttk.Button(
            member_button_row, text="Open Browser", command=self._open_selected_stack_browser
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))
        self.ttk.Button(
            member_button_row, text="Move Up", command=lambda: self._move_selected_stack_member(-1)
        ).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))
        self.ttk.Button(
            member_button_row, text="Move Down", command=lambda: self._move_selected_stack_member(1)
        ).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))

        export_dir_box = self.ttk.LabelFrame(
            self.sidebar_export_tab, text="Export Folder", padding=8
        )
        export_dir_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        export_dir_box.columnconfigure(0, weight=1)

        export_dir_entry = self.ttk.Entry(
            export_dir_box, textvariable=self.export_dir_var
        )
        export_dir_entry.grid(row=0, column=0, sticky="ew")
        self.ttk.Button(
            export_dir_box,
            text="Browse",
            command=self._browse_export_dir,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.ttk.Label(
            export_dir_box,
            textvariable=self.export_info_var,
            justify="left",
            wraplength=320,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        export_map_box = self.ttk.LabelFrame(
            self.sidebar_export_tab, text="Maps", padding=8
        )
        export_map_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        export_map_box.columnconfigure(0, weight=1)
        self.ttk.Button(
            export_map_box,
            text="Save current map as FITS+PNG",
            command=self._export_current_map_fits,
        ).grid(row=0, column=0, sticky="ew")
        self.ttk.Label(
            export_map_box,
            text="Saves the cube frame at the current visual time as FITS and also a PNG overview with the cuts at that t.",
            justify="left",
            wraplength=320,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        export_cut_box = self.ttk.LabelFrame(
            self.sidebar_export_tab, text="TD Cuts", padding=8
        )
        export_cut_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        export_cut_box.columnconfigure(0, weight=1)
        export_cut_box.columnconfigure(1, weight=1)
        self.ttk.Button(
            export_cut_box,
            text="Selected cut -> FITS+PNG",
            command=self._export_selected_cut_fits,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            export_cut_box,
            text="Stack -> FITS+PNG",
            command=self._export_stack_cut_fits,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            export_cut_box,
            text="All cuts -> FITS+PNG",
            command=self._export_all_cut_fits,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.ttk.Label(
            export_cut_box,
            text="Each export writes a FITS with the TD data and a PNG quicklook with map/cut context and TD wavelet overlays.",
            justify="left",
            wraplength=320,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        export_trace_box = self.ttk.LabelFrame(
            self.sidebar_export_tab, text="Traces / Waves", padding=8
        )
        export_trace_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        export_trace_box.columnconfigure(0, weight=1)
        export_trace_box.columnconfigure(1, weight=1)
        self.ttk.Button(
            export_trace_box,
            text="Selected cut traces -> FITS+PNG",
            command=self._export_selected_cut_trace_fits,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            export_trace_box,
            text="Stack traces -> FITS+PNG",
            command=self._export_stack_trace_fits,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            export_trace_box,
            text="All traces -> FITS+PNG",
            command=self._export_all_trace_fits,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.ttk.Label(
            export_trace_box,
            text="Writes one FITS per event/trace and a PNG quicklook with the event trace on the map plus the TD wavelet overlay.",
            justify="left",
            wraplength=320,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        export_review_box = self.ttk.LabelFrame(
            self.sidebar_export_tab, text="Review Saved FITS", padding=8
        )
        export_review_box.grid(row=4, column=0, sticky="ew")
        export_review_box.columnconfigure(0, weight=1)
        self.ttk.Button(
            export_review_box,
            text="Open saved FITS browser",
            command=self._open_saved_fits_browser,
        ).grid(row=0, column=0, sticky="ew")
        self.ttk.Label(
            export_review_box,
            text="Browse the FITS exported by this app from the current export folder and inspect TD/maps/tables.",
            justify="left",
            wraplength=320,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _build_figure(self) -> None:
        self.canvas = self.FigureCanvasTkAgg(self.figure, master=self.figure_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", self._on_global_left_shortcut)
        self.root.bind("<Right>", self._on_global_right_shortcut)
        self.root.bind("<Delete>", lambda _event: self._delete_selected_cut())
        self.root.bind("<Control-c>", lambda _event: self._copy_selected_cut())
        self.root.bind("<Control-v>", lambda _event: self._paste_cut())
        self.root.bind("<Escape>", lambda _event: self._cancel_pending_cut())
        self.root.bind("r", lambda _event: self._rotate_selected_cut(5.0))
        self.root.bind("R", lambda _event: self._rotate_selected_cut(-5.0))
        self.root.bind("n", lambda _event: self._start_draw_cut())
        self.root.bind("g", lambda _event: self._start_draw_feature_axis("line"))
        self.root.bind("G", lambda _event: self._start_draw_feature_axis("curve"))

    def _widget_blocks_timeline_shortcuts(self, widget: Any) -> bool:
        blocked_classes = {"Entry", "TEntry", "Spinbox", "TCombobox", "Text"}
        current = widget
        visited: set[int] = set()
        while current is not None:
            current_id = id(current)
            if current_id in visited:
                break
            visited.add(current_id)
            try:
                if str(current.winfo_class()) in blocked_classes:
                    return True
            except Exception:
                pass
            current = getattr(current, "master", None)
        return False

    def _on_global_left_shortcut(self, event: Any) -> str | None:
        if self._widget_blocks_timeline_shortcuts(getattr(event, "widget", None)):
            return None
        self._step_t_visual(-1)
        return "break"

    def _on_global_right_shortcut(self, event: Any) -> str | None:
        if self._widget_blocks_timeline_shortcuts(getattr(event, "widget", None)):
            return None
        self._step_t_visual(1)
        return "break"

    def _apply_layout(self) -> None:
        rows, cols = LAYOUT_PRESETS[self.layout_var.get()]
        self.visible_panels = rows * cols

        if self.active_panel_id > self.visible_panels:
            self.active_panel_id = 1

        self.figure.clear()
        grid = self.figure.add_gridspec(
            1, 2, width_ratios=[1.0, 1.15], wspace=0.18
        )
        self.map_ax = self.figure.add_subplot(grid[0, 0])
        mosaic_grid = grid[0, 1].subgridspec(rows, cols, hspace=0.28, wspace=0.22)

        self.panel_axes.clear()
        self.axis_to_panel_id.clear()
        for index in range(self.visible_panels):
            row = index // cols
            col = index % cols
            ax = self.figure.add_subplot(mosaic_grid[row, col])
            panel_id = index + 1
            self.panel_axes[panel_id] = ax
            self.axis_to_panel_id[ax] = panel_id

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
                category=UserWarning,
            )
            self.figure.tight_layout()
        self._refresh_panel_list()
        self._sync_controls_from_active_panel()

    def _save_session(self) -> None:
        default_name = self.cube_path.stem + "_td_session.json"
        save_path = self.filedialog.asksaveasfilename(
            title="Save session",
            initialdir=str(Path(__file__).resolve().parent.parent),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not save_path:
            return
        save_path_obj = Path(save_path)
        self._write_session_file(save_path_obj, autosave=False)
        self._set_status(f"Session saved to {save_path_obj}")

    def _refresh_panel_list(self) -> None:
        self.panel_listbox.delete(0, self.tk.END)
        for panel in self.panels[: self.visible_panels]:
            cut = self.cuts.get(panel.cut_id) if panel.cut_id is not None else None
            params = self._panel_td_params(panel)
            label = (
                f"{panel.name} | no cut"
                if cut is None
                else f"{panel.name} | {cut.name} | "
                f"t={int(params['t_ini'])}:{int(params['t_fin'])}:{int(params['stride'])} | "
                f"w={int(params['width'])} {params['weighting']}"
            )
            self.panel_listbox.insert(self.tk.END, label)

        if self.visible_panels > 0:
            self.panel_listbox.selection_clear(0, self.tk.END)
            self.panel_listbox.selection_set(self.active_panel_id - 1)
            self.panel_listbox.activate(self.active_panel_id - 1)

    def _refresh_cut_list(self) -> None:
        self.cut_listbox.delete(0, self.tk.END)
        ordered = sorted(self.cuts.values(), key=lambda cut: cut.cut_id)
        for cut in ordered:
            preview_cut = self._cut_preview(cut.cut_id) or cut
            marker = "L" if cut.locked else " "
            dynamic_marker = f"D{len(self._cut_dynamic_keyframes(cut.cut_id))}" if self._cut_dynamic_enabled(cut.cut_id) else "--"
            label = (
                f"{cut.cut_id:02d} {cut.name} {marker} {dynamic_marker} "
                f"({preview_cut.p0[0]:.1f},{preview_cut.p0[1]:.1f}) -> ({preview_cut.p1[0]:.1f},{preview_cut.p1[1]:.1f})"
            )
            self.cut_listbox.insert(self.tk.END, label)

        if self.selected_cut_id is None:
            return

        index = next(
            (i for i, cut in enumerate(ordered) if cut.cut_id == self.selected_cut_id),
            None,
        )
        if index is not None:
            self.cut_listbox.selection_clear(0, self.tk.END)
            self.cut_listbox.selection_set(index)
            self.cut_listbox.activate(index)

    def _feature_axis_label(self, axis: FeatureAxis) -> str:
        total_length = polyline_length(axis.points)
        return (
            f"{axis.axis_id:02d} {axis.name} | {axis.mode} | "
            f"pts={len(axis.points)} | L={total_length:.1f}"
        )

    def _selected_feature_axis(self) -> FeatureAxis | None:
        if self.selected_feature_axis_id is None:
            return None
        return self.feature_axes.get(self.selected_feature_axis_id)

    def _refresh_feature_axis_list(self) -> None:
        if not hasattr(self, "feature_axis_listbox"):
            return
        self.feature_axis_listbox.delete(0, self.tk.END)
        ordered = sorted(self.feature_axes.values(), key=lambda axis: axis.axis_id)
        for axis in ordered:
            self.feature_axis_listbox.insert(self.tk.END, self._feature_axis_label(axis))
        if self.selected_feature_axis_id not in self.feature_axes:
            self.selected_feature_axis_id = ordered[0].axis_id if ordered else None
        if self.selected_feature_axis_id is not None:
            index = next(
                (
                    idx
                    for idx, axis in enumerate(ordered)
                    if int(axis.axis_id) == int(self.selected_feature_axis_id)
                ),
                None,
            )
            if index is not None:
                self.feature_axis_listbox.selection_clear(0, self.tk.END)
                self.feature_axis_listbox.selection_set(index)
                self.feature_axis_listbox.activate(index)
        self._sync_feature_axis_controls()

    def _sync_feature_axis_controls(self) -> None:
        axis = self._selected_feature_axis()
        if self.feature_draw_mode == "curve" and not self.feature_pending_points:
            self.selected_feature_axis_name_var.set(
                "Drawing curve axis: left-click to add points. Finish Curve or right-click closes it."
            )
            return
        if self.feature_draw_mode == "curve" and self.feature_pending_points:
            self.selected_feature_axis_name_var.set(
                f"Drawing curve axis: {len(self.feature_pending_points)} point(s). "
                "Left-click adds points, Finish Curve or right-click closes it."
            )
            return
        if self.feature_draw_mode == "line" and not self.feature_pending_points:
            self.selected_feature_axis_name_var.set(
                "Drawing line axis: click the first point on the map."
            )
            return
        if self.feature_draw_mode == "line" and self.feature_pending_points:
            self.selected_feature_axis_name_var.set(
                "Drawing line axis: choose the second point on the map."
            )
            return
        if axis is None:
            self.selected_feature_axis_name_var.set(
                "No feature axis selected. Draw a line or curve to define the feature spine."
            )
            return
        self.selected_feature_axis_name_var.set(
            f"Selected: {axis.name} | mode={axis.mode} | points={len(axis.points)} | "
            f"length={polyline_length(axis.points):.1f} px"
        )

    def _stack_label(self, stack: dict[str, Any]) -> str:
        return f"{int(stack['stack_id']):02d} {stack['name']} ({len(stack.get('cut_ids') or [])})"

    def _selected_stack(self) -> dict[str, Any] | None:
        if self.active_stack_id is None:
            return None
        return self.stacks.get(self.active_stack_id)

    def _refresh_stack_list(self) -> None:
        if not hasattr(self, "stack_listbox"):
            return
        self.stack_listbox.delete(0, self.tk.END)
        ordered = sorted(self.stacks.values(), key=lambda item: int(item["stack_id"]))
        for stack in ordered:
            self.stack_listbox.insert(self.tk.END, self._stack_label(stack))
        if self.active_stack_id not in self.stacks:
            self.active_stack_id = ordered[0]["stack_id"] if ordered else None
        if self.active_stack_id is None:
            return
        index = next(
            (
                idx
                for idx, stack in enumerate(ordered)
                if int(stack["stack_id"]) == int(self.active_stack_id)
            ),
            None,
        )
        if index is not None:
            self.stack_listbox.selection_clear(0, self.tk.END)
            self.stack_listbox.selection_set(index)
            self.stack_listbox.activate(index)

    def _refresh_stack_member_list(self) -> None:
        if not hasattr(self, "stack_member_listbox"):
            return
        self.stack_member_listbox.delete(0, self.tk.END)
        stack = self._selected_stack()
        if stack is None:
            self.selected_stack_cut_id = None
            return
        cut_ids = [
            int(cut_id) for cut_id in stack.get("cut_ids", []) if int(cut_id) in self.cuts
        ]
        stack["cut_ids"] = cut_ids
        for index, cut_id in enumerate(cut_ids, start=1):
            cut = self.cuts[cut_id]
            state = self._cut_analysis(cut_id)
            event_count = len(state.get("wavelet_events") or [])
            tracking_done = "T" if state.get("crest_tracking_result") else "-"
            wavelet_done = "W" if state.get("wavelet_filter_result") else "-"
            dynamic_tag = f"D{len(self._cut_dynamic_keyframes(cut_id))}" if self._cut_dynamic_enabled(cut_id) else "--"
            self.stack_member_listbox.insert(
                self.tk.END,
                f"{index:02d} | {cut.cut_id:02d} {cut.name} | {dynamic_tag} | {tracking_done}{wavelet_done} | events={event_count}",
            )
        if self.selected_stack_cut_id not in cut_ids:
            self.selected_stack_cut_id = cut_ids[0] if cut_ids else None
        if self.selected_stack_cut_id is None:
            return
        member_index = next(
            (
                idx
                for idx, cut_id in enumerate(cut_ids)
                if int(cut_id) == int(self.selected_stack_cut_id)
            ),
            None,
        )
        if member_index is not None:
            self.stack_member_listbox.selection_clear(0, self.tk.END)
            self.stack_member_listbox.selection_set(member_index)
            self.stack_member_listbox.activate(member_index)

    def _create_stack(
        self, *, name: str | None = None, cut_ids: list[int] | None = None
    ) -> dict[str, Any]:
        stack_id = int(self.next_stack_id)
        self.next_stack_id += 1
        stack = self._make_default_stack_state(
            stack_id,
            name or f"Stack {stack_id}",
            cut_ids or [],
        )
        self.stacks[stack_id] = stack
        self.active_stack_id = stack_id
        self.selected_stack_cut_id = (cut_ids or [None])[0]
        self._record_session_change()
        self.refresh_all()
        return stack

    def _new_empty_stack(self) -> None:
        self._create_stack()
        self._set_status("Created an empty TD stack.")

    def _new_stack_from_all_cuts(self) -> None:
        ordered_cut_ids = sorted(self.cuts.keys())
        if not ordered_cut_ids:
            self._set_status("Create one or more cuts first.")
            return
        self._create_stack(cut_ids=ordered_cut_ids)
        self._set_status(f"Created a stack with {len(ordered_cut_ids)} cut(s).")

    def _rename_selected_stack(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        name = self.simpledialog.askstring(
            "Rename Stack",
            "Stack name:",
            initialvalue=str(stack["name"]),
            parent=self.root,
        )
        if name is None:
            return
        stack["name"] = str(name).strip() or str(stack["name"])
        self._record_session_change()
        self.refresh_all()

    def _delete_selected_stack(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        stack_id = int(stack["stack_id"])
        self.stacks.pop(stack_id, None)
        if self.active_stack_id == stack_id:
            self.active_stack_id = next(iter(sorted(self.stacks.keys())), None)
        self.selected_stack_cut_id = None
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Deleted stack {stack_id}.")

    def _add_cut_to_stack(self, stack_id: int, cut_id: int) -> None:
        stack = self.stacks.get(stack_id)
        if stack is None or cut_id not in self.cuts:
            return
        cut_ids = list(stack.get("cut_ids") or [])
        if cut_id not in cut_ids:
            cut_ids.append(cut_id)
            stack["cut_ids"] = cut_ids
            self.selected_stack_cut_id = cut_id
            self._record_session_change()
            self.refresh_all()

    def _add_selected_cut_to_stack(self) -> None:
        stack = self._selected_stack()
        cut = self._selected_cut()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        if cut is None:
            self._set_status("Select a cut first.")
            return
        self._add_cut_to_stack(int(stack["stack_id"]), int(cut.cut_id))
        self._set_status(f"Added {cut.name} to {stack['name']}.")

    def _add_all_cuts_to_stack(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        added = 0
        for cut_id in sorted(self.cuts.keys()):
            before = len(stack.get("cut_ids") or [])
            self._add_cut_to_stack(int(stack["stack_id"]), int(cut_id))
            if len(stack.get("cut_ids") or []) > before:
                added += 1
        if added == 0:
            self._set_status(f"{stack['name']} already contains all cuts.")
        else:
            self._set_status(f"Added {added} cut(s) to {stack['name']}.")

    def _remove_selected_cut_from_stack(self) -> None:
        stack = self._selected_stack()
        if stack is None:
            self._set_status("Select a stack first.")
            return
        if self.selected_stack_cut_id is None:
            self._set_status("Select a stack member first.")
            return
        cut_ids = [int(cut_id) for cut_id in stack.get("cut_ids") or []]
        if self.selected_stack_cut_id not in cut_ids:
            self._set_status("Select a valid stack member first.")
            return
        cut_ids.remove(int(self.selected_stack_cut_id))
        stack["cut_ids"] = cut_ids
        self.selected_stack_cut_id = cut_ids[0] if cut_ids else None
        self._record_session_change()
        self.refresh_all()

    def _move_selected_stack_member(self, direction: int) -> None:
        stack = self._selected_stack()
        if stack is None or self.selected_stack_cut_id is None:
            self._set_status("Select a stack member first.")
            return
        cut_ids = [int(cut_id) for cut_id in stack.get("cut_ids") or []]
        if self.selected_stack_cut_id not in cut_ids:
            self._set_status("Select a valid stack member first.")
            return
        index = cut_ids.index(int(self.selected_stack_cut_id))
        target = index + int(direction)
        if target < 0 or target >= len(cut_ids):
            return
        cut_ids[index], cut_ids[target] = cut_ids[target], cut_ids[index]
        stack["cut_ids"] = cut_ids
        self._record_session_change()
        self.refresh_all()

    def _on_stack_list_select(self, _event: Any) -> None:
        if not hasattr(self, "stack_listbox"):
            return
        selection = self.stack_listbox.curselection()
        if not selection:
            return
        ordered = sorted(self.stacks.values(), key=lambda item: int(item["stack_id"]))
        if selection[0] >= len(ordered):
            return
        self.active_stack_id = int(ordered[selection[0]]["stack_id"])
        self.selected_stack_cut_id = None
        self.refresh_all()

    def _on_stack_member_select(self, _event: Any) -> None:
        stack = self._selected_stack()
        if stack is None or not hasattr(self, "stack_member_listbox"):
            return
        selection = self.stack_member_listbox.curselection()
        if not selection:
            return
        cut_ids = [int(cut_id) for cut_id in stack.get("cut_ids") or [] if int(cut_id) in self.cuts]
        if selection[0] >= len(cut_ids):
            return
        self.selected_stack_cut_id = cut_ids[selection[0]]
        self.selected_cut_id = self.selected_stack_cut_id
        self.refresh_all()

    def _sync_controls_from_active_panel(self) -> None:
        panel = self.active_panel
        params = self._panel_td_params(panel)
        self.control_update_guard = True
        self.panel_t_ini_var.set(int(params["t_ini"]))
        self.panel_t_fin_var.set(int(params["t_fin"]))
        self.panel_stride_var.set(int(params["stride"]))
        self.panel_width_var.set(int(params["width"]))
        self.panel_weighting_var.set(str(params["weighting"]))
        self.control_update_guard = False

    def _cut_option_label(self, cut: Cut) -> str:
        return f"{cut.cut_id}: {cut.name}"

    def _cut_from_option(self, value: str) -> Cut | None:
        if not value:
            return None
        try:
            cut_id = int(str(value).split(":", 1)[0].strip())
        except Exception:
            return None
        return self.cuts.get(cut_id)

    def _refresh_measurement_selectors(self) -> None:
        ordered = sorted(self.cuts.values(), key=lambda cut: cut.cut_id)
        values = [self._cut_option_label(cut) for cut in ordered]
        self.reference_cut_box["values"] = values
        self.target_cut_box["values"] = values

        ref_cut = self._cut_from_option(self.reference_cut_var.get())
        target_cut = self._cut_from_option(self.target_cut_var.get())

        if ref_cut is None and ordered:
            ref_cut = ordered[0]

        if target_cut is None:
            if self.selected_cut_id is not None and self.selected_cut_id in self.cuts:
                target_cut = self.cuts[self.selected_cut_id]
            elif len(ordered) > 1:
                target_cut = ordered[1]
            elif ordered:
                target_cut = ordered[0]

        if (
            ref_cut is not None
            and target_cut is not None
            and ref_cut.cut_id == target_cut.cut_id
            and len(ordered) > 1
        ):
            for candidate in ordered:
                if candidate.cut_id != ref_cut.cut_id:
                    target_cut = candidate
                    break

        self.reference_cut_var.set("" if ref_cut is None else self._cut_option_label(ref_cut))
        self.target_cut_var.set("" if target_cut is None else self._cut_option_label(target_cut))

    def _refresh_panel_cut_selector(self) -> None:
        ordered = sorted(self.cuts.values(), key=lambda cut: cut.cut_id)
        values = [self._cut_option_label(cut) for cut in ordered]
        self.panel_selector_update_guard = True
        self.panel_cut_box["values"] = values

        panel = self.active_panel
        cut = self.cuts.get(panel.cut_id) if panel.cut_id is not None else None
        if cut is None:
            self.panel_cut_var.set("")
        else:
            self.panel_cut_var.set(self._cut_option_label(cut))
        self.panel_selector_update_guard = False

    def _refresh_measurements(self) -> None:
        ref_cut = self._cut_from_option(self.reference_cut_var.get())
        target_cut = self._cut_from_option(self.target_cut_var.get())

        if ref_cut is None or target_cut is None:
            self.measure_center_var.set("Center distance: n/a")
            self.measure_min_var.set("Min distance: n/a")
            self.measure_angle_var.set("Angle difference: n/a")
            return

        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        target_preview = self._cut_preview(target_cut.cut_id) or target_cut
        center_dist = distance(cut_center(ref_preview), cut_center(target_preview))
        min_dist = min_segment_distance(ref_preview, target_preview)
        angle_diff = angle_difference_deg(ref_preview, target_preview)

        self.measure_center_var.set(f"Center distance: {center_dist:.2f} px")
        self.measure_min_var.set(f"Min distance: {min_dist:.2f} px")
        self.measure_angle_var.set(f"Angle difference: {angle_diff:.2f} deg")

    def _sync_geometry_controls_from_selected_cut(self) -> None:
        cut = self._selected_cut()
        self.geometry_update_guard = True
        if cut is None:
            self.selected_cut_name_var.set("No cut selected")
            self.geometry_angle_var.set("")
            self.geometry_length_var.set("")
            self.geometry_x1_var.set("")
            self.geometry_y1_var.set("")
            self.geometry_x2_var.set("")
            self.geometry_y2_var.set("")
        else:
            preview_cut = self._cut_preview(cut.cut_id) or cut
            dynamic_label = " | dynamic" if self._cut_dynamic_enabled(cut.cut_id) else ""
            self.selected_cut_name_var.set(f"Selected: {cut.name}{dynamic_label}")
            self.geometry_angle_var.set(f"{cut_display_angle_deg(preview_cut):.2f}")
            self.geometry_length_var.set(f"{cut_length(preview_cut):.2f}")
            self.geometry_x1_var.set(f"{preview_cut.p0[0]:.2f}")
            self.geometry_y1_var.set(f"{preview_cut.p0[1]:.2f}")
            self.geometry_x2_var.set(f"{preview_cut.p1[0]:.2f}")
            self.geometry_y2_var.set(f"{preview_cut.p1[1]:.2f}")
        self.geometry_update_guard = False
        self._refresh_dynamic_cut_controls()

    def _refresh_dynamic_cut_controls(self) -> None:
        if not hasattr(self, "dynamic_keyframe_listbox"):
            return
        cut = self._selected_cut()
        self.dynamic_keyframe_listbox.delete(0, self.tk.END)
        if cut is None:
            self.dynamic_cut_enabled_var.set(False)
            self.dynamic_reference_frame_var.set("")
            self.dynamic_keyframe_summary_var.set("Select a cut to enable dynamic geometry.")
            return

        enabled = self._cut_dynamic_enabled(cut.cut_id)
        reference_frame = self._cut_dynamic_reference_frame(cut.cut_id)
        keyframes = self._cut_dynamic_keyframes(cut.cut_id)
        current_t = int(self.t_visual_var.get())
        self.dynamic_cut_enabled_var.set(enabled)
        self.dynamic_reference_frame_var.set(str(reference_frame))
        for frame_idx in sorted(keyframes.keys()):
            marker = " <- t" if int(frame_idx) == current_t else ""
            self.dynamic_keyframe_listbox.insert(
                self.tk.END,
                f"t={int(frame_idx):04d}{marker}",
            )
        if keyframes:
            selected_index = next(
                (
                    idx
                    for idx, frame_idx in enumerate(sorted(keyframes.keys()))
                    if int(frame_idx) == current_t
                ),
                None,
            )
            if selected_index is not None:
                self.dynamic_keyframe_listbox.selection_clear(0, self.tk.END)
                self.dynamic_keyframe_listbox.selection_set(selected_index)
                self.dynamic_keyframe_listbox.activate(selected_index)
        mode_text = "enabled" if enabled else "disabled"
        self.dynamic_keyframe_summary_var.set(
            f"Dynamic geometry {mode_text}. Ref={reference_frame}. "
            f"Keyframes={len(keyframes)}. Editing at current t writes a keyframe."
        )

    def _toggle_selected_cut_dynamic(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self.dynamic_cut_enabled_var.set(False)
            self._set_status("Select a cut first.")
            return
        state = self._cut_analysis(cut.cut_id)
        enabled = bool(self.dynamic_cut_enabled_var.get())
        state["dynamic_enabled"] = enabled
        if enabled and not state.get("dynamic_keyframes"):
            state["dynamic_reference_frame"] = clamp_int(
                int(self.t_visual_var.get()), 0, self.nt - 1
            )
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(
            f"{'Enabled' if enabled else 'Disabled'} dynamic geometry for {cut.name}."
        )

    def _capture_selected_cut_dynamic_keyframe(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("Select a cut first.")
            return
        preview_cut = self._cut_preview(cut.cut_id) or cut
        frame_idx = int(self.t_visual_var.get())
        if not self._set_cut_dynamic_keyframe(
            cut.cut_id, frame_idx, preview_cut.p0, preview_cut.p1, enable_dynamic=True
        ):
            self._set_status("Could not capture dynamic keyframe.")
            return
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Captured dynamic keyframe for {cut.name} at t={frame_idx}.")

    def _delete_selected_cut_dynamic_keyframe(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("Select a cut first.")
            return
        frame_idx = int(self.t_visual_var.get())
        if not self._delete_cut_dynamic_keyframe(cut.cut_id, frame_idx):
            self._set_status(f"{cut.name} has no keyframe at t={frame_idx}.")
            return
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Deleted dynamic keyframe for {cut.name} at t={frame_idx}.")

    def _clear_selected_cut_dynamic_keyframes(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("Select a cut first.")
            return
        state = self._cut_analysis(cut.cut_id)
        if not state.get("dynamic_keyframes"):
            self._set_status(f"{cut.name} has no dynamic keyframes.")
            return
        state["dynamic_keyframes"] = {}
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Cleared dynamic keyframes for {cut.name}.")

    def _set_selected_cut_dynamic_reference(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("Select a cut first.")
            return
        frame_idx = int(self.t_visual_var.get())
        state = self._cut_analysis(cut.cut_id)
        old_reference_frame = int(state.get("dynamic_reference_frame", 0))
        current_geometry = self._cut_geometry_for_frame(cut.cut_id, frame_idx)
        old_reference_geometry = self._cut_geometry_for_frame(cut.cut_id, old_reference_frame)
        keyframes = dict(state.get("dynamic_keyframes") or {})
        if old_reference_frame != frame_idx:
            keyframes[old_reference_frame] = {
                "p0": [
                    float(old_reference_geometry[0][0]),
                    float(old_reference_geometry[0][1]),
                ],
                "p1": [
                    float(old_reference_geometry[1][0]),
                    float(old_reference_geometry[1][1]),
                ],
            }
        keyframes.pop(frame_idx, None)
        cut.p0 = clamp_point(current_geometry[0], self.nx, self.ny)
        cut.p1 = clamp_point(current_geometry[1], self.nx, self.ny)
        state["dynamic_reference_frame"] = frame_idx
        state["dynamic_enabled"] = True
        state["dynamic_keyframes"] = keyframes
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Set dynamic reference frame of {cut.name} to t={frame_idx}.")

    def _on_dynamic_keyframe_select(self, _event: Any) -> None:
        cut = self._selected_cut()
        if cut is None or not hasattr(self, "dynamic_keyframe_listbox"):
            return
        selection = self.dynamic_keyframe_listbox.curselection()
        if not selection:
            return
        keyframe_frames = sorted(self._cut_dynamic_keyframes(cut.cut_id).keys())
        if selection[0] >= len(keyframe_frames):
            return
        self.t_visual_var.set(int(keyframe_frames[selection[0]]))
        self.refresh_all()

    def _editable_cut(self, cut: Cut | None, action: str) -> Cut | None:
        if cut is None:
            self._set_status(f"No cut selected to {action}.")
            return None
        if cut.locked:
            self._set_status(f"{cut.name} is locked.")
            return None
        return cut

    def _selected_editable_cut(self, action: str) -> Cut | None:
        return self._editable_cut(self._selected_cut(), action)

    def _apply_cut_points(
        self, cut: Cut, p0: tuple[float, float], p1: tuple[float, float], status: str
    ) -> bool:
        p0 = clamp_point(p0, self.nx, self.ny)
        p1 = clamp_point(p1, self.nx, self.ny)
        if distance(p0, p1) < 1.0:
            self._set_status("Cut too short after the requested operation.")
            return False

        if self._cut_dynamic_enabled(cut.cut_id):
            if not self._set_cut_dynamic_keyframe(
                cut.cut_id,
                int(self.t_visual_var.get()),
                p0,
                p1,
                enable_dynamic=True,
            ):
                self._set_status("Could not update dynamic cut geometry.")
                return False
        else:
            cut.p0 = p0
            cut.p1 = p1
        self.selected_cut_id = cut.cut_id
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(status)
        return True

    def _normalize_anchor_mode(self, anchor_mode: str) -> str | None:
        value = str(anchor_mode).strip().lower()
        mapping = {
            "center": "center",
            "p0": "p0",
            "p1": "p1",
            "fix p0": "p0",
            "fix p1": "p1",
        }
        return mapping.get(value)

    def _normalize_length_mode(self, length_mode: str) -> str | None:
        value = str(length_mode).strip().lower()
        mapping = {
            "symmetric": "symmetric",
            "from p0": "p0",
            "from p1": "p1",
            "fix p0": "p0",
            "fix p1": "p1",
        }
        return mapping.get(value)

    def _set_cut_angle(
        self, cut: Cut, angle_deg: float, anchor_mode: str, status_prefix: str
    ) -> bool:
        preview_cut = self._cut_preview(cut.cut_id) or cut
        desired_length = cut_length(preview_cut)
        normalized_anchor = self._normalize_anchor_mode(anchor_mode)
        if normalized_anchor == "center":
            p0, p1, actual_length = segment_from_angle_length(
                angle_deg,
                desired_length,
                "center",
                None,
                cut_center(preview_cut),
                self.nx,
                self.ny,
            )
        elif normalized_anchor == "p0":
            p0, p1, actual_length = segment_from_angle_length(
                angle_deg,
                desired_length,
                "p0",
                preview_cut.p0,
                None,
                self.nx,
                self.ny,
            )
        elif normalized_anchor == "p1":
            p0, p1, actual_length = segment_from_angle_length(
                angle_deg,
                desired_length,
                "p1",
                preview_cut.p1,
                None,
                self.nx,
                self.ny,
            )
        else:
            self._set_status(f"Unknown anchor mode: {anchor_mode}")
            return False

        limited = actual_length + 1e-6 < desired_length
        status = status_prefix
        if limited:
            status += " Limited by image bounds."
        return self._apply_cut_points(cut, p0, p1, status)

    def _set_cut_length(
        self, cut: Cut, length_value: float, length_mode: str, status_prefix: str
    ) -> bool:
        preview_cut = self._cut_preview(cut.cut_id) or cut
        angle_deg = cut_directed_angle_deg(preview_cut)
        requested_length = max(float(length_value), 1.0)

        normalized_length = self._normalize_length_mode(length_mode)
        if normalized_length == "symmetric":
            p0, p1, actual_length = segment_from_angle_length(
                angle_deg,
                requested_length,
                "center",
                None,
                cut_center(preview_cut),
                self.nx,
                self.ny,
            )
        elif normalized_length == "p0":
            p0, p1, actual_length = segment_from_angle_length(
                angle_deg,
                requested_length,
                "p0",
                preview_cut.p0,
                None,
                self.nx,
                self.ny,
            )
        elif normalized_length == "p1":
            p0, p1, actual_length = segment_from_angle_length(
                angle_deg,
                requested_length,
                "p1",
                preview_cut.p1,
                None,
                self.nx,
                self.ny,
            )
        else:
            self._set_status(f"Unknown length mode: {length_mode}")
            return False

        limited = actual_length + 1e-6 < requested_length
        status = status_prefix
        if limited:
            status += " Limited by image bounds."
        return self._apply_cut_points(cut, p0, p1, status)

    def _parse_float_var(self, value: str, field_name: str) -> float | None:
        try:
            return float(str(value).strip())
        except Exception:
            self._set_status(f"Invalid {field_name} value.")
            return None

    def _on_measurement_selection(self, _event: Any) -> None:
        self._refresh_measurements()

    def _on_panel_cut_selection(self, _event: Any) -> None:
        if self.panel_selector_update_guard:
            return
        self._assign_panel_cut_from_selector()

    def _on_geometry_entry_event(self, event: Any) -> None:
        widget = event.widget
        if widget == self.geometry_angle_entry:
            self._set_selected_cut_angle()
        elif widget == self.geometry_length_entry:
            self._set_selected_cut_length()
        elif widget == self.center_distance_entry:
            self._set_center_distance_between_cuts()
        elif widget in getattr(self, "geometry_coord_entries", []):
            self._apply_selected_cut_coords()

    def _on_angle_entry_event(self, _event: Any) -> None:
        self._set_selected_cut_angle()

    def _on_length_entry_event(self, _event: Any) -> None:
        self._set_selected_cut_length()

    def _on_coord_entry_event(self, _event: Any) -> None:
        self._apply_selected_cut_coords()

    def _on_center_distance_entry_event(self, _event: Any) -> None:
        self._set_center_distance_between_cuts()

    def _set_selected_cut_angle(self) -> None:
        if self.geometry_update_guard:
            return
        cut = self._selected_editable_cut("set angle")
        if cut is None:
            return
        angle_deg = self._parse_float_var(self.geometry_angle_var.get(), "angle")
        if angle_deg is None:
            return
        anchor_mode = self.geometry_anchor_var.get()
        self._set_cut_angle(cut, angle_deg, anchor_mode, f"Set {cut.name} angle to {angle_deg:.2f} deg.")

    def _adjust_selected_cut_angle(self, delta_deg: float) -> None:
        cut = self._selected_editable_cut("adjust angle")
        if cut is None:
            return
        new_angle = cut_display_angle_deg(cut) + delta_deg
        self._set_cut_angle(
            cut,
            new_angle,
            self.geometry_anchor_var.get(),
            f"Adjusted {cut.name} angle by {delta_deg:+.1f} deg.",
        )

    def _set_selected_cut_length(self) -> None:
        if self.geometry_update_guard:
            return
        cut = self._selected_editable_cut("set length")
        if cut is None:
            return
        length_value = self._parse_float_var(self.geometry_length_var.get(), "length")
        if length_value is None:
            return
        self._set_cut_length(
            cut,
            length_value,
            self.geometry_length_mode_var.get(),
            f"Set {cut.name} length to {length_value:.2f} px.",
        )

    def _adjust_selected_cut_length(self, delta_length: float) -> None:
        cut = self._selected_editable_cut("adjust length")
        if cut is None:
            return
        new_length = max(1.0, cut_length(cut) + delta_length)
        self._set_cut_length(
            cut,
            new_length,
            self.geometry_length_mode_var.get(),
            f"Adjusted {cut.name} length by {delta_length:+.1f} px.",
        )

    def _apply_selected_cut_coords(self) -> None:
        if self.geometry_update_guard:
            return
        cut = self._selected_editable_cut("apply coordinates")
        if cut is None:
            return

        x1 = self._parse_float_var(self.geometry_x1_var.get(), "x1")
        y1 = self._parse_float_var(self.geometry_y1_var.get(), "y1")
        x2 = self._parse_float_var(self.geometry_x2_var.get(), "x2")
        y2 = self._parse_float_var(self.geometry_y2_var.get(), "y2")
        if None in {x1, y1, x2, y2}:
            return

        self._apply_cut_points(
            cut,
            (float(x1), float(y1)),
            (float(x2), float(y2)),
            f"Applied coordinates to {cut.name}.",
        )

    def _measurement_cuts(self, action: str | None = None) -> tuple[Cut | None, Cut | None]:
        ref_cut = self._cut_from_option(self.reference_cut_var.get())
        target_cut = self._cut_from_option(self.target_cut_var.get())
        if action is not None:
            if (
                ref_cut is not None
                and target_cut is not None
                and ref_cut.cut_id == target_cut.cut_id
            ):
                self._set_status("Reference and target must be different cuts.")
                return ref_cut, None
            target_cut = self._editable_cut(target_cut, action)
        return ref_cut, target_cut

    def _copy_reference_angle(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("copy angle")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return
        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        self._set_cut_angle(
            target_cut,
            cut_display_angle_deg(ref_preview),
            "center",
            f"Copied angle from {ref_cut.name} to {target_cut.name}.",
        )

    def _copy_reference_length(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("copy length")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return
        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        self._set_cut_length(
            target_cut,
            cut_length(ref_preview),
            "symmetric",
            f"Copied length from {ref_cut.name} to {target_cut.name}.",
        )

    def _make_parallel(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("make parallel")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return
        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        self._set_cut_angle(
            target_cut,
            cut_display_angle_deg(ref_preview),
            "center",
            f"Made {target_cut.name} parallel to {ref_cut.name}.",
        )

    def _make_perpendicular(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("make perpendicular")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return
        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        self._set_cut_angle(
            target_cut,
            cut_display_angle_deg(ref_preview) + 90.0,
            "center",
            f"Made {target_cut.name} perpendicular to {ref_cut.name}.",
        )

    def _match_centers(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("match center")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return

        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        target_preview = self._cut_preview(target_cut.cut_id) or target_cut
        ref_center = cut_center(ref_preview)
        target_center = cut_center(target_preview)
        dx = ref_center[0] - target_center[0]
        dy = ref_center[1] - target_center[1]
        p0, p1 = shift_cut(target_preview, dx, dy, self.nx, self.ny)
        self._apply_cut_points(
            target_cut, p0, p1, f"Matched center of {target_cut.name} to {ref_cut.name}."
        )

    def _match_vertices(self, vertex_name: str) -> None:
        ref_cut, target_cut = self._measurement_cuts(f"match {vertex_name}")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return
        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        target_preview = self._cut_preview(target_cut.cut_id) or target_cut

        if vertex_name == "p0":
            dx = ref_preview.p0[0] - target_preview.p0[0]
            dy = ref_preview.p0[1] - target_preview.p0[1]
            status = f"Matched p0 of {target_cut.name} to {ref_cut.name}."
        elif vertex_name == "p1":
            dx = ref_preview.p1[0] - target_preview.p1[0]
            dy = ref_preview.p1[1] - target_preview.p1[1]
            status = f"Matched p1 of {target_cut.name} to {ref_cut.name}."
        else:
            self._set_status(f"Unknown vertex selector: {vertex_name}")
            return

        p0, p1 = shift_cut(target_preview, dx, dy, self.nx, self.ny)
        self._apply_cut_points(target_cut, p0, p1, status)

    def _copy_reference_geometry(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("copy geometry")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return
        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut

        self._apply_cut_points(
            target_cut,
            ref_preview.p0,
            ref_preview.p1,
            f"Copied both vertices from {ref_cut.name} to {target_cut.name}.",
        )

    def _set_center_distance_between_cuts(self) -> None:
        ref_cut, target_cut = self._measurement_cuts("set center distance")
        if ref_cut is None or target_cut is None:
            self._set_status("Select reference and target cuts first.")
            return

        desired = self._parse_float_var(
            self.center_distance_var.get(), "center distance"
        )
        if desired is None:
            return
        desired = max(0.0, desired)

        ref_preview = self._cut_preview(ref_cut.cut_id) or ref_cut
        target_preview = self._cut_preview(target_cut.cut_id) or target_cut
        ref_center = cut_center(ref_preview)
        target_center = cut_center(target_preview)
        dx = target_center[0] - ref_center[0]
        dy = target_center[1] - ref_center[1]
        current = math.hypot(dx, dy)

        if current < 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / current, dy / current

        new_center = (ref_center[0] + desired * ux, ref_center[1] + desired * uy)
        move_dx = new_center[0] - target_center[0]
        move_dy = new_center[1] - target_center[1]
        p0, p1 = shift_cut(target_preview, move_dx, move_dy, self.nx, self.ny)
        self._apply_cut_points(
            target_cut,
            p0,
            p1,
            f"Set center distance between {ref_cut.name} and {target_cut.name} to {desired:.2f} px.",
        )

    def _assign_panel_cut_from_selector(self) -> None:
        cut = self._cut_from_option(self.panel_cut_var.get())
        if cut is None:
            self._set_status("Choose a cut in the TD tab first.")
            return

        panel = self.active_panel
        panel.cut_id = cut.cut_id
        self._seed_cut_td_params_from_panel(cut.cut_id, panel)
        self.selected_cut_id = cut.cut_id
        self._invalidate_panel_cache(panel.panel_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Assigned {cut.name} to {panel.name}.")

    def _clear_active_panel_cut(self) -> None:
        panel = self.active_panel
        if panel.cut_id is None:
            self._set_status(f"{panel.name} already has no cut assigned.")
            return

        panel.cut_id = None
        self._invalidate_panel_cache(panel.panel_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Cleared cut assignment from {panel.name}.")

    def _apply_controls_to_active_panel(self) -> None:
        if self.control_update_guard:
            return

        panel = self.active_panel
        t_ini = int(self.panel_t_ini_var.get())
        t_fin = int(self.panel_t_fin_var.get())

        if t_ini > t_fin:
            t_fin = t_ini
            self.control_update_guard = True
            self.panel_t_fin_var.set(t_fin)
            self.control_update_guard = False

        panel.t_ini = clamp_int(t_ini, 0, self.nt - 1)
        panel.t_fin = clamp_int(t_fin, panel.t_ini, self.nt - 1)
        panel.stride = max(int(self.panel_stride_var.get()), 1)
        panel.width = max(int(self.panel_width_var.get()), 1)
        panel.weighting = str(self.panel_weighting_var.get())
        if panel.cut_id is not None and panel.cut_id in self.cuts:
            state = self._cut_analysis(panel.cut_id)
            state["td_params"] = {
                "t_ini": int(panel.t_ini),
                "t_fin": int(panel.t_fin),
                "stride": int(panel.stride),
                "width": int(panel.width),
                "weighting": str(panel.weighting),
            }
            state["td_cache_key"] = None
            state["td_cache_td"] = None
            state["td_cache_meta"] = None
            self._sync_panels_from_cut_td_params(panel.cut_id)
            self._invalidate_cut_dependents(panel.cut_id)
        else:
            self._invalidate_panel_cache(panel.panel_id)
        self._record_session_change()
        self.refresh_all()

    def _clear_panel_analysis_results(
        self, panel_id: int, *, clear_tracking: bool, clear_wavelet: bool
    ) -> None:
        state = self._panel_analysis(panel_id)
        state["td_cache_key"] = None
        state["td_cache_td"] = None
        state["td_cache_meta"] = None
        if clear_tracking:
            state["crest_tracking_result"] = None
            state["crest_tracking_td_key"] = None
        if clear_tracking or clear_wavelet:
            state["wavelet_filter_result"] = None
            state["wavelet_events"] = []
            state["wavelet_next_event_id"] = 1
            state["wavelet_selected_event_id"] = None

    def _invalidate_panel_cache(self, panel_id: int) -> None:
        panel = self.panels[panel_id - 1]
        panel.cache_key = None
        panel.cache_td = None
        panel.cache_meta = None
        self._clear_panel_analysis_results(
            panel_id, clear_tracking=True, clear_wavelet=True
        )
        self._clear_td_window_crest_tracking(panel_id, stale=True)

    def _invalidate_cut_dependents(self, cut_id: int) -> None:
        for panel in self.panels:
            if panel.cut_id == cut_id:
                self._invalidate_panel_cache(panel.panel_id)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _on_axis_flip_change(self) -> None:
        self._record_session_change()
        self.refresh_all()

    def _on_axis_flip_event(self, _event: Any) -> None:
        self._on_axis_flip_change()

    def _zoom_limits(
        self, lo: float, hi: float, center: float, zoom_value: float
    ) -> tuple[float, float]:
        span = hi - lo
        if span <= 0 or zoom_value <= 1.0:
            return lo, hi

        window = span / zoom_value
        center = clamp_value(center, lo, hi)
        start = center - window / 2.0
        end = center + window / 2.0

        if start < lo:
            end += lo - start
            start = lo
        if end > hi:
            start -= end - hi
            end = hi

        start = max(start, lo)
        end = min(end, hi)
        return start, end

    def _map_data_to_display(self, x: float, y: float) -> tuple[float, float]:
        if self.map_swap_xy_var.get():
            return y, x
        return x, y

    def _map_display_to_data(self, u: float, v: float) -> tuple[float, float]:
        if self.map_swap_xy_var.get():
            return v, u
        return u, v

    def _map_display_shape(self) -> tuple[int, int]:
        if self.map_swap_xy_var.get():
            return self.ny, self.nx
        return self.nx, self.ny

    def _on_t_visual_change(self, _value: str) -> None:
        self._sync_geometry_controls_from_selected_cut()
        self._refresh_measurements()
        self._refresh_cut_list()
        self._refresh_export_controls()
        self.refresh_map()
        self.refresh_td_views()

    def _on_t_ini_change(self, value: str) -> None:
        if self.control_update_guard:
            return
        self.panel_t_ini_var.set(int(float(value)))
        self._apply_controls_to_active_panel()

    def _on_t_fin_change(self, value: str) -> None:
        if self.control_update_guard:
            return
        self.panel_t_fin_var.set(int(float(value)))
        self._apply_controls_to_active_panel()

    def _on_panel_param_commit(self) -> None:
        self._apply_controls_to_active_panel()

    def _on_panel_param_event(self, _event: Any) -> None:
        self._apply_controls_to_active_panel()

    def _on_layout_change(self, _event: Any) -> None:
        self._apply_layout()
        self._record_session_change()
        self.refresh_all()

    def _step_t_visual(self, step: int) -> None:
        new_value = clamp_int(self.t_visual_var.get() + step, 0, self.nt - 1)
        self.t_visual_var.set(new_value)
        self._on_t_visual_change(str(new_value))

    def _on_panel_list_select(self, _event: Any) -> None:
        selection = self.panel_listbox.curselection()
        if not selection:
            return
        panel_id = selection[0] + 1
        if panel_id > self.visible_panels:
            return
        self.active_panel_id = panel_id
        panel = self.active_panel
        self.selected_cut_id = panel.cut_id
        self._sync_controls_from_active_panel()
        self.refresh_all()

    def _on_cut_list_select(self, _event: Any) -> None:
        selection = self.cut_listbox.curselection()
        if not selection:
            return
        ordered = sorted(self.cuts.values(), key=lambda cut: cut.cut_id)
        if selection[0] >= len(ordered):
            return
        self.selected_cut_id = ordered[selection[0]].cut_id
        self.refresh_all()

    def _on_feature_axis_list_select(self, _event: Any) -> None:
        if not hasattr(self, "feature_axis_listbox"):
            return
        selection = self.feature_axis_listbox.curselection()
        if not selection:
            return
        ordered = sorted(self.feature_axes.values(), key=lambda axis: axis.axis_id)
        if selection[0] >= len(ordered):
            return
        self.selected_feature_axis_id = ordered[selection[0]].axis_id
        self._sync_feature_axis_controls()
        self.refresh_map()

    def _create_cut(self, p0: tuple[float, float], p1: tuple[float, float]) -> Cut:
        cut = Cut(
            cut_id=self.next_cut_id,
            name=f"Cut {self.next_cut_id}",
            color=COLOR_CYCLE[(self.next_cut_id - 1) % len(COLOR_CYCLE)],
            p0=p0,
            p1=p1,
        )
        self.cuts[cut.cut_id] = cut
        self.next_cut_id += 1
        return cut

    def _create_feature_axis(
        self, points: list[tuple[float, float]], mode: str
    ) -> FeatureAxis:
        axis = FeatureAxis(
            axis_id=self.next_feature_axis_id,
            name=f"Axis {self.next_feature_axis_id}",
            color=COLOR_CYCLE[(self.next_feature_axis_id - 1) % len(COLOR_CYCLE)],
            points=[(float(point[0]), float(point[1])) for point in points],
            mode="line" if str(mode) == "line" else "curve",
        )
        self.feature_axes[axis.axis_id] = axis
        self.next_feature_axis_id += 1
        self.selected_feature_axis_id = axis.axis_id
        return axis

    def _start_draw_feature_axis(self, mode: str) -> None:
        self.draw_mode = False
        self.force_new_cut = False
        self.pending_point = None
        self.hover_point = None
        self.drag_state = None
        self.feature_draw_mode = "line" if str(mode) == "line" else "curve"
        self.feature_pending_points = []
        self.feature_hover_point = None
        self._sync_feature_axis_controls()
        if self.feature_draw_mode == "line":
            self._set_status(
                "Feature line mode: click two points on the map to define the feature axis."
            )
        else:
            self._set_status(
                "Feature curve mode: left-click to add points. Finish Curve or right-click closes it."
            )
        self.refresh_map()

    def _cancel_feature_axis_draw(self) -> None:
        if self.feature_draw_mode is None and not self.feature_pending_points:
            return
        self.feature_draw_mode = None
        self.feature_pending_points = []
        self.feature_hover_point = None
        self._sync_feature_axis_controls()
        self._set_status("Feature axis drawing cancelled.")
        self.refresh_map()

    def _append_feature_axis_point(self, x: float, y: float) -> None:
        if self.feature_draw_mode is None:
            return
        point = clamp_point((x, y), self.nx, self.ny)
        if self.feature_pending_points:
            if distance(self.feature_pending_points[-1], point) < 1.0:
                self._set_status("Choose a point farther away to define the feature axis.")
                return
        self.feature_pending_points.append(point)
        self.feature_hover_point = point
        if self.feature_draw_mode == "line" and len(self.feature_pending_points) >= 2:
            self._finish_pending_feature_axis()
            return
        self._sync_feature_axis_controls()
        if self.feature_draw_mode == "curve":
            self._set_status(
                f"Added curve point {len(self.feature_pending_points)} at ({point[0]:.1f}, {point[1]:.1f})."
            )
        else:
            self._set_status(
                f"Feature line start set at ({point[0]:.1f}, {point[1]:.1f}). Choose the second point."
            )
        self.refresh_map()

    def _finish_pending_feature_axis(self) -> None:
        if self.feature_draw_mode is None:
            self._set_status("No feature axis is being drawn.")
            return
        normalized_points: list[tuple[float, float]] = []
        for point in self.feature_pending_points:
            clamped = clamp_point(point, self.nx, self.ny)
            if normalized_points and distance(normalized_points[-1], clamped) < 1.0:
                continue
            normalized_points.append(clamped)
        if len(normalized_points) < 2 or polyline_length(normalized_points) < 1.0:
            self._set_status("Feature axis is too short; add at least two separated points.")
            return
        axis_mode = self.feature_draw_mode
        axis = self._create_feature_axis(normalized_points, axis_mode)
        self.feature_draw_mode = None
        self.feature_pending_points = []
        self.feature_hover_point = None
        self._record_session_change()
        self.refresh_all()
        self._set_status(
            f"Saved {axis.mode} feature axis {axis.name} with {len(axis.points)} point(s)."
        )

    def _delete_selected_feature_axis(self) -> None:
        axis = self._selected_feature_axis()
        if axis is None:
            self._set_status("Select a feature axis first.")
            return
        self.feature_axes.pop(axis.axis_id, None)
        if self.selected_feature_axis_id == axis.axis_id:
            self.selected_feature_axis_id = None
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Deleted feature axis {axis.name}.")

    def _generate_cuts_from_selected_feature_axis(self) -> None:
        axis = self._selected_feature_axis()
        if axis is None:
            self._set_status("Select a feature axis first.")
            return

        spacing = self._parse_float_var(self.feature_spacing_var.get(), "feature spacing")
        length_value = self._parse_float_var(self.feature_length_var.get(), "feature length")
        angle_offset = self._parse_float_var(
            self.feature_angle_offset_var.get(), "feature angle"
        )
        if spacing is None or length_value is None or angle_offset is None:
            return

        spacing = max(float(spacing), 1.0)
        length_value = max(float(length_value), 1.0)
        points = [clamp_point(point, self.nx, self.ny) for point in axis.points]
        arc_lengths = polyline_arc_lengths(points)
        total_length = 0.0 if len(arc_lengths) == 0 else float(arc_lengths[-1])
        if total_length < 1.0:
            self._set_status(f"{axis.name} is too short to generate cuts.")
            return

        positions = feature_axis_sample_positions(total_length, spacing)
        if not positions:
            self._set_status(f"{axis.name} produced no valid sample positions.")
            return
        if len(positions) > MAX_FEATURE_AXIS_CUTS:
            self._set_status(
                f"{axis.name} would generate {len(positions)} cuts. Increase spacing and try again."
            )
            return

        template_panel = self.active_panel
        generated_cut_ids: list[int] = []
        clipped_count = 0
        tangent_delta = max(min(spacing * 0.35, max(total_length * 0.15, 1.0)), 1.0)
        for index, position in enumerate(positions, start=1):
            center = polyline_point_at_length(points, arc_lengths, position)
            tx, ty = polyline_tangent_at_length(
                points,
                arc_lengths,
                position,
                delta=tangent_delta,
            )
            normal_angle = float(math.degrees(math.atan2(-ty, tx)))
            p0, p1, actual_length = segment_from_angle_length(
                normal_angle + float(angle_offset),
                length_value,
                "center",
                None,
                center,
                self.nx,
                self.ny,
            )
            if actual_length < 1.0:
                continue
            cut = self._create_cut(p0, p1)
            cut.name = f"{axis.name} {index:02d}"
            self._seed_cut_td_params_from_panel(cut.cut_id, template_panel)
            generated_cut_ids.append(cut.cut_id)
            if actual_length + 1e-6 < length_value:
                clipped_count += 1

        if not generated_cut_ids:
            self._set_status(f"{axis.name} did not yield any valid cuts.")
            return

        template_panel.cut_id = generated_cut_ids[0]
        self.selected_cut_id = generated_cut_ids[0]
        self._invalidate_panel_cache(template_panel.panel_id)

        stack_name = ""
        if bool(self.feature_create_stack_var.get()):
            stack_id = int(self.next_stack_id)
            self.next_stack_id += 1
            stack_name = f"{axis.name} Stack"
            self.stacks[stack_id] = self._make_default_stack_state(
                stack_id,
                stack_name,
                generated_cut_ids,
            )
            self.active_stack_id = stack_id
            self.selected_stack_cut_id = generated_cut_ids[0]

        self._record_session_change()
        self.refresh_all()
        message = (
            f"Generated {len(generated_cut_ids)} cuts from {axis.name} "
            f"(spacing={spacing:.1f}, length={length_value:.1f}, angle={angle_offset:.1f} deg)."
        )
        if stack_name:
            message += f" Created {stack_name}."
        if clipped_count:
            message += f" {clipped_count} cut(s) were clipped by image bounds."
        self._set_status(message)

    def _start_draw_cut(self) -> None:
        self._cancel_feature_axis_draw()
        self.draw_mode = True
        self.force_new_cut = False
        self.pending_point = None
        self.hover_point = None
        self.drag_state = None
        self._set_status(
            f"Draw mode for {self.active_panel.name}: click two points on the map."
        )
        self.refresh_map()

    def _start_add_cut(self) -> None:
        self._cancel_feature_axis_draw()
        self.draw_mode = True
        self.force_new_cut = True
        self.pending_point = None
        self.hover_point = None
        self.drag_state = None
        self._set_status(
            f"Add Cut mode for {self.active_panel.name}: the next two clicks create a new cut."
        )
        self.refresh_map()

    def _begin_pending_cut(self, x: float, y: float) -> None:
        self.pending_point = (x, y)
        self.hover_point = (x, y)
        self.drag_state = None
        self._set_status(
            f"First point set at ({x:.1f}, {y:.1f}) for {self.active_panel.name}. "
            "Click second point to finish. Esc or right-click cancels."
        )
        self.refresh_map()

    def _finish_pending_cut(self, x: float, y: float) -> None:
        if self.pending_point is None:
            return

        p0 = self.pending_point
        p1 = (x, y)
        if distance(p0, p1) < 1.0:
            self._set_status("Cut too short; choose two distinct points.")
            return

        panel = self.active_panel
        if panel.cut_id is not None and panel.cut_id in self.cuts:
            if self.force_new_cut:
                cut = self._create_cut(p0, p1)
                panel.cut_id = cut.cut_id
                self._seed_cut_td_params_from_panel(cut.cut_id, panel)
            else:
                cut = self.cuts[panel.cut_id]
                if self._cut_dynamic_enabled(cut.cut_id):
                    self._set_cut_dynamic_keyframe(
                        cut.cut_id,
                        int(self.t_visual_var.get()),
                        p0,
                        p1,
                        enable_dynamic=True,
                    )
                else:
                    cut.p0 = p0
                    cut.p1 = p1
        else:
            cut = self._create_cut(p0, p1)
            panel.cut_id = cut.cut_id
            self._seed_cut_td_params_from_panel(cut.cut_id, panel)

        self.selected_cut_id = cut.cut_id
        self.pending_point = None
        self.hover_point = None
        self.draw_mode = False
        self.force_new_cut = False
        self._invalidate_cut_dependents(cut.cut_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"{cut.name} assigned to {panel.name}.")

    def _cancel_pending_cut(self) -> None:
        if self.feature_draw_mode is not None or self.feature_pending_points:
            self._cancel_feature_axis_draw()
            return
        if self.pending_point is None and not self.draw_mode:
            return
        self.pending_point = None
        self.hover_point = None
        self.draw_mode = False
        self.force_new_cut = False
        self.drag_state = None
        self._set_status("Cut creation cancelled.")
        self.refresh_map()

    def _assign_selected_cut(self) -> None:
        if self.selected_cut_id is None or self.selected_cut_id not in self.cuts:
            self._set_status("No cut selected to assign.")
            return
        panel = self.active_panel
        panel.cut_id = self.selected_cut_id
        self._seed_cut_td_params_from_panel(self.selected_cut_id, panel)
        self._invalidate_panel_cache(panel.panel_id)
        self._sync_controls_from_active_panel()
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Assigned Cut {self.selected_cut_id} to {panel.name}.")

    def _copy_selected_cut(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            self._set_status("No cut selected to copy.")
            return
        preview_cut = self._cut_preview(cut.cut_id) or cut
        self.clipboard_cut = {
            "p0": preview_cut.p0,
            "p1": preview_cut.p1,
            "color": cut.color,
        }
        self._set_status(f"Copied {cut.name}.")

    def _paste_cut(self) -> None:
        if self.clipboard_cut is None:
            self._set_status("Clipboard is empty.")
            return

        p0 = self.clipboard_cut["p0"]
        p1 = self.clipboard_cut["p1"]
        new_cut = self._create_cut(
            clamp_point((p0[0] + 3.0, p0[1] + 3.0), self.nx, self.ny),
            clamp_point((p1[0] + 3.0, p1[1] + 3.0), self.nx, self.ny),
        )
        panel = self.active_panel
        panel.cut_id = new_cut.cut_id
        self._seed_cut_td_params_from_panel(new_cut.cut_id, panel)
        self.selected_cut_id = new_cut.cut_id
        self._invalidate_panel_cache(panel.panel_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Pasted {new_cut.name} into {panel.name}.")

    def _rotate_selected_cut(self, angle_deg: float) -> None:
        self._adjust_selected_cut_angle(angle_deg)

    def _delete_selected_cut(self) -> None:
        cut = self._selected_cut()
        if cut is None:
            return
        cut_id = cut.cut_id
        for panel in self.panels:
            if panel.cut_id == cut_id:
                panel.cut_id = None
                self._invalidate_panel_cache(panel.panel_id)
        self.cut_analysis_state.pop(cut_id, None)
        for stack in self.stacks.values():
            stack["cut_ids"] = [
                int(member_cut_id)
                for member_cut_id in (stack.get("cut_ids") or [])
                if int(member_cut_id) != int(cut_id)
            ]
        del self.cuts[cut_id]
        if self.selected_cut_id == cut_id:
            self.selected_cut_id = None
        if self.selected_stack_cut_id == cut_id:
            self.selected_stack_cut_id = None
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Deleted Cut {cut_id}.")

    def _selected_cut(self) -> Cut | None:
        if self.selected_cut_id is None:
            return None
        return self.cuts.get(self.selected_cut_id)

    def _panel_for_axis(self, axis: Any) -> int | None:
        return self.axis_to_panel_id.get(axis)

    def _find_cut_hit(
        self, x: float, y: float
    ) -> tuple[str, int] | None:
        ordered = []
        if self.selected_cut_id is not None and self.selected_cut_id in self.cuts:
            ordered.append(self.cuts[self.selected_cut_id])
        for cut in self.cuts.values():
            if self.selected_cut_id == cut.cut_id:
                continue
            ordered.append(cut)

        best: tuple[float, str, int] | None = None
        for cut in ordered:
            if not cut.visible:
                continue
            preview_cut = self._cut_preview(cut.cut_id) or cut
            d0 = distance((x, y), preview_cut.p0)
            d1 = distance((x, y), preview_cut.p1)
            ds = distance_point_to_segment((x, y), preview_cut.p0, preview_cut.p1)

            candidates = [
                (d0, "p0"),
                (d1, "p1"),
                (ds, "line"),
            ]
            for dist_value, part in candidates:
                threshold = 5.0 if part in {"p0", "p1"} else 3.0
                if dist_value <= threshold:
                    if best is None or dist_value < best[0]:
                        best = (dist_value, part, cut.cut_id)

        if best is None:
            return None
        return best[1], best[2]

    def _on_canvas_press(self, event: Any) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.map_ax:
            self._handle_map_press(
                float(event.xdata),
                float(event.ydata),
                int(getattr(event, "button", 1) or 1),
            )
            return

        panel_id = self._panel_for_axis(event.inaxes)
        if panel_id is not None:
            self.active_panel_id = panel_id
            self.selected_cut_id = self.panels[panel_id - 1].cut_id
            self._sync_controls_from_active_panel()
            self.refresh_all()
            if getattr(event, "dblclick", False):
                self._open_td_window(panel_id)

    def _handle_map_press(self, x: float, y: float, button: int = 1) -> None:
        x, y = self._map_display_to_data(x, y)
        x = clamp_value(x, 0.0, self.nx - 1.0)
        y = clamp_value(y, 0.0, self.ny - 1.0)

        if self.feature_draw_mode is not None:
            if button == 3:
                if self.feature_draw_mode == "curve" and len(self.feature_pending_points) >= 2:
                    self._finish_pending_feature_axis()
                else:
                    self._cancel_feature_axis_draw()
                return
            if button == 1:
                self._append_feature_axis_point(x, y)
            return

        if button == 3:
            self._cancel_pending_cut()
            return

        if self.pending_point is not None:
            self._finish_pending_cut(x, y)
            return

        hit = None if self.draw_mode else self._find_cut_hit(x, y)
        if hit is None:
            self._begin_pending_cut(x, y)
            return

        part, cut_id = hit
        cut = self.cuts[cut_id]
        if cut.locked:
            self.selected_cut_id = cut_id
            self.refresh_all()
            return

        self.selected_cut_id = cut_id
        self.drag_state = {
            "part": part,
            "cut_id": cut_id,
            "last": (x, y),
            "dirty": False,
        }
        self.refresh_all()

    def _on_canvas_motion(self, event: Any) -> None:
        if event.inaxes != self.map_ax or event.xdata is None or event.ydata is None:
            return

        x, y = self._map_display_to_data(float(event.xdata), float(event.ydata))
        x = clamp_value(x, 0.0, self.nx - 1.0)
        y = clamp_value(y, 0.0, self.ny - 1.0)
        if self.feature_draw_mode is not None:
            self.feature_hover_point = (x, y)
            self.refresh_map()
            return
        if self.pending_point is not None:
            self.hover_point = (x, y)
            self.refresh_map()
            return

        if self.drag_state is None:
            return

        cut = self.cuts.get(self.drag_state["cut_id"])
        if cut is None:
            self.drag_state = None
            return

        part = self.drag_state["part"]
        preview_cut = self._cut_preview(cut.cut_id) or cut
        if part == "p0":
            new_p0, new_p1 = (x, y), preview_cut.p1
        elif part == "p1":
            new_p0, new_p1 = preview_cut.p0, (x, y)
        else:
            last_x, last_y = self.drag_state["last"]
            dx = x - last_x
            dy = y - last_y
            new_p0, new_p1 = shift_cut(preview_cut, dx, dy, self.nx, self.ny)
            self.drag_state["last"] = (x, y)

        if distance(new_p0, new_p1) < 1.0:
            return
        if self._cut_dynamic_enabled(cut.cut_id):
            self._set_cut_dynamic_keyframe(
                cut.cut_id,
                int(self.t_visual_var.get()),
                new_p0,
                new_p1,
                enable_dynamic=True,
            )
        else:
            cut.p0 = new_p0
            cut.p1 = new_p1

        self.drag_state["dirty"] = True
        self._invalidate_cut_dependents(cut.cut_id)
        self.refresh_all()

    def _on_canvas_release(self, _event: Any) -> None:
        if self.drag_state is not None and self.drag_state.get("dirty"):
            cut_id = int(self.drag_state.get("cut_id", -1))
            if cut_id in self.cuts:
                self._record_session_change()
                self._set_status(
                    f"Updated geometry of {self.cuts[cut_id].name} at t={int(self.t_visual_var.get())}."
                )
        self.drag_state = None

    def _draw_map(self) -> None:
        frame = self.cube[int(self.t_visual_var.get())]
        if self.map_swap_xy_var.get():
            frame = frame.T
        vmin, vmax = frame_limits(frame)

        ax = self.map_ax
        ax.clear()
        ax.imshow(
            frame,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        for axis in self.feature_axes.values():
            if not axis.visible or len(axis.points) < 2:
                continue
            is_selected = axis.axis_id == self.selected_feature_axis_id
            display_points = [
                self._map_data_to_display(point[0], point[1]) for point in axis.points
            ]
            xs = [point[0] for point in display_points]
            ys = [point[1] for point in display_points]
            ax.plot(
                xs,
                ys,
                color=axis.color,
                linewidth=2.8 if is_selected else 1.6,
                alpha=0.95 if is_selected else 0.75,
                linestyle="-." if axis.mode == "curve" else ":",
            )
            ax.scatter(
                xs,
                ys,
                color=axis.color,
                s=32 if is_selected else 18,
                zorder=3,
                alpha=0.95 if is_selected else 0.8,
            )
            mid_index = len(display_points) // 2
            mid_point = display_points[mid_index]
            ax.text(
                mid_point[0],
                mid_point[1],
                axis.name,
                color=axis.color,
                fontsize=8,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
            )

        for cut in self.cuts.values():
            if not cut.visible:
                continue
            preview_cut = self._cut_preview(cut.cut_id) or cut
            is_selected = cut.cut_id == self.selected_cut_id
            lw = 3.0 if is_selected else 1.8
            alpha = 1.0 if is_selected else 0.85
            p0_disp = self._map_data_to_display(preview_cut.p0[0], preview_cut.p0[1])
            p1_disp = self._map_data_to_display(preview_cut.p1[0], preview_cut.p1[1])
            ax.plot(
                [p0_disp[0], p1_disp[0]],
                [p0_disp[1], p1_disp[1]],
                color=cut.color,
                linewidth=lw,
                alpha=alpha,
                linestyle="--" if self._cut_dynamic_enabled(cut.cut_id) else "-",
            )
            ax.scatter(
                [p0_disp[0], p1_disp[0]],
                [p0_disp[1], p1_disp[1]],
                color=cut.color,
                s=36 if is_selected else 24,
                zorder=3,
            )
            mid_x = 0.5 * (p0_disp[0] + p1_disp[0])
            mid_y = 0.5 * (p0_disp[1] + p1_disp[1])
            ax.text(
                mid_x,
                mid_y,
                cut.name + (" [dyn]" if self._cut_dynamic_enabled(cut.cut_id) else ""),
                color=cut.color,
                fontsize=8,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
            )

        if self.pending_point is not None:
            pending_disp = self._map_data_to_display(
                self.pending_point[0], self.pending_point[1]
            )
            ax.scatter(
                [pending_disp[0]],
                [pending_disp[1]],
                color="yellow",
                s=48,
                marker="x",
                zorder=4,
            )
            if self.hover_point is not None:
                hover_disp = self._map_data_to_display(
                    self.hover_point[0], self.hover_point[1]
                )
                ax.plot(
                    [pending_disp[0], hover_disp[0]],
                    [pending_disp[1], hover_disp[1]],
                    color="yellow",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                )

        if self.feature_pending_points:
            pending_display_points = [
                self._map_data_to_display(point[0], point[1])
                for point in self.feature_pending_points
            ]
            pending_xs = [point[0] for point in pending_display_points]
            pending_ys = [point[1] for point in pending_display_points]
            ax.plot(
                pending_xs,
                pending_ys,
                color="deepskyblue",
                linewidth=2.0,
                linestyle="--",
                alpha=0.95,
            )
            ax.scatter(
                pending_xs,
                pending_ys,
                color="deepskyblue",
                s=40,
                marker="o",
                zorder=4,
            )
            if self.feature_hover_point is not None:
                hover_disp = self._map_data_to_display(
                    self.feature_hover_point[0], self.feature_hover_point[1]
                )
                last_disp = pending_display_points[-1]
                ax.plot(
                    [last_disp[0], hover_disp[0]],
                    [last_disp[1], hover_disp[1]],
                    color="deepskyblue",
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.9,
                )

        disp_nx, disp_ny = self._map_display_shape()
        if self.map_flip_x_var.get():
            ax.set_xlim(disp_nx - 0.5, -0.5)
        else:
            ax.set_xlim(-0.5, disp_nx - 0.5)

        if self.map_flip_y_var.get():
            ax.set_ylim(disp_ny - 0.5, -0.5)
        else:
            ax.set_ylim(-0.5, disp_ny - 0.5)

        if self.map_swap_xy_var.get():
            ax.set_xlabel("y [pixel]")
            ax.set_ylabel("x [pixel]")
        else:
            ax.set_xlabel("x [pixel]")
            ax.set_ylabel("y [pixel]")
        ax.set_title(f"Map at t={int(self.t_visual_var.get())}")

    def _panel_td(self, panel: TDPanel) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        if panel.cut_id is None or panel.cut_id not in self.cuts:
            return None, None

        cut = self.cuts[panel.cut_id]
        params = self._panel_td_params(panel)
        state = self._cut_analysis(cut.cut_id)
        key = cut_cache_key(cut, params) + self._cut_geometry_signature(cut.cut_id)
        if (
            state.get("td_cache_key") == key
            and state.get("td_cache_td") is not None
            and state.get("td_cache_meta") is not None
        ):
            return state["td_cache_td"], state["td_cache_meta"]

        try:
            td, meta = compute_td(
                self.cube,
                cut,
                int(params["t_ini"]),
                int(params["t_fin"]),
                int(params["stride"]),
                int(params["width"]),
                str(params["weighting"]),
                dynamic_geometry=self._dynamic_cut_geometry_samples(
                    cut.cut_id,
                    int(params["t_ini"]),
                    int(params["t_fin"]),
                    int(params["stride"]),
                ),
            )
        except ValueError as exc:
            state["td_cache_key"] = key
            state["td_cache_td"] = None
            state["td_cache_meta"] = {"error": str(exc)}
            panel.cache_key = key
            panel.cache_td = None
            panel.cache_meta = dict(state["td_cache_meta"])
            return None, state["td_cache_meta"]

        state["td_cache_key"] = key
        state["td_cache_td"] = td
        state["td_cache_meta"] = meta
        panel.cache_key = key
        panel.cache_td = td
        panel.cache_meta = meta
        return td, meta

    def _draw_td_axis(
        self,
        ax: Any,
        panel_id: int,
        use_zoom: bool = True,
        title_fontsize: float = 9.0,
        display_mode: str = "raw",
    ) -> None:
        display_mode = "raw"
        current_t = int(self.t_visual_var.get())
        td_aspect = "equal" if self.td_aspect_var.get() == "equal" else "auto"
        td_zoom = float(str(self.td_zoom_var.get()).rstrip("x") or "1")
        panel = self.panels[panel_id - 1]
        cut = self.cuts.get(panel.cut_id) if panel.cut_id is not None else None

        ax.clear()
        td, meta = self._panel_td(panel)
        if cut is None:
            ax.text(
                0.5,
                0.5,
                f"{panel.name}\nno cut assigned",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(panel.name)
        elif td is None or meta is None:
            error_text = "invalid TD"
            if meta is not None and "error" in meta:
                error_text = meta["error"]
            ax.text(
                0.5,
                0.5,
                f"{panel.name}\n{error_text}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(panel_title(panel, cut))
        else:
            distances = meta["distances"]
            t_indices = meta["t_indices"]
            td_plot = td
            display_note = ""
            visual_fit = display_mode in {"visual spline", "visual spline fit"}
            if visual_fit:
                td_plot = td_visual_spline(td)
                display_note = " | visual spline fit"
            if len(t_indices) == 1:
                t_min = float(t_indices[0])
                t_max = float(t_indices[0]) + 1.0
            else:
                t_min = float(t_indices[0])
                t_max = float(t_indices[-1])

            vmin, vmax = frame_limits(td)
            dist_max = float(distances[-1])
            time_center = (
                float(current_t)
                if panel.t_ini <= current_t <= panel.t_fin
                else 0.5 * (t_min + t_max)
            )
            dist_center = 0.5 * dist_max

            if self.td_swap_axes_var.get():
                plot_x_size = td_plot.shape[0]
                plot_y_size = td_plot.shape[1]
                if visual_fit:
                    image_extent = [0.0, plot_x_size - 1.0, 0.0, plot_y_size - 1.0]
                    line_pos = map_value_to_display(current_t, t_min, t_max, plot_x_size)
                    x_center = map_value_to_display(time_center, t_min, t_max, plot_x_size)
                    y_center = map_value_to_display(dist_center, 0.0, dist_max, plot_y_size)
                else:
                    image_extent = [t_min, t_max, 0.0, dist_max]
                    line_pos = current_t
                    x_center = time_center
                    y_center = dist_center
                ax.imshow(
                    td_plot.T,
                    origin="lower",
                    aspect=td_aspect,
                    cmap="gray",
                    extent=image_extent,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                if panel.t_ini <= current_t <= panel.t_fin:
                    ax.axvline(
                        line_pos,
                        color=cut.color,
                        linestyle="--",
                        linewidth=1.0,
                    )

                if use_zoom:
                    if visual_fit:
                        x0, x1 = self._zoom_limits(0.0, plot_x_size - 1.0, x_center, td_zoom)
                        y0, y1 = self._zoom_limits(0.0, plot_y_size - 1.0, y_center, td_zoom)
                    else:
                        x0, x1 = self._zoom_limits(t_min, t_max, time_center, td_zoom)
                        y0, y1 = self._zoom_limits(0.0, dist_max, dist_center, td_zoom)
                else:
                    if visual_fit:
                        x0, x1 = 0.0, plot_x_size - 1.0
                        y0, y1 = 0.0, plot_y_size - 1.0
                    else:
                        x0, x1 = t_min, t_max
                        y0, y1 = 0.0, dist_max

                if self.td_flip_x_var.get():
                    ax.set_xlim(x1, x0)
                else:
                    ax.set_xlim(x0, x1)
                if self.td_flip_y_var.get():
                    ax.set_ylim(y1, y0)
                else:
                    ax.set_ylim(y0, y1)
                ax.set_xlabel("time index")
                ax.set_ylabel("distance [pixel]")
                if visual_fit:
                    xticks, xticklabels = display_ticks(t_min, t_max, plot_x_size)
                    yticks, yticklabels = display_ticks(0.0, dist_max, plot_y_size)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)
            else:
                plot_x_size = td_plot.shape[1]
                plot_y_size = td_plot.shape[0]
                if visual_fit:
                    image_extent = [0.0, plot_x_size - 1.0, 0.0, plot_y_size - 1.0]
                    line_pos = map_value_to_display(current_t, t_min, t_max, plot_y_size)
                    x_center = map_value_to_display(dist_center, 0.0, dist_max, plot_x_size)
                    y_center = map_value_to_display(time_center, t_min, t_max, plot_y_size)
                else:
                    image_extent = [0.0, dist_max, t_min, t_max]
                    line_pos = current_t
                    x_center = dist_center
                    y_center = time_center
                ax.imshow(
                    td_plot,
                    origin="lower",
                    aspect=td_aspect,
                    cmap="gray",
                    extent=image_extent,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                if panel.t_ini <= current_t <= panel.t_fin:
                    ax.axhline(
                        line_pos,
                        color=cut.color,
                        linestyle="--",
                        linewidth=1.0,
                    )

                if use_zoom:
                    if visual_fit:
                        x0, x1 = self._zoom_limits(0.0, plot_x_size - 1.0, x_center, td_zoom)
                        y0, y1 = self._zoom_limits(0.0, plot_y_size - 1.0, y_center, td_zoom)
                    else:
                        x0, x1 = self._zoom_limits(0.0, dist_max, dist_center, td_zoom)
                        y0, y1 = self._zoom_limits(t_min, t_max, time_center, td_zoom)
                else:
                    if visual_fit:
                        x0, x1 = 0.0, plot_x_size - 1.0
                        y0, y1 = 0.0, plot_y_size - 1.0
                    else:
                        x0, x1 = 0.0, dist_max
                        y0, y1 = t_min, t_max

                if self.td_flip_x_var.get():
                    ax.set_xlim(x1, x0)
                else:
                    ax.set_xlim(x0, x1)
                if self.td_flip_y_var.get():
                    ax.set_ylim(y1, y0)
                else:
                    ax.set_ylim(y0, y1)
                ax.set_xlabel("distance [pixel]")
                ax.set_ylabel("time index")
                if visual_fit:
                    xticks, xticklabels = display_ticks(0.0, dist_max, plot_x_size)
                    yticks, yticklabels = display_ticks(t_min, t_max, plot_y_size)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)

            ax.set_title(panel_title(panel, cut) + display_note, fontsize=title_fontsize)
            if display_note:
                ax.text(
                    0.99,
                    0.01,
                    "visual only",
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=max(title_fontsize - 2.0, 8.0),
                    color="tab:orange",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
                )

        for spine in ax.spines.values():
            spine.set_linewidth(2.5 if panel_id == self.active_panel_id else 1.0)
            if panel_id == self.active_panel_id:
                spine.set_edgecolor("crimson")
            elif cut is not None:
                spine.set_edgecolor(cut.color)
            else:
                spine.set_edgecolor("0.6")

    def _draw_td_panels(self) -> None:
        for panel_id, ax in self.panel_axes.items():
            self._draw_td_axis(ax, panel_id, use_zoom=True)

    def _td_window_panel(self, panel_id: int) -> TDPanel | None:
        if 1 <= panel_id <= len(self.panels):
            return self.panels[panel_id - 1]
        return None

    def _td_window_cut(self, panel_id: int, action: str) -> Cut | None:
        panel = self._td_window_panel(panel_id)
        if panel is None:
            self._set_status("Invalid TD window panel.")
            return None
        cut = self.cuts.get(panel.cut_id) if panel.cut_id is not None else None
        if cut is None:
            self._set_status(f"{panel.name} has no cut assigned.")
            return None
        return self._editable_cut(cut, action)

    def _td_window_tracking_key(self, panel_id: int) -> tuple[Any, ...] | None:
        panel = self._td_window_panel(panel_id)
        if panel is None or panel.cut_id is None or panel.cut_id not in self.cuts:
            return None
        return cut_cache_key(self.cuts[panel.cut_id], panel) + self._cut_geometry_signature(panel.cut_id)

    def _clear_td_window_wavelet_filter(
        self, panel_id: int, stale: bool = False, refresh: bool = False
    ) -> None:
        self._clear_panel_analysis_results(
            panel_id, clear_tracking=False, clear_wavelet=True
        )
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        existing["wavelet_filter_result"] = None
        existing["wavelet_events"] = []
        existing["wavelet_next_event_id"] = 1
        existing["wavelet_selected_event_id"] = None
        summary_var = existing.get("wavelet_summary_var")
        physics_var = existing.get("wavelet_physics_var")
        events_summary_var = existing.get("wavelet_events_summary_var")
        selected_var = existing.get("wavelet_selected_var")
        diag_var = existing.get("wavelet_diag_var")
        if summary_var is not None:
            if stale:
                summary_var.set("Wavelet filter cleared because tracking changed.")
            else:
                summary_var.set("No wavelet filter results.")
        if physics_var is not None:
            physics_var.set("")
        if events_summary_var is not None:
            events_summary_var.set("")
        if selected_var is not None:
            selected_var.set("")
        if diag_var is not None:
            diag_var.set("")
        tree = existing.get("wavelet_events_tree")
        if tree is not None:
            existing["wavelet_table_updating"] = True
            try:
                children = tree.get_children()
                if children:
                    tree.delete(*children)
            finally:
                existing["wavelet_table_updating"] = False
        self._refresh_td_window_wavelet_diagnostics(panel_id)
        if not stale:
            self._record_session_change()
        if refresh:
            self._refresh_td_window(panel_id)

    def _clear_td_window_crest_tracking(
        self, panel_id: int, stale: bool = False, refresh: bool = False
    ) -> None:
        self._clear_panel_analysis_results(
            panel_id, clear_tracking=True, clear_wavelet=True
        )
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        existing["crest_tracking_result"] = None
        existing["crest_tracking_td_key"] = None
        summary_var = existing.get("crest_summary_var")
        if summary_var is None:
            return
        if stale:
            summary_var.set("Crest tracking cleared because the TD map changed.")
        else:
            summary_var.set("No crest tracking results.")
        self._clear_td_window_wavelet_filter(panel_id, stale=stale, refresh=False)
        if not stale:
            self._record_session_change()
        if refresh:
            self._refresh_td_window(panel_id)

    def _sync_td_window_crest_tracking(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        tracking_result = existing.get("crest_tracking_result")
        if tracking_result is None:
            return

        current_key = self._td_window_tracking_key(panel_id)
        if existing.get("crest_tracking_td_key") != current_key:
            self._clear_td_window_crest_tracking(panel_id, stale=True)

    def _refresh_td_window_crest_summary(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        tracking_result = existing.get("crest_tracking_result")
        summary_var = existing.get("crest_summary_var")
        if summary_var is None:
            return
        if not tracking_result:
            summary_var.set("No crest tracking results.")
            return
        threads = tracking_result.get("threads") or []
        located = tracking_result.get("located") or {}
        located_count = int(np.count_nonzero(np.asarray(located.get("errs", [])) > 0))
        thread_count = len(threads)
        longest = max((int(th.get("length", 0)) for th in threads), default=0)
        summary_var.set(
            f"Located {located_count} crest bins. Threads: {thread_count}. Longest: {longest}."
        )

    def _parse_td_window_crest_tracking_params(
        self, panel_id: int
    ) -> dict[str, Any] | None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return None

        try:
            cad = float(str(existing["crest_cad_var"].get()).strip())
            res = float(str(existing["crest_res_var"].get()).strip())
            grad = float(str(existing["crest_grad_var"].get()).strip())
            min_tlen = int(float(str(existing["crest_min_tlen_var"].get()).strip()))
            max_dist_jump = int(
                float(str(existing["crest_max_dist_jump_var"].get()).strip())
            )
            max_time_skip = int(
                float(str(existing["crest_max_time_skip_var"].get()).strip())
            )
        except Exception:
            self._set_status(f"Invalid crest-tracking parameter for {panel.name}.")
            return None

        if cad <= 0.0 or res <= 0.0 or grad < 0.0:
            self._set_status(f"Crest-tracking cadence, resolution and gradient must be valid for {panel.name}.")
            return None
        if min_tlen < 1 or max_dist_jump < 0 or max_time_skip < 1:
            self._set_status(f"Crest-tracking integer limits are invalid for {panel.name}.")
            return None

        params = {
            "cad": cad,
            "res": res,
            "grad": grad,
            "min_tlen": min_tlen,
            "max_dist_jump": max_dist_jump,
            "max_time_skip": max_time_skip,
            "invert": bool(existing["crest_invert_var"].get()),
            "gauss": bool(existing["crest_gauss_var"].get()),
        }
        self._panel_analysis(panel_id)["crest_params"] = dict(params)
        self._update_td_window_preset_from_values(panel_id)
        return params

    def _run_td_window_crest_tracking(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return

        api, import_error = load_local_nuwt_api()
        if api is None:
            existing["crest_summary_var"].set(f"NUWT unavailable: {import_error}")
            self._set_status(f"NUWT import failed for {panel.name}.")
            return

        params = self._parse_td_window_crest_tracking_params(panel_id)
        if params is None:
            return

        td, meta = self._panel_td(panel)
        if td is None or meta is None:
            existing["crest_summary_var"].set("No valid TD map available for crest tracking.")
            self._set_status(f"{panel.name} has no valid TD map for crest tracking.")
            return

        td_nuwt = np.asarray(td, dtype=np.float64).T
        finite = np.isfinite(td_nuwt)
        if not np.any(finite):
            existing["crest_summary_var"].set("TD map contains no finite values.")
            self._set_status(f"{panel.name} has no finite TD values for crest tracking.")
            return
        fill_value = float(np.nanmin(td_nuwt[finite]))
        if not np.all(finite):
            td_nuwt = np.where(finite, td_nuwt, fill_value)

        tracking_mode = "gauss fit" if params["gauss"] else "nearest pixel"
        self._set_status(
            f"Running crest tracking for {panel.name} ({tracking_mode})..."
        )
        self.root.update_idletasks()

        try:
            located = api["locate_things"](
                td_nuwt,
                invert=params["invert"],
                grad=params["grad"],
                res=params["res"],
                cad=params["cad"],
                nearest_pixel=not params["gauss"],
            )
            threads, _ = api["follow_threads"](
                located,
                min_tlen=params["min_tlen"],
                max_dist_jump=params["max_dist_jump"],
                max_time_skip=params["max_time_skip"],
            )
            threads = api["patch_up_threads"](
                threads, fit_flag=0, simp_fill=False, debug=False
            )
        except Exception as exc:
            existing["crest_tracking_result"] = None
            existing["crest_tracking_td_key"] = None
            existing["crest_summary_var"].set(
                f"Crest tracking failed: {type(exc).__name__}: {exc}"
            )
            self._set_status(f"Crest tracking failed for {panel.name}.")
            self._refresh_td_window(panel_id)
            return

        located_count = int(np.count_nonzero(np.asarray(located["errs"]) > 0))
        thread_count = len(threads)
        longest = max((int(th.get("length", 0)) for th in threads), default=0)

        existing["crest_tracking_result"] = {
            "located": located,
            "threads": threads,
            "params": params,
        }
        existing["crest_tracking_td_key"] = self._td_window_tracking_key(panel_id)
        state = self._panel_analysis(panel_id)
        state["crest_tracking_result"] = self._clone_wavelet_payload(
            existing["crest_tracking_result"]
        )
        state["crest_tracking_td_key"] = self._clone_wavelet_payload(
            existing["crest_tracking_td_key"]
        )
        self._clear_td_window_wavelet_filter(panel_id, stale=True, refresh=False)
        existing["crest_summary_var"].set(
            f"Located {located_count} crest bins. Threads: {thread_count}. Longest: {longest}."
        )
        self._record_session_change()
        self._refresh_td_window(panel_id)
        self._set_status(
            f"Crest tracking completed for {panel.name} ({tracking_mode}): "
            f"{thread_count} thread(s)."
        )

    def _parse_td_window_wavelet_filter_params(
        self, panel_id: int
    ) -> dict[str, Any] | None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return None

        try:
            cad = float(str(existing["crest_cad_var"].get()).strip())
            res = float(str(existing["crest_res_var"].get()).strip())
            p_min = float(str(existing["wavelet_p_min_var"].get()).strip())
            p_max = float(str(existing["wavelet_p_max_var"].get()).strip())
            power_ratio_thresh = float(
                str(existing["wavelet_power_ratio_var"].get()).strip()
            )
            segment_power_frac = float(
                str(existing["wavelet_segment_frac_var"].get()).strip()
            )
            min_points_segment = int(
                float(str(existing["wavelet_min_points_seg_var"].get()).strip())
            )
            min_amp_arcsec = float(
                str(existing["wavelet_min_amp_var"].get()).strip()
            )
            max_jump_pix = float(str(existing["wavelet_max_jump_var"].get()).strip())
            min_points_cut_seg = int(
                float(str(existing["wavelet_min_points_cut_var"].get()).strip())
            )
            rms_amp_ratio_max = float(
                str(existing["wavelet_rms_amp_ratio_var"].get()).strip()
            )
            km_per_arcsec = float(
                str(existing["wavelet_km_per_arcsec_var"].get()).strip()
            )
            density_text = str(existing["wavelet_density_var"].get()).strip()
            phase_speed_text = str(existing["wavelet_phase_speed_var"].get()).strip()
            density_kg_m3 = float(density_text) if density_text else float("nan")
            phase_speed_km_s = (
                float(phase_speed_text) if phase_speed_text else float("nan")
            )
        except Exception:
            self._set_status(f"Invalid wavelet-filter parameter for {panel.name}.")
            return None

        if cad <= 0.0 or res <= 0.0:
            self._set_status(f"Wavelet cadence and pixel scale must be valid for {panel.name}.")
            return None
        if p_min <= 0.0 or p_max <= p_min:
            self._set_status(f"Wavelet period range is invalid for {panel.name}.")
            return None
        if power_ratio_thresh < 0.0 or not (0.0 < segment_power_frac <= 1.0):
            self._set_status(f"Wavelet power thresholds are invalid for {panel.name}.")
            return None
        if min_points_segment < 3 or min_points_cut_seg < 3 or max_jump_pix < 0.0:
            self._set_status(f"Wavelet point/jump limits are invalid for {panel.name}.")
            return None
        if min_amp_arcsec < 0.0 or rms_amp_ratio_max <= 0.0:
            self._set_status(f"Wavelet amplitude criteria are invalid for {panel.name}.")
            return None
        if km_per_arcsec <= 0.0:
            self._set_status(f"Wavelet km/arcsec conversion is invalid for {panel.name}.")
            return None
        if np.isfinite(density_kg_m3) and density_kg_m3 <= 0.0:
            self._set_status(f"Wavelet density must be positive for {panel.name}.")
            return None
        if np.isfinite(phase_speed_km_s) and phase_speed_km_s <= 0.0:
            self._set_status(f"Wavelet phase speed must be positive for {panel.name}.")
            return None

        params = {
            "cad": cad,
            "res": res,
            "p_min": p_min,
            "p_max": p_max,
            "power_ratio_thresh": power_ratio_thresh,
            "segment_power_frac": segment_power_frac,
            "min_points_segment": min_points_segment,
            "min_amp_arcsec": min_amp_arcsec,
            "max_jump_pix": max_jump_pix,
            "min_points_cut_seg": min_points_cut_seg,
            "rms_amp_ratio_max": rms_amp_ratio_max,
            "km_per_arcsec": km_per_arcsec,
            "density_kg_m3": density_kg_m3,
            "phase_speed_km_s": phase_speed_km_s,
        }
        self._panel_analysis(panel_id)["wavelet_params"] = {
            key: value for key, value in params.items() if key not in {"cad", "res"}
        }
        self._update_td_window_preset_from_values(panel_id)
        return params

    def _set_td_window_selected_event_thread_filter(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if existing is None or panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        thread_index = self._wavelet_event_thread_index(selected_event)
        if thread_index is None:
            self._set_status(f"The selected event in {panel.name} has no valid thread index.")
            return
        filter_text = self._format_wavelet_thread_filter_text([thread_index])
        existing["wavelet_thread_filter_var"].set(filter_text)
        self._panel_analysis(panel_id)["wavelet_thread_filter_text"] = filter_text
        self._record_session_change()
        self._set_status(f"Set wavelet thread filter to {filter_text} for {panel.name}.")

    def _apply_td_window_selected_event_thread_to_stack(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if existing is None or panel is None or panel.cut_id is None:
            return
        if selected_event is None:
            self._set_status(f"Select a wavelet event first for {panel.name}.")
            return
        stack_id = self._resolve_td_window_target_stack_id(panel_id)
        if stack_id is None or int(stack_id) not in self.stacks:
            self._set_status(
                f"{panel.name} is not tied to a unique stack. Use Stack Browser -> Thread->Stack."
            )
            return
        thread_index = self._wavelet_event_thread_index(selected_event)
        if thread_index is None:
            self._set_status(f"The selected event in {panel.name} has no valid thread index.")
            return
        stack = self.stacks[int(stack_id)]
        cut_ids = [
            int(cut_id)
            for cut_id in (stack.get("cut_ids") or [])
            if int(cut_id) in self.cuts
        ]
        applied_cut_ids, skipped_cut_ids = self._apply_wavelet_thread_filter_to_cuts(
            cut_ids,
            [thread_index],
        )
        filter_text = self._format_wavelet_thread_filter_text([thread_index])
        if not applied_cut_ids and skipped_cut_ids:
            self._set_status(
                f"Thread filter {filter_text} could not be applied in {stack['name']}; "
                "some cuts do not have that tracked thread."
            )
            return
        message = (
            f"Applied wavelet thread filter {filter_text} to {len(applied_cut_ids)} cut(s) in {stack['name']}."
        )
        if skipped_cut_ids:
            message += f" Skipped {len(skipped_cut_ids)} cut(s) without that thread."
        self._set_status(message)

    def _format_wavelet_segment_physics(
        self, segment: dict[str, Any], prefix: str
    ) -> str:
        return (
            f"{prefix} thread {int(segment.get('thread_index', -1)) + 1}, "
            f"seg {int(segment.get('seg_id', -1))}, "
            f"wseg {int(segment.get('wseg_id', -1))} | "
            f"A={float(segment.get('fit_amp_arcsec', float('nan'))):.3f}'' "
            f"({float(segment.get('fit_amp_km', float('nan'))):.1f} km) | "
            f"P={float(segment.get('peak_period_s', float('nan'))):.2f} s | "
            f"f={float(segment.get('freq_mhz', float('nan'))):.2f} mHz | "
            f"v={float(segment.get('velocity_amp_km_s', float('nan'))):.2f} km/s | "
            f"ratio={float(segment.get('power_ratio', float('nan'))):.2f}"
        )

    def _clone_wavelet_payload(self, payload: Any) -> Any:
        if isinstance(payload, np.ndarray):
            return np.asarray(payload, dtype=np.float64).copy()
        if isinstance(payload, dict):
            return {key: self._clone_wavelet_payload(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._clone_wavelet_payload(value) for value in payload]
        if isinstance(payload, tuple):
            return tuple(self._clone_wavelet_payload(value) for value in payload)
        return payload

    def _panel_wavelet_events_snapshot(self, panel_id: int) -> list[dict[str, Any]]:
        existing = self.td_windows.get(panel_id)
        if existing is not None:
            events = existing.get("wavelet_events") or []
        else:
            events = (self._panel_analysis(panel_id).get("wavelet_events") or [])
        out = events if isinstance(events, list) else []
        for event in out:
            self._ensure_wavelet_event_fields(event)
        return out

    def _cut_wavelet_events_snapshot(self, cut_id: int) -> list[dict[str, Any]]:
        state = self.cut_analysis_state.get(cut_id)
        events = [] if state is None else (state.get("wavelet_events") or [])
        out = events if isinstance(events, list) else []
        for event in out:
            self._ensure_wavelet_event_fields(event)
        return out

    def _wavelet_event_ref(
        self, panel_id: int, event_id: int | None
    ) -> dict[str, Any] | None:
        if event_id is None:
            return None
        for event in self._panel_wavelet_events_snapshot(panel_id):
            if int(event.get("event_id", -1)) == int(event_id):
                self._ensure_wavelet_event_fields(event)
                return event
        return None

    def _wavelet_event_ref_by_cut(
        self, cut_id: int, event_id: int | None
    ) -> dict[str, Any] | None:
        if event_id is None:
            return None
        for event in self._cut_wavelet_events_snapshot(cut_id):
            if int(event.get("event_id", -1)) == int(event_id):
                self._ensure_wavelet_event_fields(event)
                return event
        return None

    def _ensure_wavelet_event_fields(self, event: dict[str, Any]) -> None:
        event.setdefault("manual_decision", None)
        event.setdefault("customized", False)
        event.setdefault("split_children_ids", [])
        event.setdefault("review_locked", False)
        event.setdefault("review_notes", "")
        event.setdefault("history", [])
        event.setdefault("link_group_id", None)
        event.setdefault("propagation_class", "")
        event.setdefault("confidence_score", float("nan"))
        event.setdefault("confidence_label", "")
        if event.get("history") is None:
            event["history"] = []

    def _event_source_overlap_fraction(
        self, source_a: np.ndarray, source_b: np.ndarray
    ) -> float:
        if source_a.size == 0 or source_b.size == 0:
            return 0.0
        overlap = np.intersect1d(
            np.asarray(np.round(source_a), dtype=np.int64),
            np.asarray(np.round(source_b), dtype=np.int64),
        )
        denom = max(min(source_a.size, source_b.size), 1)
        return float(overlap.size / denom)

    def _wavelet_segment_matches_locked_event(
        self, segment: dict[str, Any], event: dict[str, Any]
    ) -> bool:
        if not bool(event.get("review_locked")):
            return False
        analysis_a = segment or {}
        analysis_b = event.get("analysis") or {}
        if (
            int(analysis_a.get("thread_index", -9999))
            != int(analysis_b.get("thread_index", -9999))
        ):
            return False
        source_a = np.asarray(segment.get("source_t_idx", []), dtype=np.float64)
        source_b = np.asarray(event.get("source_t_idx", []), dtype=np.float64)
        return self._event_source_overlap_fraction(source_a, source_b) >= 0.5

    def _append_wavelet_event_history(
        self,
        event: dict[str, Any],
        action: str,
        *,
        note: str = "",
        details: str = "",
    ) -> None:
        self._ensure_wavelet_event_fields(event)
        entry = {
            "timestamp": self._timestamp_now(),
            "action": str(action),
            "note": str(note),
            "details": str(details),
            "status": self._td_window_wavelet_event_status(event),
        }
        event["history"].append(entry)

    def _wavelet_event_confidence_details(
        self, event: dict[str, Any]
    ) -> tuple[float, str]:
        self._ensure_wavelet_event_fields(event)
        analysis = event.get("analysis") or {}
        params = dict(event.get("current_params") or event.get("base_params") or {})
        flags = self._td_window_wavelet_event_qa_flags(event)
        score = 45.0
        power_ratio = float(analysis.get("power_ratio", float("nan")))
        thresh = max(float(params.get("power_ratio_thresh", 1.0)), 1e-6)
        if np.isfinite(power_ratio):
            score += min(max((power_ratio - thresh) / thresh, 0.0), 2.0) * 15.0
        point_count = int(np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64).size)
        target_points = max(int(params.get("min_points_segment", 3)), 3)
        score += min(point_count / target_points, 2.0) * 8.0
        peak_period = float(analysis.get("peak_period_s", float("nan")))
        duration = float(analysis.get("duration_s", float("nan")))
        if np.isfinite(duration) and np.isfinite(peak_period) and peak_period > 0.0:
            score += min(duration / (2.0 * peak_period), 1.5) * 8.0
        residual = float(
            analysis.get(
                "fit_rms_over_amp",
                analysis.get("rms_amp_ratio", float("nan")),
            )
        )
        rms_limit = float(params.get("rms_amp_ratio_max", float("nan")))
        if np.isfinite(residual) and np.isfinite(rms_limit) and rms_limit > 0.0:
            score -= min(residual / rms_limit, 2.0) * 10.0
        if "few_points" in flags:
            score -= 10.0
        if "period_edge" in flags:
            score -= 10.0
        if "high_residual" in flags:
            score -= 18.0
        status = self._td_window_wavelet_event_status(event)
        if status == "manual accepted":
            score += 6.0
        elif status == "manual rejected":
            score -= 20.0
        elif status == "custom accepted":
            score += 3.0
        elif status == "custom rejected":
            score -= 12.0
        if bool(event.get("review_locked")):
            score += 4.0
        score = float(max(0.0, min(100.0, score)))
        if score >= 80.0:
            label = "high"
        elif score >= 55.0:
            label = "medium"
        else:
            label = "low"
        event["confidence_score"] = score
        event["confidence_label"] = label
        return score, label

    def _wavelet_event_confidence_score(self, event: dict[str, Any]) -> float:
        score, _label = self._wavelet_event_confidence_details(event)
        return score

    def _wavelet_event_link_refs(
        self, event: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._ensure_wavelet_event_fields(event)
        group_id = event.get("link_group_id")
        if not group_id:
            return []
        refs: list[dict[str, Any]] = []
        for panel in self.panels:
            panel_events = self._panel_wavelet_events_snapshot(panel.panel_id)
            cut = self.cuts.get(panel.cut_id) if panel.cut_id is not None else None
            for candidate in panel_events:
                self._ensure_wavelet_event_fields(candidate)
                if candidate.get("link_group_id") != group_id:
                    continue
                refs.append(
                    {
                        "panel_id": panel.panel_id,
                        "panel_name": panel.name,
                        "cut_name": "" if cut is None else cut.name,
                        "event_id": int(candidate.get("event_id", -1)),
                    }
                )
        refs.sort(key=lambda item: (item["panel_id"], item["event_id"]))
        return refs

    def _wavelet_event_link_count(self, event: dict[str, Any]) -> int:
        refs = self._wavelet_event_link_refs(event)
        if not refs:
            return 0
        return max(len(refs) - 1, 0)

    def _wavelet_edit_state_snapshot(self, panel_id: int) -> dict[str, Any]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            state = self._panel_analysis(panel_id)
            return {
                "events": self._clone_wavelet_payload(state.get("wavelet_events") or []),
                "selected_event_id": state.get("wavelet_selected_event_id"),
                "next_event_id": int(state.get("wavelet_next_event_id", 1)),
            }
        return {
            "events": self._clone_wavelet_payload(existing.get("wavelet_events") or []),
            "selected_event_id": existing.get("wavelet_selected_event_id"),
            "next_event_id": int(existing.get("wavelet_next_event_id", 1)),
        }

    def _push_wavelet_undo_state(self, panel_id: int, label: str) -> None:
        snapshot = self._wavelet_edit_state_snapshot(panel_id)
        snapshot["label"] = str(label)
        state = self._panel_analysis(panel_id)
        undo_stack = state.setdefault("wavelet_undo_stack", [])
        undo_stack.append(self._clone_wavelet_payload(snapshot))
        if len(undo_stack) > MAX_WAVELET_HISTORY:
            del undo_stack[: len(undo_stack) - MAX_WAVELET_HISTORY]
        state["wavelet_redo_stack"] = []
        existing = self.td_windows.get(panel_id)
        if existing is not None:
            existing["wavelet_undo_stack"] = self._clone_wavelet_payload(undo_stack)
            existing["wavelet_redo_stack"] = []

    def _restore_wavelet_edit_state(
        self, panel_id: int, snapshot: dict[str, Any], *, clear_target_stack: bool = False
    ) -> None:
        state = self._panel_analysis(panel_id)
        state["wavelet_events"] = self._clone_wavelet_payload(snapshot.get("events") or [])
        state["wavelet_selected_event_id"] = snapshot.get("selected_event_id")
        state["wavelet_next_event_id"] = int(snapshot.get("next_event_id", 1))
        existing = self.td_windows.get(panel_id)
        if existing is not None:
            existing["wavelet_events"] = self._clone_wavelet_payload(state["wavelet_events"])
            existing["wavelet_selected_event_id"] = state["wavelet_selected_event_id"]
            existing["wavelet_next_event_id"] = state["wavelet_next_event_id"]
            if clear_target_stack:
                existing["wavelet_redo_stack"] = self._clone_wavelet_payload(
                    state.get("wavelet_redo_stack") or []
                )
            self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)

    def _undo_td_window_wavelet_edit(self, panel_id: int) -> None:
        state = self._panel_analysis(panel_id)
        undo_stack = state.setdefault("wavelet_undo_stack", [])
        if not undo_stack:
            self._set_status(f"No wavelet edits to undo for P{panel_id}.")
            return
        redo_stack = state.setdefault("wavelet_redo_stack", [])
        redo_stack.append(self._wavelet_edit_state_snapshot(panel_id))
        snapshot = undo_stack.pop()
        self._restore_wavelet_edit_state(panel_id, snapshot, clear_target_stack=True)
        existing = self.td_windows.get(panel_id)
        if existing is not None:
            existing["wavelet_undo_stack"] = self._clone_wavelet_payload(undo_stack)
            existing["wavelet_redo_stack"] = self._clone_wavelet_payload(redo_stack)
        self._record_session_change()
        self._set_status(f"Undid wavelet edit for P{panel_id}.")

    def _redo_td_window_wavelet_edit(self, panel_id: int) -> None:
        state = self._panel_analysis(panel_id)
        redo_stack = state.setdefault("wavelet_redo_stack", [])
        if not redo_stack:
            self._set_status(f"No wavelet edits to redo for P{panel_id}.")
            return
        undo_stack = state.setdefault("wavelet_undo_stack", [])
        undo_stack.append(self._wavelet_edit_state_snapshot(panel_id))
        snapshot = redo_stack.pop()
        self._restore_wavelet_edit_state(panel_id, snapshot, clear_target_stack=True)
        existing = self.td_windows.get(panel_id)
        if existing is not None:
            existing["wavelet_undo_stack"] = self._clone_wavelet_payload(undo_stack)
            existing["wavelet_redo_stack"] = self._clone_wavelet_payload(redo_stack)
        self._record_session_change()
        self._set_status(f"Redid wavelet edit for P{panel_id}.")

    def _best_wavelet_segment(
        self, segments: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not segments:
            return None
        accepted = [segment for segment in segments if segment.get("accepted")]
        pool = accepted or segments
        best = max(
            pool,
            key=lambda segment: (
                float(segment.get("power_ratio", float("-inf"))),
                float(segment.get("duration_s", float("-inf"))),
                float(segment.get("fit_amp_arcsec", float("-inf"))),
            ),
        )
        return best

    def _lightweight_wavelet_filter_result(
        self,
        *,
        segments: list[dict[str, Any]],
        params: dict[str, Any],
        best_segment: dict[str, Any] | None,
        preserved_locked_event_ids: list[int],
        warnings: list[str],
    ) -> dict[str, Any]:
        best_ref = None
        if best_segment is not None:
            best_ref = {
                "thread_index": int(best_segment.get("thread_index", -1)),
                "seg_id": int(best_segment.get("seg_id", -1)),
                "wseg_id": int(best_segment.get("wseg_id", -1)),
                "accepted": bool(best_segment.get("accepted")),
                "power_ratio": float(best_segment.get("power_ratio", float("nan"))),
                "duration_s": float(best_segment.get("duration_s", float("nan"))),
                "fit_amp_arcsec": float(best_segment.get("fit_amp_arcsec", float("nan"))),
            }
        return {
            "segment_count": int(len(segments)),
            "accepted_count": int(sum(1 for segment in segments if segment.get("accepted"))),
            "with_segment_count": int(sum(1 for segment in segments if segment.get("has_segment"))),
            "params": dict(params),
            "best_segment": best_ref,
            "warnings": [str(item) for item in warnings if str(item)],
            "preserved_locked_event_ids": [
                int(event_id) for event_id in preserved_locked_event_ids
            ],
            "segments": [],
        }

    def _make_td_window_wavelet_event(
        self,
        event_id: int,
        segment: dict[str, Any],
        params: dict[str, Any],
        *,
        origin: str = "auto",
        parent_event_id: int | None = None,
    ) -> dict[str, Any]:
        base_segment = self._clone_wavelet_payload(segment)
        base_source_t_idx = np.asarray(
            segment.get("source_t_idx", segment.get("wave_t_idx", [])),
            dtype=np.float64,
        ).copy()
        base_source_y_idx = np.asarray(
            segment.get("source_y_idx", segment.get("wave_y_idx", [])),
            dtype=np.float64,
        ).copy()
        event = {
            "event_id": int(event_id),
            "parent_event_id": parent_event_id,
            "origin": origin,
            "manual_decision": None,
            "customized": False,
            "split_children_ids": [],
            "base_source_t_idx": base_source_t_idx,
            "base_source_y_idx": base_source_y_idx,
            "source_t_idx": base_source_t_idx.copy(),
            "source_y_idx": base_source_y_idx.copy(),
            "base_analysis": base_segment,
            "analysis": self._clone_wavelet_payload(base_segment),
            "base_params": dict(params),
            "current_params": dict(params),
            "diagnostic": None,
            "review_locked": False,
            "review_notes": "",
            "history": [],
            "link_group_id": None,
        }
        self._append_wavelet_event_history(
            event,
            "created",
            details=f"origin={origin}",
        )
        self._wavelet_event_confidence_details(event)
        return event

    def _td_window_wavelet_events(self, panel_id: int) -> list[dict[str, Any]]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return []
        events = existing.get("wavelet_events")
        if not isinstance(events, list):
            return []
        for event in events:
            self._ensure_wavelet_event_fields(event)
            self._wavelet_event_confidence_details(event)
        return events

    def _td_window_wavelet_event_by_id(
        self, panel_id: int, event_id: int | None
    ) -> dict[str, Any] | None:
        if event_id is None:
            return None
        for event in self._td_window_wavelet_events(panel_id):
            if int(event.get("event_id", -1)) == int(event_id):
                return event
        return None

    def _td_window_wavelet_selected_event(
        self, panel_id: int
    ) -> dict[str, Any] | None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return None
        return self._td_window_wavelet_event_by_id(
            panel_id, existing.get("wavelet_selected_event_id")
        )

    def _selected_td_window_wavelet_event_editable(
        self, panel_id: int, action: str
    ) -> dict[str, Any] | None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None:
            return None
        if selected_event is None:
            self._set_status(f"Select a wavelet event first for {panel.name}.")
            return None
        self._ensure_wavelet_event_fields(selected_event)
        if bool(selected_event.get("review_locked")):
            self._set_status(
                f"Unlock the selected wavelet event before trying to {action} in {panel.name}."
            )
            return None
        return selected_event

    def _toggle_td_window_selected_wavelet_lock(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        self._push_wavelet_undo_state(panel_id, "toggle lock")
        self._ensure_wavelet_event_fields(selected_event)
        selected_event["review_locked"] = not bool(selected_event.get("review_locked"))
        self._append_wavelet_event_history(
            selected_event,
            "lock" if selected_event["review_locked"] else "unlock",
            details=f"panel={panel.name}",
        )
        self._wavelet_event_confidence_details(selected_event)
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(
            f"{'Locked' if selected_event['review_locked'] else 'Unlocked'} selected wavelet event in {panel.name}."
        )

    def _edit_td_window_selected_wavelet_note(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        self._ensure_wavelet_event_fields(selected_event)
        current_note = str(selected_event.get("review_notes", ""))
        note = self.simpledialog.askstring(
            "Event Note",
            f"Reviewer note for {panel.name} event {int(selected_event.get('event_id', -1))}:",
            initialvalue=current_note,
            parent=self.td_windows.get(panel_id, {}).get("top"),
        )
        if note is None:
            return
        self._push_wavelet_undo_state(panel_id, "edit note")
        selected_event["review_notes"] = str(note)
        self._append_wavelet_event_history(
            selected_event,
            "note",
            note=str(note),
        )
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Updated reviewer note for {panel.name}.")

    def _show_td_window_selected_wavelet_history(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        self._ensure_wavelet_event_fields(selected_event)
        top = self.tk.Toplevel(self.td_windows.get(panel_id, {}).get("top", self.root))
        top.title(
            f"History - {panel.name} event {int(selected_event.get('event_id', -1))}"
        )
        top.geometry("760x420")
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        text = self.tk.Text(top, wrap="word")
        text.grid(row=0, column=0, sticky="nsew")
        scroll = self.ttk.Scrollbar(top, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        lines = [
            f"Event {int(selected_event.get('event_id', -1))} | panel {panel.name}",
            f"Status: {self._td_window_wavelet_event_status(selected_event)}",
            f"Link group: {selected_event.get('link_group_id') or '-'}",
            f"Locked: {bool(selected_event.get('review_locked'))}",
            f"Note: {selected_event.get('review_notes') or '-'}",
            "",
        ]
        history_entries = selected_event.get("history") or []
        if not history_entries:
            lines.append("No history entries.")
        else:
            for item in history_entries:
                lines.append(
                    f"{item.get('timestamp', '')} | {item.get('action', '')} | "
                    f"{item.get('status', '')} | {item.get('details', '')}"
                )
                if item.get("note"):
                    lines.append(f"note: {item.get('note')}")
                lines.append("")
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    def _copy_td_window_selected_wavelet_link_source(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        self.link_source_event_ref = {
            "panel_id": panel_id,
            "event_id": int(selected_event.get("event_id", -1)),
        }
        self._set_status(
            f"Copied link source from {panel.name} event {int(selected_event.get('event_id', -1))}."
        )

    def _link_td_window_selected_wavelet_to_source(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._selected_td_window_wavelet_event_editable(panel_id, "link")
        if panel is None or selected_event is None:
            return
        if self.link_source_event_ref is None:
            self._set_status("Copy a link source event first.")
            return
        src_panel_id = int(self.link_source_event_ref.get("panel_id", -1))
        src_event_id = int(self.link_source_event_ref.get("event_id", -1))
        if src_panel_id == panel_id and src_event_id == int(selected_event.get("event_id", -1)):
            self._set_status("Select a different target event to link.")
            return
        source_event = self._td_window_wavelet_event_by_id(src_panel_id, src_event_id)
        if source_event is None:
            source_events = self._panel_wavelet_events_snapshot(src_panel_id)
            source_event = next(
                (evt for evt in source_events if int(evt.get("event_id", -1)) == src_event_id),
                None,
            )
        if source_event is None:
            self._set_status("The copied link source no longer exists.")
            return
        self._ensure_wavelet_event_fields(source_event)
        self._push_wavelet_undo_state(panel_id, "link event")
        if src_panel_id != panel_id:
            src_state = self._panel_analysis(src_panel_id)
            src_undo = src_state.setdefault("wavelet_undo_stack", [])
            src_undo.append(self._clone_wavelet_payload(self._wavelet_edit_state_snapshot(src_panel_id)))
            if len(src_undo) > MAX_WAVELET_HISTORY:
                del src_undo[: len(src_undo) - MAX_WAVELET_HISTORY]
            src_state["wavelet_redo_stack"] = []
            existing_src = self.td_windows.get(src_panel_id)
            if existing_src is not None:
                existing_src["wavelet_undo_stack"] = self._clone_wavelet_payload(src_undo)
                existing_src["wavelet_redo_stack"] = []
        group_id = source_event.get("link_group_id") or selected_event.get("link_group_id")
        if not group_id:
            group_id = self._next_link_group_label()
        source_event["link_group_id"] = group_id
        selected_event["link_group_id"] = group_id
        self._append_wavelet_event_history(
            source_event,
            "link",
            details=f"group={group_id} target=P{panel_id}:{int(selected_event.get('event_id', -1))}",
        )
        self._append_wavelet_event_history(
            selected_event,
            "link",
            details=f"group={group_id} source=P{src_panel_id}:{src_event_id}",
        )
        self._record_session_change()
        self._refresh_all_open_td_window_wavelet_views(redraw_td=True)
        self._set_status(
            f"Linked {panel.name} event {int(selected_event.get('event_id', -1))} to group {group_id}."
        )

    def _clear_td_window_selected_wavelet_link(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._selected_td_window_wavelet_event_editable(panel_id, "clear link")
        if panel is None or selected_event is None:
            return
        self._ensure_wavelet_event_fields(selected_event)
        if not selected_event.get("link_group_id"):
            self._set_status(f"Selected event in {panel.name} is not linked.")
            return
        self._push_wavelet_undo_state(panel_id, "clear link")
        group_id = str(selected_event.get("link_group_id"))
        selected_event["link_group_id"] = None
        self._append_wavelet_event_history(
            selected_event,
            "unlink",
            details=f"group={group_id}",
        )
        self._record_session_change()
        self._refresh_all_open_td_window_wavelet_views(redraw_td=True)
        self._set_status(f"Cleared link group {group_id} from selected event in {panel.name}.")

    def _show_td_window_selected_wavelet_links(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        self._ensure_wavelet_event_fields(selected_event)
        group_id = str(selected_event.get("link_group_id") or "").strip() or None
        self._open_link_groups_window(
            initial_group_id=group_id,
            initial_member_ref=(
                int(panel.cut_id) if panel.cut_id is not None else int(panel_id),
                int(selected_event.get("event_id", -1)),
            ),
        )
        if group_id is None:
            self._set_status(
                f"Opened link group viewer. {panel.name} event {int(selected_event.get('event_id', -1))} is not linked yet."
            )

    def _sync_linked_wavelet_group_from_cut_event(
        self, cut_id: int, event_id: int
    ) -> None:
        source_event = self._wavelet_event_ref_by_cut(cut_id, event_id)
        cut = self.cuts.get(cut_id)
        if cut is None or source_event is None:
            self._set_status("The selected linked event source is no longer available.")
            return
        group_id = str(source_event.get("link_group_id") or "").strip()
        if not group_id:
            self._set_status(
                f"{cut.name} event {int(source_event.get('event_id', -1))} is not linked."
            )
            return

        group_members: list[tuple[int, dict[str, Any]]] = []
        for current_cut_id in sorted(self.cuts.keys()):
            for event in self._cut_wavelet_events_snapshot(current_cut_id):
                self._ensure_wavelet_event_fields(event)
                if str(event.get("link_group_id") or "") == group_id:
                    group_members.append((int(current_cut_id), event))
        if len(group_members) < 2:
            self._set_status(f"Link group {group_id} only has one event.")
            return

        affected_cut_ids = sorted({int(item[0]) for item in group_members})
        for affected_cut_id in affected_cut_ids:
            self._push_wavelet_undo_state_for_cut(
                affected_cut_id, f"sync linked group {group_id}"
            )

        source_manual_decision = source_event.get("manual_decision")
        source_locked = bool(source_event.get("review_locked"))
        source_note = str(source_event.get("review_notes", ""))
        source_status = self._td_window_wavelet_event_status(source_event)
        updated_count = 0
        for target_cut_id, target_event in group_members:
            if target_cut_id == cut_id and int(target_event.get("event_id", -1)) == int(event_id):
                continue
            target_event["manual_decision"] = source_manual_decision
            target_event["review_locked"] = source_locked
            target_event["review_notes"] = source_note
            self._append_wavelet_event_history(
                target_event,
                "sync-in",
                note=source_note,
                details=(
                    f"group={group_id} source=cut{cut_id}:{event_id} "
                    f"status={source_status}"
                ),
            )
            self._wavelet_event_confidence_details(target_event)
            updated_count += 1

        self._append_wavelet_event_history(
            source_event,
            "sync-out",
            note=source_note,
            details=f"group={group_id} targets={updated_count}",
        )
        self._wavelet_event_confidence_details(source_event)
        self._record_session_change()
        for affected_cut_id in affected_cut_ids:
            self._sync_open_td_windows_from_cut(affected_cut_id)
        self._set_status(
            f"Synced {cut.name} event {event_id} to {updated_count} linked event(s) in {group_id}."
        )

    def _sync_linked_wavelet_group_from_event(self, panel_id: int, event_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        if panel is not None and panel.cut_id is not None and panel.cut_id in self.cuts:
            self._sync_linked_wavelet_group_from_cut_event(int(panel.cut_id), event_id)
            return
        source_event = self._wavelet_event_ref(panel_id, event_id)
        if panel is None or source_event is None:
            self._set_status("The selected linked event source is no longer available.")
            return
        group_id = str(source_event.get("link_group_id") or "").strip()
        if not group_id:
            self._set_status(
                f"{panel.name} event {int(source_event.get('event_id', -1))} is not linked."
            )
            return

        group_members: list[tuple[int, dict[str, Any]]] = []
        for candidate_panel in self.panels:
            for event in self._panel_wavelet_events_snapshot(candidate_panel.panel_id):
                self._ensure_wavelet_event_fields(event)
                if str(event.get("link_group_id") or "") == group_id:
                    group_members.append((candidate_panel.panel_id, event))
        if len(group_members) < 2:
            self._set_status(f"Link group {group_id} only has one event.")
            return

        for affected_panel_id in sorted({int(item[0]) for item in group_members}):
            self._push_wavelet_undo_state(
                affected_panel_id, f"sync linked group {group_id}"
            )

        source_manual_decision = source_event.get("manual_decision")
        source_locked = bool(source_event.get("review_locked"))
        source_note = str(source_event.get("review_notes", ""))
        source_status = self._td_window_wavelet_event_status(source_event)
        updated_count = 0
        for target_panel_id, target_event in group_members:
            if target_panel_id == panel_id and int(target_event.get("event_id", -1)) == int(event_id):
                continue
            target_event["manual_decision"] = source_manual_decision
            target_event["review_locked"] = source_locked
            target_event["review_notes"] = source_note
            self._append_wavelet_event_history(
                target_event,
                "sync-in",
                note=source_note,
                details=(
                    f"group={group_id} source=P{panel_id}:{event_id} "
                    f"status={source_status}"
                ),
            )
            self._wavelet_event_confidence_details(target_event)
            updated_count += 1

        self._append_wavelet_event_history(
            source_event,
            "sync-out",
            note=source_note,
            details=f"group={group_id} targets={updated_count}",
        )
        self._wavelet_event_confidence_details(source_event)
        self._record_session_change()
        self._refresh_all_open_td_window_wavelet_views(redraw_td=True)
        self._refresh_link_groups_window()
        self._set_status(
            f"Synced {panel.name} event {event_id} to {updated_count} linked event(s) in {group_id}."
        )

    def _sync_td_window_selected_wavelet_group(self, panel_id: int) -> None:
        panel = self._td_window_panel(panel_id)
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if panel is None or selected_event is None:
            self._set_status(f"Select a wavelet event first for P{panel_id}.")
            return
        self._sync_linked_wavelet_group_from_event(
            panel_id, int(selected_event.get("event_id", -1))
        )

    def _td_window_wavelet_event_status(self, event: dict[str, Any]) -> str:
        if event.get("split_children_ids"):
            return "split parent"
        manual_decision = event.get("manual_decision")
        if manual_decision == "accepted":
            return "manual accepted"
        if manual_decision == "rejected":
            return "manual rejected"
        analysis = event.get("analysis") or {}
        if event.get("customized"):
            return "custom accepted" if analysis.get("accepted") else "custom rejected"
        return "auto accepted" if analysis.get("accepted") else "auto rejected"

    def _td_window_wavelet_event_is_counted(self, event: dict[str, Any]) -> bool:
        return self._td_window_wavelet_event_status(event) in {
            "auto accepted",
            "custom accepted",
            "manual accepted",
        }

    def _td_window_wavelet_event_reason(self, event: dict[str, Any]) -> str:
        status = self._td_window_wavelet_event_status(event)
        if status == "manual accepted":
            return "accepted manually"
        if status == "manual rejected":
            return "rejected manually"
        if status == "split parent":
            return "replaced by split children"
        return str((event.get("analysis") or {}).get("decision_reason", "") or "")

    def _td_window_wavelet_advanced_filter_values(
        self, panel_id: int
    ) -> dict[str, Any]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            values = dict(
                (self._panel_analysis(panel_id).get("wavelet_advanced_filters") or {})
            )
            return {
                "qa": str(values.get("qa", "all")),
                "locked": str(values.get("locked", "all")),
                "linked": str(values.get("linked", "all")),
                "score_min": self._safe_float_text(values.get("score_min", ""), float("nan")),
                "period_min": self._safe_float_text(values.get("period_min", ""), float("nan")),
                "period_max": self._safe_float_text(values.get("period_max", ""), float("nan")),
                "amp_min": self._safe_float_text(values.get("amp_min", ""), float("nan")),
                "amp_max": self._safe_float_text(values.get("amp_max", ""), float("nan")),
                "energy_min": self._safe_float_text(values.get("energy_min", ""), float("nan")),
                "energy_max": self._safe_float_text(values.get("energy_max", ""), float("nan")),
            }
        return {
            "qa": str(existing["wavelet_filter_qa_var"].get() or "all"),
            "locked": str(existing["wavelet_filter_locked_var"].get() or "all"),
            "linked": str(existing["wavelet_filter_linked_var"].get() or "all"),
            "score_min": self._safe_float_text(existing["wavelet_filter_score_min_var"].get(), float("nan")),
            "period_min": self._safe_float_text(existing["wavelet_filter_period_min_var"].get(), float("nan")),
            "period_max": self._safe_float_text(existing["wavelet_filter_period_max_var"].get(), float("nan")),
            "amp_min": self._safe_float_text(existing["wavelet_filter_amp_min_var"].get(), float("nan")),
            "amp_max": self._safe_float_text(existing["wavelet_filter_amp_max_var"].get(), float("nan")),
            "energy_min": self._safe_float_text(existing["wavelet_filter_energy_min_var"].get(), float("nan")),
            "energy_max": self._safe_float_text(existing["wavelet_filter_energy_max_var"].get(), float("nan")),
        }

    def _td_window_wavelet_advanced_filter_match(
        self, event: dict[str, Any], filters: dict[str, Any]
    ) -> bool:
        self._ensure_wavelet_event_fields(event)
        analysis = event.get("analysis") or {}
        flags = self._td_window_wavelet_event_qa_flags(event)
        qa_filter = str(filters.get("qa", "all"))
        if qa_filter == "flagged" and not flags:
            return False
        if qa_filter == "clean" and flags:
            return False
        if qa_filter in {"few_points", "period_edge", "high_residual"} and qa_filter not in flags:
            return False
        locked_filter = str(filters.get("locked", "all"))
        if locked_filter == "locked" and not bool(event.get("review_locked")):
            return False
        if locked_filter == "unlocked" and bool(event.get("review_locked")):
            return False
        linked_filter = str(filters.get("linked", "all"))
        link_count = self._wavelet_event_link_count(event)
        if linked_filter == "linked" and link_count <= 0:
            return False
        if linked_filter == "unlinked" and link_count > 0:
            return False
        confidence_score = self._wavelet_event_confidence_score(event)
        if np.isfinite(float(filters.get("score_min", float("nan")))):
            if confidence_score < float(filters["score_min"]):
                return False
        value_map = {
            "period_min": float(analysis.get("peak_period_s", float("nan"))),
            "period_max": float(analysis.get("peak_period_s", float("nan"))),
            "amp_min": float(analysis.get("fit_amp_arcsec", float("nan"))),
            "amp_max": float(analysis.get("fit_amp_arcsec", float("nan"))),
            "energy_min": float(analysis.get("specific_energy_j_kg", float("nan"))),
            "energy_max": float(analysis.get("specific_energy_j_kg", float("nan"))),
        }
        if np.isfinite(float(filters.get("period_min", float("nan")))) and (
            not np.isfinite(value_map["period_min"]) or value_map["period_min"] < float(filters["period_min"])
        ):
            return False
        if np.isfinite(float(filters.get("period_max", float("nan")))) and (
            not np.isfinite(value_map["period_max"]) or value_map["period_max"] > float(filters["period_max"])
        ):
            return False
        if np.isfinite(float(filters.get("amp_min", float("nan")))) and (
            not np.isfinite(value_map["amp_min"]) or value_map["amp_min"] < float(filters["amp_min"])
        ):
            return False
        if np.isfinite(float(filters.get("amp_max", float("nan")))) and (
            not np.isfinite(value_map["amp_max"]) or value_map["amp_max"] > float(filters["amp_max"])
        ):
            return False
        if np.isfinite(float(filters.get("energy_min", float("nan")))) and (
            not np.isfinite(value_map["energy_min"]) or value_map["energy_min"] < float(filters["energy_min"])
        ):
            return False
        if np.isfinite(float(filters.get("energy_max", float("nan")))) and (
            not np.isfinite(value_map["energy_max"]) or value_map["energy_max"] > float(filters["energy_max"])
        ):
            return False
        return True

    def _td_window_wavelet_event_filter_match(
        self, event: dict[str, Any], filter_name: str, filters: dict[str, Any]
    ) -> bool:
        status = self._td_window_wavelet_event_status(event)
        if filter_name == "all":
            base_match = True
        elif filter_name == "accepted":
            base_match = self._td_window_wavelet_event_is_counted(event)
        elif filter_name == "rejected":
            base_match = (not self._td_window_wavelet_event_is_counted(event)) and (
                status != "split parent"
            )
        elif filter_name == "manual":
            base_match = status.startswith("manual") or status.startswith("custom")
        elif filter_name == "split":
            base_match = status == "split parent"
        else:
            base_match = True
        if not base_match:
            return False
        return self._td_window_wavelet_advanced_filter_match(event, filters)

    def _td_window_wavelet_event_row(
        self, event: dict[str, Any]
    ) -> tuple[str, ...]:
        self._ensure_wavelet_event_fields(event)
        analysis = event.get("analysis") or {}
        confidence_score = self._wavelet_event_confidence_score(event)
        return (
            str(int(event.get("event_id", -1))),
            self._td_window_wavelet_event_status(event),
            str(event.get("origin", "")),
            str(int(analysis.get("thread_index", -1)) + 1),
            str(int(analysis.get("seg_id", -1))),
            str(int(analysis.get("wseg_id", -1))),
            f"{float(analysis.get('peak_period_s', float('nan'))):.2f}",
            f"{float(analysis.get('freq_mhz', float('nan'))):.2f}",
            f"{float(analysis.get('fit_amp_arcsec', float('nan'))):.3f}",
            f"{float(analysis.get('fit_amp_km', float('nan'))):.1f}",
            f"{float(analysis.get('velocity_amp_km_s', float('nan'))):.2f}",
            f"{float(analysis.get('accel_amp_km_s2', float('nan'))):.3f}",
            f"{float(analysis.get('specific_energy_j_kg', float('nan'))):.3e}",
            f"{float(analysis.get('duration_s', float('nan'))):.2f}",
            f"{float(analysis.get('power_ratio', float('nan'))):.2f}",
            f"{confidence_score:.1f}",
            "yes" if bool(event.get("review_locked")) else "no",
            str(self._wavelet_event_link_count(event)),
            ",".join(self._td_window_wavelet_event_qa_flags(event)),
            self._td_window_wavelet_event_reason(event),
        )

    def _td_window_wavelet_table_tag(self, event: dict[str, Any]) -> str:
        return self._td_window_wavelet_event_status(event).replace(" ", "_")

    def _analyze_td_window_wavelet_event_source(
        self,
        panel_id: int,
        source_t_idx: np.ndarray,
        source_y_idx: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        panel = self._td_window_panel(panel_id)
        if panel is None:
            return None
        api, import_error = load_local_wavelet_filter_api()
        if api is None:
            self._set_status(f"Wavelet filter import failed for {panel.name}: {import_error}")
            return None
        try:
            analysis = api["analyze_tracked_segment_with_wavelet"](
                source_t_idx,
                source_y_idx,
                cadence=params["cad"],
                pix_scale=params["res"],
                km_per_arcsec=params["km_per_arcsec"],
                p_min=params["p_min"],
                p_max=params["p_max"],
                power_ratio_thresh=params["power_ratio_thresh"],
                segment_power_frac=params["segment_power_frac"],
                min_points_segment=params["min_points_segment"],
                min_amp_arcsec=params["min_amp_arcsec"],
                rms_amp_ratio_max=params["rms_amp_ratio_max"],
                density_kg_m3=params["density_kg_m3"],
                phase_speed_km_s=params["phase_speed_km_s"],
            )
        except Exception as exc:
            self._set_status(
                f"Wavelet analysis failed for {panel.name}: {type(exc).__name__}: {exc}"
            )
            return None
        selected = self._best_wavelet_segment(analysis.get("candidates", []))
        if selected is None:
            return None
        return {
            "selected": selected,
            "diag": self._clone_wavelet_payload(analysis.get("diag", {})),
            "candidates": self._clone_wavelet_payload(analysis.get("candidates", [])),
            "t_seg_s": self._clone_wavelet_payload(analysis.get("t_seg_s", np.array([], dtype=np.float64))),
            "y_arcsec": self._clone_wavelet_payload(analysis.get("y_arcsec", np.array([], dtype=np.float64))),
            "source_t_idx": np.asarray(source_t_idx, dtype=np.float64).copy(),
            "source_y_idx": np.asarray(source_y_idx, dtype=np.float64).copy(),
            "params": dict(params),
        }

    def _set_td_window_wavelet_event_analysis(
        self, event: dict[str, Any], diagnostic: dict[str, Any], *, customized: bool
    ) -> None:
        self._ensure_wavelet_event_fields(event)
        selected = self._clone_wavelet_payload(diagnostic["selected"])
        current_analysis = event.get("analysis") or event.get("base_analysis") or {}
        for key in ("thread_index", "seg_id"):
            if key not in selected:
                selected[key] = current_analysis.get(key, -1)
        event["analysis"] = selected
        event["diagnostic"] = self._clone_wavelet_payload(diagnostic)
        event["current_params"] = dict(diagnostic["params"])
        event["customized"] = bool(customized)
        self._wavelet_event_confidence_details(event)

    def _refresh_td_window_wavelet_table(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        tree = existing.get("wavelet_events_tree")
        if tree is None:
            return
        filter_name = str(existing["wavelet_events_filter_var"].get() or "all")
        advanced_filters = self._td_window_wavelet_advanced_filter_values(panel_id)
        selected_event_id = existing.get("wavelet_selected_event_id")
        existing["wavelet_table_updating"] = True
        try:
            children = tree.get_children()
            if children:
                tree.delete(*children)

            visible_ids: list[int] = []
            for event in self._td_window_wavelet_events(panel_id):
                if not self._td_window_wavelet_event_filter_match(
                    event, filter_name, advanced_filters
                ):
                    continue
                iid = f"evt-{int(event['event_id'])}"
                visible_ids.append(int(event["event_id"]))
                tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=self._td_window_wavelet_event_row(event),
                    tags=(self._td_window_wavelet_table_tag(event),),
                )

            target_iid: str | None = None
            if selected_event_id in visible_ids:
                target_iid = f"evt-{int(selected_event_id)}"
            elif visible_ids:
                existing["wavelet_selected_event_id"] = visible_ids[0]
                target_iid = f"evt-{visible_ids[0]}"
            else:
                existing["wavelet_selected_event_id"] = None

            if target_iid is not None:
                current_selection = tree.selection()
                if current_selection != (target_iid,):
                    tree.selection_set(target_iid)
                tree.focus(target_iid)
            else:
                tree.selection_remove(tree.selection())
        finally:
            existing["wavelet_table_updating"] = False

        events = self._td_window_wavelet_events(panel_id)
        counted = sum(1 for event in events if self._td_window_wavelet_event_is_counted(event))
        rejected = sum(
            1
            for event in events
            if (not self._td_window_wavelet_event_is_counted(event))
            and self._td_window_wavelet_event_status(event) != "split parent"
        )
        manual = sum(
            1
            for event in events
            if self._td_window_wavelet_event_status(event).startswith("manual")
            or self._td_window_wavelet_event_status(event).startswith("custom")
        )
        split_parents = sum(
            1 for event in events if self._td_window_wavelet_event_status(event) == "split parent"
        )
        visible_count = len(tree.get_children())
        existing["wavelet_events_summary_var"].set(
            f"Events: {len(events)} | visible: {visible_count} | counted: {counted} | "
            f"rejected: {rejected} | manual/custom: {manual} | split parents: {split_parents}"
        )

    def _refresh_td_window_wavelet_diagnostics(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        diag_canvas = existing.get("diag_canvas")
        diag_fig = existing.get("diag_figure")
        diag_axes = existing.get("diag_axes")
        if diag_fig is None or diag_canvas is None or diag_axes is None:
            return

        for axis in diag_axes:
            axis.clear()

        if selected_event is None:
            existing["wavelet_selected_var"].set("Select a wavelet event to inspect it.")
            existing["wavelet_diag_var"].set("")
            diag_axes[0].text(0.5, 0.5, "Select an event in the table.", ha="center", va="center", transform=diag_axes[0].transAxes)
            for axis in diag_axes[1:]:
                axis.axis("off")
            diag_canvas.draw_idle()
            return

        diagnostic = selected_event.get("diagnostic")
        if diagnostic is None:
            diagnostic = self._analyze_td_window_wavelet_event_source(
                panel_id,
                np.asarray(selected_event.get("source_t_idx", []), dtype=np.float64),
                np.asarray(selected_event.get("source_y_idx", []), dtype=np.float64),
                dict(selected_event.get("current_params") or selected_event.get("base_params") or {}),
            )
            if diagnostic is not None:
                selected_event["diagnostic"] = self._clone_wavelet_payload(diagnostic)
        if diagnostic is None:
            existing["wavelet_selected_var"].set("Could not build diagnostics for the selected event.")
            existing["wavelet_diag_var"].set("")
            diag_axes[0].text(0.5, 0.5, "Diagnostics unavailable.", ha="center", va="center", transform=diag_axes[0].transAxes)
            for axis in diag_axes[1:]:
                axis.axis("off")
            diag_canvas.draw_idle()
            return

        analysis = selected_event.get("analysis") or {}
        confidence_score, confidence_label = self._wavelet_event_confidence_details(
            selected_event
        )
        link_refs = self._wavelet_event_link_refs(selected_event)
        linked_targets = [
            f"{item['panel_name']}:{item['event_id']}"
            for item in link_refs
            if not (
                int(item.get("panel_id", -1)) == int(panel_id)
                and int(item.get("event_id", -1))
                == int(selected_event.get("event_id", -1))
            )
        ]
        note_text = str(selected_event.get("review_notes") or "").strip()
        diag = diagnostic.get("diag") or {}
        t_seg = np.asarray(diagnostic.get("t_seg_s", []), dtype=np.float64)
        y_arc = np.asarray(diagnostic.get("y_arcsec", []), dtype=np.float64)
        trend = np.asarray(diag.get("trend", []), dtype=np.float64)
        y_detr = np.asarray(diag.get("y_detr", []), dtype=np.float64)
        wave_t = np.asarray(analysis.get("wave_t_s", []), dtype=np.float64)
        wave_y_arc = np.asarray(analysis.get("wave_y_arcsec", []), dtype=np.float64)
        wave_y_detr = np.asarray(analysis.get("wave_y_detr_arcsec", []), dtype=np.float64)
        wave_model = np.asarray(analysis.get("wave_model_arcsec", []), dtype=np.float64)
        model_detr = np.array([], dtype=np.float64)
        if wave_model.size == wave_y_arc.size == wave_y_detr.size and wave_model.size > 0:
            model_detr = wave_model - (wave_y_arc - wave_y_detr)

        existing["wavelet_selected_var"].set(
            self._format_wavelet_segment_physics(
                analysis,
                f"Event {int(selected_event.get('event_id', -1))} [{self._td_window_wavelet_event_status(selected_event)}]:",
            )
            + f" | conf={confidence_score:.1f}/100 ({confidence_label})"
            + f" | lock={'yes' if bool(selected_event.get('review_locked')) else 'no'}"
            + f" | links={max(len(link_refs) - 1, 0)}"
            + (f" | linked_to={','.join(linked_targets[:4])}" if linked_targets else "")
            + (f" | note={note_text}" if note_text else "")
        )
        existing["wavelet_diag_var"].set(
            f"Reason: {self._td_window_wavelet_event_reason(selected_event)} | "
            f"QA: {','.join(self._td_window_wavelet_event_qa_flags(selected_event)) or '-'} | "
            f"E/m={float(analysis.get('specific_energy_j_kg', float('nan'))):.3e} J/kg | "
            f"a={float(analysis.get('accel_amp_km_s2', float('nan'))):.3f} km/s^2 | "
            f"rho={selected_event.get('current_params', {}).get('density_kg_m3', float('nan')):.3e} kg/m^3 | "
            f"F={float(analysis.get('energy_flux_w_m2', float('nan'))):.3e} W/m^2 | "
            "diag: drag-left=trim, right-click=split"
        )

        raw_ax, detr_ax, scal_ax, spec_ax = diag_axes
        if t_seg.size and y_arc.size == t_seg.size:
            raw_ax.plot(t_seg, y_arc, color="0.3", linewidth=1.3, label="signal")
        if trend.size == t_seg.size and trend.size > 0:
            raw_ax.plot(t_seg, trend, color="tab:orange", linewidth=1.1, label="trend")
        if wave_t.size:
            raw_ax.axvspan(float(wave_t[0]), float(wave_t[-1]), color="tab:green", alpha=0.15)

        interaction = existing.get("diag_interaction")
        if interaction and interaction.get("mode") == "trim":
            left = min(
                float(interaction.get("start", 0.0)),
                float(interaction.get("current", interaction.get("start", 0.0))),
            )
            right = max(
                float(interaction.get("start", 0.0)),
                float(interaction.get("current", interaction.get("start", 0.0))),
            )
            raw_ax.axvspan(left, right, color="gold", alpha=0.18)
        raw_ax.set_title("Signal + Trend", fontsize=9)
        raw_ax.set_xlabel("time [s]")
        raw_ax.set_ylabel("disp. [arcsec]")
        raw_ax.legend(loc="best", fontsize=8)

        if t_seg.size and y_detr.size == t_seg.size:
            detr_ax.plot(t_seg, y_detr, color="0.4", linewidth=1.1, label="detrended")
        if wave_t.size and wave_y_detr.size == wave_t.size:
            detr_ax.plot(wave_t, wave_y_detr, color="tab:blue", linewidth=1.5, label="selected")
        if wave_t.size and model_detr.size == wave_t.size:
            detr_ax.plot(wave_t, model_detr, color="tab:red", linewidth=1.2, label="sine fit")
        if interaction and interaction.get("mode") == "trim":
            detr_ax.axvspan(left, right, color="gold", alpha=0.18)

        split_markers: list[float] = []
        split_text = str(existing["wavelet_split_frames_var"].get()).strip()
        if split_text:
            try:
                split_markers = sorted(
                    {
                        float(token.strip())
                        for token in split_text.split(",")
                        if token.strip()
                    }
                )
            except Exception:
                split_markers = []
        for split_marker in split_markers:
            raw_ax.axvline(split_marker, color="tab:purple", linestyle=":", linewidth=1.0)
            detr_ax.axvline(split_marker, color="tab:purple", linestyle=":", linewidth=1.0)
        detr_ax.set_title("Detrended Segment", fontsize=9)
        detr_ax.set_xlabel("time [s]")
        detr_ax.set_ylabel("disp. [arcsec]")
        detr_ax.legend(loc="best", fontsize=8)

        periods = np.asarray(diag.get("periods", []), dtype=np.float64)
        power = np.asarray(diag.get("power", []), dtype=np.float64)
        if power.ndim == 2 and periods.size == power.shape[0] and t_seg.size == power.shape[1]:
            scal_ax.imshow(
                power,
                aspect="auto",
                origin="lower",
                extent=[float(t_seg[0]), float(t_seg[-1]), float(periods[0]), float(periods[-1])],
                cmap="viridis",
            )
            if np.isfinite(float(diag.get("peak_period", float("nan")))):
                scal_ax.axhline(float(diag.get("peak_period")), color="w", linestyle="--", linewidth=1.0)
            if wave_t.size:
                scal_ax.axvspan(float(wave_t[0]), float(wave_t[-1]), color="tab:red", alpha=0.12)
        else:
            scal_ax.text(0.5, 0.5, "No scalogram available.", ha="center", va="center", transform=scal_ax.transAxes)
        scal_ax.set_title("Wavelet Power", fontsize=9)
        scal_ax.set_xlabel("time [s]")
        scal_ax.set_ylabel("period [s]")

        global_ws = np.asarray(diag.get("global_ws", []), dtype=np.float64)
        if periods.size and global_ws.size == periods.size:
            spec_ax.plot(periods, global_ws, color="tab:purple", linewidth=1.2)
            if np.isfinite(float(diag.get("peak_period", float("nan")))):
                spec_ax.axvline(float(diag.get("peak_period")), color="tab:red", linestyle="--", linewidth=1.0)
        else:
            spec_ax.text(0.5, 0.5, "No global spectrum.", ha="center", va="center", transform=spec_ax.transAxes)
        spec_ax.set_title("Global Spectrum", fontsize=9)
        spec_ax.set_xlabel("period [s]")
        spec_ax.set_ylabel("power")

        diag_fig.tight_layout()
        diag_canvas.draw_idle()

    def _refresh_td_window_wavelet_views(
        self, panel_id: int, *, redraw_td: bool = True
    ) -> None:
        _trace_stack_wavelet(
            f"refresh_td_window_wavelet_views start panel={panel_id} redraw_td={bool(redraw_td)}"
        )
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        self._refresh_td_window_wavelet_table(panel_id)
        events = self._td_window_wavelet_events(panel_id)
        _job_id, running_job = self._background_wavelet_job_for_panel(panel_id)
        if running_job is not None:
            existing["wavelet_summary_var"].set(
                str(running_job.get("message", "Running wavelet filter in background..."))
            )
        elif events:
            accepted_count = sum(1 for event in events if (event.get("analysis") or {}).get("accepted"))
            with_segment_count = sum(
                1 for event in events if (event.get("analysis") or {}).get("has_segment")
            )
            existing["wavelet_summary_var"].set(
                f"Wavelet accepted {accepted_count}/{len(events)} segment(s). "
                f"Selected by wavelet: {with_segment_count}."
            )
        elif existing.get("wavelet_filter_result"):
            existing["wavelet_summary_var"].set("Wavelet filter produced no events.")
        else:
            existing["wavelet_summary_var"].set("No wavelet filter results.")
        counted_events = [
            event for event in events if self._td_window_wavelet_event_is_counted(event)
        ]
        if counted_events:
            best_event = max(
                counted_events,
                key=lambda event: (
                    float((event.get("analysis") or {}).get("power_ratio", float("-inf"))),
                    float((event.get("analysis") or {}).get("duration_s", float("-inf"))),
                ),
            )
            existing["wavelet_physics_var"].set(
                self._format_wavelet_segment_physics(
                    best_event.get("analysis") or {},
                    f"Best counted [{self._td_window_wavelet_event_status(best_event)}]:",
                )
            )
        else:
            best_segment = self._best_wavelet_segment(
                [event.get("analysis") or {} for event in events]
            )
            existing["wavelet_physics_var"].set(
                self._format_wavelet_segment_physics(best_segment, "Best candidate:")
                if best_segment
                else ""
            )
        self._refresh_td_window_wavelet_diagnostics(panel_id)
        if redraw_td:
            _trace_stack_wavelet(f"refresh_td_window_wavelet_views before_refresh_td_window panel={panel_id}")
            self._refresh_td_window(panel_id)
        _trace_stack_wavelet(f"refresh_td_window_wavelet_views end panel={panel_id}")

    def _on_td_window_wavelet_filter_mode_change(self, panel_id: int) -> None:
        self._sync_panel_analysis_state_from_window(panel_id)
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=False)

    def _on_td_window_wavelet_event_select(self, panel_id: int, _event: Any = None) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        if existing.get("wavelet_table_updating"):
            return
        tree = existing.get("wavelet_events_tree")
        if tree is None:
            return
        previous_event_id = existing.get("wavelet_selected_event_id")
        selection = tree.selection()
        if not selection:
            existing["wavelet_selected_event_id"] = None
        else:
            iid = str(selection[0])
            if iid.startswith("evt-"):
                try:
                    existing["wavelet_selected_event_id"] = int(iid.split("-", 1)[1])
                except Exception:
                    existing["wavelet_selected_event_id"] = None
        if existing.get("wavelet_selected_event_id") == previous_event_id:
            return
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)

    def _recompute_td_window_selected_wavelet_event(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        selected_event = self._selected_td_window_wavelet_event_editable(
            panel_id, "recompute"
        )
        if selected_event is None:
            return
        if selected_event.get("split_children_ids"):
            self._set_status(f"Reset the split parent before recomputing it in {panel.name}.")
            return
        params = self._parse_td_window_wavelet_filter_params(panel_id)
        if params is None:
            return
        diagnostic = self._analyze_td_window_wavelet_event_source(
            panel_id,
            np.asarray(selected_event.get("source_t_idx", []), dtype=np.float64),
            np.asarray(selected_event.get("source_y_idx", []), dtype=np.float64),
            params,
        )
        if diagnostic is None:
            return
        self._push_wavelet_undo_state(panel_id, "recompute event")
        self._set_td_window_wavelet_event_analysis(
            selected_event, diagnostic, customized=True
        )
        self._append_wavelet_event_history(
            selected_event,
            "recompute",
            details=f"panel={panel.name}",
        )
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Recomputed selected wavelet event for {panel.name}.")

    def _accept_td_window_selected_wavelet_event(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        selected_event = self._selected_td_window_wavelet_event_editable(
            panel_id, "accept"
        )
        if selected_event is None:
            return
        wave_t_idx = np.asarray(
            (selected_event.get("analysis") or {}).get("wave_t_idx", []),
            dtype=np.float64,
        )
        if wave_t_idx.size < 2:
            self._set_status(
                f"The selected event has no valid wave segment to accept in {panel.name}."
            )
            return
        self._push_wavelet_undo_state(panel_id, "accept event")
        selected_event["manual_decision"] = "accepted"
        self._append_wavelet_event_history(selected_event, "accept")
        self._wavelet_event_confidence_details(selected_event)
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Marked selected wavelet event as accepted for {panel.name}.")

    def _reject_td_window_selected_wavelet_event(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        selected_event = self._selected_td_window_wavelet_event_editable(
            panel_id, "reject"
        )
        if selected_event is None:
            return
        self._push_wavelet_undo_state(panel_id, "reject event")
        selected_event["manual_decision"] = "rejected"
        self._append_wavelet_event_history(selected_event, "reject")
        self._wavelet_event_confidence_details(selected_event)
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Marked selected wavelet event as rejected for {panel.name}.")

    def _reset_td_window_selected_wavelet_event(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        selected_event = self._selected_td_window_wavelet_event_editable(
            panel_id, "reset"
        )
        if selected_event is None:
            return
        self._push_wavelet_undo_state(panel_id, "reset event")
        if selected_event.get("split_children_ids"):
            child_ids = set(int(child_id) for child_id in selected_event["split_children_ids"])
            existing["wavelet_events"] = [
                event
                for event in self._td_window_wavelet_events(panel_id)
                if int(event.get("event_id", -1)) not in child_ids
            ]
            selected_event["split_children_ids"] = []
        selected_event["source_t_idx"] = np.asarray(
            selected_event.get("base_source_t_idx", []), dtype=np.float64
        ).copy()
        selected_event["source_y_idx"] = np.asarray(
            selected_event.get("base_source_y_idx", []), dtype=np.float64
        ).copy()
        selected_event["analysis"] = self._clone_wavelet_payload(
            selected_event.get("base_analysis") or {}
        )
        selected_event["current_params"] = dict(
            selected_event.get("base_params") or {}
        )
        selected_event["manual_decision"] = None
        selected_event["customized"] = False
        selected_event["diagnostic"] = None
        self._append_wavelet_event_history(selected_event, "reset")
        self._wavelet_event_confidence_details(selected_event)
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Reset selected wavelet event for {panel.name}.")

    def _trim_td_window_selected_wavelet_event(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        selected_event = self._selected_td_window_wavelet_event_editable(
            panel_id, "trim"
        )
        if selected_event is None:
            return
        if selected_event.get("split_children_ids"):
            self._set_status(f"Reset the split parent before trimming it in {panel.name}.")
            return
        try:
            trim_start_text = str(existing["wavelet_trim_start_var"].get()).strip()
            trim_end_text = str(existing["wavelet_trim_end_var"].get()).strip()
            trim_start = float(trim_start_text) if trim_start_text else float("nan")
            trim_end = float(trim_end_text) if trim_end_text else float("nan")
        except Exception:
            self._set_status(f"Invalid trim range for {panel.name}.")
            return
        source_t_idx = np.asarray(selected_event.get("source_t_idx", []), dtype=np.float64)
        source_y_idx = np.asarray(selected_event.get("source_y_idx", []), dtype=np.float64)
        if source_t_idx.size < 3 or source_y_idx.size != source_t_idx.size:
            self._set_status(f"Selected event has no valid source segment in {panel.name}.")
            return
        if not np.isfinite(trim_start):
            trim_start = float(source_t_idx[0])
        if not np.isfinite(trim_end):
            trim_end = float(source_t_idx[-1])
        if trim_end <= trim_start:
            self._set_status(f"Trim end must be larger than trim start for {panel.name}.")
            return
        mask = (source_t_idx >= trim_start) & (source_t_idx <= trim_end)
        if np.count_nonzero(mask) < 3:
            self._set_status(f"Trimmed segment is too short for {panel.name}.")
            return
        params = self._parse_td_window_wavelet_filter_params(panel_id)
        if params is None:
            return
        diagnostic = self._analyze_td_window_wavelet_event_source(
            panel_id,
            source_t_idx[mask],
            source_y_idx[mask],
            params,
        )
        if diagnostic is None:
            return
        self._push_wavelet_undo_state(panel_id, "trim event")
        selected_event["source_t_idx"] = np.asarray(source_t_idx[mask], dtype=np.float64).copy()
        selected_event["source_y_idx"] = np.asarray(source_y_idx[mask], dtype=np.float64).copy()
        selected_event["origin"] = "manual-trim"
        self._set_td_window_wavelet_event_analysis(
            selected_event, diagnostic, customized=True
        )
        self._append_wavelet_event_history(
            selected_event,
            "trim",
            details=f"{float(source_t_idx[mask][0]):.2f}-{float(source_t_idx[mask][-1]):.2f}",
        )
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Trimmed selected wavelet event for {panel.name}.")

    def _split_td_window_selected_wavelet_event(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        selected_event = self._selected_td_window_wavelet_event_editable(
            panel_id, "split"
        )
        if selected_event is None:
            return
        if selected_event.get("split_children_ids"):
            self._set_status(f"Selected event is already split in {panel.name}.")
            return
        split_text = str(existing["wavelet_split_frames_var"].get()).strip()
        if not split_text:
            self._set_status(f"Enter one or more split frame indices for {panel.name}.")
            return
        try:
            split_frames = sorted(
                {float(token.strip()) for token in split_text.split(",") if token.strip()}
            )
        except Exception:
            self._set_status(f"Invalid split frame list for {panel.name}.")
            return
        source_t_idx = np.asarray(selected_event.get("source_t_idx", []), dtype=np.float64)
        source_y_idx = np.asarray(selected_event.get("source_y_idx", []), dtype=np.float64)
        if source_t_idx.size < 6 or source_y_idx.size != source_t_idx.size:
            self._set_status(f"Selected event is too short to split in {panel.name}.")
            return
        valid_splits = [
            value
            for value in split_frames
            if float(source_t_idx[0]) < value < float(source_t_idx[-1])
        ]
        if not valid_splits:
            self._set_status(f"No split points fall inside the selected event in {panel.name}.")
            return
        params = self._parse_td_window_wavelet_filter_params(panel_id)
        if params is None:
            return
        self._push_wavelet_undo_state(panel_id, "split event")
        boundaries = [float(source_t_idx[0])] + valid_splits + [float(source_t_idx[-1]) + 1e-9]
        next_event_id = int(existing.get("wavelet_next_event_id", 1))
        child_ids: list[int] = []
        new_events: list[dict[str, Any]] = []
        for idx in range(len(boundaries) - 1):
            left = boundaries[idx]
            right = boundaries[idx + 1]
            if idx == len(boundaries) - 2:
                mask = (source_t_idx >= left) & (source_t_idx <= right)
            else:
                mask = (source_t_idx >= left) & (source_t_idx < right)
            if np.count_nonzero(mask) < 3:
                continue
            diagnostic = self._analyze_td_window_wavelet_event_source(
                panel_id,
                source_t_idx[mask],
                source_y_idx[mask],
                params,
            )
            if diagnostic is None:
                continue
            child_event = self._make_td_window_wavelet_event(
                next_event_id,
                diagnostic["selected"],
                params,
                origin="manual-split",
                parent_event_id=int(selected_event.get("event_id", -1)),
            )
            child_event["source_t_idx"] = np.asarray(source_t_idx[mask], dtype=np.float64).copy()
            child_event["source_y_idx"] = np.asarray(source_y_idx[mask], dtype=np.float64).copy()
            child_event["base_source_t_idx"] = child_event["source_t_idx"].copy()
            child_event["base_source_y_idx"] = child_event["source_y_idx"].copy()
            parent_analysis = selected_event.get("analysis") or {}
            child_event["analysis"]["thread_index"] = parent_analysis.get("thread_index", -1)
            child_event["analysis"]["seg_id"] = parent_analysis.get("seg_id", -1)
            child_event["base_analysis"]["thread_index"] = parent_analysis.get("thread_index", -1)
            child_event["base_analysis"]["seg_id"] = parent_analysis.get("seg_id", -1)
            child_event["diagnostic"] = self._clone_wavelet_payload(diagnostic)
            self._append_wavelet_event_history(
                child_event,
                "split-child",
                details=f"parent={int(selected_event.get('event_id', -1))}",
            )
            child_ids.append(next_event_id)
            new_events.append(child_event)
            next_event_id += 1
        if not new_events:
            self._set_status(f"Could not create split child events for {panel.name}.")
            return
        selected_event["split_children_ids"] = child_ids
        self._append_wavelet_event_history(
            selected_event,
            "split",
            details=",".join(f"{value:.2f}" for value in valid_splits),
        )
        existing["wavelet_events"].extend(new_events)
        existing["wavelet_next_event_id"] = next_event_id
        existing["wavelet_selected_event_id"] = child_ids[0]
        self._record_session_change()
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=True)
        self._set_status(f"Split selected wavelet event into {len(new_events)} child event(s) for {panel.name}.")

    def _run_td_window_wavelet_filter(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        _trace_stack_wavelet(f"run_td_window_wavelet_filter start panel={panel_id} panel_name={panel.name}")

        _job_id, running_job = self._background_wavelet_job_for_panel(panel_id)
        if running_job is not None:
            self._set_status(f"Wavelet filter is already running for {panel.name}.")
            self._update_wavelet_job_widgets(panel_id)
            return

        tracking_result = existing.get("crest_tracking_result")
        if not tracking_result:
            existing["wavelet_summary_var"].set(
                "Run crest tracking first to use the wavelet filter."
            )
            existing["wavelet_physics_var"].set("")
            self._set_status(f"Run crest tracking before wavelet filtering {panel.name}.")
            return

        params = self._parse_td_window_wavelet_filter_params(panel_id)
        if params is None:
            return

        td, meta = self._panel_td(panel)
        if td is None or meta is None:
            existing["wavelet_summary_var"].set(
                "No valid TD map available for wavelet filtering."
            )
            existing["wavelet_physics_var"].set("")
            self._set_status(f"{panel.name} has no valid TD map for wavelet filtering.")
            return

        threads = tracking_result.get("threads") or []
        if not threads:
            existing["wavelet_summary_var"].set(
                "No tracked threads available for wavelet filtering."
            )
            existing["wavelet_physics_var"].set("")
            self._set_status(f"{panel.name} has no tracked threads for wavelet filtering.")
            return
        thread_filter_text = str(existing["wavelet_thread_filter_var"].get() or "").strip()
        selected_thread_indices, filter_error = self._parse_wavelet_thread_filter_text(
            thread_filter_text,
            max_threads=len(threads),
        )
        if filter_error is not None:
            existing["wavelet_summary_var"].set(filter_error)
            existing["wavelet_physics_var"].set("")
            self._set_status(f"{panel.name}: {filter_error}")
            return
        thread_entries = [
            {"thread_index": int(idx), "thread": thread}
            for idx, thread in enumerate(threads)
            if selected_thread_indices is None or int(idx) in selected_thread_indices
        ]
        if not thread_entries:
            existing["wavelet_summary_var"].set("Wavelet thread filter matched no tracked threads.")
            existing["wavelet_physics_var"].set("")
            self._set_status(f"{panel.name}: wavelet thread filter matched no tracked threads.")
            return

        job_id, job = self._new_background_job(
            kind="wavelet", panel_id=panel_id, panel_name=panel.name
        )
        filter_label = (
            self._format_wavelet_thread_filter_text(selected_thread_indices)
            if selected_thread_indices
            else "all"
        )
        existing["wavelet_summary_var"].set(
            f"Running wavelet filter in background... threads {filter_label}"
        )
        existing["wavelet_physics_var"].set("")
        job["stage"] = "starting"
        job["current"] = 0
        job["total"] = max(len(thread_entries), 1)
        job["message"] = (
            f"{panel.name}: queued wavelet filter "
            f"({len(thread_entries)} thread(s), filter={filter_label})"
        )
        thread = threading.Thread(
            target=self._wavelet_worker,
            args=(
                job_id,
                panel_id,
                panel.name,
                self._clone_wavelet_payload(thread_entries),
                np.asarray(meta["t_indices"], dtype=np.float64).copy(),
                dict(params),
                job["cancel_event"],
            ),
            daemon=True,
        )
        job["thread"] = thread
        thread.start()
        self._panel_analysis(panel_id)["wavelet_thread_filter_text"] = thread_filter_text
        self._update_wavelet_job_widgets(panel_id)
        self._set_status(
            f"Running wavelet filter in background for {panel.name} (threads {filter_label})..."
        )

    def _draw_td_window_crest_overlay(
        self, ax: Any, panel_id: int, meta: dict[str, Any] | None
    ) -> None:
        if meta is None:
            return

        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        tracking_result = existing.get("crest_tracking_result")
        if not tracking_result:
            return

        threads = tracking_result.get("threads") or []
        if not threads:
            return

        distances = np.asarray(meta["distances"], dtype=np.float64)
        t_indices = np.asarray(meta["t_indices"], dtype=np.float64)
        if distances.size == 0 or t_indices.size == 0:
            return

        dist_index = np.arange(distances.size, dtype=np.float64)
        dist_hi = dist_index[-1]

        for idx, thread in enumerate(threads):
            pos = np.asarray(thread.get("pos", []), dtype=np.float64)
            if pos.size != t_indices.size:
                continue
            mask = np.isfinite(pos) & (pos >= 0.0) & (pos <= dist_hi + 1e-9)
            if np.count_nonzero(mask) < 2:
                continue
            dist_vals = np.interp(pos[mask], dist_index, distances)
            time_vals = t_indices[mask]
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            if self.td_swap_axes_var.get():
                ax.plot(time_vals, dist_vals, color=color, linewidth=1.5, alpha=0.85)
            else:
                ax.plot(dist_vals, time_vals, color=color, linewidth=1.5, alpha=0.85)

        events = existing.get("wavelet_events")
        if events:
            selected_event_id = existing.get("wavelet_selected_event_id")
            for event in events:
                if not self._td_window_wavelet_event_is_counted(event):
                    continue
                analysis = event.get("analysis") or {}
                wave_t_idx = np.asarray(analysis.get("wave_t_idx", []), dtype=np.float64)
                wave_y_idx = np.asarray(analysis.get("wave_y_idx", []), dtype=np.float64)
                if wave_t_idx.size < 2 or wave_y_idx.size != wave_t_idx.size:
                    continue
                mask = np.isfinite(wave_y_idx) & (wave_y_idx >= 0.0) & (wave_y_idx <= dist_hi + 1e-9)
                if np.count_nonzero(mask) < 2:
                    continue
                dist_vals = np.interp(wave_y_idx[mask], dist_index, distances)
                time_vals = wave_t_idx[mask]
                status = self._td_window_wavelet_event_status(event)
                color = "lime"
                if status == "manual accepted":
                    color = "deepskyblue"
                elif status == "custom accepted":
                    color = "cyan"
                linewidth = 2.6
                alpha = 0.95
                if int(event.get("event_id", -1)) == int(selected_event_id or -1):
                    color = "gold"
                    linewidth = 3.0
                    alpha = 1.0
                if self.td_swap_axes_var.get():
                    ax.plot(time_vals, dist_vals, color=color, linewidth=linewidth, alpha=alpha)
                else:
                    ax.plot(dist_vals, time_vals, color=color, linewidth=linewidth, alpha=alpha)
            return

        wavelet_result = existing.get("wavelet_filter_result")
        if not wavelet_result:
            return

        for segment in wavelet_result.get("segments", []):
            if not segment.get("accepted"):
                continue
            wave_t_idx = np.asarray(segment.get("wave_t_idx", []), dtype=np.float64)
            wave_y_idx = np.asarray(segment.get("wave_y_idx", []), dtype=np.float64)
            if wave_t_idx.size < 2 or wave_y_idx.size != wave_t_idx.size:
                continue
            mask = np.isfinite(wave_y_idx) & (wave_y_idx >= 0.0) & (wave_y_idx <= dist_hi + 1e-9)
            if np.count_nonzero(mask) < 2:
                continue
            dist_vals = np.interp(wave_y_idx[mask], dist_index, distances)
            time_vals = wave_t_idx[mask]
            if self.td_swap_axes_var.get():
                ax.plot(time_vals, dist_vals, color="lime", linewidth=2.6, alpha=0.95)
            else:
                ax.plot(dist_vals, time_vals, color="lime", linewidth=2.6, alpha=0.95)

    def _toggle_td_window_editor(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        is_open = not bool(existing["edit_var"].get())
        existing["edit_var"].set(is_open)

        if is_open:
            existing["edit_frame"].grid()
            existing["editor_toggle_var"].set("Hide editor")
            self._set_status(
                f"Detached TD controls enabled for {self.panels[panel_id - 1].name}."
            )
        else:
            existing["edit_frame"].grid_remove()
            existing["editor_toggle_var"].set("Show editor")
            self._set_status(
                f"Detached TD controls hidden for {self.panels[panel_id - 1].name}."
            )

    def _on_td_window_display_mode_change(self, panel_id: int) -> None:
        self._refresh_td_window(panel_id)

    def _sync_td_window_controls(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return

        cut = self.cuts.get(panel.cut_id) if panel.cut_id is not None else None
        params = self._panel_td_params(panel)
        existing["control_guard"] = True
        existing["panel_info_var"].set(
            f"{panel.name} | no cut"
            if cut is None
            else f"{panel.name} | {cut.name} | "
            f"t={int(params['t_ini'])}:{int(params['t_fin'])}:{int(params['stride'])} | "
            f"w={int(params['width'])} {params['weighting']}"
        )
        existing["panel_t_ini_var"].set(str(int(params["t_ini"])))
        existing["panel_t_fin_var"].set(str(int(params["t_fin"])))
        existing["panel_stride_var"].set(str(int(params["stride"])))
        existing["panel_width_var"].set(str(int(params["width"])))
        existing["panel_weighting_var"].set(str(params["weighting"]))

        if cut is None:
            existing["cut_info_var"].set("No cut assigned to this panel.")
            existing["cut_angle_var"].set("")
            existing["cut_length_var"].set("")
            existing["center_x_var"].set("")
            existing["center_y_var"].set("")
            existing["cut_x1_var"].set("")
            existing["cut_y1_var"].set("")
            existing["cut_x2_var"].set("")
            existing["cut_y2_var"].set("")
        else:
            preview_cut = self._cut_preview(cut.cut_id) or cut
            center = cut_center(preview_cut)
            dynamic_label = " | dynamic" if self._cut_dynamic_enabled(cut.cut_id) else ""
            existing["cut_info_var"].set(
                f"{cut.name}{dynamic_label} | angle={cut_display_angle_deg(preview_cut):.2f} deg | "
                f"length={cut_length(preview_cut):.2f} px"
            )
            existing["cut_angle_var"].set(f"{cut_display_angle_deg(preview_cut):.2f}")
            existing["cut_length_var"].set(f"{cut_length(preview_cut):.2f}")
            existing["center_x_var"].set(f"{center[0]:.2f}")
            existing["center_y_var"].set(f"{center[1]:.2f}")
            existing["cut_x1_var"].set(f"{preview_cut.p0[0]:.2f}")
            existing["cut_y1_var"].set(f"{preview_cut.p0[1]:.2f}")
            existing["cut_x2_var"].set(f"{preview_cut.p1[0]:.2f}")
            existing["cut_y2_var"].set(f"{preview_cut.p1[1]:.2f}")
        existing["control_guard"] = False

    def _apply_td_window_panel_controls(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None or existing.get("control_guard"):
            return

        try:
            t_ini = int(float(str(existing["panel_t_ini_var"].get()).strip()))
            t_fin = int(float(str(existing["panel_t_fin_var"].get()).strip()))
            stride = int(float(str(existing["panel_stride_var"].get()).strip()))
            width = int(float(str(existing["panel_width_var"].get()).strip()))
        except Exception:
            self._set_status("Invalid detached TD control value.")
            return

        if t_ini > t_fin:
            t_fin = t_ini
            existing["control_guard"] = True
            existing["panel_t_fin_var"].set(str(t_fin))
            existing["control_guard"] = False

        panel.t_ini = clamp_int(t_ini, 0, self.nt - 1)
        panel.t_fin = clamp_int(t_fin, panel.t_ini, self.nt - 1)
        panel.stride = max(stride, 1)
        panel.width = max(width, 1)
        panel.weighting = str(existing["panel_weighting_var"].get() or "uniform")
        if panel.cut_id is not None and panel.cut_id in self.cuts:
            state = self._cut_analysis(panel.cut_id)
            state["td_params"] = {
                "t_ini": int(panel.t_ini),
                "t_fin": int(panel.t_fin),
                "stride": int(panel.stride),
                "width": int(panel.width),
                "weighting": str(panel.weighting),
            }
            state["td_cache_key"] = None
            state["td_cache_td"] = None
            state["td_cache_meta"] = None
            self._sync_panels_from_cut_td_params(panel.cut_id)
            self._invalidate_cut_dependents(panel.cut_id)
        else:
            self._invalidate_panel_cache(panel.panel_id)
        self._record_session_change()
        self.refresh_all()
        self._set_status(f"Updated detached TD controls for {panel.name}.")

    def _td_window_time_range(self, meta: dict[str, Any]) -> tuple[float, float]:
        t_indices = meta["t_indices"]
        if len(t_indices) == 1:
            return float(t_indices[0]), float(t_indices[0]) + 1.0
        return float(t_indices[0]), float(t_indices[-1])

    def _ensure_td_window_roi(self, panel_id: int, meta: dict[str, Any]) -> tuple[float, float, float, float]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return 0.0, 1.0, 0.0, 1.0

        t_min, t_max = self._td_window_time_range(meta)
        dist_max = float(meta["distances"][-1])
        total_t = max(t_max - t_min, 1.0)
        total_d = max(dist_max, 1.0)

        try:
            t_span = float(str(existing["roi_t_span_var"].get()).strip())
        except Exception:
            t_span = max(5.0, 0.25 * total_t)
        try:
            d_span = float(str(existing["roi_d_span_var"].get()).strip())
        except Exception:
            d_span = max(2.0, 0.35 * total_d)

        t_span = clamp_value(t_span, min(1.0, total_t), total_t)
        d_span = clamp_value(d_span, min(1.0, total_d), total_d)

        if existing.get("roi_center_t") is None:
            current_t = float(self.t_visual_var.get())
            existing["roi_center_t"] = (
                current_t if t_min <= current_t <= t_max else 0.5 * (t_min + t_max)
            )
        if existing.get("roi_center_d") is None:
            existing["roi_center_d"] = 0.5 * dist_max

        roi_center_t = float(existing["roi_center_t"])
        roi_center_d = float(existing["roi_center_d"])

        if total_t <= t_span + 1e-9:
            roi_center_t = 0.5 * (t_min + t_max)
        else:
            roi_center_t = clamp_value(roi_center_t, t_min + 0.5 * t_span, t_max - 0.5 * t_span)
        if total_d <= d_span + 1e-9:
            roi_center_d = 0.5 * dist_max
        else:
            roi_center_d = clamp_value(roi_center_d, 0.5 * d_span, dist_max - 0.5 * d_span)

        existing["roi_center_t"] = roi_center_t
        existing["roi_center_d"] = roi_center_d
        existing["control_guard"] = True
        existing["roi_t_span_var"].set(f"{t_span:.2f}")
        existing["roi_d_span_var"].set(f"{d_span:.2f}")
        existing["control_guard"] = False

        return (
            roi_center_t - 0.5 * t_span,
            roi_center_t + 0.5 * t_span,
            roi_center_d - 0.5 * d_span,
            roi_center_d + 0.5 * d_span,
        )

    def _on_td_window_roi_toggle(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        existing["roi_dragging"] = False
        self._record_session_change()
        self._refresh_td_window(panel_id)
        if existing["roi_enabled_var"].get():
            self._set_status(f"ROI zoom enabled for {self.panels[panel_id - 1].name}.")
        else:
            self._set_status(f"ROI zoom hidden for {self.panels[panel_id - 1].name}.")

    def _apply_td_window_roi_settings(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None or existing.get("control_guard"):
            return
        td, meta = self._panel_td(panel)
        if td is None or meta is None:
            self._set_status(f"{panel.name} has no valid TD for ROI zoom.")
            return
        self._ensure_td_window_roi(panel_id, meta)
        self._record_session_change()
        self._refresh_td_window(panel_id)
        self._set_status(f"Updated ROI box for {panel.name}.")

    def _open_td_window_roi_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        top = existing.get("top")
        if top is None or not top.winfo_exists():
            return

        roi_top = existing.get("roi_top")
        if roi_top is not None and roi_top.winfo_exists():
            roi_top.deiconify()
            roi_top.lift()
            self._refresh_td_window(panel_id)
            return

        roi_top = self.tk.Toplevel(top)
        roi_top.title(f"ROI TD - {panel.name}")
        roi_top.geometry("980x420")
        roi_top.rowconfigure(0, weight=1)
        roi_top.columnconfigure(0, weight=1)
        frame = self.ttk.Frame(roi_top, padding=8)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        roi_fig = self.Figure(figsize=(8.6, 3.2), dpi=120)
        roi_ax = roi_fig.add_subplot(111)
        roi_canvas = self.FigureCanvasTkAgg(roi_fig, master=frame)
        roi_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        existing["roi_top"] = roi_top
        existing["roi_figure"] = roi_fig
        existing["roi_ax"] = roi_ax
        existing["roi_canvas"] = roi_canvas
        existing["roi_window_toggle_var"].set("Close ROI window")
        roi_top.protocol(
            "WM_DELETE_WINDOW",
            lambda pid=panel_id: self._close_td_window_roi_window(pid),
        )
        self._refresh_td_window(panel_id)

    def _close_td_window_roi_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        roi_top = existing.get("roi_top")
        if roi_top is not None and roi_top.winfo_exists():
            roi_top.destroy()
        existing["roi_top"] = None
        existing["roi_figure"] = None
        existing["roi_ax"] = None
        existing["roi_canvas"] = None
        toggle_var = existing.get("roi_window_toggle_var")
        if toggle_var is not None:
            toggle_var.set("ROI window")

    def _toggle_td_window_roi_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        roi_top = existing.get("roi_top")
        if roi_top is not None and roi_top.winfo_exists():
            self._close_td_window_roi_window(panel_id)
        else:
            self._open_td_window_roi_window(panel_id)

    def _open_td_window_diag_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None:
            return
        top = existing.get("top")
        if top is None or not top.winfo_exists():
            return

        diag_top = existing.get("diag_top")
        if diag_top is not None and diag_top.winfo_exists():
            diag_top.deiconify()
            diag_top.lift()
            self._refresh_td_window_wavelet_diagnostics(panel_id)
            return

        diag_top = self.tk.Toplevel(top)
        diag_top.title(f"Wavelet Diagnostic - {panel.name}")
        diag_top.geometry("1080x760")
        diag_top.rowconfigure(0, weight=1)
        diag_top.columnconfigure(0, weight=1)
        frame = self.ttk.Frame(diag_top, padding=8)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        diag_fig = self.Figure(figsize=(9.2, 6.4), dpi=120)
        diag_axes = tuple(np.ravel(diag_fig.subplots(2, 2)))
        diag_canvas = self.FigureCanvasTkAgg(diag_fig, master=frame)
        diag_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        diag_canvas.mpl_connect(
            "button_press_event",
            lambda event, pid=panel_id: self._on_td_window_diag_press(pid, event),
        )
        diag_canvas.mpl_connect(
            "motion_notify_event",
            lambda event, pid=panel_id: self._on_td_window_diag_motion(pid, event),
        )
        diag_canvas.mpl_connect(
            "button_release_event",
            lambda event, pid=panel_id: self._on_td_window_diag_release(pid, event),
        )

        existing["diag_top"] = diag_top
        existing["diag_figure"] = diag_fig
        existing["diag_axes"] = diag_axes
        existing["diag_canvas"] = diag_canvas
        existing["diag_window_toggle_var"].set("Close Wavelet diag")
        diag_top.protocol(
            "WM_DELETE_WINDOW",
            lambda pid=panel_id: self._close_td_window_diag_window(pid),
        )
        self._refresh_td_window_wavelet_diagnostics(panel_id)

    def _close_td_window_diag_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        diag_top = existing.get("diag_top")
        if diag_top is not None and diag_top.winfo_exists():
            diag_top.destroy()
        existing["diag_top"] = None
        existing["diag_figure"] = None
        existing["diag_axes"] = None
        existing["diag_canvas"] = None
        toggle_var = existing.get("diag_window_toggle_var")
        if toggle_var is not None:
            toggle_var.set("Wavelet diag window")

    def _toggle_td_window_diag_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        diag_top = existing.get("diag_top")
        if diag_top is not None and diag_top.winfo_exists():
            self._close_td_window_diag_window(panel_id)
        else:
            self._open_td_window_diag_window(panel_id)

    def _diag_edit_axes(self, panel_id: int) -> tuple[Any, Any] | tuple[None, None]:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return None, None
        diag_axes = existing.get("diag_axes")
        if not isinstance(diag_axes, tuple) or len(diag_axes) < 2:
            return None, None
        return diag_axes[0], diag_axes[1]

    def _on_td_window_diag_press(self, panel_id: int, event: Any) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if existing is None or panel is None or event.xdata is None:
            return
        raw_ax, detr_ax = self._diag_edit_axes(panel_id)
        if event.inaxes not in {raw_ax, detr_ax}:
            return
        selected_event = self._td_window_wavelet_selected_event(panel_id)
        if selected_event is None:
            return
        source_t_idx = np.asarray(selected_event.get("source_t_idx", []), dtype=np.float64)
        if source_t_idx.size < 3:
            return
        clicked_t = float(event.xdata)
        if clicked_t < float(source_t_idx[0]) or clicked_t > float(source_t_idx[-1]):
            return
        button = int(getattr(event, "button", 1) or 1)
        if button == 1:
            existing["diag_interaction"] = {
                "mode": "trim",
                "start": clicked_t,
                "current": clicked_t,
            }
            self._refresh_td_window_wavelet_diagnostics(panel_id)
            return
        if button == 3:
            existing["wavelet_split_frames_var"].set(f"{clicked_t:.2f}")
            self._split_td_window_selected_wavelet_event(panel_id)

    def _on_td_window_diag_motion(self, panel_id: int, event: Any) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None or event.xdata is None:
            return
        interaction = existing.get("diag_interaction")
        if not interaction or interaction.get("mode") != "trim":
            return
        raw_ax, detr_ax = self._diag_edit_axes(panel_id)
        if event.inaxes not in {raw_ax, detr_ax}:
            return
        interaction["current"] = float(event.xdata)
        self._refresh_td_window_wavelet_diagnostics(panel_id)

    def _on_td_window_diag_release(self, panel_id: int, event: Any) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        interaction = existing.get("diag_interaction")
        if not interaction or interaction.get("mode") != "trim":
            return
        end_value = float(event.xdata) if event.xdata is not None else float(interaction.get("current", interaction.get("start", 0.0)))
        start_value = float(interaction.get("start", end_value))
        existing["diag_interaction"] = None
        left = min(start_value, end_value)
        right = max(start_value, end_value)
        if right - left < 1e-6:
            self._refresh_td_window_wavelet_diagnostics(panel_id)
            return
        existing["wavelet_trim_start_var"].set(f"{left:.2f}")
        existing["wavelet_trim_end_var"].set(f"{right:.2f}")
        self._trim_td_window_selected_wavelet_event(panel_id)

    def _draw_td_window_roi(
        self, panel_id: int, td: np.ndarray | None, meta: dict[str, Any] | None
    ) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        if td is None or meta is None or not existing["roi_enabled_var"].get():
            roi_ax = existing.get("roi_ax")
            if roi_ax is not None:
                roi_ax.clear()
                roi_ax.text(
                    0.5,
                    0.5,
                    "Enable ROI box to create a second TD panel.",
                    ha="center",
                    va="center",
                    transform=roi_ax.transAxes,
                )
                roi_ax.set_xticks([])
                roi_ax.set_yticks([])
                roi_ax.set_title("ROI TD", fontsize=10)
            return

        t0, t1, d0, d1 = self._ensure_td_window_roi(panel_id, meta)
        main_ax = existing["ax"]
        if self.td_swap_axes_var.get():
            rect_x, rect_y = t0, d0
            rect_w, rect_h = t1 - t0, d1 - d0
        else:
            rect_x, rect_y = d0, t0
            rect_w, rect_h = d1 - d0, t1 - t0

        main_ax.add_patch(
            self.Rectangle(
                (rect_x, rect_y),
                rect_w,
                rect_h,
                fill=False,
                linewidth=1.6,
                linestyle="--",
                edgecolor="tab:orange",
            )
        )

        roi_ax = existing.get("roi_ax")
        if roi_ax is None:
            return
        roi_ax.clear()

        t_min, t_max = self._td_window_time_range(meta)
        dist_max = float(meta["distances"][-1])
        vmin, vmax = frame_limits(td)
        td_aspect = "equal" if self.td_aspect_var.get() == "equal" else "auto"

        if self.td_swap_axes_var.get():
            roi_ax.imshow(
                td.T,
                origin="lower",
                aspect=td_aspect,
                cmap="gray",
                extent=[t_min, t_max, 0.0, dist_max],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if self.td_flip_x_var.get():
                roi_ax.set_xlim(t1, t0)
            else:
                roi_ax.set_xlim(t0, t1)
            if self.td_flip_y_var.get():
                roi_ax.set_ylim(d1, d0)
            else:
                roi_ax.set_ylim(d0, d1)
            roi_ax.set_xlabel("time index")
            roi_ax.set_ylabel("distance [pixel]")
            self._draw_td_window_crest_overlay(roi_ax, panel_id, meta)
        else:
            roi_ax.imshow(
                td,
                origin="lower",
                aspect=td_aspect,
                cmap="gray",
                extent=[0.0, dist_max, t_min, t_max],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if self.td_flip_x_var.get():
                roi_ax.set_xlim(d1, d0)
            else:
                roi_ax.set_xlim(d0, d1)
            if self.td_flip_y_var.get():
                roi_ax.set_ylim(t1, t0)
            else:
                roi_ax.set_ylim(t0, t1)
            roi_ax.set_xlabel("distance [pixel]")
            roi_ax.set_ylabel("time index")
            self._draw_td_window_crest_overlay(roi_ax, panel_id, meta)

        roi_ax.set_title("ROI TD", fontsize=10)
        roi_ax.tick_params(labelsize=8)
        for spine in roi_ax.spines.values():
            spine.set_edgecolor("tab:orange")
            spine.set_linewidth(1.4)

    def _on_td_window_press(self, panel_id: int, event: Any) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if (
            existing is None
            or panel is None
            or not existing["roi_enabled_var"].get()
            or event.inaxes is not existing["ax"]
            or event.xdata is None
            or event.ydata is None
        ):
            return

        td, meta = self._panel_td(panel)
        if td is None or meta is None:
            return

        t0, t1, d0, d1 = self._ensure_td_window_roi(panel_id, meta)
        if self.td_swap_axes_var.get():
            clicked_t = float(event.xdata)
            clicked_d = float(event.ydata)
            inside = t0 <= clicked_t <= t1 and d0 <= clicked_d <= d1
        else:
            clicked_d = float(event.xdata)
            clicked_t = float(event.ydata)
            inside = d0 <= clicked_d <= d1 and t0 <= clicked_t <= t1

        if inside:
            existing["roi_drag_offset_t"] = clicked_t - float(existing["roi_center_t"])
            existing["roi_drag_offset_d"] = clicked_d - float(existing["roi_center_d"])
        else:
            existing["roi_center_t"] = clicked_t
            existing["roi_center_d"] = clicked_d
            existing["roi_drag_offset_t"] = 0.0
            existing["roi_drag_offset_d"] = 0.0

        existing["roi_dragging"] = True
        self._refresh_td_window(panel_id)

    def _on_td_window_motion(self, panel_id: int, event: Any) -> None:
        existing = self.td_windows.get(panel_id)
        panel = self._td_window_panel(panel_id)
        if (
            existing is None
            or panel is None
            or not existing.get("roi_dragging")
            or event.inaxes is not existing["ax"]
            or event.xdata is None
            or event.ydata is None
        ):
            return

        if self.td_swap_axes_var.get():
            clicked_t = float(event.xdata)
            clicked_d = float(event.ydata)
        else:
            clicked_d = float(event.xdata)
            clicked_t = float(event.ydata)

        existing["roi_center_t"] = clicked_t - float(existing.get("roi_drag_offset_t", 0.0))
        existing["roi_center_d"] = clicked_d - float(existing.get("roi_drag_offset_d", 0.0))
        self._refresh_td_window(panel_id)

    def _on_td_window_release(self, panel_id: int, _event: Any) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is not None:
            existing["roi_dragging"] = False

    def _set_td_window_cut_center(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None or existing.get("control_guard"):
            return
        cut = self._td_window_cut(panel_id, "set detached center")
        if cut is None:
            return
        center_x = self._parse_float_var(existing["center_x_var"].get(), "center x")
        center_y = self._parse_float_var(existing["center_y_var"].get(), "center y")
        if center_x is None or center_y is None:
            return

        old_center = cut_center(cut)
        dx = float(center_x) - old_center[0]
        dy = float(center_y) - old_center[1]
        p0, p1 = shift_cut(cut, dx, dy, self.nx, self.ny)
        self._apply_cut_points(
            cut,
            p0,
            p1,
            f"Moved center of {cut.name} from detached TD window.",
        )

    def _apply_td_window_cut_coords(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None or existing.get("control_guard"):
            return
        cut = self._td_window_cut(panel_id, "apply detached vertices")
        if cut is None:
            return
        x1 = self._parse_float_var(existing["cut_x1_var"].get(), "x1")
        y1 = self._parse_float_var(existing["cut_y1_var"].get(), "y1")
        x2 = self._parse_float_var(existing["cut_x2_var"].get(), "x2")
        y2 = self._parse_float_var(existing["cut_y2_var"].get(), "y2")
        if None in {x1, y1, x2, y2}:
            return

        self._apply_cut_points(
            cut,
            (float(x1), float(y1)),
            (float(x2), float(y2)),
            f"Applied detached vertex coordinates to {cut.name}.",
        )

    def _set_td_window_cut_angle(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None or existing.get("control_guard"):
            return
        cut = self._td_window_cut(panel_id, "set detached angle")
        if cut is None:
            return
        angle_deg = self._parse_float_var(existing["cut_angle_var"].get(), "angle")
        if angle_deg is None:
            return
        anchor_mode = str(existing["cut_anchor_var"].get()).strip()
        self._set_cut_angle(
            cut,
            angle_deg,
            anchor_mode,
            f"Set {cut.name} angle from detached TD window to {angle_deg:.2f} deg.",
        )

    def _adjust_td_window_cut_angle(self, panel_id: int, delta_deg: float) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        cut = self._td_window_cut(panel_id, "adjust detached angle")
        if cut is None:
            return
        anchor_mode = str(existing["cut_anchor_var"].get()).strip()
        self._set_cut_angle(
            cut,
            cut_display_angle_deg(cut) + delta_deg,
            anchor_mode,
            f"Adjusted {cut.name} angle by {delta_deg:+.1f} deg from detached TD window.",
        )

    def _set_td_window_cut_length(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None or existing.get("control_guard"):
            return
        cut = self._td_window_cut(panel_id, "set detached length")
        if cut is None:
            return
        length_value = self._parse_float_var(existing["cut_length_var"].get(), "length")
        if length_value is None:
            return
        length_mode = str(existing["cut_length_mode_var"].get()).strip()
        self._set_cut_length(
            cut,
            length_value,
            length_mode,
            f"Set {cut.name} length from detached TD window to {length_value:.2f} px.",
        )

    def _adjust_td_window_cut_length(self, panel_id: int, delta_length: float) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        cut = self._td_window_cut(panel_id, "adjust detached length")
        if cut is None:
            return
        length_mode = str(existing["cut_length_mode_var"].get()).strip()
        self._set_cut_length(
            cut,
            max(1.0, cut_length(cut) + delta_length),
            length_mode,
            f"Adjusted {cut.name} length by {delta_length:+.1f} px from detached TD window.",
        )

    def _open_td_window(
        self,
        panel_id: int | None = None,
        *,
        source_stack_id: int | None = None,
    ) -> None:
        panel_id = self.active_panel_id if panel_id is None else panel_id
        if panel_id < 1 or panel_id > len(self.panels):
            return
        panel_state = self._panel_analysis(panel_id)

        existing = self.td_windows.get(panel_id)
        if existing is not None:
            existing["source_stack_id"] = (
                None if source_stack_id is None else int(source_stack_id)
            )
            top = existing["top"]
            if top.winfo_exists():
                top.deiconify()
                top.lift()
                self._refresh_td_window(panel_id)
                return
            del self.td_windows[panel_id]

        top = self.tk.Toplevel(self.root)
        top.title(f"TD Window - {self.panels[panel_id - 1].name}")
        top.geometry("1180x860")
        top.rowconfigure(2, weight=1)
        top.columnconfigure(0, weight=1)

        header_frame = self.ttk.Frame(top, padding=(8, 8, 8, 0))
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.columnconfigure(0, weight=1)

        edit_var = self.tk.BooleanVar(value=False)
        roi_enabled_var = self.tk.BooleanVar(
            value=bool(panel_state.get("roi_enabled", False))
        )
        editor_toggle_var = self.tk.StringVar(value="Show editor")
        roi_window_toggle_var = self.tk.StringVar(value="ROI window")
        diag_window_toggle_var = self.tk.StringVar(value="Wavelet diag window")
        panel_info_var = self.tk.StringVar(value="")
        cut_info_var = self.tk.StringVar(value="")
        panel_t_ini_var = self.tk.StringVar(value="0")
        panel_t_fin_var = self.tk.StringVar(value=str(self.nt - 1))
        panel_stride_var = self.tk.StringVar(value="1")
        panel_width_var = self.tk.StringVar(value="1")
        panel_weighting_var = self.tk.StringVar(value="uniform")
        roi_t_span_var = self.tk.StringVar(value=str(panel_state.get("roi_t_span", "")))
        roi_d_span_var = self.tk.StringVar(value=str(panel_state.get("roi_d_span", "")))
        cut_angle_var = self.tk.StringVar(value="")
        cut_length_var = self.tk.StringVar(value="")
        cut_anchor_var = self.tk.StringVar(value=str(self.geometry_anchor_var.get()))
        cut_length_mode_var = self.tk.StringVar(
            value=str(self.geometry_length_mode_var.get())
        )
        crest_params = dict(DEFAULT_CREST_TRACKING)
        crest_params.update(panel_state.get("crest_params") or {})
        wavelet_params = dict(DEFAULT_WAVELET_FILTER)
        wavelet_params.update(panel_state.get("wavelet_params") or {})
        crest_summary_var = self.tk.StringVar(value="No crest tracking results.")
        crest_cad_var = self.tk.StringVar(
            value=f"{float(crest_params['cad']):.2f}"
        )
        crest_res_var = self.tk.StringVar(
            value=f"{float(crest_params['res']):.2f}"
        )
        crest_grad_var = self.tk.StringVar(
            value=f"{float(crest_params['grad']):.2f}"
        )
        crest_min_tlen_var = self.tk.StringVar(
            value=str(int(crest_params["min_tlen"]))
        )
        crest_max_dist_jump_var = self.tk.StringVar(
            value=str(int(crest_params["max_dist_jump"]))
        )
        crest_max_time_skip_var = self.tk.StringVar(
            value=str(int(crest_params["max_time_skip"]))
        )
        crest_invert_var = self.tk.BooleanVar(
            value=bool(crest_params["invert"])
        )
        crest_gauss_var = self.tk.BooleanVar(
            value=bool(crest_params["gauss"])
        )
        wavelet_summary_var = self.tk.StringVar(value="No wavelet filter results.")
        wavelet_physics_var = self.tk.StringVar(value="")
        wavelet_p_min_var = self.tk.StringVar(
            value=f"{float(wavelet_params['p_min']):.2f}"
        )
        wavelet_p_max_var = self.tk.StringVar(
            value=f"{float(wavelet_params['p_max']):.2f}"
        )
        wavelet_power_ratio_var = self.tk.StringVar(
            value=f"{float(wavelet_params['power_ratio_thresh']):.2f}"
        )
        wavelet_segment_frac_var = self.tk.StringVar(
            value=f"{float(wavelet_params['segment_power_frac']):.2f}"
        )
        wavelet_min_points_seg_var = self.tk.StringVar(
            value=str(int(wavelet_params["min_points_segment"]))
        )
        wavelet_min_amp_var = self.tk.StringVar(
            value=f"{float(wavelet_params['min_amp_arcsec']):.3f}"
        )
        wavelet_max_jump_var = self.tk.StringVar(
            value=f"{float(wavelet_params['max_jump_pix']):.2f}"
        )
        wavelet_min_points_cut_var = self.tk.StringVar(
            value=str(int(wavelet_params["min_points_cut_seg"]))
        )
        wavelet_rms_amp_ratio_var = self.tk.StringVar(
            value=f"{float(wavelet_params['rms_amp_ratio_max']):.2f}"
        )
        wavelet_km_per_arcsec_var = self.tk.StringVar(
            value=f"{float(wavelet_params['km_per_arcsec']):.2f}"
        )
        wavelet_density_var = self.tk.StringVar(
            value=(
                ""
                if not np.isfinite(float(wavelet_params["density_kg_m3"]))
                else f"{float(wavelet_params['density_kg_m3']):.3e}"
            )
        )
        wavelet_phase_speed_var = self.tk.StringVar(
            value=(
                ""
                if not np.isfinite(float(wavelet_params["phase_speed_km_s"]))
                else f"{float(wavelet_params['phase_speed_km_s']):.2f}"
            )
        )
        wavelet_thread_filter_var = self.tk.StringVar(
            value=str(panel_state.get("wavelet_thread_filter_text", "") or "")
        )
        preset_name = str(panel_state.get("preset_name", "custom") or "custom")
        if preset_name not in PARAMETER_PRESETS:
            preset_name = self._matching_parameter_preset_name(
                dict(crest_params), dict(wavelet_params)
            )
        preset_var = self.tk.StringVar(value=preset_name)
        wavelet_events_summary_var = self.tk.StringVar(value="")
        wavelet_selected_var = self.tk.StringVar(
            value="Select a wavelet event to inspect it."
        )
        wavelet_diag_var = self.tk.StringVar(value="")
        wavelet_progress_var = self.tk.StringVar(value="Idle.")
        wavelet_events_filter_var = self.tk.StringVar(
            value=str(panel_state.get("wavelet_events_filter", "accepted"))
        )
        advanced_filters = dict(
            self._make_default_panel_analysis_state()["wavelet_advanced_filters"]
        )
        advanced_filters.update(panel_state.get("wavelet_advanced_filters") or {})
        wavelet_filter_qa_var = self.tk.StringVar(
            value=str(advanced_filters.get("qa", "all"))
        )
        wavelet_filter_locked_var = self.tk.StringVar(
            value=str(advanced_filters.get("locked", "all"))
        )
        wavelet_filter_linked_var = self.tk.StringVar(
            value=str(advanced_filters.get("linked", "all"))
        )
        wavelet_filter_score_min_var = self.tk.StringVar(
            value=str(advanced_filters.get("score_min", ""))
        )
        wavelet_filter_period_min_var = self.tk.StringVar(
            value=str(advanced_filters.get("period_min", ""))
        )
        wavelet_filter_period_max_var = self.tk.StringVar(
            value=str(advanced_filters.get("period_max", ""))
        )
        wavelet_filter_amp_min_var = self.tk.StringVar(
            value=str(advanced_filters.get("amp_min", ""))
        )
        wavelet_filter_amp_max_var = self.tk.StringVar(
            value=str(advanced_filters.get("amp_max", ""))
        )
        wavelet_filter_energy_min_var = self.tk.StringVar(
            value=str(advanced_filters.get("energy_min", ""))
        )
        wavelet_filter_energy_max_var = self.tk.StringVar(
            value=str(advanced_filters.get("energy_max", ""))
        )
        wavelet_trim_start_var = self.tk.StringVar(value="")
        wavelet_trim_end_var = self.tk.StringVar(value="")
        wavelet_split_frames_var = self.tk.StringVar(value="")
        center_x_var = self.tk.StringVar(value="")
        center_y_var = self.tk.StringVar(value="")
        cut_x1_var = self.tk.StringVar(value="")
        cut_y1_var = self.tk.StringVar(value="")
        cut_x2_var = self.tk.StringVar(value="")
        cut_y2_var = self.tk.StringVar(value="")

        self.ttk.Label(header_frame, textvariable=panel_info_var).grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Checkbutton(
            header_frame,
            text="ROI box",
            variable=roi_enabled_var,
            command=lambda pid=panel_id: self._on_td_window_roi_toggle(pid),
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.ttk.Button(
            header_frame,
            textvariable=roi_window_toggle_var,
            command=lambda pid=panel_id: self._toggle_td_window_roi_window(pid),
        ).grid(row=0, column=2, sticky="e", padx=(12, 0))
        self.ttk.Button(
            header_frame,
            textvariable=diag_window_toggle_var,
            command=lambda pid=panel_id: self._toggle_td_window_diag_window(pid),
        ).grid(row=0, column=3, sticky="e", padx=(12, 0))
        self.ttk.Button(
            header_frame,
            textvariable=editor_toggle_var,
            command=lambda pid=panel_id: self._toggle_td_window_editor(pid),
        ).grid(row=0, column=4, sticky="e", padx=(12, 0))

        edit_container, edit_scroll_body, edit_canvas = self._create_scrolled_frame(
            top, padding=0, height=340
        )
        edit_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))
        edit_scroll_body.columnconfigure(0, weight=1)

        edit_frame = self.ttk.LabelFrame(
            edit_scroll_body, text="Detached TD Controls", padding=8
        )
        edit_frame.grid(row=0, column=0, sticky="ew")
        for idx in range(8):
            edit_frame.columnconfigure(idx, weight=1)

        self.ttk.Label(edit_frame, textvariable=cut_info_var).grid(
            row=0, column=0, columnspan=8, sticky="ew"
        )

        self.ttk.Label(edit_frame, text="t_ini").grid(row=1, column=0, sticky="w", pady=(8, 0))
        t_ini_entry = self.ttk.Entry(edit_frame, textvariable=panel_t_ini_var, width=8)
        t_ini_entry.grid(row=1, column=1, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(edit_frame, text="t_fin").grid(row=1, column=2, sticky="w", pady=(8, 0))
        t_fin_entry = self.ttk.Entry(edit_frame, textvariable=panel_t_fin_var, width=8)
        t_fin_entry.grid(row=1, column=3, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(edit_frame, text="stride").grid(row=1, column=4, sticky="w", pady=(8, 0))
        stride_entry = self.ttk.Entry(edit_frame, textvariable=panel_stride_var, width=8)
        stride_entry.grid(row=1, column=5, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Button(
            edit_frame,
            text="Apply TD",
            command=lambda pid=panel_id: self._apply_td_window_panel_controls(pid),
        ).grid(row=1, column=6, columnspan=2, sticky="ew", pady=(8, 0))

        self.ttk.Label(edit_frame, text="width").grid(row=2, column=0, sticky="w", pady=(8, 0))
        width_box = self.ttk.Combobox(
            edit_frame,
            textvariable=panel_width_var,
            values=["1", "3", "5", "7", "9"],
            state="readonly",
            width=8,
        )
        width_box.grid(row=2, column=1, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(edit_frame, text="weighting").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        weighting_box = self.ttk.Combobox(
            edit_frame,
            textvariable=panel_weighting_var,
            values=["uniform", "gaussian"],
            state="readonly",
            width=10,
        )
        weighting_box.grid(row=2, column=3, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(edit_frame, text="ROI time").grid(row=2, column=4, sticky="w", pady=(8, 0))
        roi_t_entry = self.ttk.Entry(edit_frame, textvariable=roi_t_span_var, width=8)
        roi_t_entry.grid(row=2, column=5, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(edit_frame, text="ROI dist").grid(row=2, column=6, sticky="w", pady=(8, 0))
        roi_d_entry = self.ttk.Entry(edit_frame, textvariable=roi_d_span_var, width=8)
        roi_d_entry.grid(row=2, column=7, sticky="ew", padx=(6, 0), pady=(8, 0))

        self.ttk.Label(edit_frame, text="center x").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )
        center_x_entry = self.ttk.Entry(edit_frame, textvariable=center_x_var, width=8)
        center_x_entry.grid(row=3, column=1, sticky="ew", padx=(6, 8), pady=(10, 0))
        self.ttk.Label(edit_frame, text="center y").grid(
            row=3, column=2, sticky="w", pady=(10, 0)
        )
        center_y_entry = self.ttk.Entry(edit_frame, textvariable=center_y_var, width=8)
        center_y_entry.grid(row=3, column=3, sticky="ew", padx=(6, 8), pady=(10, 0))
        self.ttk.Button(
            edit_frame,
            text="Set center",
            command=lambda pid=panel_id: self._set_td_window_cut_center(pid),
        ).grid(row=3, column=4, columnspan=2, sticky="ew", pady=(10, 0))
        self.ttk.Button(
            edit_frame,
            text="Apply ROI",
            command=lambda pid=panel_id: self._apply_td_window_roi_settings(pid),
        ).grid(row=3, column=6, columnspan=2, sticky="ew", pady=(10, 0))

        vertices_frame = self.ttk.Frame(edit_frame)
        vertices_frame.grid(row=4, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        for idx in range(4):
            vertices_frame.columnconfigure(idx, weight=1)

        self.ttk.Label(vertices_frame, text="x1").grid(row=0, column=0, sticky="w")
        self.ttk.Label(vertices_frame, text="y1").grid(row=0, column=1, sticky="w")
        self.ttk.Label(vertices_frame, text="x2").grid(row=0, column=2, sticky="w")
        self.ttk.Label(vertices_frame, text="y2").grid(row=0, column=3, sticky="w")

        x1_entry = self.ttk.Entry(vertices_frame, textvariable=cut_x1_var, width=8)
        y1_entry = self.ttk.Entry(vertices_frame, textvariable=cut_y1_var, width=8)
        x2_entry = self.ttk.Entry(vertices_frame, textvariable=cut_x2_var, width=8)
        y2_entry = self.ttk.Entry(vertices_frame, textvariable=cut_y2_var, width=8)
        for idx, entry in enumerate((x1_entry, y1_entry, x2_entry, y2_entry)):
            entry.grid(row=1, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))

        self.ttk.Button(
            edit_frame,
            text="Apply vertices",
            command=lambda pid=panel_id: self._apply_td_window_cut_coords(pid),
        ).grid(row=5, column=0, columnspan=8, sticky="ew", pady=(8, 0))

        self.ttk.Label(edit_frame, text="Length mode").grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )
        length_mode_box = self.ttk.Combobox(
            edit_frame,
            textvariable=cut_length_mode_var,
            values=["symmetric", "fix p0", "fix p1"],
            state="readonly",
            width=10,
        )
        length_mode_box.grid(row=6, column=1, sticky="ew", padx=(6, 8), pady=(10, 0))
        self.ttk.Label(edit_frame, text="Anchor").grid(row=6, column=2, sticky="w", pady=(10, 0))
        anchor_box = self.ttk.Combobox(
            edit_frame,
            textvariable=cut_anchor_var,
            values=["center", "fix p0", "fix p1"],
            state="readonly",
            width=10,
        )
        anchor_box.grid(row=6, column=3, sticky="ew", padx=(6, 8), pady=(10, 0))

        self.ttk.Label(edit_frame, text="Length").grid(row=7, column=0, sticky="w", pady=(10, 0))
        length_entry = self.ttk.Entry(edit_frame, textvariable=cut_length_var, width=8)
        length_entry.grid(row=7, column=1, sticky="ew", padx=(6, 8), pady=(10, 0))
        self.ttk.Button(
            edit_frame,
            text="Set length",
            command=lambda pid=panel_id: self._set_td_window_cut_length(pid),
        ).grid(row=7, column=2, columnspan=2, sticky="ew", pady=(10, 0))

        length_buttons = self.ttk.Frame(edit_frame)
        length_buttons.grid(row=8, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        for idx, delta in enumerate([-20, -10, -5, 5, 10, 20]):
            self.ttk.Button(
                length_buttons,
                text=f"{delta:+d}",
                command=lambda pid=panel_id, d=delta: self._adjust_td_window_cut_length(pid, d),
            ).grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))
            length_buttons.columnconfigure(idx, weight=1)

        self.ttk.Label(edit_frame, text="Angle").grid(row=9, column=0, sticky="w", pady=(10, 0))
        angle_entry = self.ttk.Entry(edit_frame, textvariable=cut_angle_var, width=8)
        angle_entry.grid(row=9, column=1, sticky="ew", padx=(6, 8), pady=(10, 0))
        self.ttk.Button(
            edit_frame,
            text="Set angle",
            command=lambda pid=panel_id: self._set_td_window_cut_angle(pid),
        ).grid(row=9, column=2, columnspan=2, sticky="ew", pady=(10, 0))

        angle_buttons = self.ttk.Frame(edit_frame)
        angle_buttons.grid(row=10, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        for idx, delta in enumerate([-10, -5, -1, 1, 5, 10]):
            self.ttk.Button(
                angle_buttons,
                text=f"{delta:+d}",
                command=lambda pid=panel_id, d=delta: self._adjust_td_window_cut_angle(pid, d),
            ).grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))
            angle_buttons.columnconfigure(idx, weight=1)

        analysis_frame = self.ttk.LabelFrame(
            edit_frame, text="Crest Tracking (NUWT)", padding=8
        )
        analysis_frame.grid(row=11, column=0, columnspan=8, sticky="ew", pady=(12, 0))
        for idx in (1, 3, 5, 7):
            analysis_frame.columnconfigure(idx, weight=1)

        self.ttk.Label(
            analysis_frame, textvariable=crest_summary_var
        ).grid(row=0, column=0, columnspan=8, sticky="w")

        self.ttk.Label(analysis_frame, text="cad [s]").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        crest_cad_entry = self.ttk.Entry(
            analysis_frame, textvariable=crest_cad_var, width=8
        )
        crest_cad_entry.grid(row=1, column=1, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(analysis_frame, text="res [arcsec/px]").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        crest_res_entry = self.ttk.Entry(
            analysis_frame, textvariable=crest_res_var, width=8
        )
        crest_res_entry.grid(row=1, column=3, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(analysis_frame, text="grad").grid(
            row=1, column=4, sticky="w", pady=(8, 0)
        )
        crest_grad_entry = self.ttk.Entry(
            analysis_frame, textvariable=crest_grad_var, width=8
        )
        crest_grad_entry.grid(row=1, column=5, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Checkbutton(
            analysis_frame,
            text="invert",
            variable=crest_invert_var,
        ).grid(row=1, column=6, columnspan=2, sticky="w", pady=(8, 0))

        self.ttk.Label(analysis_frame, text="min thread").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        crest_min_tlen_entry = self.ttk.Entry(
            analysis_frame, textvariable=crest_min_tlen_var, width=8
        )
        crest_min_tlen_entry.grid(
            row=2, column=1, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(analysis_frame, text="max dist jump").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        crest_max_dist_jump_entry = self.ttk.Entry(
            analysis_frame, textvariable=crest_max_dist_jump_var, width=8
        )
        crest_max_dist_jump_entry.grid(
            row=2, column=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(analysis_frame, text="max time skip").grid(
            row=2, column=4, sticky="w", pady=(8, 0)
        )
        crest_max_time_skip_entry = self.ttk.Entry(
            analysis_frame, textvariable=crest_max_time_skip_var, width=8
        )
        crest_max_time_skip_entry.grid(
            row=2, column=5, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Checkbutton(
            analysis_frame,
            text="gauss fit (slow)",
            variable=crest_gauss_var,
        ).grid(row=2, column=6, columnspan=2, sticky="w", pady=(8, 0))

        self.ttk.Button(
            analysis_frame,
            text="Run tracking",
            command=lambda pid=panel_id: self._run_td_window_crest_tracking(pid),
        ).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0), padx=(0, 4))
        self.ttk.Button(
            analysis_frame,
            text="Clear tracking",
            command=lambda pid=panel_id: self._clear_td_window_crest_tracking(
                pid, refresh=True
            ),
        ).grid(row=3, column=4, columnspan=4, sticky="ew", pady=(10, 0), padx=(4, 0))

        wavelet_frame = self.ttk.LabelFrame(
            edit_frame, text="Wavelet Filter", padding=8
        )
        wavelet_frame.grid(row=12, column=0, columnspan=8, sticky="ew", pady=(12, 0))
        for idx in (1, 3, 5, 7):
            wavelet_frame.columnconfigure(idx, weight=1)

        self.ttk.Label(
            wavelet_frame, textvariable=wavelet_summary_var
        ).grid(row=0, column=0, columnspan=8, sticky="w")
        self.ttk.Label(
            wavelet_frame,
            textvariable=wavelet_physics_var,
            wraplength=520,
            justify="left",
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(4, 0))

        self.ttk.Label(wavelet_frame, text="P min [s]").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_p_min_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_p_min_var, width=8
        )
        wavelet_p_min_entry.grid(row=2, column=1, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(wavelet_frame, text="P max [s]").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        wavelet_p_max_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_p_max_var, width=8
        )
        wavelet_p_max_entry.grid(row=2, column=3, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(wavelet_frame, text="power ratio").grid(
            row=2, column=4, sticky="w", pady=(8, 0)
        )
        wavelet_power_ratio_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_power_ratio_var, width=8
        )
        wavelet_power_ratio_entry.grid(
            row=2, column=5, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="segment frac").grid(
            row=2, column=6, sticky="w", pady=(8, 0)
        )
        wavelet_segment_frac_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_segment_frac_var, width=8
        )
        wavelet_segment_frac_entry.grid(
            row=2, column=7, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        self.ttk.Label(wavelet_frame, text="min pts seg").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_min_points_seg_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_min_points_seg_var, width=8
        )
        wavelet_min_points_seg_entry.grid(
            row=3, column=1, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="min amp [arcsec]").grid(
            row=3, column=2, sticky="w", pady=(8, 0)
        )
        wavelet_min_amp_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_min_amp_var, width=8
        )
        wavelet_min_amp_entry.grid(
            row=3, column=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="max jump [px]").grid(
            row=3, column=4, sticky="w", pady=(8, 0)
        )
        wavelet_max_jump_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_max_jump_var, width=8
        )
        wavelet_max_jump_entry.grid(
            row=3, column=5, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="min pts cut").grid(
            row=3, column=6, sticky="w", pady=(8, 0)
        )
        wavelet_min_points_cut_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_min_points_cut_var, width=8
        )
        wavelet_min_points_cut_entry.grid(
            row=3, column=7, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        self.ttk.Label(wavelet_frame, text="rms/amp max").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_rms_amp_ratio_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_rms_amp_ratio_var, width=8
        )
        wavelet_rms_amp_ratio_entry.grid(
            row=4, column=1, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="km / arcsec").grid(
            row=4, column=2, sticky="w", pady=(8, 0)
        )
        wavelet_km_per_arcsec_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_km_per_arcsec_var, width=8
        )
        wavelet_km_per_arcsec_entry.grid(
            row=4, column=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="density [kg/m3]").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_density_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_density_var, width=10
        )
        wavelet_density_entry.grid(
            row=5, column=1, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(wavelet_frame, text="phase speed [km/s]").grid(
            row=5, column=2, sticky="w", pady=(8, 0)
        )
        wavelet_phase_speed_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_phase_speed_var, width=10
        )
        wavelet_phase_speed_entry.grid(
            row=5, column=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        wavelet_run_button = self.ttk.Button(
            wavelet_frame,
            text="Run wavelet filter",
            command=lambda pid=panel_id: self._run_td_window_wavelet_filter(pid),
        )
        wavelet_run_button.grid(
            row=4, column=4, columnspan=2, sticky="ew", pady=(8, 0), padx=(0, 4)
        )
        self.ttk.Button(
            wavelet_frame,
            text="Clear wavelet",
            command=lambda pid=panel_id: self._clear_td_window_wavelet_filter(
                pid, refresh=True
            ),
        ).grid(row=4, column=6, columnspan=2, sticky="ew", pady=(8, 0), padx=(4, 0))
        wavelet_cancel_button = self.ttk.Button(
            wavelet_frame,
            text="Cancel",
            command=lambda pid=panel_id: self._cancel_td_window_wavelet_job(pid),
            state="disabled",
        )
        wavelet_cancel_button.grid(
            row=5, column=4, columnspan=2, sticky="ew", pady=(8, 0), padx=(0, 4)
        )
        wavelet_progressbar = self.ttk.Progressbar(
            wavelet_frame, mode="determinate", maximum=1.0, value=0.0
        )
        wavelet_progressbar.grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0), padx=(0, 8)
        )
        self.ttk.Label(
            wavelet_frame, textvariable=wavelet_progress_var, justify="left"
        ).grid(row=5, column=6, columnspan=2, sticky="w", pady=(8, 0), padx=(4, 0))
        self.ttk.Label(wavelet_frame, text="Preset").grid(
            row=6, column=0, sticky="w", pady=(8, 0)
        )
        preset_box = self.ttk.Combobox(
            wavelet_frame,
            textvariable=preset_var,
            values=list(PARAMETER_PRESETS.keys()),
            state="readonly",
            width=14,
        )
        preset_box.grid(row=6, column=1, columnspan=2, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Button(
            wavelet_frame,
            text="Apply preset",
            command=lambda pid=panel_id: self._apply_td_window_parameter_preset(pid),
        ).grid(row=6, column=3, columnspan=2, sticky="ew", padx=(0, 4), pady=(8, 0))
        self.ttk.Button(
            wavelet_frame,
            text="Export report",
            command=self._export_curated_report,
        ).grid(row=6, column=5, columnspan=3, sticky="ew", padx=(4, 0), pady=(8, 0))
        self.ttk.Label(wavelet_frame, text="Threads").grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_thread_filter_entry = self.ttk.Entry(
            wavelet_frame, textvariable=wavelet_thread_filter_var, width=14
        )
        wavelet_thread_filter_entry.grid(
            row=7, column=1, columnspan=2, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Button(
            wavelet_frame,
            text="Use event thread",
            command=lambda pid=panel_id: self._set_td_window_selected_event_thread_filter(pid),
        ).grid(row=7, column=3, columnspan=2, sticky="ew", padx=(0, 4), pady=(8, 0))
        self.ttk.Button(
            wavelet_frame,
            text="Thread -> stack",
            command=lambda pid=panel_id: self._apply_td_window_selected_event_thread_to_stack(pid),
        ).grid(row=7, column=5, columnspan=3, sticky="ew", padx=(4, 0), pady=(8, 0))

        events_frame = self.ttk.LabelFrame(
            edit_frame, text="Wavelet Events", padding=8
        )
        events_frame.grid(row=13, column=0, columnspan=8, sticky="ew", pady=(12, 0))
        for idx in range(8):
            events_frame.columnconfigure(idx, weight=1)

        self.ttk.Label(
            events_frame, textvariable=wavelet_events_summary_var
        ).grid(row=0, column=0, columnspan=6, sticky="w")
        self.ttk.Label(events_frame, text="Filter").grid(row=0, column=6, sticky="e")
        wavelet_filter_box = self.ttk.Combobox(
            events_frame,
            textvariable=wavelet_events_filter_var,
            values=["all", "accepted", "rejected", "manual", "split"],
            state="readonly",
            width=10,
        )
        wavelet_filter_box.grid(row=0, column=7, sticky="ew", padx=(6, 0))

        advanced_filter_frame = self.ttk.LabelFrame(
            events_frame, text="Advanced Filter", padding=6
        )
        advanced_filter_frame.grid(
            row=1, column=0, columnspan=8, sticky="ew", pady=(8, 0)
        )
        for idx in range(8):
            advanced_filter_frame.columnconfigure(idx, weight=1)
        self.ttk.Label(advanced_filter_frame, text="QA").grid(row=0, column=0, sticky="w")
        wavelet_filter_qa_box = self.ttk.Combobox(
            advanced_filter_frame,
            textvariable=wavelet_filter_qa_var,
            values=["all", "flagged", "clean", "few_points", "period_edge", "high_residual"],
            state="readonly",
            width=12,
        )
        wavelet_filter_qa_box.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        self.ttk.Label(advanced_filter_frame, text="Lock").grid(row=0, column=2, sticky="w")
        wavelet_filter_locked_box = self.ttk.Combobox(
            advanced_filter_frame,
            textvariable=wavelet_filter_locked_var,
            values=["all", "locked", "unlocked"],
            state="readonly",
            width=10,
        )
        wavelet_filter_locked_box.grid(row=0, column=3, sticky="ew", padx=(6, 8))
        self.ttk.Label(advanced_filter_frame, text="Link").grid(row=0, column=4, sticky="w")
        wavelet_filter_linked_box = self.ttk.Combobox(
            advanced_filter_frame,
            textvariable=wavelet_filter_linked_var,
            values=["all", "linked", "unlinked"],
            state="readonly",
            width=10,
        )
        wavelet_filter_linked_box.grid(row=0, column=5, sticky="ew", padx=(6, 8))
        self.ttk.Label(advanced_filter_frame, text="score >=").grid(
            row=0, column=6, sticky="w"
        )
        wavelet_filter_score_min_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_score_min_var, width=8
        )
        wavelet_filter_score_min_entry.grid(
            row=0, column=7, sticky="ew", padx=(6, 0)
        )

        self.ttk.Label(advanced_filter_frame, text="P min").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_filter_period_min_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_period_min_var, width=8
        )
        wavelet_filter_period_min_entry.grid(
            row=1, column=1, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(advanced_filter_frame, text="P max").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        wavelet_filter_period_max_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_period_max_var, width=8
        )
        wavelet_filter_period_max_entry.grid(
            row=1, column=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(advanced_filter_frame, text="A min").grid(
            row=1, column=4, sticky="w", pady=(8, 0)
        )
        wavelet_filter_amp_min_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_amp_min_var, width=8
        )
        wavelet_filter_amp_min_entry.grid(
            row=1, column=5, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(advanced_filter_frame, text="A max").grid(
            row=1, column=6, sticky="w", pady=(8, 0)
        )
        wavelet_filter_amp_max_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_amp_max_var, width=8
        )
        wavelet_filter_amp_max_entry.grid(
            row=1, column=7, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        self.ttk.Label(advanced_filter_frame, text="E min").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        wavelet_filter_energy_min_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_energy_min_var, width=8
        )
        wavelet_filter_energy_min_entry.grid(
            row=2, column=1, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Label(advanced_filter_frame, text="E max").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        wavelet_filter_energy_max_entry = self.ttk.Entry(
            advanced_filter_frame, textvariable=wavelet_filter_energy_max_var, width=8
        )
        wavelet_filter_energy_max_entry.grid(
            row=2, column=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        self.ttk.Button(
            advanced_filter_frame,
            text="Apply filters",
            command=lambda pid=panel_id: self._apply_td_window_advanced_filters(pid),
        ).grid(row=2, column=4, columnspan=2, sticky="ew", padx=(4, 4), pady=(8, 0))
        self.ttk.Button(
            advanced_filter_frame,
            text="Clear filters",
            command=lambda pid=panel_id: self._clear_td_window_advanced_filters(pid),
        ).grid(row=2, column=6, columnspan=2, sticky="ew", padx=(4, 0), pady=(8, 0))

        self.ttk.Label(
            events_frame,
            textvariable=wavelet_selected_var,
            wraplength=520,
            justify="left",
        ).grid(row=2, column=0, columnspan=8, sticky="w", pady=(6, 0))
        self.ttk.Label(
            events_frame,
            textvariable=wavelet_diag_var,
            wraplength=520,
            justify="left",
        ).grid(row=3, column=0, columnspan=8, sticky="w", pady=(4, 0))

        tree_frame = self.ttk.Frame(events_frame)
        tree_frame.grid(row=4, column=0, columnspan=8, sticky="nsew", pady=(8, 0))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(4, weight=1)
        wavelet_events_tree = self.ttk.Treeview(
            tree_frame,
            columns=(
                "id",
                "status",
                "origin",
                "thread",
                "seg",
                "wseg",
                "period",
                "freq",
                "amp_arc",
                "amp_km",
                "vel",
                "accel",
                "energy",
                "dur",
                "ratio",
                "score",
                "lock",
                "links",
                "flags",
                "reason",
            ),
            show="headings",
            height=10,
        )
        tree_y = self.ttk.Scrollbar(
            tree_frame, orient="vertical", command=wavelet_events_tree.yview
        )
        tree_x = self.ttk.Scrollbar(
            tree_frame, orient="horizontal", command=wavelet_events_tree.xview
        )
        wavelet_events_tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        wavelet_events_tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        headings = {
            "id": ("ID", 50),
            "status": ("Status", 110),
            "origin": ("Origin", 90),
            "thread": ("Thr", 50),
            "seg": ("Seg", 50),
            "wseg": ("Wseg", 55),
            "period": ("P [s]", 70),
            "freq": ("f [mHz]", 70),
            "amp_arc": ("A ['']", 70),
            "amp_km": ("A [km]", 75),
            "vel": ("v [km/s]", 80),
            "accel": ("a [km/s2]", 85),
            "energy": ("E/m [J/kg]", 100),
            "dur": ("dur [s]", 70),
            "ratio": ("ratio", 60),
            "score": ("Conf", 62),
            "lock": ("Lock", 55),
            "links": ("Links", 55),
            "flags": ("QA", 110),
            "reason": ("Reason", 140),
        }
        for column, (label, width) in headings.items():
            wavelet_events_tree.heading(column, text=label)
            wavelet_events_tree.column(
                column,
                width=width,
                stretch=(column in {"reason", "flags"}),
            )
        wavelet_events_tree.tag_configure("auto_accepted", foreground="green4")
        wavelet_events_tree.tag_configure("auto_rejected", foreground="firebrick")
        wavelet_events_tree.tag_configure("custom_accepted", foreground="darkcyan")
        wavelet_events_tree.tag_configure("custom_rejected", foreground="darkorange")
        wavelet_events_tree.tag_configure("manual_accepted", foreground="dodgerblue4")
        wavelet_events_tree.tag_configure("manual_rejected", foreground="red4")
        wavelet_events_tree.tag_configure("split_parent", foreground="gray45")

        actions_frame = self.ttk.Frame(events_frame)
        actions_frame.grid(row=5, column=0, columnspan=8, sticky="ew", pady=(8, 0))
        for idx in range(10):
            actions_frame.columnconfigure(idx, weight=1)
        self.ttk.Button(
            actions_frame,
            text="Recompute selected",
            command=lambda pid=panel_id: self._recompute_td_window_selected_wavelet_event(pid),
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 4))
        self.ttk.Button(
            actions_frame,
            text="Accept",
            command=lambda pid=panel_id: self._accept_td_window_selected_wavelet_event(pid),
        ).grid(row=0, column=2, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="Reject",
            command=lambda pid=panel_id: self._reject_td_window_selected_wavelet_event(pid),
        ).grid(row=0, column=3, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="Reset",
            command=lambda pid=panel_id: self._reset_td_window_selected_wavelet_event(pid),
        ).grid(row=0, column=4, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="Undo",
            command=lambda pid=panel_id: self._undo_td_window_wavelet_edit(pid),
        ).grid(row=0, column=5, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="Redo",
            command=lambda pid=panel_id: self._redo_td_window_wavelet_edit(pid),
        ).grid(row=0, column=6, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="Lock/Unlock",
            command=lambda pid=panel_id: self._toggle_td_window_selected_wavelet_lock(pid),
        ).grid(row=0, column=7, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="Note",
            command=lambda pid=panel_id: self._edit_td_window_selected_wavelet_note(pid),
        ).grid(row=0, column=8, sticky="ew", padx=4)
        self.ttk.Button(
            actions_frame,
            text="History",
            command=lambda pid=panel_id: self._show_td_window_selected_wavelet_history(pid),
        ).grid(row=0, column=9, sticky="ew", padx=(4, 0))
        self.ttk.Button(
            actions_frame,
            text="Copy link src",
            command=lambda pid=panel_id: self._copy_td_window_selected_wavelet_link_source(pid),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 4), pady=(8, 0))
        self.ttk.Button(
            actions_frame,
            text="Link to src",
            command=lambda pid=panel_id: self._link_td_window_selected_wavelet_to_source(pid),
        ).grid(row=1, column=2, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        self.ttk.Button(
            actions_frame,
            text="Clear link",
            command=lambda pid=panel_id: self._clear_td_window_selected_wavelet_link(pid),
        ).grid(row=1, column=4, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        self.ttk.Button(
            actions_frame,
            text="Sync group",
            command=lambda pid=panel_id: self._sync_td_window_selected_wavelet_group(pid),
        ).grid(row=1, column=6, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        self.ttk.Button(
            actions_frame,
            text="Show links",
            command=lambda pid=panel_id: self._show_td_window_selected_wavelet_links(pid),
        ).grid(row=1, column=8, columnspan=2, sticky="ew", padx=(4, 0), pady=(8, 0))
        wavelet_trim_start_entry = self.ttk.Entry(
            actions_frame, textvariable=wavelet_trim_start_var, width=8
        )
        wavelet_trim_start_entry.grid(row=2, column=1, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Label(actions_frame, text="trim start").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.ttk.Label(actions_frame, text="trim end").grid(row=2, column=2, sticky="w", pady=(8, 0))
        wavelet_trim_end_entry = self.ttk.Entry(
            actions_frame, textvariable=wavelet_trim_end_var, width=8
        )
        wavelet_trim_end_entry.grid(row=2, column=3, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Button(
            actions_frame,
            text="Trim selected",
            command=lambda pid=panel_id: self._trim_td_window_selected_wavelet_event(pid),
        ).grid(row=2, column=4, columnspan=2, sticky="ew", padx=(4, 4), pady=(8, 0))
        self.ttk.Label(actions_frame, text="split frames").grid(row=3, column=0, sticky="w", pady=(8, 0))
        wavelet_split_entry = self.ttk.Entry(
            actions_frame, textvariable=wavelet_split_frames_var, width=16
        )
        wavelet_split_entry.grid(row=3, column=1, columnspan=3, sticky="ew", padx=(6, 8), pady=(8, 0))
        self.ttk.Button(
            actions_frame,
            text="Split selected",
            command=lambda pid=panel_id: self._split_td_window_selected_wavelet_event(pid),
        ).grid(row=3, column=4, columnspan=2, sticky="ew", padx=(4, 4), pady=(8, 0))

        edit_container.grid_remove()

        main_plot_frame = self.ttk.Frame(top, padding=(8, 0, 8, 8))
        main_plot_frame.grid(row=2, column=0, sticky="nsew")
        main_plot_frame.rowconfigure(0, weight=1)
        main_plot_frame.columnconfigure(0, weight=1)

        fig = self.Figure(figsize=(8.6, 5.2), dpi=120)
        ax = fig.add_subplot(111)
        canvas = self.FigureCanvasTkAgg(fig, master=main_plot_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        canvas.mpl_connect(
            "button_press_event",
            lambda event, pid=panel_id: self._on_td_window_press(pid, event),
        )
        canvas.mpl_connect(
            "motion_notify_event",
            lambda event, pid=panel_id: self._on_td_window_motion(pid, event),
        )
        canvas.mpl_connect(
            "button_release_event",
            lambda event, pid=panel_id: self._on_td_window_release(pid, event),
        )

        for widget in (t_ini_entry, t_fin_entry, stride_entry):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._apply_td_window_panel_controls(pid),
            )
            widget.bind(
                "<FocusOut>",
                lambda _event, pid=panel_id: self._apply_td_window_panel_controls(pid),
            )
        width_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, pid=panel_id: self._apply_td_window_panel_controls(pid),
        )
        weighting_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, pid=panel_id: self._apply_td_window_panel_controls(pid),
        )
        length_entry.bind(
            "<Return>",
            lambda _event, pid=panel_id: self._set_td_window_cut_length(pid),
        )
        angle_entry.bind(
            "<Return>",
            lambda _event, pid=panel_id: self._set_td_window_cut_angle(pid),
        )
        for widget in (roi_t_entry, roi_d_entry):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._apply_td_window_roi_settings(pid),
            )
        for widget in (center_x_entry, center_y_entry):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._set_td_window_cut_center(pid),
            )
        for widget in (x1_entry, y1_entry, x2_entry, y2_entry):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._apply_td_window_cut_coords(pid),
            )
        for widget in (
            crest_cad_entry,
            crest_res_entry,
            crest_grad_entry,
            crest_min_tlen_entry,
            crest_max_dist_jump_entry,
            crest_max_time_skip_entry,
        ):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._run_td_window_crest_tracking(pid),
            )
        for widget in (
            wavelet_p_min_entry,
            wavelet_p_max_entry,
            wavelet_power_ratio_entry,
            wavelet_segment_frac_entry,
            wavelet_min_points_seg_entry,
            wavelet_min_amp_entry,
            wavelet_max_jump_entry,
            wavelet_min_points_cut_entry,
            wavelet_rms_amp_ratio_entry,
            wavelet_km_per_arcsec_entry,
            wavelet_density_entry,
            wavelet_phase_speed_entry,
        ):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._run_td_window_wavelet_filter(pid),
            )
        preset_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, pid=panel_id: self._apply_td_window_parameter_preset(pid),
        )
        wavelet_filter_box.bind(
            "<<ComboboxSelected>>",
            lambda _event, pid=panel_id: self._on_td_window_wavelet_filter_mode_change(pid),
        )
        for widget in (
            wavelet_filter_qa_box,
            wavelet_filter_locked_box,
            wavelet_filter_linked_box,
        ):
            widget.bind(
                "<<ComboboxSelected>>",
                lambda _event, pid=panel_id: self._apply_td_window_advanced_filters(pid),
            )
        for widget in (
            wavelet_filter_score_min_entry,
            wavelet_filter_period_min_entry,
            wavelet_filter_period_max_entry,
            wavelet_filter_amp_min_entry,
            wavelet_filter_amp_max_entry,
            wavelet_filter_energy_min_entry,
            wavelet_filter_energy_max_entry,
        ):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._apply_td_window_advanced_filters(pid),
            )
            widget.bind(
                "<FocusOut>",
                lambda _event, pid=panel_id: self._apply_td_window_advanced_filters(pid),
            )
        wavelet_events_tree.bind(
            "<<TreeviewSelect>>",
            lambda _event, pid=panel_id: self._on_td_window_wavelet_event_select(pid),
        )
        for widget in (
            wavelet_trim_start_entry,
            wavelet_trim_end_entry,
        ):
            widget.bind(
                "<Return>",
                lambda _event, pid=panel_id: self._trim_td_window_selected_wavelet_event(pid),
            )
        wavelet_split_entry.bind(
            "<Return>",
            lambda _event, pid=panel_id: self._split_td_window_selected_wavelet_event(pid),
        )

        self.td_windows[panel_id] = {
            "top": top,
            "plot_panes": None,
            "figure": fig,
            "ax": ax,
            "canvas": canvas,
            "roi_top": None,
            "roi_figure": None,
            "roi_ax": None,
            "roi_canvas": None,
            "diag_top": None,
            "diag_figure": None,
            "diag_axes": None,
            "diag_canvas": None,
            "diag_interaction": None,
            "edit_var": edit_var,
            "editor_toggle_var": editor_toggle_var,
            "roi_window_toggle_var": roi_window_toggle_var,
            "diag_window_toggle_var": diag_window_toggle_var,
            "edit_frame": edit_container,
            "edit_canvas": edit_canvas,
            "panel_info_var": panel_info_var,
            "cut_info_var": cut_info_var,
            "panel_t_ini_var": panel_t_ini_var,
            "panel_t_fin_var": panel_t_fin_var,
            "panel_stride_var": panel_stride_var,
            "panel_width_var": panel_width_var,
            "panel_weighting_var": panel_weighting_var,
            "roi_enabled_var": roi_enabled_var,
            "roi_t_span_var": roi_t_span_var,
            "roi_d_span_var": roi_d_span_var,
            "roi_center_t": self._clone_wavelet_payload(panel_state.get("roi_center_t")),
            "roi_center_d": self._clone_wavelet_payload(panel_state.get("roi_center_d")),
            "roi_dragging": False,
            "roi_drag_offset_t": 0.0,
            "roi_drag_offset_d": 0.0,
            "cut_angle_var": cut_angle_var,
            "cut_length_var": cut_length_var,
            "cut_anchor_var": cut_anchor_var,
            "cut_length_mode_var": cut_length_mode_var,
            "crest_summary_var": crest_summary_var,
            "crest_cad_var": crest_cad_var,
            "crest_res_var": crest_res_var,
            "crest_grad_var": crest_grad_var,
            "crest_min_tlen_var": crest_min_tlen_var,
            "crest_max_dist_jump_var": crest_max_dist_jump_var,
            "crest_max_time_skip_var": crest_max_time_skip_var,
            "crest_invert_var": crest_invert_var,
            "crest_gauss_var": crest_gauss_var,
            "crest_tracking_result": self._clone_wavelet_payload(
                panel_state.get("crest_tracking_result")
            ),
            "crest_tracking_td_key": self._clone_wavelet_payload(
                panel_state.get("crest_tracking_td_key")
            ),
            "wavelet_summary_var": wavelet_summary_var,
            "wavelet_physics_var": wavelet_physics_var,
            "wavelet_p_min_var": wavelet_p_min_var,
            "wavelet_p_max_var": wavelet_p_max_var,
            "wavelet_power_ratio_var": wavelet_power_ratio_var,
            "wavelet_segment_frac_var": wavelet_segment_frac_var,
            "wavelet_min_points_seg_var": wavelet_min_points_seg_var,
            "wavelet_min_amp_var": wavelet_min_amp_var,
            "wavelet_max_jump_var": wavelet_max_jump_var,
            "wavelet_min_points_cut_var": wavelet_min_points_cut_var,
            "wavelet_rms_amp_ratio_var": wavelet_rms_amp_ratio_var,
            "wavelet_km_per_arcsec_var": wavelet_km_per_arcsec_var,
            "wavelet_density_var": wavelet_density_var,
            "wavelet_phase_speed_var": wavelet_phase_speed_var,
            "wavelet_thread_filter_var": wavelet_thread_filter_var,
            "preset_var": preset_var,
            "wavelet_filter_result": self._clone_wavelet_payload(
                panel_state.get("wavelet_filter_result")
            ),
            "wavelet_events": self._clone_wavelet_payload(
                panel_state.get("wavelet_events") or []
            ),
            "wavelet_next_event_id": int(panel_state.get("wavelet_next_event_id", 1)),
            "wavelet_selected_event_id": panel_state.get("wavelet_selected_event_id"),
            "wavelet_undo_stack": self._clone_wavelet_payload(
                panel_state.get("wavelet_undo_stack") or []
            ),
            "wavelet_redo_stack": self._clone_wavelet_payload(
                panel_state.get("wavelet_redo_stack") or []
            ),
            "wavelet_events_tree": wavelet_events_tree,
            "wavelet_events_summary_var": wavelet_events_summary_var,
            "wavelet_selected_var": wavelet_selected_var,
            "wavelet_diag_var": wavelet_diag_var,
            "wavelet_progress_var": wavelet_progress_var,
            "wavelet_progressbar": wavelet_progressbar,
            "wavelet_run_button": wavelet_run_button,
            "wavelet_cancel_button": wavelet_cancel_button,
            "wavelet_events_filter_var": wavelet_events_filter_var,
            "wavelet_filter_qa_var": wavelet_filter_qa_var,
            "wavelet_filter_locked_var": wavelet_filter_locked_var,
            "wavelet_filter_linked_var": wavelet_filter_linked_var,
            "wavelet_filter_score_min_var": wavelet_filter_score_min_var,
            "wavelet_filter_period_min_var": wavelet_filter_period_min_var,
            "wavelet_filter_period_max_var": wavelet_filter_period_max_var,
            "wavelet_filter_amp_min_var": wavelet_filter_amp_min_var,
            "wavelet_filter_amp_max_var": wavelet_filter_amp_max_var,
            "wavelet_filter_energy_min_var": wavelet_filter_energy_min_var,
            "wavelet_filter_energy_max_var": wavelet_filter_energy_max_var,
            "wavelet_trim_start_var": wavelet_trim_start_var,
            "wavelet_trim_end_var": wavelet_trim_end_var,
            "wavelet_split_frames_var": wavelet_split_frames_var,
            "center_x_var": center_x_var,
            "center_y_var": center_y_var,
            "cut_x1_var": cut_x1_var,
            "cut_y1_var": cut_y1_var,
            "cut_x2_var": cut_x2_var,
            "cut_y2_var": cut_y2_var,
            "control_guard": False,
            "wavelet_table_updating": False,
            "source_stack_id": None if source_stack_id is None else int(source_stack_id),
        }
        for event in self.td_windows[panel_id]["wavelet_events"]:
            self._ensure_wavelet_event_fields(event)
            self._wavelet_event_confidence_details(event)

        top.protocol("WM_DELETE_WINDOW", lambda pid=panel_id: self._close_td_window(pid))
        self._sync_td_window_controls(panel_id)
        self._refresh_td_window(panel_id)
        self._refresh_td_window_wavelet_views(panel_id, redraw_td=False)
        self._update_wavelet_job_widgets(panel_id)
        self._set_status(
            f"Opened TD window for {self.panels[panel_id - 1].name}. "
            "Use Show editor to open the linked TD/cut controls for that panel."
        )

    def _close_td_window(self, panel_id: int) -> None:
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return
        self._sync_panel_analysis_state_from_window(panel_id)
        self._close_td_window_roi_window(panel_id)
        self._close_td_window_diag_window(panel_id)
        existing = self.td_windows.pop(panel_id, None)
        if existing is None:
            return
        top = existing["top"]
        if top.winfo_exists():
            top.destroy()

    def _close_active_td_window(self) -> None:
        self._close_td_window(self.active_panel_id)

    def _refresh_td_window(self, panel_id: int) -> None:
        _trace_stack_wavelet(f"refresh_td_window start panel={panel_id}")
        existing = self.td_windows.get(panel_id)
        if existing is None:
            return

        top = existing["top"]
        if not top.winfo_exists():
            self._close_td_window_roi_window(panel_id)
            self._close_td_window_diag_window(panel_id)
            del self.td_windows[panel_id]
            return

        panel_name = self.panels[panel_id - 1].name
        top.title(f"TD Window - {panel_name}")
        ax = existing["ax"]
        self._sync_td_window_controls(panel_id)
        self._sync_td_window_crest_tracking(panel_id)
        self._refresh_td_window_crest_summary(panel_id)
        self._draw_td_axis(
            ax,
            panel_id,
            use_zoom=False,
            title_fontsize=11.0,
        )
        td, meta = self._panel_td(self.panels[panel_id - 1])
        self._draw_td_window_crest_overlay(ax, panel_id, meta)
        self._draw_td_window_roi(panel_id, td, meta)
        existing["canvas"].draw_idle()
        roi_canvas = existing.get("roi_canvas")
        if roi_canvas is not None:
            roi_canvas.draw_idle()
        _trace_stack_wavelet(f"refresh_td_window end panel={panel_id}")

    def _refresh_td_windows(self) -> None:
        for panel_id in list(self.td_windows.keys()):
            self._refresh_td_window(panel_id)

    def refresh_map(self) -> None:
        self._draw_map()
        self.canvas.draw_idle()

    def refresh_td_views(self) -> None:
        self._draw_td_panels()
        self.canvas.draw_idle()
        self._refresh_td_windows()
        self._refresh_metrics_window()
        self._refresh_link_groups_window()
        self._refresh_propagation_window()
        self._refresh_all_stack_browsers()

    def refresh_all(self) -> None:
        self._refresh_measurement_selectors()
        self._refresh_measurements()
        self._refresh_panel_cut_selector()
        self._sync_geometry_controls_from_selected_cut()
        self._refresh_export_controls()
        self._refresh_panel_list()
        self._refresh_cut_list()
        self._refresh_feature_axis_list()
        self._refresh_stack_list()
        self._refresh_stack_member_list()
        self._draw_map()
        self._draw_td_panels()
        self.canvas.draw_idle()
        self._refresh_td_windows()
        self._refresh_metrics_window()
        self._refresh_link_groups_window()
        self._refresh_propagation_window()
        self._refresh_all_stack_browsers()

    def _on_app_close(self) -> None:
        self._sync_all_panel_analysis_state_from_windows()
        self._cancel_all_background_jobs()
        try:
            self._write_session_file(self.autosave_path, autosave=True)
        except Exception:
            pass
        self._close_all_td_windows()
        self._close_metrics_window()
        self._close_link_groups_window()
        self._close_propagation_window()
        self._close_saved_fits_browser()
        self._close_all_stack_browsers()
        if self.root.winfo_exists():
            self.root.destroy()


def main() -> None:
    args = parse_args()
    try:
        app = TDMosaicApp(args.cube, cube_axis_order=args.cube_order)
        app.run()
    except Exception as exc:
        if exc.__class__.__name__ == "TclError":
            raise SystemExit(
                "Could not open the GUI. Start it from a session with a working display."
            ) from exc
        raise


if __name__ == "__main__":
    main()
