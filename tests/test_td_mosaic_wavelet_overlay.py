from __future__ import annotations

import unittest

import numpy as np

from app_code.td_mosaic_app import TDMosaicApp


class _FakeTree:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.selection_value: tuple[str, ...] = ()
        self.focus_value: str | None = None

    def get_children(self) -> tuple[str, ...]:
        return tuple(str(row["iid"]) for row in self.rows)

    def delete(self, *children: str) -> None:
        child_set = set(children)
        self.rows = [row for row in self.rows if str(row["iid"]) not in child_set]

    def insert(self, _parent: str, _index: str, **kwargs: object) -> None:
        self.rows.append(dict(kwargs))

    def selection_set(self, iid: str) -> None:
        self.selection_value = (iid,)

    def focus(self, iid: str) -> None:
        self.focus_value = iid


class _FakeAxis:
    def __init__(self) -> None:
        self.plots: list[dict[str, object]] = []

    def plot(self, *args: object, **kwargs: object) -> None:
        self.plots.append({"args": args, "kwargs": kwargs})


class _FakeVar:
    def __init__(self) -> None:
        self.value: object = None

    def set(self, value: object) -> None:
        self.value = value


class TDMosaicWaveletOverlayTests(unittest.TestCase):
    def test_hidden_advanced_wavelet_filters_no_longer_apply_from_stale_state(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        app.nt = 8
        app.panels = []
        app.cuts = {}
        app.td_windows = {}
        app.panel_analysis_state = {
            3: {
                "wavelet_advanced_filters": {
                    "qa": "clean",
                    "score_min": "95",
                    "period_min": "20",
                }
            }
        }

        filters = app._td_window_wavelet_advanced_filter_values(3)

        self.assertEqual(filters["qa"], "all")
        self.assertEqual(filters["locked"], "all")
        self.assertEqual(filters["linked"], "all")
        self.assertTrue(np.isnan(filters["score_min"]))
        self.assertTrue(np.isnan(filters["period_min"]))

    def test_best_wavelet_segment_prefers_cleaner_fit_over_higher_power_ratio(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        params = {
            "power_ratio_thresh": 1.75,
            "min_points_segment": 5,
            "rms_amp_ratio_max": 1.1,
        }
        cleaner = {
            "accepted": True,
            "power_ratio": 2.4,
            "duration_s": 42.0,
            "peak_period_s": 18.0,
            "fit_amp_arcsec": 0.055,
            "fit_rms_over_amp": 0.22,
            "fit_point_count": 11,
            "mode_rank": 1,
        }
        noisier = {
            "accepted": True,
            "power_ratio": 3.9,
            "duration_s": 42.0,
            "peak_period_s": 18.0,
            "fit_amp_arcsec": 0.055,
            "fit_rms_over_amp": 1.05,
            "fit_point_count": 11,
            "mode_rank": 0,
        }

        best = app._best_wavelet_segment([noisier, cleaner], params=params)

        self.assertIs(best, cleaner)

    def test_wavelet_rerun_replaces_previous_events_instead_of_appending(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        app.panels = []
        app.td_windows = {}
        app.panel_analysis_state = {
            3: {
                "wavelet_filter_result": {"segment_count": 4},
                "wavelet_events": [
                    {
                        "event_id": 7,
                        "review_locked": True,
                        "analysis": {
                            "thread_index": 9,
                            "seg_id": 9,
                            "wseg_id": 9,
                        },
                    }
                ],
                "wavelet_next_event_id": 8,
                "wavelet_selected_event_id": 7,
            }
        }
        app._record_session_change = lambda: None
        app._set_status = lambda _text: None
        params = {
            "power_ratio_thresh": 1.75,
            "min_points_segment": 3,
            "rms_amp_ratio_max": 1.1,
        }
        segment = {
            "thread_index": 0,
            "seg_id": 0,
            "wseg_id": 1,
            "accepted": True,
            "has_segment": True,
            "mode_rank": 0,
            "duration_s": 24.0,
            "peak_period_s": 12.0,
            "fit_amp_arcsec": 0.05,
            "fit_amp_km": 36.0,
            "fit_rms_over_amp": 0.25,
            "power_ratio": 2.2,
            "freq_mhz": 83.3,
            "velocity_amp_km_s": 1.4,
            "source_t_idx": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "source_y_idx": np.array([0.0, 0.8, 0.1, -0.7], dtype=np.float64),
            "wave_t_idx": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "wave_y_idx": np.array([0.0, 0.8, 0.1, -0.7], dtype=np.float64),
        }

        app._apply_background_wavelet_results(3, [segment], params, "P3")

        state = app.panel_analysis_state[3]
        self.assertEqual(len(state["wavelet_events"]), 1)
        self.assertEqual(state["wavelet_events"][0]["event_id"], 1)
        self.assertFalse(state["wavelet_events"][0]["review_locked"])
        self.assertEqual(state["wavelet_next_event_id"], 2)
        self.assertEqual(state["wavelet_selected_event_id"], 1)

    def test_replacement_wavelet_payload_preserves_distinct_modes_per_source_segment(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        params = {
            "power_ratio_thresh": 1.75,
            "min_points_segment": 5,
            "rms_amp_ratio_max": 1.1,
        }
        better = {
            "thread_index": 4,
            "seg_id": 1,
            "wseg_id": 0,
            "accepted": True,
            "has_segment": True,
            "mode_rank": 0,
            "duration_s": 36.0,
            "peak_period_s": 22.0,
            "fit_amp_arcsec": 0.080,
            "fit_rms_over_amp": 0.20,
            "fit_point_count": 12,
            "power_ratio": 2.8,
            "source_t_idx": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "source_y_idx": np.array([0.0, 0.8, 0.0, -0.7, 0.0], dtype=np.float64),
            "wave_t_idx": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "wave_y_idx": np.array([0.0, 0.8, 0.0, -0.7, 0.0], dtype=np.float64),
        }
        alternate = {
            "thread_index": 4,
            "seg_id": 1,
            "wseg_id": 1,
            "accepted": True,
            "has_segment": True,
            "mode_rank": 1,
            "duration_s": 34.0,
            "peak_period_s": 22.0,
            "fit_amp_arcsec": 0.050,
            "fit_rms_over_amp": 0.48,
            "fit_point_count": 12,
            "power_ratio": 3.0,
            "source_t_idx": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "source_y_idx": np.array([0.0, 0.8, 0.0, -0.7, 0.0], dtype=np.float64),
            "wave_t_idx": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "wave_y_idx": np.array([0.0, 0.8, 0.0, -0.7, 0.0], dtype=np.float64),
        }
        duplicate = {
            **better,
            "fit_amp_arcsec": 0.060,
            "fit_rms_over_amp": 0.35,
            "power_ratio": 2.3,
        }
        other_source = {
            "thread_index": 4,
            "seg_id": 2,
            "wseg_id": 0,
            "accepted": True,
            "has_segment": True,
            "mode_rank": 0,
            "duration_s": 20.0,
            "peak_period_s": 18.0,
            "fit_amp_arcsec": 0.040,
            "fit_rms_over_amp": 0.25,
            "fit_point_count": 8,
            "power_ratio": 2.2,
            "source_t_idx": np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
            "source_y_idx": np.array([0.0, -0.4, 0.0, 0.3], dtype=np.float64),
            "wave_t_idx": np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
            "wave_y_idx": np.array([0.0, -0.4, 0.0, 0.3], dtype=np.float64),
        }

        payload = app._replacement_wavelet_run_payload(
            [alternate, better, duplicate, other_source],
            params,
        )

        kept = payload["filtered_segments"]
        self.assertEqual(len(kept), 3)
        self.assertEqual(
            {
                (int(item["seg_id"]), int(item["mode_rank"]), int(item["wseg_id"]))
                for item in kept
            },
            {(1, 0, 0), (1, 1, 1), (2, 0, 0)},
        )
        self.assertIn(
            "removed overlapping duplicate wavelet candidate(s): 1",
            payload["warnings"],
        )

    def test_best_wavelet_event_id_includes_mode_rank(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        params = {
            "power_ratio_thresh": 1.75,
            "min_points_segment": 5,
            "rms_amp_ratio_max": 1.1,
        }
        base = {
            "thread_index": 2,
            "seg_id": 3,
            "wseg_id": 0,
            "accepted": True,
            "has_segment": True,
            "duration_s": 24.0,
            "peak_period_s": 12.0,
            "fit_point_count": 12,
            "source_t_idx": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "source_y_idx": np.array([0.0, 0.3, 0.0, -0.3], dtype=np.float64),
            "wave_t_idx": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "wave_y_idx": np.array([0.0, 0.3, 0.0, -0.3], dtype=np.float64),
        }
        weaker = {
            **base,
            "mode_rank": 0,
            "fit_amp_arcsec": 0.040,
            "fit_rms_over_amp": 0.55,
            "power_ratio": 2.0,
        }
        stronger = {
            **base,
            "mode_rank": 1,
            "fit_amp_arcsec": 0.080,
            "fit_rms_over_amp": 0.18,
            "power_ratio": 2.6,
        }

        payload = app._replacement_wavelet_run_payload([weaker, stronger], params)

        self.assertEqual(len(payload["events"]), 2)
        self.assertEqual(payload["best_event_id"], 2)

    def test_refresh_wavelet_views_uses_event_params_for_best_candidate(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        app.td_windows = {
            3: {
                "wavelet_summary_var": _FakeVar(),
                "wavelet_physics_var": _FakeVar(),
                "wavelet_filter_result": {"segment_count": 1},
            }
        }
        app._refresh_td_window_wavelet_table = lambda panel_id: None
        app._td_window_wavelet_events = lambda panel_id: [
            {
                "event_id": 1,
                "analysis": {
                    "thread_index": 0,
                    "seg_id": 1,
                    "wseg_id": 0,
                    "accepted": False,
                    "has_segment": True,
                    "mode_rank": 0,
                    "duration_s": 18.0,
                    "peak_period_s": 12.0,
                    "fit_amp_arcsec": 0.04,
                    "fit_rms_over_amp": 0.45,
                    "fit_point_count": 7,
                    "power_ratio": 1.4,
                },
                "current_params": {
                    "power_ratio_thresh": 1.75,
                    "min_points_segment": 5,
                    "rms_amp_ratio_max": 1.1,
                },
            }
        ]
        app._background_wavelet_job_for_panel = lambda panel_id: (None, None)
        app._td_window_wavelet_event_is_counted = lambda event: False
        app._format_wavelet_segment_physics = lambda analysis, prefix: prefix
        app._refresh_td_window_wavelet_diagnostics = lambda panel_id: None
        app._refresh_td_window = lambda panel_id: None

        app._refresh_td_window_wavelet_views(3, redraw_td=False)

        self.assertEqual(app.td_windows[3]["wavelet_physics_var"].value, "Best candidate:")

    def test_overlay_segment_uses_wave_points_when_available(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        event = {
            "analysis": {
                "wave_t_idx": np.array([10.0, 11.0, 12.0], dtype=np.float64),
                "wave_y_idx": np.array([2.0, 2.5, 3.0], dtype=np.float64),
            },
            "source_t_idx": np.array([20.0, 21.0], dtype=np.float64),
            "source_y_idx": np.array([4.0, 5.0], dtype=np.float64),
        }

        t_idx, y_idx = app._wavelet_event_overlay_samples(
            event, allow_source_fallback=True
        )

        np.testing.assert_array_equal(t_idx, np.array([10.0, 11.0, 12.0]))
        np.testing.assert_array_equal(y_idx, np.array([2.0, 2.5, 3.0]))

    def test_overlay_segment_uses_source_points_for_selected_rejected_event(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        event = {
            "analysis": {
                "wave_t_idx": np.array([], dtype=np.float64),
                "wave_y_idx": np.array([], dtype=np.float64),
            },
            "source_t_idx": np.array([20.0, 21.0, 22.0], dtype=np.float64),
            "source_y_idx": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        }

        t_idx, y_idx = app._wavelet_event_overlay_samples(
            event, allow_source_fallback=True
        )

        np.testing.assert_array_equal(t_idx, np.array([20.0, 21.0, 22.0]))
        np.testing.assert_array_equal(y_idx, np.array([4.0, 5.0, 6.0]))

    def test_overlay_segment_keeps_source_trace_hidden_without_fallback(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        event = {
            "analysis": {
                "wave_t_idx": np.array([], dtype=np.float64),
                "wave_y_idx": np.array([], dtype=np.float64),
            },
            "source_t_idx": np.array([20.0, 21.0, 22.0], dtype=np.float64),
            "source_y_idx": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        }

        t_idx, y_idx = app._wavelet_event_overlay_samples(
            event, allow_source_fallback=False
        )

        self.assertEqual(t_idx.size, 0)
        self.assertEqual(y_idx.size, 0)

    def test_stack_browser_tree_uses_wavelet_status_tags(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        app.cuts = {7: object()}
        app._cut_wavelet_events_snapshot = lambda cut_id: [
            {
                "event_id": 12,
                "manual_decision": "rejected",
                "analysis": {},
            }
        ]
        app._stack_browser_event_row = lambda event: ("12", "-", "-", "manual rejected")

        tree = _FakeTree()
        browser = {"tree_updating": False}

        TDMosaicApp._refresh_stack_browser_tree(app, tree, 7, browser, "current", None)

        self.assertEqual(len(tree.rows), 1)
        self.assertEqual(tree.rows[0]["tags"], ("manual_rejected",))
        self.assertEqual(browser["current_selected_event_id"], 12)
        self.assertEqual(tree.selection_value, ("current-12",))

    def test_td_editor_overlay_draws_selected_rejected_event_from_source_trace(self) -> None:
        app = TDMosaicApp.__new__(TDMosaicApp)
        app.td_windows = {
            3: {
                "crest_tracking_result": {
                    "threads": [
                        {"pos": np.array([1.0, 1.0, 1.0], dtype=np.float64)},
                    ]
                },
                "wavelet_events": [
                    {
                        "event_id": 21,
                        "manual_decision": "rejected",
                        "analysis": {
                            "wave_t_idx": np.array([], dtype=np.float64),
                            "wave_y_idx": np.array([], dtype=np.float64),
                        },
                        "source_t_idx": np.array([0.0, 1.0, 2.0], dtype=np.float64),
                        "source_y_idx": np.array([0.2, 0.4, 0.6], dtype=np.float64),
                    }
                ],
                "wavelet_selected_event_id": 21,
            }
        }
        app.td_swap_axes_var = type("Var", (), {"get": lambda self: False})()

        ax = _FakeAxis()
        meta = {
            "distances": np.array([0.0, 10.0, 20.0], dtype=np.float64),
            "t_indices": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        }

        app._draw_td_window_crest_overlay(ax, 3, meta)

        self.assertGreaterEqual(len(ax.plots), 2)
        selected_plot = ax.plots[-1]
        self.assertEqual(selected_plot["kwargs"]["color"], "gold")
        np.testing.assert_array_equal(
            np.asarray(selected_plot["args"][0], dtype=np.float64),
            np.array([2.0, 4.0, 6.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            np.asarray(selected_plot["args"][1], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
        )


if __name__ == "__main__":
    unittest.main()
