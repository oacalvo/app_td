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


class TDMosaicWaveletOverlayTests(unittest.TestCase):
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
