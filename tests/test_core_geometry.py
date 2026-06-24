from __future__ import annotations

import unittest

import numpy as np

from app_code.core import Cut, compute_td, cut_point_at_distance


class CoreGeometryTests(unittest.TestCase):
    def test_cut_point_at_distance_uses_curve_arc_length(self) -> None:
        cut = Cut(
            cut_id=1,
            name="Curve",
            color="red",
            p0=(1.0, 1.0),
            p1=(4.0, 4.0),
            mode="curve",
            curve_points=[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0)],
        )

        self.assertEqual(cut_point_at_distance(cut, 4.0), (4.0, 2.0))

    def test_compute_td_samples_curved_cut_by_arc_length(self) -> None:
        y_grid, x_grid = np.mgrid[0:8, 0:8]
        frame = (x_grid + y_grid).astype(np.float32)
        cube = frame[None, :, :]
        cut = Cut(
            cut_id=1,
            name="Curve",
            color="red",
            p0=(1.0, 1.0),
            p1=(4.0, 4.0),
            mode="curve",
            curve_points=[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0)],
        )

        td, meta = compute_td(
            cube,
            cut,
            t_ini=0,
            t_fin=0,
            stride=1,
            width=1,
            weighting="uniform",
        )

        np.testing.assert_allclose(meta["distances"], np.arange(7, dtype=np.float32))
        np.testing.assert_allclose(
            td[0],
            np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
