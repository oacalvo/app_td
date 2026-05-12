from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app_code.core import Cut, TDPanel
from app_code.td_mosaic_app import TDMosaicApp


class _FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class ExhaustiveStudyTests(unittest.TestCase):
    def test_generate_experiment_applies_explicit_td_width_and_weighting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = TDMosaicApp.__new__(TDMosaicApp)
            app.nt = 12
            app.nx = 128
            app.ny = 128
            app.cube_path = Path(tmpdir) / "cube.fits"
            app.cut_analysis_state = {}
            app.panel_analysis_state = {}
            app.stacks = {}
            app.experiments = {}
            app.next_experiment_id = 1
            app.next_stack_id = 1
            app.next_cut_id = 2
            app.active_stack_id = None
            app.selected_stack_cut_id = None
            app.active_experiment_id = None
            app.selected_cut_id = 1
            app.active_panel_id = 1
            app.t_visual_var = _FakeVar(0)
            app.panel_width_var = _FakeVar("1")
            app.panel_weighting_var = _FakeVar("uniform")
            app.experiment_name_var = _FakeVar("Width Study")
            app.experiment_displacement_step_var = _FakeVar("5")
            app.experiment_displacement_limit_var = _FakeVar("0")
            app.experiment_angle_min_var = _FakeVar("0")
            app.experiment_angle_max_var = _FakeVar("0")
            app.experiment_angle_step_var = _FakeVar("5")
            app.experiment_width_var = _FakeVar("7")
            app.experiment_weighting_var = _FakeVar("gaussian")
            app.experiment_save_td_fits_var = _FakeVar(True)
            app.experiment_save_cell_json_var = _FakeVar(True)
            app.experiment_save_important_images_var = _FakeVar(True)
            app.panels = [
                TDPanel(
                    panel_id=1,
                    name="P1",
                    cut_id=1,
                    t_ini=2,
                    t_fin=10,
                    stride=2,
                    width=1,
                    weighting="uniform",
                )
            ]
            app.cuts = {
                1: Cut(
                    cut_id=1,
                    name="Base",
                    color="red",
                    p0=(10.0, 20.0),
                    p1=(60.0, 20.0),
                )
            }

            app._record_session_change = lambda: None
            app.refresh_all = lambda: None
            app._set_status = lambda _text: None
            app._resolved_export_dir = lambda: Path(tmpdir)
            app._safe_export_slug = lambda text: str(text).replace(" ", "_")
            app._cut_preview = lambda cut_id, frame_idx: app.cuts[cut_id]

            def _create_cut(p0, p1):
                cut_id = int(app.next_cut_id)
                app.next_cut_id += 1
                cut = Cut(
                    cut_id=cut_id,
                    name=f"Cut {cut_id}",
                    color="red",
                    p0=(float(p0[0]), float(p0[1])),
                    p1=(float(p1[0]), float(p1[1])),
                )
                app.cuts[cut_id] = cut
                return cut

            app._create_cut = _create_cut

            app._generate_exhaustive_experiment()

            self.assertIn(1, app.experiments)
            experiment = app.experiments[1]
            self.assertEqual(experiment["td_width"], 7)
            self.assertEqual(experiment["td_weighting"], "gaussian")
            generated_cut_id = int(experiment["cut_ids"][0])
            params = app._cut_td_params(generated_cut_id)
            self.assertEqual(params["width"], 7)
            self.assertEqual(params["weighting"], "gaussian")


if __name__ == "__main__":
    unittest.main()
