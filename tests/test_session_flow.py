from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app_code.td_mosaic_app import TDMosaicApp, _session_startup_defaults


class _DummyMessageBox:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, str]] = []

    def showwarning(self, title: str, message: str) -> None:
        self.warnings.append((title, message))


class SessionFlowTests(unittest.TestCase):
    def test_session_startup_defaults_reads_cube_and_axis_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.json"
            cube_path = Path(tmpdir) / "cube.fits"
            payload = {
                "cube_path": str(cube_path),
                "cube_axis_order": "TYX",
            }
            session_path.write_text(json.dumps(payload), encoding="utf-8")

            resolved_cube, resolved_order = _session_startup_defaults(session_path)

            self.assertEqual(resolved_cube, cube_path.resolve())
            self.assertEqual(resolved_order, "TYX")

    def test_load_session_restarts_with_relocated_cube_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            session_path = tmp / "session.json"
            old_cube = tmp / "old_location.fits"
            replacement_cube = tmp / "replacement.fits"
            replacement_cube.write_text("placeholder", encoding="utf-8")
            session_path.write_text(
                json.dumps(
                    {
                        "cube_path": str(old_cube),
                        "cube_axis_order": "TYX",
                    }
                ),
                encoding="utf-8",
            )

            restart_calls: list[dict[str, object]] = []
            choose_calls: list[dict[str, object]] = []
            confirm_calls: list[dict[str, object]] = []

            dummy = SimpleNamespace()
            dummy.cube_path = tmp / "current.fits"
            dummy.cube_axis_order = "TYX"
            dummy.messagebox = _DummyMessageBox()

            def _choose_cube_dialog(**kwargs):
                choose_calls.append(kwargs)
                return replacement_cube, "TYX"

            def _confirm_restart_with_warning(**kwargs):
                confirm_calls.append(kwargs)
                return True

            def _restart_app_with_cube(
                cube_path: Path,
                cube_axis_order: str,
                *,
                session_path: Path | None = None,
                session_cube_override: Path | None = None,
            ) -> None:
                restart_calls.append(
                    {
                        "cube_path": cube_path,
                        "cube_axis_order": cube_axis_order,
                        "session_path": session_path,
                        "session_cube_override": session_cube_override,
                    }
                )

            dummy._choose_cube_dialog = _choose_cube_dialog
            dummy._confirm_restart_with_warning = _confirm_restart_with_warning
            dummy._restart_app_with_cube = _restart_app_with_cube

            TDMosaicApp._load_session_from_path(dummy, session_path)

            self.assertEqual(len(dummy.messagebox.warnings), 1)
            self.assertEqual(len(choose_calls), 1)
            self.assertEqual(len(confirm_calls), 1)
            self.assertEqual(len(restart_calls), 1)
            self.assertEqual(restart_calls[0]["cube_path"], replacement_cube)
            self.assertEqual(restart_calls[0]["cube_axis_order"], "TYX")
            self.assertEqual(restart_calls[0]["session_path"], session_path.resolve())
            self.assertEqual(
                restart_calls[0]["session_cube_override"], replacement_cube
            )

    def test_load_session_restarts_with_existing_foreign_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            session_path = tmp / "session.json"
            foreign_cube = tmp / "foreign.fits"
            foreign_cube.write_text("placeholder", encoding="utf-8")
            session_path.write_text(
                json.dumps(
                    {
                        "cube_path": str(foreign_cube),
                        "cube_axis_order": "TYX",
                    }
                ),
                encoding="utf-8",
            )

            restart_calls: list[dict[str, object]] = []
            confirm_calls: list[dict[str, object]] = []

            dummy = SimpleNamespace()
            dummy.cube_path = tmp / "current.fits"
            dummy.cube_axis_order = "TYX"
            dummy.messagebox = _DummyMessageBox()

            def _confirm_restart_with_warning(**kwargs):
                confirm_calls.append(kwargs)
                return True

            def _restart_app_with_cube(
                cube_path: Path,
                cube_axis_order: str,
                *,
                session_path: Path | None = None,
                session_cube_override: Path | None = None,
            ) -> None:
                restart_calls.append(
                    {
                        "cube_path": cube_path,
                        "cube_axis_order": cube_axis_order,
                        "session_path": session_path,
                        "session_cube_override": session_cube_override,
                    }
                )

            dummy._confirm_restart_with_warning = _confirm_restart_with_warning
            dummy._restart_app_with_cube = _restart_app_with_cube

            TDMosaicApp._load_session_from_path(dummy, session_path)

            self.assertEqual(len(confirm_calls), 1)
            self.assertEqual(len(restart_calls), 1)
            self.assertEqual(restart_calls[0]["cube_path"], foreign_cube.resolve())
            self.assertEqual(restart_calls[0]["cube_axis_order"], "TYX")
            self.assertEqual(restart_calls[0]["session_path"], session_path.resolve())
            self.assertIsNone(restart_calls[0]["session_cube_override"])


if __name__ == "__main__":
    unittest.main()
