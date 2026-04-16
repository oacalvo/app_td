from __future__ import annotations

import unittest

import numpy as np

from app_code.td_wavelet_filter import (
    DEFAULT_WAVELET_FILTER,
    _candidate_decision,
    analyze_tracked_segment_with_wavelet,
)


class WaveletFilterTests(unittest.TestCase):
    def test_candidate_rejects_only_when_support_or_power_is_too_low(self) -> None:
        accepted, reason, warnings = _candidate_decision(
            has_segment=True,
            is_wave_like=False,
            power_ratio=1.10,
            power_ratio_thresh=DEFAULT_WAVELET_FILTER["power_ratio_thresh"],
            amp_arcsec=0.20,
            min_amp_arcsec=DEFAULT_WAVELET_FILTER["min_amp_arcsec"],
            rms_amp_ratio=0.4,
            rms_amp_ratio_max=DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"],
            point_count=16,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "low power ratio")
        self.assertEqual(warnings, [])

    def test_candidate_keeps_low_amplitude_and_residual_as_warnings(self) -> None:
        accepted, reason, warnings = _candidate_decision(
            has_segment=True,
            is_wave_like=True,
            power_ratio=2.10,
            power_ratio_thresh=DEFAULT_WAVELET_FILTER["power_ratio_thresh"],
            amp_arcsec=0.01,
            min_amp_arcsec=DEFAULT_WAVELET_FILTER["min_amp_arcsec"],
            rms_amp_ratio=1.35,
            rms_amp_ratio_max=DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"],
            point_count=18,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertIn("low amplitude", warnings)
        self.assertIn("high fit residual", warnings)

    def test_candidate_accepts_borderline_power_with_warning(self) -> None:
        accepted, reason, warnings = _candidate_decision(
            has_segment=True,
            is_wave_like=False,
            power_ratio=1.45,
            power_ratio_thresh=DEFAULT_WAVELET_FILTER["power_ratio_thresh"],
            amp_arcsec=0.08,
            min_amp_arcsec=DEFAULT_WAVELET_FILTER["min_amp_arcsec"],
            rms_amp_ratio=0.5,
            rms_amp_ratio_max=DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"],
            point_count=14,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertIn("borderline power ratio", warnings)

    def test_clear_sine_wave_is_detected_with_default_filter(self) -> None:
        rng = np.random.default_rng(1234)
        t_idx = np.arange(96, dtype=np.float64)
        y_idx = (
            10.0
            + 4.0 * np.sin(2.0 * np.pi * t_idx / 24.0)
            + 0.25 * rng.normal(size=t_idx.size)
        )

        analysis = analyze_tracked_segment_with_wavelet(
            t_idx,
            y_idx,
            cadence=1.0,
            pix_scale=0.05,
            km_per_arcsec=DEFAULT_WAVELET_FILTER["km_per_arcsec"],
            p_min=10.0,
            p_max=40.0,
            power_ratio_thresh=DEFAULT_WAVELET_FILTER["power_ratio_thresh"],
            segment_power_frac=DEFAULT_WAVELET_FILTER["segment_power_frac"],
            min_points_segment=DEFAULT_WAVELET_FILTER["min_points_segment"],
            min_amp_arcsec=DEFAULT_WAVELET_FILTER["min_amp_arcsec"],
            rms_amp_ratio_max=DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"],
        )

        candidates = analysis["candidates"]
        self.assertTrue(candidates)
        self.assertTrue(any(candidate.get("has_segment") for candidate in candidates))
        self.assertTrue(any(candidate.get("accepted") for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
