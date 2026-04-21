from __future__ import annotations

import unittest

import numpy as np

from app_code.td_wavelet_filter import (
    DEFAULT_WAVELET_FILTER,
    _candidate_decision,
    _apply_detrend,
    analyze_tracked_segment_with_wavelet,
    fit_sine_with_trend,
    fit_wavelet_guided_oscillation,
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

    def test_cubic_trend_does_not_flatten_segment_fit_amplitude(self) -> None:
        rng = np.random.default_rng(7)
        t_idx = np.arange(120, dtype=np.float64)
        centered = t_idx - np.mean(t_idx)
        cubic_trend = 0.00035 * centered**3 / np.max(np.abs(centered))
        y_idx = (
            8.0
            + cubic_trend
            + 4.0 * np.sin(2.0 * np.pi * t_idx / 24.0 + 0.4)
            + 0.18 * rng.normal(size=t_idx.size)
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

        accepted = [candidate for candidate in analysis["candidates"] if candidate.get("accepted")]
        self.assertTrue(accepted)
        best = max(
            accepted,
            key=lambda candidate: float(candidate.get("fit_amp_arcsec", float("-inf"))),
        )
        self.assertGreater(float(best.get("fit_amp_arcsec", 0.0)), 0.12)
        self.assertEqual(
            len(best.get("wave_model_detr_arcsec", [])),
            len(best.get("wave_y_detr_arcsec", [])),
        )

    def test_fit_sine_refines_short_period_guess_for_single_cycle_signal(self) -> None:
        rng = np.random.default_rng(11)
        t_idx = np.arange(28, dtype=np.float64)
        true_period = 24.0
        true_amp = 1.8
        y_idx = (
            true_amp * np.sin(2.0 * np.pi * (t_idx - np.mean(t_idx)) / true_period + 0.5)
            + 0.08 * rng.normal(size=t_idx.size)
        )

        y_model, fit_amp, omega = fit_sine_with_trend(
            t_idx,
            y_idx,
            14.0,
            baseline_degree=0,
        )

        fit_period = (2.0 * np.pi) / abs(omega)
        self.assertGreater(fit_amp, 1.35)
        self.assertAlmostEqual(fit_period, true_period, delta=3.0)
        self.assertEqual(len(y_model), len(y_idx))

    def test_linear_detrend_preserves_clear_sine_with_small_drift(self) -> None:
        rng = np.random.default_rng(23)
        t_idx = np.arange(80, dtype=np.float64)
        sine = 1.5 * np.sin(2.0 * np.pi * t_idx / 24.0 + 0.3)
        drift = 0.01 * (t_idx - np.mean(t_idx))
        y_idx = sine + drift + 0.05 * rng.normal(size=t_idx.size)

        y_detr, trend = _apply_detrend(t_idx, y_idx, method="poly", degree=1)

        recovered_amp = 0.5 * (
            float(np.nanmax(y_detr)) - float(np.nanmin(y_detr))
        )
        trend_error = float(np.sqrt(np.mean((trend - drift) ** 2)))

        self.assertGreater(recovered_amp, 1.35)
        self.assertLess(trend_error, 0.20)

    def test_wavelet_guided_oscillation_beats_rigid_sine_for_variable_period(self) -> None:
        t_idx = np.arange(90, dtype=np.float64)
        ridge_periods = 16.0 + 8.0 * (t_idx / float(t_idx[-1]))
        omega_inst = 2.0 * np.pi / ridge_periods
        phase = np.zeros_like(t_idx)
        phase[1:] = np.cumsum(0.5 * (omega_inst[1:] + omega_inst[:-1]) * np.diff(t_idx))
        y_idx = 1.4 * np.sin(phase + 0.35)

        guided_model, guided_amp, _ = fit_wavelet_guided_oscillation(
            t_idx,
            y_idx,
            float(np.nanmedian(ridge_periods)),
            ridge_periods=ridge_periods,
            baseline_degree=0,
        )
        rigid_model, rigid_amp, _ = fit_sine_with_trend(
            t_idx,
            y_idx,
            float(np.nanmedian(ridge_periods)),
            baseline_degree=0,
        )

        guided_mse = float(np.mean((y_idx - guided_model) ** 2))
        rigid_mse = float(np.mean((y_idx - rigid_model) ** 2))

        self.assertGreater(guided_amp, 1.0)
        self.assertGreater(rigid_amp, 0.2)
        self.assertLess(guided_mse, rigid_mse * 0.35)


if __name__ == "__main__":
    unittest.main()
