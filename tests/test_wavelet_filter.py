from __future__ import annotations

import unittest

import numpy as np

from app_code.td_wavelet_filter import (
    DEFAULT_WAVELET_FILTER,
    _candidate_decision,
    _apply_detrend,
    _prefer_full_source_fit,
    _ridge_power_segments,
    analyze_tracked_segment_with_wavelet,
    fit_sine_with_trend,
    fit_wavelet_guided_oscillation,
)


class WaveletFilterTests(unittest.TestCase):
    def test_ridge_power_segments_ignore_isolated_spike_when_cutting_segment(self) -> None:
        t_idx = np.arange(24, dtype=np.float64)
        ridge_power = np.full(t_idx.shape, 0.45, dtype=np.float64)
        ridge_power[6:18] = 4.0
        ridge_power[11] = 20.0
        ridge_valid = np.ones(t_idx.shape, dtype=bool)

        segments_idx, segments_time, threshold, power_smooth = _ridge_power_segments(
            t_idx,
            ridge_power,
            ridge_valid,
            segment_power_frac=DEFAULT_WAVELET_FILTER["segment_power_frac"],
            min_points_segment=5,
        )

        self.assertEqual(len(segments_idx), 1)
        start, end = segments_idx[0]
        self.assertLessEqual(start, 7)
        self.assertGreaterEqual(end, 16)
        self.assertEqual(segments_time[0], (float(t_idx[start]), float(t_idx[end])))
        self.assertTrue(np.isfinite(threshold))
        self.assertLess(threshold, 4.0)
        self.assertLess(float(np.nanmax(power_smooth)), 10.0)

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

    def test_candidate_rejects_low_amplitude_even_with_good_power(self) -> None:
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

        self.assertFalse(accepted)
        self.assertEqual(reason, "low amplitude")
        self.assertEqual(warnings, [])

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

    def test_candidate_keeps_strong_temporal_fit_despite_low_power_ratio(self) -> None:
        accepted, reason, warnings = _candidate_decision(
            has_segment=True,
            is_wave_like=False,
            power_ratio=0.50,
            power_ratio_thresh=DEFAULT_WAVELET_FILTER["power_ratio_thresh"],
            amp_arcsec=0.036,
            min_amp_arcsec=DEFAULT_WAVELET_FILTER["min_amp_arcsec"],
            rms_amp_ratio=0.50,
            rms_amp_ratio_max=DEFAULT_WAVELET_FILTER["rms_amp_ratio_max"],
            point_count=18,
            duration_s=28.0,
            peak_period_s=20.0,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertIn("low power ratio", warnings)
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

    def test_full_source_fit_can_override_wavelet_crop_when_cleaner(self) -> None:
        source_support = {
            "fit_params": {
                "fit_amp_arcsec": 0.070,
                "fit_period_s": 18.0,
            },
            "rms_amp_ratio": 0.34,
            "fit_point_count": 18,
            "duration_s": 30.0,
            "span_amp_arcsec": 0.075,
        }
        wave_support = {
            "fit_params": {
                "fit_amp_arcsec": 0.062,
                "fit_period_s": 18.0,
            },
            "rms_amp_ratio": 0.48,
            "fit_point_count": 10,
            "duration_s": 20.0,
            "span_amp_arcsec": 0.074,
        }

        self.assertTrue(
            _prefer_full_source_fit(
                source_support,
                wave_support,
                min_points_segment=8,
            )
        )

    def test_wavelet_guided_oscillation_uses_ridge_seed_and_linear_trend(self) -> None:
        rng = np.random.default_rng(29)
        t_idx = np.arange(60, dtype=np.float64)
        true_period = 42.0
        ridge_periods = true_period + 1.5 * np.sin(2.0 * np.pi * t_idx / float(t_idx[-1]))
        centered = t_idx - np.mean(t_idx)
        y_idx = (
            1.6 * np.sin(2.0 * np.pi * centered / true_period + 0.4)
            + 0.035 * centered
            + 0.04 * rng.normal(size=t_idx.size)
        )

        guided_model, guided_amp, guided_omega = fit_wavelet_guided_oscillation(
            t_idx,
            y_idx,
            14.0,
            ridge_periods=ridge_periods,
            baseline_degree=1,
        )
        rigid_model, rigid_amp, rigid_omega = fit_sine_with_trend(
            t_idx,
            y_idx,
            14.0,
            baseline_degree=1,
        )

        guided_period = (2.0 * np.pi) / abs(guided_omega)
        rigid_period = (2.0 * np.pi) / abs(rigid_omega)
        guided_mse = float(np.mean((y_idx - guided_model) ** 2))
        rigid_mse = float(np.mean((y_idx - rigid_model) ** 2))

        self.assertGreater(guided_amp, 1.1)
        self.assertLess(rigid_amp, guided_amp * 0.25)
        self.assertAlmostEqual(guided_period, true_period, delta=3.0)
        self.assertLess(rigid_period, 35.5)
        self.assertLess(guided_mse, rigid_mse * 0.3)

    def test_wavelet_guided_oscillation_respects_zero_baseline_for_detrended_signal(self) -> None:
        rng = np.random.default_rng(31)
        t_idx = np.arange(54, dtype=np.float64)
        true_period = 20.0
        ridge_periods = true_period + 0.8 * np.sin(2.0 * np.pi * t_idx / float(t_idx[-1]))
        y_detr = (
            1.45 * np.sin(2.0 * np.pi * (t_idx - np.mean(t_idx)) / true_period + 0.2)
            + 0.05 * rng.normal(size=t_idx.size)
        )

        model, fit_amp, omega = fit_wavelet_guided_oscillation(
            t_idx,
            y_detr,
            18.0,
            ridge_periods=ridge_periods,
            baseline_degree=0,
        )

        fit_period = (2.0 * np.pi) / abs(omega)
        self.assertGreater(fit_amp, 1.0)
        self.assertAlmostEqual(fit_period, true_period, delta=2.5)
        self.assertEqual(len(model), len(y_detr))


if __name__ == "__main__":
    unittest.main()
