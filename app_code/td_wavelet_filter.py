#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np
import pywt
from scipy.optimize import minimize_scalar


DEFAULT_WAVELET_FILTER = {
    "p_min": 10.0,
    "p_max": 100.0,
    "power_ratio_thresh": 1.75,
    "segment_power_frac": 0.22,
    "min_points_segment": 12,
    "min_amp_arcsec": 0.03,
    "max_jump_pix": 2.5,
    "min_points_cut_seg": 6,
    "rms_amp_ratio_max": 1.1,
    "km_per_arcsec": 725.27,
    "density_kg_m3": float("nan"),
    "phase_speed_km_s": float("nan"),
}


def _mad_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1e-12
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    return 1.4826 * mad + 1e-12


def _detrend_poly(t: np.ndarray, y: np.ndarray, degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
    tt = np.asarray(t, dtype=np.float64) - float(np.nanmean(t))
    yy = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(tt) & np.isfinite(yy)
    valid_count = int(np.count_nonzero(finite))
    if valid_count < 2:
        trend = np.full_like(yy, float(np.nanmedian(yy)), dtype=np.float64)
        return yy - trend, trend

    max_degree = max(valid_count - 1, 0)
    degree = int(np.clip(int(degree), 0, max_degree))
    coeffs = np.polyfit(tt[finite], yy[finite], degree)
    trend = np.polyval(coeffs, tt)
    return yy - trend, trend


def _apply_detrend(
    t: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "poly",
    degree: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    yy = np.asarray(y, dtype=np.float64)
    if method == "none":
        trend = np.full_like(yy, float(np.nanmedian(yy)), dtype=np.float64)
        return yy - trend, trend
    if method == "poly":
        return _detrend_poly(t, yy, degree=degree)
    raise ValueError(f"Unknown detrend method: {method}")


def _cone_of_influence_frequency(t: np.ndarray, wavelet_name: str) -> np.ndarray:
    cf = float(pywt.central_frequency(wavelet_name))
    tt = np.asarray(t, dtype=np.float64)
    edge_distance = np.minimum(tt - tt[0], tt[-1] - tt)
    return cf * np.sqrt(2.0) / np.maximum(edge_distance, 1e-12)


def _continuous_ridge_indices(
    score: np.ndarray,
    valid_mask: np.ndarray,
    penalty: float,
) -> np.ndarray:
    n_freqs, n_times = score.shape
    freq_idx = np.arange(n_freqs, dtype=np.float64)
    dp = np.full((n_freqs, n_times), -np.inf, dtype=np.float64)
    parent = np.full((n_freqs, n_times), -1, dtype=np.int64)
    ridge = np.zeros(n_times, dtype=np.int64)
    valid_cols = np.any(valid_mask, axis=0)
    if not np.any(valid_cols):
        return ridge

    start_t = int(np.argmax(valid_cols))
    end_t = int(n_times - 1 - np.argmax(valid_cols[::-1]))
    dp[:, start_t] = np.where(valid_mask[:, start_t], score[:, start_t], -np.inf)

    for tidx in range(start_t + 1, end_t + 1):
        if not valid_cols[tidx]:
            continue
        prev = dp[:, tidx - 1]
        if not np.isfinite(prev).any():
            dp[:, tidx] = np.where(valid_mask[:, tidx], score[:, tidx], -np.inf)
            continue
        valid_now = np.flatnonzero(valid_mask[:, tidx])
        for fidx in valid_now:
            transitions = prev - penalty * (freq_idx - float(fidx)) ** 2
            best_prev = int(np.argmax(transitions))
            dp[fidx, tidx] = score[fidx, tidx] + transitions[best_prev]
            parent[fidx, tidx] = best_prev

    last_scores = dp[:, end_t]
    if np.isfinite(last_scores).any():
        ridge[end_t] = int(np.argmax(last_scores))
    else:
        fallback = np.sum(np.where(valid_mask, score, 0.0), axis=1)
        ridge[end_t] = int(np.argmax(fallback))

    for tidx in range(end_t, start_t, -1):
        prev = int(parent[ridge[tidx], tidx])
        ridge[tidx - 1] = ridge[tidx] if prev < 0 else prev

    ridge[:start_t] = ridge[start_t]
    ridge[end_t + 1 :] = ridge[end_t]
    return ridge


def _peak_frequency_from_spectrum(freq_grid: np.ndarray, spectrum: np.ndarray) -> float:
    finite = np.isfinite(spectrum)
    if not np.any(finite):
        return float("nan")

    spec = np.asarray(spectrum, dtype=np.float64)
    peak_idx = int(np.nanargmax(spec))
    if 0 < peak_idx < freq_grid.size - 1:
        y0, y1, y2 = spec[peak_idx - 1 : peak_idx + 2]
        if np.all(np.isfinite([y0, y1, y2])):
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                step = float(np.nanmedian(np.diff(freq_grid)))
                return float(freq_grid[peak_idx] + delta * step)
    return float(freq_grid[peak_idx])


def _spectrum_peak_indices(
    spectrum: np.ndarray,
    *,
    max_peaks: int = 3,
    min_spacing: int = 3,
) -> list[int]:
    spec = np.asarray(spectrum, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(spec))
    if finite_idx.size == 0:
        return []

    candidates: list[tuple[float, int]] = []
    for idx in finite_idx:
        left = spec[idx - 1] if idx > 0 else float("-inf")
        right = spec[idx + 1] if idx + 1 < spec.size else float("-inf")
        value = spec[idx]
        if value >= left and value >= right:
            candidates.append((float(value), int(idx)))
    if not candidates:
        peak_idx = int(np.nanargmax(spec))
        candidates = [(float(spec[peak_idx]), peak_idx)]

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[int] = []
    spacing = max(int(min_spacing), 1)
    for _value, idx in candidates:
        if any(abs(idx - other) < spacing for other in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= max(int(max_peaks), 1):
            break
    return selected


def _smooth_masked_series(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    window: int = 5,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    if arr.size == 0:
        return np.array([], dtype=np.float64)

    smoothed = np.full(arr.shape, float("nan"), dtype=np.float64)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return smoothed
    if valid_count == 1:
        smoothed[valid] = arr[valid]
        return smoothed

    valid_idx = np.flatnonzero(valid)
    filled = np.interp(
        np.arange(arr.size, dtype=np.float64),
        valid_idx.astype(np.float64),
        arr[valid_idx],
        left=float(arr[valid_idx[0]]),
        right=float(arr[valid_idx[-1]]),
    )
    smooth_window = min(max(int(window), 1), arr.size)
    if smooth_window % 2 == 0:
        smooth_window -= 1
    if smooth_window < 3:
        smoothed = np.asarray(filled, dtype=np.float64)
    else:
        kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
        pad = smooth_window // 2
        padded = np.pad(filled, pad, mode="edge")
        smoothed = np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)
    smoothed[~valid] = float("nan")
    return smoothed


def _bridge_short_false_gaps(mask: np.ndarray, *, max_gap: int) -> np.ndarray:
    bridged = np.asarray(mask, dtype=bool).copy()
    if bridged.size == 0 or max_gap <= 0:
        return bridged

    true_idx = np.flatnonzero(bridged)
    if true_idx.size < 2:
        return bridged

    prev = int(true_idx[0])
    for current_idx in true_idx[1:]:
        gap = int(current_idx) - prev - 1
        if 0 < gap <= int(max_gap):
            bridged[prev + 1 : int(current_idx)] = True
        prev = int(current_idx)
    return bridged


def _segments_from_active_mask(
    mask: np.ndarray,
    *,
    min_points: int,
) -> list[tuple[int, int]]:
    active_idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if active_idx.size < max(int(min_points), 1):
        return []

    segments: list[tuple[int, int]] = []
    start = int(active_idx[0])
    prev = int(active_idx[0])
    for idx in active_idx[1:]:
        idx = int(idx)
        if idx != prev + 1:
            if (prev - start + 1) >= int(min_points):
                segments.append((start, prev))
            start = idx
        prev = idx
    if (prev - start + 1) >= int(min_points):
        segments.append((start, prev))
    return segments


def _ridge_power_segments(
    t: np.ndarray,
    ridge_power: np.ndarray,
    ridge_valid: np.ndarray,
    *,
    segment_power_frac: float,
    min_points_segment: int,
) -> tuple[list[tuple[int, int]], list[tuple[float, float]], float, np.ndarray]:
    power_time = np.asarray(ridge_power, dtype=np.float64)
    valid = np.asarray(ridge_valid, dtype=bool) & np.isfinite(power_time)
    if power_time.size == 0 or np.count_nonzero(valid) == 0:
        return [], [], float("nan"), np.array([], dtype=np.float64)

    power_smooth = _smooth_masked_series(power_time, valid, window=5)
    threshold_source = power_smooth
    source_valid = valid & np.isfinite(threshold_source)
    if np.count_nonzero(source_valid) == 0:
        threshold_source = power_time
        source_valid = valid

    threshold = float(segment_power_frac) * float(np.nanmax(threshold_source[source_valid]))
    active_mask = valid & np.isfinite(threshold_source) & (threshold_source >= threshold)
    max_gap = max(1, min(2, int(max(min_points_segment, 1)) // 6))
    active_mask = _bridge_short_false_gaps(active_mask, max_gap=max_gap)
    segments_idx = _segments_from_active_mask(
        active_mask,
        min_points=max(int(min_points_segment), 1),
    )
    t_arr = np.asarray(t, dtype=np.float64)
    segments_time = [(float(t_arr[i0]), float(t_arr[i1])) for (i0, i1) in segments_idx]
    return segments_idx, segments_time, threshold, power_smooth


def _mode_segments_from_peak(
    t: np.ndarray,
    freq_grid: np.ndarray,
    power: np.ndarray,
    score_masked: np.ndarray,
    coi_valid: np.ndarray,
    global_ws: np.ndarray,
    mean_power: float,
    trend: np.ndarray,
    y_detr: np.ndarray,
    coi_boundary_freq: np.ndarray,
    *,
    peak_idx: int,
    mode_rank: int,
    power_ratio_thresh: float,
    segment_power_frac: float,
    min_points_segment: int,
    ridge_penalty: float,
    band_half_width: int,
) -> dict[str, Any]:
    peak_idx = int(np.clip(int(peak_idx), 0, freq_grid.size - 1))
    peak_freq = float(freq_grid[peak_idx])
    peak_period = float(1.0 / max(peak_freq, 1e-12))
    peak_power = float(global_ws[peak_idx])
    power_ratio = peak_power / mean_power if mean_power > 0 else float("inf")

    freq_band = np.abs(np.arange(freq_grid.size) - peak_idx) <= max(int(band_half_width), 1)
    mode_valid = coi_valid & freq_band[:, None]
    if not np.any(mode_valid):
        mode_valid = coi_valid

    ridge_idx = _continuous_ridge_indices(
        np.where(mode_valid, score_masked, -np.inf),
        mode_valid,
        ridge_penalty,
    )
    ridge_freqs = freq_grid[ridge_idx]
    ridge_periods = 1.0 / np.maximum(ridge_freqs, 1e-12)
    ridge_power = power[ridge_idx, np.arange(t.size)]
    power_time = ridge_power
    ridge_valid = mode_valid[ridge_idx, np.arange(t.size)]

    segments_idx, segments_time, threshold, power_time_smooth = _ridge_power_segments(
        t,
        power_time,
        ridge_valid,
        segment_power_frac=segment_power_frac,
        min_points_segment=min_points_segment,
    )

    return {
        "mode_rank": int(mode_rank),
        "peak_idx": int(peak_idx),
        "peak_freq": peak_freq,
        "peak_period": peak_period,
        "peak_power": peak_power,
        "mean_power": float(mean_power),
        "power_ratio": float(power_ratio),
        "is_wave_like": bool(power_ratio >= power_ratio_thresh) if np.isfinite(power_ratio) else False,
        "power_time": power_time,
        "power_time_smooth": power_time_smooth,
        "power_threshold": threshold,
        "ridge_periods": ridge_periods,
        "ridge_power": ridge_power,
        "segments_idx": segments_idx,
        "segments_time": segments_time,
        "trend": trend,
        "y_detr": y_detr,
        "coi_boundary_period": 1.0 / np.maximum(coi_boundary_freq, 1e-12),
    }


def _segment_bounds_from_jumps(
    y_idx: np.ndarray,
    *,
    max_jump_pix: float,
    min_points: int,
) -> list[tuple[int, int]]:
    if y_idx.size < min_points:
        return []

    jump_pos = np.where(np.abs(np.diff(y_idx)) > max_jump_pix)[0]
    if jump_pos.size == 0:
        return [(0, int(y_idx.size))]

    segments: list[tuple[int, int]] = []
    start = 0
    for jump_idx in jump_pos:
        end = int(jump_idx + 1)
        if end - start >= min_points:
            segments.append((start, end))
        start = end

    if y_idx.size - start >= min_points:
        segments.append((start, int(y_idx.size)))
    return segments


def split_thread_on_jumps(
    t_idx: np.ndarray,
    y_idx: np.ndarray,
    max_jump_pix: float = 1.5,
    min_points: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(t_idx)
    t_idx = np.asarray(t_idx, dtype=np.float64)[order]
    y_idx = np.asarray(y_idx, dtype=np.float64)[order]

    if t_idx.size < min_points:
        return []

    bounds = _segment_bounds_from_jumps(
        y_idx,
        max_jump_pix=max_jump_pix,
        min_points=min_points,
    )
    return [(t_idx[start:end], y_idx[start:end]) for start, end in bounds]


def fit_sine_with_trend(
    t: np.ndarray,
    y: np.ndarray,
    period_guess: float,
    *,
    fit_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    baseline_degree: int = 0,
) -> tuple[np.ndarray, float, float]:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size == 0 or y.size != t.size:
        raise ValueError("t and y must have the same non-zero length")
    if not np.isfinite(period_guess) or period_guess <= 0.0:
        raise ValueError("period_guess must be finite and positive")

    t_centered = t - np.mean(t)
    valid = np.isfinite(t_centered) & np.isfinite(y)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.size != t.size:
            raise ValueError("fit_mask must match t and y")
        valid &= fit_mask
    baseline_degree = max(int(baseline_degree), 0)
    min_required = baseline_degree + 3
    if np.count_nonzero(valid) < min_required:
        raise ValueError(f"at least {min_required} valid samples are required")

    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.size != t.size:
            raise ValueError("weights must match t and y")
        positive = np.isfinite(weights_arr[valid]) & (weights_arr[valid] > 0.0)
        if np.count_nonzero(positive) < min_required:
            raise ValueError(f"at least {min_required} positively weighted samples are required")

    def _fit_for_period(period: float) -> tuple[np.ndarray, float, float, float]:
        omega = 2.0 * np.pi / float(period)
        basis = [
            np.sin(omega * t_centered),
            np.cos(omega * t_centered),
        ]
        basis.extend(t_centered**power for power in range(baseline_degree + 1))
        design = np.column_stack(basis)
        design_fit = design[valid]
        y_fit = y[valid]
        if weights_arr is not None:
            w_fit = weights_arr[valid]
            positive = np.isfinite(w_fit) & (w_fit > 0.0)
            scale = np.sqrt(w_fit[positive])[:, None]
            beta, _, _, _ = np.linalg.lstsq(
                design_fit[positive] * scale,
                y_fit[positive] * scale[:, 0],
                rcond=None,
            )
        else:
            beta, _, _, _ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
        y_model = design @ beta
        amp = float(np.hypot(beta[0], beta[1]))
        residuals = y_fit - y_model[valid]
        mse = float(np.mean(residuals**2)) if residuals.size else float("inf")
        return y_model, amp, omega, mse

    y_model, amp, omega, best_mse = _fit_for_period(float(period_guess))

    valid_t = t[valid]
    if valid_t.size >= max(min_required + 1, 5):
        diffs = np.diff(valid_t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        dt = float(np.nanmedian(diffs)) if diffs.size else 1.0
        span = float(valid_t[-1] - valid_t[0])
        lower_period = max(2.0 * dt, 0.5 * float(period_guess))
        upper_period = min(
            2.5 * float(period_guess),
            max(1.15 * float(period_guess), 2.0 * span),
        )
        if np.isfinite(lower_period) and np.isfinite(upper_period) and upper_period > lower_period * 1.02:
            def _objective(period: float) -> float:
                try:
                    _model, _amp, _omega, mse = _fit_for_period(float(period))
                except Exception:
                    return float("inf")
                regularization = 0.03 * ((float(period) / float(period_guess)) - 1.0) ** 2
                return float(mse * (1.0 + regularization))

            result = minimize_scalar(
                _objective,
                bounds=(lower_period, upper_period),
                method="bounded",
                options={"xatol": max(1e-3, 0.01 * dt)},
            )
            if bool(result.success):
                refined_model, refined_amp, refined_omega, refined_mse = _fit_for_period(
                    float(result.x)
                )
                if np.isfinite(refined_mse) and refined_mse < best_mse * 0.995:
                    y_model = refined_model
                    amp = refined_amp
                    omega = refined_omega

    return y_model, amp, omega


def _smooth_positive_periods(periods: np.ndarray) -> np.ndarray:
    period_arr = np.asarray(periods, dtype=np.float64)
    valid = np.isfinite(period_arr) & (period_arr > 0.0)
    if np.count_nonzero(valid) < 2:
        return np.array([], dtype=np.float64)

    valid_idx = np.flatnonzero(valid)
    filled = np.interp(
        np.arange(period_arr.size, dtype=np.float64),
        valid_idx.astype(np.float64),
        period_arr[valid_idx],
        left=float(period_arr[valid_idx[0]]),
        right=float(period_arr[valid_idx[-1]]),
    )
    window = min(5, period_arr.size if period_arr.size % 2 == 1 else period_arr.size - 1)
    if window < 3:
        return np.asarray(filled, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(filled, pad, mode="edge")
    return np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)


def _wavelet_seed_period(
    ridge_periods: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    ridge_smoothed = _smooth_positive_periods(ridge_periods)
    if ridge_smoothed.size == 0:
        return float("nan")

    usable = np.isfinite(ridge_smoothed) & (ridge_smoothed > 0.0)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.size != ridge_smoothed.size:
            raise ValueError("valid_mask must match ridge_periods")
        usable &= valid_mask
    if np.count_nonzero(usable) == 0:
        return float("nan")
    return float(np.nanmedian(ridge_smoothed[usable]))


def fit_wavelet_guided_oscillation(
    t: np.ndarray,
    y: np.ndarray,
    period_guess: float,
    *,
    ridge_periods: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    baseline_degree: int = 0,
) -> tuple[np.ndarray, float, float]:
    if ridge_periods is None:
        return fit_sine_with_trend(
            t,
            y,
            period_guess,
            fit_mask=fit_mask,
            weights=weights,
            baseline_degree=baseline_degree,
        )

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ridge_periods = np.asarray(ridge_periods, dtype=np.float64)
    if ridge_periods.size != t.size:
        return fit_sine_with_trend(
            t,
            y,
            period_guess,
            fit_mask=fit_mask,
            weights=weights,
            baseline_degree=baseline_degree,
        )
    if t.size == 0 or y.size != t.size:
        raise ValueError("t and y must have the same non-zero length")

    valid = np.isfinite(t) & np.isfinite(y)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.size != t.size:
            raise ValueError("fit_mask must match t and y")
        valid &= fit_mask

    seed_period = _wavelet_seed_period(ridge_periods, valid_mask=valid)
    if not np.isfinite(seed_period) or seed_period <= 0.0:
        seed_period = float(period_guess)

    # Use the wavelet ridge only as a period seed; the final model is a simple
    # sinusoid with at least a linear trend over the selected segment.
    return fit_sine_with_trend(
        t,
        y,
        seed_period,
        fit_mask=fit_mask,
        weights=weights,
        baseline_degree=max(int(baseline_degree), 1),
    )


def compute_segment_physical_params(
    t: np.ndarray,
    y_arcsec: np.ndarray,
    peak_period: float,
    km_per_arcsec: float,
    density_kg_m3: float = float("nan"),
    phase_speed_km_s: float = float("nan"),
    ridge_periods: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    params = {
        "fit_amp_arcsec": float("nan"),
        "fit_amp_km": float("nan"),
        "fit_period_s": float("nan"),
        "peak_to_peak_arcsec": float("nan"),
        "peak_to_peak_km": float("nan"),
        "freq_hz": float("nan"),
        "freq_mhz": float("nan"),
        "omega_rad_s": float("nan"),
        "velocity_amp_km_s": float("nan"),
        "accel_amp_km_s2": float("nan"),
        "specific_energy_j_kg": float("nan"),
        "kinetic_energy_density_j_m3": float("nan"),
        "energy_flux_w_m2": float("nan"),
        "fit_rms_arcsec": float("nan"),
        "fit_rms_over_amp": float("nan"),
    }

    if (
        not np.isfinite(peak_period)
        or peak_period <= 0.0
        or not np.isfinite(km_per_arcsec)
        or km_per_arcsec <= 0.0
    ):
        return params

    t = np.asarray(t, dtype=np.float64)
    y_arcsec = np.asarray(y_arcsec, dtype=np.float64)
    if t.size < 3 or y_arcsec.size != t.size:
        return params

    try:
        y_model, fit_amp_arcsec, omega = fit_wavelet_guided_oscillation(
            t,
            y_arcsec,
            peak_period,
            ridge_periods=ridge_periods,
            fit_mask=fit_mask,
            weights=weights,
            baseline_degree=0,
        )
    except Exception:
        return params

    fit_eval = np.isfinite(t) & np.isfinite(y_arcsec)
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.size == t.size:
            fit_eval &= fit_mask
    residuals = y_arcsec[fit_eval] - y_model[fit_eval]
    fit_rms_arcsec = float(np.sqrt(np.mean(residuals**2))) if residuals.size else float("nan")
    fit_amp_km = float(fit_amp_arcsec * km_per_arcsec)
    fit_period_s = float((2.0 * np.pi) / abs(omega)) if abs(omega) > 1e-12 else float("nan")
    freq_hz = (
        float(1.0 / fit_period_s)
        if np.isfinite(fit_period_s) and fit_period_s > 0.0
        else float("nan")
    )
    peak_to_peak_arcsec = float(2.0 * fit_amp_arcsec)
    peak_to_peak_km = float(2.0 * fit_amp_km)
    velocity_amp_km_s = float(abs(omega) * fit_amp_km)
    accel_amp_km_s2 = float((omega**2) * abs(fit_amp_km))
    velocity_amp_m_s = float(1000.0 * velocity_amp_km_s)
    specific_energy_j_kg = float(0.5 * velocity_amp_m_s**2)
    kinetic_energy_density_j_m3 = float("nan")
    energy_flux_w_m2 = float("nan")
    if np.isfinite(density_kg_m3) and density_kg_m3 > 0.0:
        kinetic_energy_density_j_m3 = float(0.5 * density_kg_m3 * velocity_amp_m_s**2)
        if np.isfinite(phase_speed_km_s) and phase_speed_km_s > 0.0:
            energy_flux_w_m2 = float(
                kinetic_energy_density_j_m3 * (1000.0 * phase_speed_km_s)
            )

    params.update(
        {
            "fit_amp_arcsec": float(fit_amp_arcsec),
            "fit_amp_km": fit_amp_km,
            "fit_period_s": fit_period_s,
            "peak_to_peak_arcsec": peak_to_peak_arcsec,
            "peak_to_peak_km": peak_to_peak_km,
            "freq_hz": freq_hz,
            "freq_mhz": float(1000.0 * freq_hz),
            "omega_rad_s": float(omega),
            "velocity_amp_km_s": velocity_amp_km_s,
            "accel_amp_km_s2": accel_amp_km_s2,
            "specific_energy_j_kg": specific_energy_j_kg,
            "kinetic_energy_density_j_m3": kinetic_energy_density_j_m3,
            "energy_flux_w_m2": energy_flux_w_m2,
            "fit_rms_arcsec": fit_rms_arcsec,
            "fit_rms_over_amp": (
                float(fit_rms_arcsec / fit_amp_arcsec)
                if abs(fit_amp_arcsec) > 1e-12
                else float("inf")
            ),
        }
    )
    return params


def wavelet_select_segment(
    t: np.ndarray,
    y_arcsec: np.ndarray,
    *,
    p_min: float,
    p_max: float,
    power_ratio_thresh: float,
    segment_power_frac: float,
    min_points_segment: int,
    wavelet_name: str = "cmor1.5-1.0",
    n_scales: int = 60,
    detrend_method: str = "poly",
    detrend_degree: int = 1,
    edge_fraction: float = 0.12,
    ridge_penalty: float = 0.18,
) -> tuple[bool, list[tuple[float, float]], dict[str, Any]]:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y_arcsec, dtype=np.float64)

    if t.size < min_points_segment:
        return False, [], {}

    dt = float(np.median(np.diff(t)))
    total_span = float(t[-1] - t[0])
    if not np.isfinite(dt) or dt <= 0.0 or total_span <= 0.0:
        return False, [], {}

    try:
        y_detr, trend = _apply_detrend(
            t,
            y,
            method=detrend_method,
            degree=detrend_degree,
        )
    except Exception:
        return False, [], {}

    series_scale = _mad_std(y_detr)
    y_norm = y_detr / series_scale

    freq_min = 1.0 / max(float(p_max), 1e-9)
    freq_max = 1.0 / max(float(p_min), 1e-9)
    freq_grid = np.linspace(freq_min, freq_max, int(max(n_scales, 16)))
    cf = float(pywt.central_frequency(wavelet_name))
    scales = cf / (freq_grid * dt)
    coeffs, _ = pywt.cwt(y_norm, scales, wavelet_name, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    periods = 1.0 / np.maximum(freq_grid, 1e-12)

    edge = max(3, int(round(float(edge_fraction) * t.size)))
    if 2 * edge >= t.size:
        edge = max(1, t.size // 6)
    valid_time = np.zeros(t.size, dtype=bool)
    valid_time[edge : max(edge + 1, t.size - edge)] = True
    if not np.any(valid_time):
        valid_time[:] = True

    coi_boundary_freq = _cone_of_influence_frequency(t, wavelet_name)
    coi_valid = valid_time[None, :] & (freq_grid[:, None] >= coi_boundary_freq[None, :])

    global_ws = np.full(freq_grid.size, np.nan, dtype=np.float64)
    valid_counts = np.sum(coi_valid, axis=1)
    if np.any(valid_counts > 0):
        masked_power_sum = np.sum(np.where(coi_valid, power, 0.0), axis=1)
        supported = valid_counts > 0
        global_ws[supported] = masked_power_sum[supported] / valid_counts[supported]
    if np.all(~np.isfinite(global_ws)):
        global_ws = np.nanmean(power, axis=1)

    if not np.any(np.isfinite(global_ws)):
        return False, [], {}

    dom_freq = _peak_frequency_from_spectrum(freq_grid, global_ws)
    if not np.isfinite(dom_freq) or dom_freq <= 0.0:
        return False, [], {}
    mean_power = float(np.nanmean(global_ws))

    score = np.log(power + 1e-12)
    score -= np.nanmean(score, axis=0, keepdims=True)
    score_masked = np.where(coi_valid, score, -np.inf)
    dom_idx = int(np.nanargmin(np.abs(freq_grid - dom_freq)))
    peak_indices = _spectrum_peak_indices(
        global_ws,
        max_peaks=3,
        min_spacing=max(2, int(round(freq_grid.size / 12))),
    )
    if dom_idx not in peak_indices:
        peak_indices.insert(0, dom_idx)
    else:
        peak_indices = [dom_idx] + [idx for idx in peak_indices if idx != dom_idx]

    band_half_width = max(2, int(round(freq_grid.size / 10)))
    modes = [
        _mode_segments_from_peak(
            t,
            freq_grid,
            power,
            score_masked,
            coi_valid,
            global_ws,
            mean_power,
            trend,
            y_detr,
            coi_boundary_freq,
            peak_idx=peak_idx,
            mode_rank=mode_rank,
            power_ratio_thresh=power_ratio_thresh,
            segment_power_frac=segment_power_frac,
            min_points_segment=min_points_segment,
            ridge_penalty=ridge_penalty,
            band_half_width=band_half_width,
        )
        for mode_rank, peak_idx in enumerate(peak_indices)
    ]
    primary_mode = modes[0] if modes else None
    peak_period = float("nan") if primary_mode is None else float(primary_mode["peak_period"])
    peak_power = float("nan") if primary_mode is None else float(primary_mode["peak_power"])
    power_ratio = float("nan") if primary_mode is None else float(primary_mode["power_ratio"])
    is_wave_like = False if primary_mode is None else bool(primary_mode["is_wave_like"])
    ridge_periods = (
        np.array([], dtype=np.float64)
        if primary_mode is None
        else np.asarray(primary_mode["ridge_periods"], dtype=np.float64)
    )
    ridge_power = (
        np.array([], dtype=np.float64)
        if primary_mode is None
        else np.asarray(primary_mode["ridge_power"], dtype=np.float64)
    )
    power_time = (
        None
        if primary_mode is None
        else np.asarray(primary_mode["power_time"], dtype=np.float64)
    )
    threshold = (
        float("nan") if primary_mode is None else float(primary_mode["power_threshold"])
    )
    contiguous_segments = [] if primary_mode is None else list(primary_mode["segments_idx"])
    segments_time = [] if primary_mode is None else list(primary_mode["segments_time"])
    diag = {
        "periods": periods,
        "power": power,
        "global_ws": global_ws,
        "peak_period": peak_period,
        "peak_power": peak_power,
        "mean_power": mean_power,
        "power_ratio": power_ratio,
        "power_time": power_time,
        "power_t": t,
        "power_threshold": threshold,
        "t": t,
        "trend": trend,
        "y_detr": y_detr,
        "ridge_periods": ridge_periods,
        "ridge_power": ridge_power,
        "coi_boundary_period": 1.0 / np.maximum(coi_boundary_freq, 1e-12),
        "detrend_method": detrend_method,
        "segments_idx": contiguous_segments,
        "segments_time": segments_time,
        "modes": modes,
    }
    return is_wave_like, segments_time, diag


def _candidate_decision(
    *,
    has_segment: bool,
    is_wave_like: bool,
    power_ratio: float,
    power_ratio_thresh: float,
    amp_arcsec: float,
    min_amp_arcsec: float,
    rms_amp_ratio: float,
    rms_amp_ratio_max: float,
    point_count: int,
    duration_s: float = float("nan"),
    peak_period_s: float = float("nan"),
) -> tuple[bool, str, list[str]]:
    if not has_segment:
        return False, "no wavelet segment", []
    if point_count < 3:
        return False, "too few points", []
    if np.isfinite(min_amp_arcsec) and min_amp_arcsec > 0.0:
        if (not np.isfinite(amp_arcsec)) or amp_arcsec < min_amp_arcsec:
            return False, "low amplitude", []

    warnings: list[str] = []
    soft_power_floor = max(float(power_ratio_thresh) * 0.8, 1.15)
    hard_power_floor = 0.45
    strong_fit_support = (
        point_count >= 8
        and np.isfinite(duration_s)
        and np.isfinite(peak_period_s)
        and peak_period_s > 0.0
        and (duration_s / peak_period_s) >= 0.75
        and np.isfinite(amp_arcsec)
        and (
            (not np.isfinite(min_amp_arcsec))
            or min_amp_arcsec <= 0.0
            or amp_arcsec >= min_amp_arcsec
        )
        and np.isfinite(rms_amp_ratio)
        and np.isfinite(rms_amp_ratio_max)
        and rms_amp_ratio_max > 0.0
        and rms_amp_ratio <= (0.8 * rms_amp_ratio_max)
    )
    if (not np.isfinite(power_ratio)) or power_ratio < soft_power_floor:
        if (
            np.isfinite(power_ratio)
            and power_ratio >= hard_power_floor
            and strong_fit_support
        ):
            warnings.append("low power ratio")
        else:
            return False, "low power ratio", warnings
    if (not is_wave_like) or power_ratio < power_ratio_thresh:
        warnings.append("borderline power ratio")
    if (
        np.isfinite(rms_amp_ratio)
        and np.isfinite(rms_amp_ratio_max)
        and rms_amp_ratio > rms_amp_ratio_max
    ):
        warnings.append("high fit residual")
    return True, "accepted", warnings


def analyze_tracked_segment_with_wavelet(
    t_idx_seg: np.ndarray,
    y_idx_seg: np.ndarray,
    *,
    cadence: float,
    pix_scale: float,
    km_per_arcsec: float,
    p_min: float,
    p_max: float,
    power_ratio_thresh: float,
    segment_power_frac: float,
    min_points_segment: int,
    min_amp_arcsec: float,
    rms_amp_ratio_max: float,
    density_kg_m3: float = float("nan"),
    phase_speed_km_s: float = float("nan"),
    fit_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    t_idx_seg = np.asarray(t_idx_seg, dtype=np.float64)
    y_idx_seg = np.asarray(y_idx_seg, dtype=np.float64)
    if fit_mask is None:
        fit_mask = np.ones(t_idx_seg.shape, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.size != t_idx_seg.size:
            fit_mask = np.ones(t_idx_seg.shape, dtype=bool)
    if t_idx_seg.size == 0 or y_idx_seg.size != t_idx_seg.size:
        return {
            "candidates": [],
            "diag": {},
            "t_seg_s": np.array([], dtype=np.float64),
            "y_arcsec": np.array([], dtype=np.float64),
            "source_fit_mask": np.array([], dtype=bool),
        }

    t0_frames = float(t_idx_seg[0])
    t_seg = (t_idx_seg - t0_frames) * cadence
    y_arc = y_idx_seg * pix_scale

    is_wave, segments_time, diag = wavelet_select_segment(
        t_seg,
        y_arc,
        p_min=p_min,
        p_max=p_max,
        power_ratio_thresh=power_ratio_thresh,
        segment_power_frac=segment_power_frac,
        min_points_segment=min_points_segment,
    )
    power_ratio = float(diag.get("power_ratio", np.nan))
    peak_period = float(diag.get("peak_period", np.nan))
    peak_power = float(diag.get("peak_power", np.nan))
    mean_power = float(diag.get("mean_power", np.nan))
    raw_modes = [
        mode for mode in (diag.get("modes") or []) if isinstance(mode, dict)
    ]
    if not raw_modes:
        raw_modes = [
            {
                "mode_rank": 0,
                "peak_period": peak_period,
                "peak_power": peak_power,
                "mean_power": mean_power,
                "power_ratio": power_ratio,
                "is_wave_like": bool(is_wave),
                "segments_time": list(segments_time),
            }
        ]

    candidates: list[dict[str, Any]] = []
    if (
        not raw_modes
        or not any((mode.get("segments_time") or []) for mode in raw_modes)
        or "peak_period" not in diag
    ):
        primary_mode = raw_modes[0] if raw_modes else {}
        candidates.append(
            {
                "mode_rank": int(primary_mode.get("mode_rank", 0)),
                "wseg_id": -1,
                "has_segment": False,
                "accepted": False,
                "amp_arcsec": 0.0,
                "peak_period_s": float(primary_mode.get("peak_period", peak_period)),
                "power_ratio": float(primary_mode.get("power_ratio", power_ratio)),
                "peak_power": float(primary_mode.get("peak_power", peak_power)),
                "mean_power": float(primary_mode.get("mean_power", mean_power)),
                "duration_s": float("nan"),
                "wave_t_idx": np.array([], dtype=np.float64),
                "wave_y_idx": np.array([], dtype=np.float64),
                "wave_t_s": np.array([], dtype=np.float64),
                "wave_y_arcsec": np.array([], dtype=np.float64),
                "wave_y_detr_arcsec": np.array([], dtype=np.float64),
                "wave_model_arcsec": np.array([], dtype=np.float64),
                "wave_model_detr_arcsec": np.array([], dtype=np.float64),
                "rms_amp_ratio": float("nan"),
                "fit_point_count": 0,
                "interp_point_count": 0,
                "decision_reason": "no wavelet segment",
                "decision_warnings": [],
                **compute_segment_physical_params(
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    float(primary_mode.get("peak_period", peak_period)),
                    km_per_arcsec,
                    density_kg_m3=density_kg_m3,
                    phase_speed_km_s=phase_speed_km_s,
                ),
            }
        )
        return {
            "candidates": candidates,
            "diag": diag,
            "t_seg_s": t_seg,
            "y_arcsec": y_arc,
            "source_t_idx": t_idx_seg,
            "source_y_idx": y_idx_seg,
            "source_fit_mask": fit_mask,
        }

    for mode in raw_modes:
        mode_rank = int(mode.get("mode_rank", 0))
        mode_peak_period = float(mode.get("peak_period", peak_period))
        mode_peak_power = float(mode.get("peak_power", peak_power))
        mode_mean_power = float(mode.get("mean_power", mean_power))
        mode_power_ratio = float(mode.get("power_ratio", power_ratio))
        mode_is_wave = bool(mode.get("is_wave_like", is_wave))
        mode_trend = np.asarray(mode.get("trend", diag.get("trend", [])), dtype=np.float64)
        mode_y_detr = np.asarray(mode.get("y_detr", diag.get("y_detr", [])), dtype=np.float64)
        mode_ridge_periods = np.asarray(
            mode.get("ridge_periods", diag.get("ridge_periods", [])),
            dtype=np.float64,
        )
        for wseg_id, (t_start, t_end) in enumerate(mode.get("segments_time") or []):
            wave_mask = (t_seg >= t_start) & (t_seg <= t_end)
            wave_t_idx = t_idx_seg[wave_mask]
            wave_y_idx = y_idx_seg[wave_mask]
            wave_t_s = t_seg[wave_mask]
            wave_y_arc = y_arc[wave_mask]
            wave_fit_mask = fit_mask[wave_mask]
            wave_ridge_periods = (
                np.asarray(mode_ridge_periods[wave_mask], dtype=np.float64)
                if mode_ridge_periods.size == t_seg.size
                else np.array([], dtype=np.float64)
            )
            wave_trend = (
                mode_trend[wave_mask]
                if mode_trend.size == t_seg.size
                else np.zeros_like(wave_y_arc, dtype=np.float64)
            )
            has_segment = wave_t_idx.size > 0
            duration_s = float(t_end - t_start) if has_segment else float("nan")
            fit_point_count = int(np.count_nonzero(wave_fit_mask))
            interp_point_count = int(wave_fit_mask.size - fit_point_count)

            amp = 0.0
            accepted = False
            wave_y_detr = np.array([], dtype=np.float64)
            wave_model = np.array([], dtype=np.float64)
            wave_model_detr = np.array([], dtype=np.float64)
            rms_amp_ratio = float("nan")
            decision_warnings: list[str] = []
            if has_segment:
                if mode_y_detr.size == t_seg.size:
                    wave_y_detr = np.asarray(mode_y_detr[wave_mask], dtype=np.float64)
                else:
                    baseline = np.full_like(wave_y_arc, float(np.nanmedian(wave_y_arc)), dtype=np.float64)
                    wave_y_detr = wave_y_arc - baseline
            fit_params = compute_segment_physical_params(
                wave_t_s if has_segment else np.array([], dtype=np.float64),
                wave_y_detr if has_segment else np.array([], dtype=np.float64),
                mode_peak_period,
                km_per_arcsec,
                density_kg_m3=density_kg_m3,
                phase_speed_km_s=phase_speed_km_s,
                ridge_periods=(
                    wave_ridge_periods
                    if wave_ridge_periods.size == wave_t_s.size
                    else None
                ),
                fit_mask=wave_fit_mask if has_segment else None,
            )
            fit_amp_arcsec = float(fit_params.get("fit_amp_arcsec", float("nan")))
            amp_metric = fit_amp_arcsec if np.isfinite(fit_amp_arcsec) else 0.0
            span_amp_arcsec = float("nan")
            if has_segment and fit_point_count >= 3:
                try:
                    span_amp_arcsec = 0.5 * (
                        float(np.nanmax(wave_y_detr[wave_fit_mask]))
                        - float(np.nanmin(wave_y_detr[wave_fit_mask]))
                    )
                    amp_metric = (
                        fit_amp_arcsec if np.isfinite(fit_amp_arcsec) else span_amp_arcsec
                    )
                    amp = float(amp_metric)
                    if np.isfinite(mode_peak_period) and mode_peak_period > 0.0:
                        wave_model_detr, _, _ = fit_wavelet_guided_oscillation(
                            wave_t_s,
                            wave_y_detr,
                            mode_peak_period,
                            ridge_periods=(
                                wave_ridge_periods
                                if wave_ridge_periods.size == wave_t_s.size
                                else None
                            ),
                            fit_mask=wave_fit_mask,
                            baseline_degree=0,
                        )
                        wave_model = wave_model_detr + wave_trend
                except Exception:
                    amp = 0.0
                    wave_y_detr = np.array([], dtype=np.float64)
                    wave_model = np.array([], dtype=np.float64)
                    wave_model_detr = np.array([], dtype=np.float64)
                    span_amp_arcsec = float("nan")
                    decision_warnings.append("segment fit failed")

                if np.isfinite(mode_peak_period) and mode_peak_period > 0.0:
                    try:
                        wave_model_detr, fit_amp, _ = fit_wavelet_guided_oscillation(
                            wave_t_s,
                            wave_y_detr,
                            mode_peak_period,
                            ridge_periods=(
                                wave_ridge_periods
                                if wave_ridge_periods.size == wave_t_s.size
                                else None
                            ),
                            fit_mask=wave_fit_mask,
                            baseline_degree=0,
                        )
                        wave_model = wave_model_detr + wave_trend
                        residuals = wave_y_detr[wave_fit_mask] - wave_model_detr[wave_fit_mask]
                        fit_rms = float(np.sqrt(np.mean(residuals**2)))
                        rms_amp_ratio = (
                            fit_rms / fit_amp
                            if np.isfinite(fit_amp) and fit_amp > 1e-9
                            else float("inf")
                        )
                    except Exception:
                        rms_amp_ratio = float("nan")
                        decision_warnings.append("fit check unavailable")

            accepted, decision_reason, hard_warnings = _candidate_decision(
                has_segment=has_segment,
                is_wave_like=mode_is_wave,
                power_ratio=mode_power_ratio,
                power_ratio_thresh=power_ratio_thresh,
                amp_arcsec=amp,
                min_amp_arcsec=min_amp_arcsec,
                rms_amp_ratio=rms_amp_ratio,
                rms_amp_ratio_max=rms_amp_ratio_max,
                point_count=fit_point_count,
                duration_s=duration_s,
                peak_period_s=mode_peak_period,
            )
            decision_warnings.extend(hard_warnings)
            if decision_warnings:
                seen_warnings: set[str] = set()
                decision_warnings = [
                    warning
                    for warning in decision_warnings
                    if not (warning in seen_warnings or seen_warnings.add(warning))
                ]

            candidates.append(
                {
                    "mode_rank": mode_rank,
                    "wseg_id": wseg_id,
                    "has_segment": has_segment,
                    "accepted": accepted,
                    "amp_arcsec": float(amp),
                    "span_amp_arcsec": float(span_amp_arcsec),
                    "amplitude_method": "fit",
                    "fit_model": (
                        "wavelet-seeded-sine"
                        if wave_ridge_periods.size == wave_t_s.size
                        else "sine"
                    ),
                    "peak_period_s": mode_peak_period,
                    "power_ratio": mode_power_ratio,
                    "peak_power": mode_peak_power,
                    "mean_power": mode_mean_power,
                    "duration_s": duration_s,
                    "wave_t_idx": np.asarray(wave_t_idx, dtype=np.float64),
                    "wave_y_idx": np.asarray(wave_y_idx, dtype=np.float64),
                    "wave_t_s": np.asarray(wave_t_s, dtype=np.float64),
                    "wave_y_arcsec": np.asarray(wave_y_arc, dtype=np.float64),
                    "wave_y_detr_arcsec": np.asarray(wave_y_detr, dtype=np.float64),
                    "wave_model_arcsec": np.asarray(wave_model, dtype=np.float64),
                    "wave_model_detr_arcsec": np.asarray(wave_model_detr, dtype=np.float64),
                    "rms_amp_ratio": rms_amp_ratio,
                    "fit_point_count": fit_point_count,
                    "interp_point_count": interp_point_count,
                    "decision_reason": decision_reason,
                    "decision_warnings": decision_warnings,
                    **fit_params,
                }
            )

    return {
        "candidates": candidates,
        "diag": diag,
        "t_seg_s": t_seg,
        "y_arcsec": y_arc,
        "source_t_idx": t_idx_seg,
        "source_y_idx": y_idx_seg,
        "source_fit_mask": fit_mask,
    }


def analyze_tracked_threads_with_wavelets(
    threads: list[dict[str, Any]],
    t_indices: np.ndarray,
    *,
    cadence: float,
    pix_scale: float,
    km_per_arcsec: float,
    p_min: float,
    p_max: float,
    power_ratio_thresh: float,
    segment_power_frac: float,
    min_points_segment: int,
    min_amp_arcsec: float,
    max_jump_pix: float,
    min_points_cut_seg: int,
    rms_amp_ratio_max: float,
    density_kg_m3: float = float("nan"),
    phase_speed_km_s: float = float("nan"),
) -> list[dict[str, Any]]:
    t_indices = np.asarray(t_indices, dtype=np.float64)
    results: list[dict[str, Any]] = []

    for thread_index, thread in enumerate(threads):
        pos = np.asarray(thread.get("pos", []), dtype=np.float64)
        if pos.size != t_indices.size:
            continue
        bin_flags = np.asarray(thread.get("bin_flags", []), dtype=np.int64)
        if bin_flags.size != pos.size:
            bin_flags = np.full(pos.shape, 2, dtype=np.int64)

        valid = np.isfinite(pos) & (pos >= 0.0)
        if np.count_nonzero(valid) < min_points_cut_seg:
            continue

        t_valid = np.asarray(t_indices[valid], dtype=np.float64)
        y_valid = np.asarray(pos[valid], dtype=np.float64)
        fit_valid = np.asarray(bin_flags[valid] != 1, dtype=bool)

        order = np.argsort(t_valid)
        t_valid = t_valid[order]
        y_valid = y_valid[order]
        fit_valid = fit_valid[order]

        bounds = _segment_bounds_from_jumps(
            y_valid,
            max_jump_pix=max_jump_pix,
            min_points=min_points_cut_seg,
        )

        for seg_id, (start, end) in enumerate(bounds):
            t_idx_seg = t_valid[start:end]
            y_idx_seg = y_valid[start:end]
            fit_mask_seg = fit_valid[start:end]
            analysis = analyze_tracked_segment_with_wavelet(
                t_idx_seg,
                y_idx_seg,
                cadence=cadence,
                pix_scale=pix_scale,
                km_per_arcsec=km_per_arcsec,
                p_min=p_min,
                p_max=p_max,
                power_ratio_thresh=power_ratio_thresh,
                segment_power_frac=segment_power_frac,
                min_points_segment=min_points_segment,
                min_amp_arcsec=min_amp_arcsec,
                rms_amp_ratio_max=rms_amp_ratio_max,
                density_kg_m3=density_kg_m3,
                phase_speed_km_s=phase_speed_km_s,
                fit_mask=fit_mask_seg,
            )

            for candidate in analysis["candidates"]:
                results.append(
                    {
                        "thread_index": thread_index,
                        "seg_id": seg_id,
                        **candidate,
                        "source_t_idx": np.asarray(t_idx_seg, dtype=np.float64),
                        "source_y_idx": np.asarray(y_idx_seg, dtype=np.float64),
                        "source_fit_mask": np.asarray(fit_mask_seg, dtype=bool),
                    }
                )
    from collections import Counter
    print("decision reasons:", Counter(r["decision_reason"] for r in results))
    return results


__all__ = [
    "DEFAULT_WAVELET_FILTER",
    "analyze_tracked_threads_with_wavelets",
    "analyze_tracked_segment_with_wavelet",
    "compute_segment_physical_params",
    "fit_wavelet_guided_oscillation",
    "fit_sine_with_trend",
    "split_thread_on_jumps",
    "wavelet_select_segment",
]
