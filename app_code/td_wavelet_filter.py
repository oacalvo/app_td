#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np
import pywt


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

    jump_pos = np.where(np.abs(np.diff(y_idx)) > max_jump_pix)[0]
    if jump_pos.size == 0:
        return [(t_idx, y_idx)] if t_idx.size >= min_points else []

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for jump_idx in jump_pos:
        end = jump_idx + 1
        if end - start >= min_points:
            segments.append((t_idx[start:end], y_idx[start:end]))
        start = jump_idx + 1

    if t_idx.size - start >= min_points:
        segments.append((t_idx[start:], y_idx[start:]))
    return segments


def fit_sine_with_trend(
    t: np.ndarray,
    y: np.ndarray,
    period_guess: float,
) -> tuple[np.ndarray, float, float]:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    t_centered = t - np.mean(t)
    omega = 2.0 * np.pi / period_guess
    design = np.column_stack(
        [
            np.sin(omega * t_centered),
            np.cos(omega * t_centered),
            np.ones_like(t_centered),
            t_centered,
        ]
    )
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    y_model = design @ beta
    amp = float(np.hypot(beta[0], beta[1]))
    return y_model, amp, omega


def compute_segment_physical_params(
    t: np.ndarray,
    y_arcsec: np.ndarray,
    peak_period: float,
    km_per_arcsec: float,
    density_kg_m3: float = float("nan"),
    phase_speed_km_s: float = float("nan"),
) -> dict[str, float]:
    params = {
        "fit_amp_arcsec": float("nan"),
        "fit_amp_km": float("nan"),
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

    y_model, fit_amp_arcsec, omega = fit_sine_with_trend(t, y_arcsec, peak_period)
    residuals = y_arcsec - y_model
    fit_rms_arcsec = float(np.sqrt(np.mean(residuals**2)))
    fit_amp_km = float(fit_amp_arcsec * km_per_arcsec)
    freq_hz = float(1.0 / peak_period)
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
    scale_min: float = 1.0,
    scale_max: float = 1000.0,
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
        coef = np.polyfit(t, y, deg=2)
        trend = np.polyval(coef, t)
    except Exception:
        return False, [], {}

    y_detr = y - trend
    y_norm = (y_detr - np.mean(y_detr)) / (np.std(y_detr) + 1e-9)

    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), int(n_scales))
    coeffs, freqs = pywt.cwt(y_norm, scales, wavelet_name, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    periods = 1.0 / (freqs + 1e-9)
    global_ws = np.nanmean(power, axis=1)

    period_mask = (periods >= p_min) & (periods <= p_max)
    if not np.any(period_mask):
        return False, [], {}

    period_sel = periods[period_mask]
    global_sel = global_ws[period_mask]
    if not np.any(np.isfinite(global_sel)):
        return False, [], {}

    peak_idx_sel = int(np.nanargmax(global_sel))
    peak_period = float(period_sel[peak_idx_sel])
    peak_power = float(global_sel[peak_idx_sel])
    mean_power = float(np.nanmean(global_sel))
    power_ratio = peak_power / mean_power if mean_power > 0 else float("inf")
    is_wave_like = power_ratio >= power_ratio_thresh

    full_idx = int(np.where(period_mask)[0][peak_idx_sel])
    power_time = power[full_idx, :]
    if np.all(np.isnan(power_time)):
        diag = {
            "periods": periods,
            "power": power,
            "global_ws": global_ws,
            "peak_period": peak_period,
            "peak_power": peak_power,
            "mean_power": mean_power,
            "power_ratio": power_ratio,
            "power_time": None,
            "power_threshold": float("nan"),
            "t": t,
            "trend": trend,
            "y_detr": y_detr,
            "segments_idx": [],
            "segments_time": [],
        }
        return False, [], diag

    threshold = float(segment_power_frac) * float(np.nanmax(power_time))
    idx = np.where(power_time >= threshold)[0]
    if idx.size < min_points_segment:
        diag = {
            "periods": periods,
            "power": power,
            "global_ws": global_ws,
            "peak_period": peak_period,
            "peak_power": peak_power,
            "mean_power": mean_power,
            "power_ratio": power_ratio,
            "power_time": power_time,
            "power_threshold": threshold,
            "t": t,
            "trend": trend,
            "y_detr": y_detr,
            "segments_idx": [],
            "segments_time": [],
        }
        return False, [], diag

    contiguous_segments: list[tuple[int, int]] = []
    start = int(idx[0])
    for pos in range(1, idx.size):
        if idx[pos] != idx[pos - 1] + 1:
            contiguous_segments.append((start, int(idx[pos - 1])))
            start = int(idx[pos])
    contiguous_segments.append((start, int(idx[-1])))
    contiguous_segments = [
        (i0, i1)
        for (i0, i1) in contiguous_segments
        if (i1 - i0 + 1) >= min_points_segment
    ]

    segments_time = [(float(t[i0]), float(t[i1])) for (i0, i1) in contiguous_segments]
    diag = {
        "periods": periods,
        "power": power,
        "global_ws": global_ws,
        "peak_period": peak_period,
        "peak_power": peak_power,
        "mean_power": mean_power,
        "power_ratio": power_ratio,
        "power_time": power_time,
        "power_threshold": threshold,
        "t": t,
        "trend": trend,
        "y_detr": y_detr,
        "segments_idx": contiguous_segments,
        "segments_time": segments_time,
    }
    return is_wave_like, segments_time, diag


def _decision_reason(
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
) -> str:
    if not has_segment:
        return "no wavelet segment"
    if point_count < 3:
        return "too few points"
    if not is_wave_like or power_ratio < power_ratio_thresh:
        return "low power ratio"
    if amp_arcsec < min_amp_arcsec:
        return "low amplitude"
    if np.isfinite(rms_amp_ratio) and rms_amp_ratio > rms_amp_ratio_max:
        return "high fit residual"
    return "accepted"


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
) -> dict[str, Any]:
    t_idx_seg = np.asarray(t_idx_seg, dtype=np.float64)
    y_idx_seg = np.asarray(y_idx_seg, dtype=np.float64)
    if t_idx_seg.size == 0 or y_idx_seg.size != t_idx_seg.size:
        return {"candidates": [], "diag": {}, "t_seg_s": np.array([], dtype=np.float64), "y_arcsec": np.array([], dtype=np.float64)}

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

    candidates: list[dict[str, Any]] = []
    if not segments_time or "peak_period" not in diag:
        candidates.append(
            {
                "wseg_id": -1,
                "has_segment": False,
                "accepted": False,
                "amp_arcsec": 0.0,
                "peak_period_s": peak_period,
                "power_ratio": power_ratio,
                "peak_power": peak_power,
                "mean_power": mean_power,
                "duration_s": float("nan"),
                "wave_t_idx": np.array([], dtype=np.float64),
                "wave_y_idx": np.array([], dtype=np.float64),
                "wave_t_s": np.array([], dtype=np.float64),
                "wave_y_arcsec": np.array([], dtype=np.float64),
                "wave_y_detr_arcsec": np.array([], dtype=np.float64),
                "wave_model_arcsec": np.array([], dtype=np.float64),
                "rms_amp_ratio": float("nan"),
                "decision_reason": "no wavelet segment",
                **compute_segment_physical_params(
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    peak_period,
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
        }

    for wseg_id, (t_start, t_end) in enumerate(segments_time):
        wave_mask = (t_seg >= t_start) & (t_seg <= t_end)
        wave_t_idx = t_idx_seg[wave_mask]
        wave_y_idx = y_idx_seg[wave_mask]
        wave_t_s = t_seg[wave_mask]
        wave_y_arc = y_arc[wave_mask]
        has_segment = wave_t_idx.size > 0
        duration_s = float(t_end - t_start) if has_segment else float("nan")

        amp = 0.0
        accepted = False
        wave_y_detr = np.array([], dtype=np.float64)
        wave_model = np.array([], dtype=np.float64)
        rms_amp_ratio = float("nan")
        fit_params = compute_segment_physical_params(
            wave_t_s if has_segment else np.array([], dtype=np.float64),
            wave_y_arc if has_segment else np.array([], dtype=np.float64),
            peak_period,
            km_per_arcsec,
            density_kg_m3=density_kg_m3,
            phase_speed_km_s=phase_speed_km_s,
        )
        if has_segment and wave_t_idx.size >= 3:
            try:
                coef = np.polyfit(wave_t_s, wave_y_arc, deg=2)
                trend = np.polyval(coef, wave_t_s)
                wave_y_detr = wave_y_arc - trend
                amp = 0.5 * (
                    float(np.nanmax(wave_y_detr)) - float(np.nanmin(wave_y_detr))
                )
                if np.isfinite(peak_period) and peak_period > 0.0:
                    wave_model, _, _ = fit_sine_with_trend(
                        wave_t_s, wave_y_arc, peak_period
                    )
            except Exception:
                amp = 0.0
                wave_y_detr = np.array([], dtype=np.float64)
                wave_model = np.array([], dtype=np.float64)

            if is_wave and power_ratio >= power_ratio_thresh and amp >= min_amp_arcsec:
                accepted = True

            if accepted and np.isfinite(peak_period) and peak_period > 0.0:
                try:
                    coef = np.polyfit(wave_t_s, wave_y_arc, deg=2)
                    trend = np.polyval(coef, wave_t_s)
                    wave_y_detr = wave_y_arc - trend
                    wave_model, fit_amp, _ = fit_sine_with_trend(
                        wave_t_s, wave_y_arc, peak_period
                    )
                    residuals = wave_y_arc - wave_model
                    fit_rms = float(np.sqrt(np.mean(residuals**2)))
                    rms_amp_ratio = (
                        fit_rms / fit_amp if np.isfinite(fit_amp) and fit_amp > 1e-9 else float("inf")
                    )
                    if rms_amp_ratio > rms_amp_ratio_max:
                        accepted = False
                except Exception:
                    rms_amp_ratio = float("inf")
                    accepted = False

        candidates.append(
            {
                "wseg_id": wseg_id,
                "has_segment": has_segment,
                "accepted": accepted,
                "amp_arcsec": float(amp),
                "peak_period_s": peak_period,
                "power_ratio": power_ratio,
                "peak_power": peak_power,
                "mean_power": mean_power,
                "duration_s": duration_s,
                "wave_t_idx": np.asarray(wave_t_idx, dtype=np.float64),
                "wave_y_idx": np.asarray(wave_y_idx, dtype=np.float64),
                "wave_t_s": np.asarray(wave_t_s, dtype=np.float64),
                "wave_y_arcsec": np.asarray(wave_y_arc, dtype=np.float64),
                "wave_y_detr_arcsec": np.asarray(wave_y_detr, dtype=np.float64),
                "wave_model_arcsec": np.asarray(wave_model, dtype=np.float64),
                "rms_amp_ratio": rms_amp_ratio,
                "decision_reason": _decision_reason(
                    has_segment=has_segment,
                    is_wave_like=bool(is_wave),
                    power_ratio=power_ratio,
                    power_ratio_thresh=power_ratio_thresh,
                    amp_arcsec=amp,
                    min_amp_arcsec=min_amp_arcsec,
                    rms_amp_ratio=rms_amp_ratio,
                    rms_amp_ratio_max=rms_amp_ratio_max,
                    point_count=int(wave_t_idx.size),
                ),
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

        valid = np.isfinite(pos) & (pos >= 0.0)
        if np.count_nonzero(valid) < min_points_cut_seg:
            continue

        t_valid = t_indices[valid]
        y_valid = pos[valid]
        split_segments = split_thread_on_jumps(
            t_valid,
            y_valid,
            max_jump_pix=max_jump_pix,
            min_points=min_points_cut_seg,
        )

        for seg_id, (t_idx_seg, y_idx_seg) in enumerate(split_segments):
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
            )

            for candidate in analysis["candidates"]:
                results.append(
                    {
                        "thread_index": thread_index,
                        "seg_id": seg_id,
                        **candidate,
                        "source_t_idx": np.asarray(t_idx_seg, dtype=np.float64),
                        "source_y_idx": np.asarray(y_idx_seg, dtype=np.float64),
                    }
                )

    return results


__all__ = [
    "DEFAULT_WAVELET_FILTER",
    "analyze_tracked_threads_with_wavelets",
    "analyze_tracked_segment_with_wavelet",
    "compute_segment_physical_params",
    "fit_sine_with_trend",
    "split_thread_on_jumps",
    "wavelet_select_segment",
]
