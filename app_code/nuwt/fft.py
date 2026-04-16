import numpy as np
from scipy.stats import chi2

from .statistics import signif_noise_spec, signif_noise_spec_bootstrap, test_residuals


def _get_apod_window(tn, p=0.4, win_function="split_cosine_bell"):
    w_t = np.ones(tn, dtype=np.float64)
    if win_function in ("split_cosine_bell", "hann"):
        num_taper = int(tn * (p / 2.0))
        if num_taper > 0:
            w_t[:num_taper] = np.sin(np.pi * np.arange(num_taper) / (tn * p)) ** 2
            w_t[-num_taper:] = w_t[:num_taper][::-1]
        cpg = np.sum(w_t) / len(w_t)
    else:
        cpg = 1.0
    return w_t, cpg


def _bootstrap_1d_data(data_in, errors_in, num_resamples=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    data_in = np.asarray(data_in, dtype=float)
    errors_in = np.asarray(errors_in, dtype=float)
    len_data = data_in.size
    output = rng.standard_normal((len_data, num_resamples))
    mean_array = np.repeat(data_in[:, None], num_resamples, axis=1)
    sigma_array = np.repeat(errors_in[:, None], num_resamples, axis=1)
    output = output * sigma_array + mean_array
    return output


def _smooth_1d_along_axis(arr, axis=0):
    # simple 3-point moving average with edge truncation
    kernel = np.array([1, 1, 1], dtype=float) / 3.0
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis, arr)


def apply_fft(
    threads,
    res=0.0,
    cad=0.0,
    km_per_arcsec=725.27,
    detrend=False,
    pad_fft=False,
    fill_pad=False,
    pad_length=1000,
    bootstrap=False,
    num_bootstrap=1000,
    include_nyquist=False,
    min_freq_cutoff=1e-4,
    min_cyc_cutoff=0.75,
    window_func="split_cosine_bell",
    window_param=0.4,
    signif_levels=0.95,
    vaughn=False,
    adjacent_peaks=False,
    recover_power=False,
    vel_amp_mode=False,
):
    fft_spec = []
    fft_peaks = []

    if window_func.lower() == "hann":
        window_param = 1.0
    if window_param == 1.0:
        window_func = "hann"
    if window_func.lower() == "rectangular":
        window_param = 0.0

    if pad_fft:
        allow_adj_pks = "no"
    else:
        allow_adj_pks = "yes" if adjacent_peaks else "no"

    if res:
        amp_dx = res * km_per_arcsec
        amp_units = "km"
    else:
        amp_dx = 1.0
        amp_units = "pixels"

    if cad:
        freq_dt = 1.0 / cad
        freq_units = "Hz"
        period_units = "s"
    else:
        freq_dt = 1.0
        freq_units = "timesteps^{-1}"
        period_units = "timesteps"

    vel_amp_units = f"{amp_units}/{period_units}"

    convert_amp_to_pxls = 1.0 / amp_dx
    convert_freq_to_timesteps = 1.0 / freq_dt
    pow_units = f"{amp_units}^2 {period_units}"
    convert_pow_to_pxls = convert_amp_to_pxls ** 2

    for h, th in enumerate(threads):
        tpos = th["pos"].copy()
        terr = th["err_pos"].copy()

        t_start = th.get("start_bin", -1)
        t_end = th.get("end_bin", -1)
        len_thread = th.get("length", -1)
        if len_thread <= 0:
            len_thread = 1

        # skip if not enough data
        if np.sum(tpos > 0.0) <= 2:
            spec_th_fft = {
                "power": np.zeros(1),
                "err_power": np.zeros(1),
                "power_units": pow_units,
                "power_to_pxls": convert_pow_to_pxls,
                "amplitude": np.zeros(1),
                "err_amplitude": np.zeros(1),
                "amp_units": amp_units,
                "amp_to_pxls": convert_amp_to_pxls,
                "freq": np.zeros(1),
                "freq_units": freq_units,
                "freq_to_timesteps": convert_freq_to_timesteps,
                "phase": np.zeros(1),
                "err_phase": np.zeros(1),
                "trend": np.zeros(1),
                "trend_poly_degree": 0,
                "apod_window": np.zeros(1),
                "window_func": "not_applicable",
                "window_param": -1.0,
                "signif_vals": np.zeros(1),
                "fft_length": -1,
                "signif_test": "not_applicable",
                "signif_level": 0.0,
                "bin_flags": np.zeros(1, dtype=int),
                "enbw": -1.0,
                "cpg": -1.0,
            }
            peaks_th_fft = {
                "analysis_method": "FFT",
                "peak_power": np.zeros(4),
                "peak_amplitude": np.zeros(4),
                "peak_freq": np.zeros(4),
                "peak_vel_amp": np.zeros(4),
                "peak_phase": np.zeros(4),
                "err_peak_power": np.zeros(4),
                "err_peak_amplitude": np.zeros(4),
                "err_peak_phase": np.zeros(4),
                "peak_bin": -np.ones(4, dtype=int),
                "KS_stat": -np.ones(5),
                "KS_prob": -np.ones(5),
                "AD_stat": -np.ones(5),
                "AD_crit": -np.ones(5),
                "LB_stat": -np.ones(5),
                "LB_chisqrd": -np.ones(5),
                "num_signif_peaks": 0,
                "num_saved_waves": 0,
                "fft_length": -1,
                "window_func": "not_applicable",
                "window_param": -1.0,
                "enbw": -1.0,
                "cpg": -1.0,
                "signif_level": 0.0,
                "signif_test": "not_applicable",
                "adjacent_peaks": allow_adj_pks,
                "user_qual_flag": -1,
                "auto_qual_flag": 1,
                "power_to_pxls": convert_pow_to_pxls,
                "power_units": pow_units,
                "amp_to_pxls": convert_amp_to_pxls,
                "amp_units": amp_units,
                "freq_to_timesteps": convert_freq_to_timesteps,
                "freq_units": freq_units,
                "vel_amp_units": vel_amp_units,
            }
            fft_spec.append(spec_th_fft)
            fft_peaks.append(peaks_th_fft)
            continue

        # fill zeroes if any (should be patched already)
        zero_idx = np.where(tpos == 0.0)[0]
        for zi in zero_idx:
            if zi > 0:
                tpos[zi] = tpos[zi - 1]
                terr[zi] = 1.0

        if t_start >= 0:
            tpos = tpos[t_start : t_end + 1]
            terr = terr[t_start : t_end + 1]

        if vel_amp_mode:
            dv = (np.roll(tpos, -1) - np.roll(tpos, 1)) * 0.5
            dv[0] = tpos[1] - tpos[0]
            dv[-1] = tpos[-1] - tpos[-2]
            dv_err = (np.roll(terr, -1) - np.roll(terr, 1)) * 0.5
            dv_err[0] = terr[1] - terr[0]
            dv_err[-1] = terr[-1] - terr[-2]
            tpos = dv
            terr = dv_err

        len_thread = tpos.size

        # trend
        if detrend:
            x = np.arange(len_thread)
            coef = np.polyfit(x, tpos, 1)
            trend = np.polyval(coef, x)
            trend_poly_degree = 1
        else:
            trend = np.full(len_thread, np.mean(tpos))
            trend_poly_degree = 0

        oscil = tpos - trend

        s_len = oscil.size
        apodt, cpg = _get_apod_window(s_len, p=window_param, win_function=window_func)

        if fill_pad and pad_fft:
            num_pad_zeros = pad_length - len_thread
            if num_pad_zeros <= 0:
                num_pad_zeros = 1
        else:
            num_pad_zeros = pad_length

        if bootstrap:
            boot_arr = _bootstrap_1d_data(oscil, terr, num_resamples=num_bootstrap)
            if not pad_fft:
                oscil_w = oscil * apodt
                boot_arr = boot_arr * apodt[:, None]
                n_len = s_len
            else:
                oscil_w = np.concatenate([oscil * apodt, np.zeros(num_pad_zeros)])
                boot_arr = _smooth_1d_along_axis(boot_arr, axis=0)
                boot_arr = np.concatenate(
                    [boot_arr * apodt[:, None], np.zeros((num_pad_zeros, num_bootstrap))], axis=0
                )
                n_len = s_len + num_pad_zeros
        else:
            if not pad_fft:
                oscil_w = oscil * apodt
                n_len = oscil_w.size
            else:
                oscil_w = np.concatenate([oscil * apodt, np.zeros(num_pad_zeros)])
                n_len = oscil_w.size

        if include_nyquist:
            end_fft_index = int(np.ceil((n_len - 1) / 2.0))
        else:
            end_fft_index = int(np.floor((n_len - 1) / 2.0))

        if end_fft_index <= 0:
            end_fft_index = 1

        df = 1.0 / n_len
        f = np.arange(n_len) * df
        f = f[1 : end_fft_index + 1]
        nf = f.size

        if min_cyc_cutoff and min_cyc_cutoff > 0:
            max_possible_period = float(len_thread) / min_cyc_cutoff
            min_freq_cutoff = 1.0 / max_possible_period

        # FFT
        if bootstrap:
            fft_of_boot = np.fft.fft(boot_arr, axis=0)
            boot_pow = (2.0 * (np.abs(fft_of_boot)) ** 2)[1 : end_fft_index + 1, :]
            boot_phase = np.angle(fft_of_boot)[1 : end_fft_index + 1, :]
            pow_ = np.mean(boot_pow, axis=1)
            phase = np.mean(boot_phase, axis=1)
            pow_err = np.std(boot_pow, axis=1)
            phase_err = np.std(boot_phase, axis=1)
        else:
            fft_of_oscill = np.fft.fft(oscil_w)
            pow_ = (2.0 * (np.abs(fft_of_oscill)) ** 2)[1 : end_fft_index + 1]
            phase = np.angle(fft_of_oscill)[1 : end_fft_index + 1]
            pow_err = None
            phase_err = None

        # power/amplitude scaling (Heinzel FFT notes)
        S1 = np.sum(apodt)
        S2 = np.sum(apodt ** 2)
        power_correction = cad * (n_len ** 2) / S2 if cad else (n_len ** 2) / S2
        amp_correction = n_len / S1
        signif_correction = (cad * (S1 ** 2) / S2) if cad else (S1 ** 2) / S2

        spec_th_fft = {
            "power": pow_ * power_correction * (amp_dx ** 2),
            "err_power": np.zeros(nf),
            "power_units": pow_units,
            "power_to_pxls": convert_pow_to_pxls,
            "amplitude": 2.0 * np.sqrt(pow_ / 2.0) * amp_correction * amp_dx,
            "err_amplitude": np.zeros(nf),
            "amp_units": amp_units,
            "amp_to_pxls": convert_amp_to_pxls,
            "freq": f * freq_dt,
            "freq_units": freq_units,
            "freq_to_timesteps": convert_freq_to_timesteps,
            "phase": phase,
            "err_phase": np.zeros(nf),
            "trend": trend,
            "trend_poly_degree": trend_poly_degree,
            "apod_window": apodt,
            "window_func": window_func,
            "window_param": window_param,
            "signif_vals": np.zeros(nf),
            "fft_length": n_len,
            "signif_test": "not_applicable",
            "signif_level": signif_levels,
            "bin_flags": np.zeros(nf, dtype=int),
            "enbw": (1.0 / cad) * (S2 / (S1 ** 2)) if cad else (S2 / (S1 ** 2)),
            "cpg": cpg,
        }

        if bootstrap and pow_err is not None:
            spec_th_fft["err_power"] = pow_err * power_correction * (amp_dx ** 2)
            spec_th_fft["err_amplitude"] = 2.0 * np.sqrt(pow_err / 2.0) * amp_correction * amp_dx
            spec_th_fft["err_phase"] = phase_err

        # significance tests
        if np.var(pow_) > 1e-30:
            if vaughn:
                spec_th_fft["signif_test"] = "Vaughan_2005"
                npw = s_len * pow_ / np.var(oscil)
                prob = 1.0 - chi2.cdf(2.0 * npw, 2)
                nprob = np.zeros_like(prob)
                log_pp = s_len * np.log(1.0 - prob)
                mask = log_pp > -30.0
                nprob[mask] = np.exp(log_pp[mask])
                spec_th_fft["signif_vals"] = nprob
                loc_signif_pow = np.where(nprob > signif_levels)[0]
            else:
                spec_th_fft["signif_test"] = "Torrence_&_Compo_1998"
                sig_vals = signif_noise_spec(oscil, signif_levels, f.size, color="white", bonferroni=True)
                spec_th_fft["signif_vals"] = sig_vals * signif_correction * (amp_dx ** 2)
                loc_signif_pow = np.where(spec_th_fft["power"] > spec_th_fft["signif_vals"])[0]

            # bin flags
            spec_th_fft["bin_flags"][:] = 0
            spec_th_fft["bin_flags"][loc_signif_pow] = 1

            temp_pow = spec_th_fft["power"].copy()
            temp_pow[spec_th_fft["bin_flags"] < 1] = 0
            compare_right = temp_pow - np.roll(temp_pow, -1)
            compare_left = temp_pow - np.roll(temp_pow, 1)
            compare_right[-1] = 0
            compare_left[0] = 0

            if allow_adj_pks == "no":
                loc_signif_peaks = np.where((temp_pow > 0) & (compare_right >= 0) & (compare_left >= 0))[0]
            else:
                loc_signif_peaks = np.where(temp_pow > 0)[0]

            spec_th_fft["bin_flags"][loc_signif_peaks] = 2
        else:
            loc_signif_peaks = np.array([], dtype=int)

        # save peaks
        peaks_th_fft = {
            "analysis_method": "FFT",
            "peak_power": np.zeros(4),
            "peak_amplitude": np.zeros(4),
            "peak_freq": np.zeros(4),
            "peak_vel_amp": np.zeros(4),
            "peak_phase": np.zeros(4),
            "err_peak_power": np.zeros(4),
            "err_peak_amplitude": np.zeros(4),
            "err_peak_phase": np.zeros(4),
            "peak_bin": -np.ones(4, dtype=int),
            "KS_stat": -np.ones(5),
            "KS_prob": -np.ones(5),
            "AD_stat": -np.ones(5),
            "AD_crit": -np.ones(5),
            "LB_stat": -np.ones(5),
            "LB_chisqrd": -np.ones(5),
            "num_signif_peaks": len(loc_signif_peaks),
            "num_saved_waves": 0,
            "fft_length": n_len,
            "window_func": window_func,
            "window_param": window_param,
            "enbw": spec_th_fft["enbw"],
            "cpg": cpg,
            "signif_level": signif_levels,
            "signif_test": spec_th_fft["signif_test"],
            "adjacent_peaks": allow_adj_pks,
            "user_qual_flag": -1,
            "auto_qual_flag": 1,
            "power_to_pxls": convert_pow_to_pxls,
            "power_units": pow_units,
            "amp_to_pxls": convert_amp_to_pxls,
            "amp_units": amp_units,
            "freq_to_timesteps": convert_freq_to_timesteps,
            "freq_units": freq_units,
            "vel_amp_units": vel_amp_units,
        }

        if loc_signif_peaks.size > 0:
            pow_at_peaks = spec_th_fft["power"][loc_signif_peaks]
            sorted_indices = np.argsort(pow_at_peaks)[::-1]
            sav_p = 0
            for jj in range(len(sorted_indices)):
                bin_of_peak = loc_signif_peaks[sorted_indices[jj]]
                if sav_p < 4 and (f[bin_of_peak] > min_freq_cutoff):
                    peaks_th_fft["peak_power"][sav_p] = spec_th_fft["power"][bin_of_peak]
                    peaks_th_fft["peak_amplitude"][sav_p] = spec_th_fft["amplitude"][bin_of_peak]
                    peaks_th_fft["peak_freq"][sav_p] = spec_th_fft["freq"][bin_of_peak]
                    peaks_th_fft["peak_vel_amp"][sav_p] = (
                        2 * np.pi * spec_th_fft["amplitude"][bin_of_peak] * spec_th_fft["freq"][bin_of_peak]
                    )
                    peaks_th_fft["peak_phase"][sav_p] = phase[bin_of_peak]
                    peaks_th_fft["peak_bin"][sav_p] = bin_of_peak
                    if bootstrap and pow_err is not None:
                        peaks_th_fft["err_peak_power"][sav_p] = spec_th_fft["err_power"][bin_of_peak]
                        peaks_th_fft["err_peak_amplitude"][sav_p] = spec_th_fft["err_amplitude"][bin_of_peak]
                        peaks_th_fft["err_peak_phase"][sav_p] = spec_th_fft["err_phase"][bin_of_peak]
                    sav_p += 1
            peaks_th_fft["num_saved_waves"] = sav_p

        # recover leaked power
        if recover_power and allow_adj_pks == "no":
            for w in range(4):
                center_bin = peaks_th_fft["peak_bin"][w]
                if center_bin > -1:
                    left_bin = max(center_bin - 1, 0)
                    right_bin = min(center_bin + 1, nf - 1)
                    sum_weights = np.ones(right_bin - left_bin + 1)
                    if center_bin >= 2 and spec_th_fft["bin_flags"][left_bin - 1] == 2:
                        sum_weights[0] = 0.5
                    if center_bin <= nf - 3 and spec_th_fft["bin_flags"][right_bin + 1] == 2:
                        sum_weights[-1] = 0.5
                    tot_power = np.sum(spec_th_fft["power"][left_bin : right_bin + 1] * sum_weights)
                    mean_weights = (
                        spec_th_fft["power"][left_bin : right_bin + 1] / tot_power * sum_weights
                        if tot_power != 0
                        else sum_weights
                    )
                    mean_freq = np.sum(f[left_bin : right_bin + 1] * mean_weights) / np.sum(mean_weights)
                    mean_phase = np.sum(phase[left_bin : right_bin + 1] * mean_weights) / np.sum(mean_weights)
                    tot_amp = 2.0 * np.sqrt(tot_power / 2.0)
                    peaks_th_fft["peak_power"][w] = tot_power
                    peaks_th_fft["peak_amplitude"][w] = tot_amp
                    peaks_th_fft["peak_freq"][w] = mean_freq * freq_dt
                    peaks_th_fft["peak_phase"][w] = mean_phase

        # GOF tests of residuals
        combined_wave_vals = np.zeros(s_len)
        total_fit_params = 0
        for w in range(peaks_th_fft["num_saved_waves"] + 1):
            if w > 0 and peaks_th_fft["peak_bin"][w - 1] > -1:
                amp = peaks_th_fft["peak_amplitude"][w - 1]
                freq = peaks_th_fft["peak_freq"][w - 1]
                ph = peaks_th_fft["peak_phase"][w - 1]
                add_wave = amp * np.cos(2.0 * np.pi * freq * np.arange(s_len) * (cad if cad else 1.0) + ph)
                combined_wave_vals = combined_wave_vals + add_wave
                total_fit_params += 3
            residuals = tpos - (combined_wave_vals + trend)
            resid_stats = test_residuals(residuals, num_fit_params=total_fit_params, signif_level=signif_levels)
            peaks_th_fft["KS_stat"][w] = resid_stats["KS_stat"]
            peaks_th_fft["KS_prob"][w] = resid_stats["KS_prob"]
            peaks_th_fft["AD_stat"][w] = resid_stats["AD_stat"]
            peaks_th_fft["AD_crit"][w] = resid_stats["AD_crit"]
            peaks_th_fft["LB_stat"][w] = resid_stats["LB_stat"]
            peaks_th_fft["LB_chisqrd"][w] = resid_stats["LB_chisqrd"]

        fft_spec.append(spec_th_fft)
        fft_peaks.append(peaks_th_fft)

    return fft_spec, fft_peaks
