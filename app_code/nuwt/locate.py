import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2


def _gauss_plus_linear(x, a, x0, sigma, c0, c1):
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c0 + c1 * x


def _chisq_cvf(p, dof):
    # IDL CHISQR_CVF uses the upper-tail probability.
    return chi2.isf(p, dof)


def _normalize_sm_width(sm_width):
    widths = np.atleast_1d(sm_width).astype(int)
    if widths.size == 0:
        widths = np.array([1, 3], dtype=int)
    elif widths.size == 1:
        widths = np.array([0, widths[0]], dtype=int)
    else:
        widths = widths[:2]
    return tuple(int(v) for v in widths)


def _smooth_along_axis(arr, width, axis):
    if width <= 1:
        return arr.copy()

    pad_before = width // 2
    pad_after = width - 1 - pad_before
    kernel = np.ones(width, dtype=np.float64) / float(width)

    def _smooth_1d(vec):
        padded = np.pad(vec, (pad_before, pad_after), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    return np.apply_along_axis(_smooth_1d, axis, arr)


def _smooth_2d(arr, sm_width):
    out = np.array(arr, dtype=np.float64, copy=True)
    if len(sm_width) >= 1:
        out = _smooth_along_axis(out, sm_width[0], axis=0)
    if len(sm_width) >= 2:
        out = _smooth_along_axis(out, sm_width[1], axis=1)
    return out


def _mad_std(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0.0:
        return 1.4826 * mad

    std = float(np.std(finite))
    return std if np.isfinite(std) else 0.0


def _local_peak_metrics(img_slice, center_idx, half_width, err_slice=None):
    left = max(0, center_idx - half_width)
    right = min(img_slice.size, center_idx + half_width + 1)
    window = np.asarray(img_slice[left:right], dtype=np.float64)
    local_center = int(center_idx - left)

    left_shoulder = window[:local_center]
    right_shoulder = window[local_center + 1 :]
    if left_shoulder.size and right_shoulder.size:
        baseline = max(float(np.median(left_shoulder)), float(np.median(right_shoulder)))
        shoulder = np.concatenate([left_shoulder, right_shoulder])
    elif left_shoulder.size or right_shoulder.size:
        shoulder = np.concatenate([left_shoulder, right_shoulder])
        baseline = float(np.median(shoulder))
    else:
        shoulder = window
        baseline = float(window[local_center]) if window.size else 0.0

    prominence = float(img_slice[center_idx] - baseline)
    noise = _mad_std(shoulder)

    if err_slice is not None:
        err_window = np.asarray(err_slice[left:right], dtype=np.float64)
        err_shoulder = np.concatenate(
            [err_window[:local_center], err_window[local_center + 1 :]]
        )
        valid_err = err_shoulder[np.isfinite(err_shoulder) & (err_shoulder > 0.0)]
        if valid_err.size:
            noise = max(noise, float(np.median(valid_err)))

    noise = max(noise, 1e-9)
    snr = prominence / noise if prominence > 0.0 else 0.0
    return baseline, prominence, noise, snr


def locate_things(
    td_diagram,
    errors=None,
    res=None,
    cad=None,
    km_per_arcsec=725.27,
    set_dx_units="arcsec",
    set_dt_units="min",
    invert=False,
    bad_data=-999,
    despike=False,
    spike_sigma=3.0,
    smooth=False,
    sm_width=(1, 3),
    num_search_bins=5,
    grad=0.5,
    percent_grad=None,
    nearest_pixel=False,
    meas_size=7,
    simp_grad=False,
    weighted_mean=False,
    cut_chisq=None,
    shift_cut=1.5,
    full_gauss=False,
    min_prominence=0.0,
    min_snr=1.0,
):
    """Port of NUWT_LOCATE_THINGS.pro (core peak detection).

    ``min_prominence`` and ``min_snr`` are accepted for local app compatibility
    but are not part of the original Auto-NUWT peak filter and are not applied.
    """
    td = np.array(td_diagram, dtype=np.float64, copy=True)
    nx, nt = td.shape
    sm_width = _normalize_sm_width(sm_width)
    _ = min_prominence, min_snr

    if invert:
        # Preserve bad_data (<= bad_data) on inversion.
        mask_bad = td <= bad_data
        max_intensity_val = np.nanmax(td)
        td = -td + max_intensity_val
        td[mask_bad] = td_diagram[mask_bad]

    if despike:
        # Replace spikes within each time slice.
        for j in range(nt):
            slice_j = td[:, j]
            mu = np.nanmean(slice_j)
            sigma = np.nanstd(slice_j)
            thresh = mu + spike_sigma * sigma
            spikes = slice_j > thresh
            # Replace with average of neighbors.
            for i in np.where(spikes)[0]:
                if 0 < i < nx - 1:
                    slice_j[i] = 0.5 * (slice_j[i - 1] + slice_j[i + 1])

    if errors is None:
        td_errs = np.ones_like(td)
    else:
        td_errs = np.array(errors, dtype=np.float64, copy=True)
        nonzero = td_errs != 0
        if np.any(nonzero):
            min_nonzero = np.min(np.abs(td_errs[nonzero]))
            td_errs[~nonzero] = min_nonzero
        else:
            td_errs[:] = 1.0

    if smooth:
        mask_bad = td <= bad_data
        if np.any(mask_bad):
            bad_td_vals = td[mask_bad].copy()
            bad_err_vals = td_errs[mask_bad].copy()
            td = _smooth_2d(td, sm_width)
            td_errs = _smooth_2d(td_errs, sm_width)
            td[mask_bad] = bad_td_vals
            td_errs[mask_bad] = bad_err_vals
        else:
            td = _smooth_2d(td, sm_width)
            td_errs = _smooth_2d(td_errs, sm_width)

    if res is None:
        res = 1.0
        set_dx_units = "pixels"
    else:
        if set_dx_units == "pixels":
            set_dx_units = "arcsec"

    if cad is None:
        cad = 1.0
        set_dt_units = "timesteps"
    else:
        if set_dt_units == "timesteps":
            set_dt_units = "min"

    if set_dx_units == "pixels":
        dx = 1.0
    elif set_dx_units == "arcsec":
        dx = res
    elif set_dx_units == "m":
        dx = res * km_per_arcsec * 1000.0
    elif set_dx_units == "km":
        dx = res * km_per_arcsec
    elif set_dx_units == "Mm":
        dx = res * km_per_arcsec / 1000.0
    else:
        dx = res
        set_dx_units = "arcsec"

    if set_dt_units == "timesteps":
        dt = 1.0
    elif set_dt_units == "s":
        dt = cad
    elif set_dt_units == "min":
        dt = cad / 60.0
    elif set_dt_units == "hr":
        dt = cad / 3600.0
    else:
        dt = cad / 60.0
        set_dt_units = "min"

    if meas_size % 2 == 0:
        meas_size += 1
    meas_size = int(meas_size)
    half_meas = meas_size // 2

    if cut_chisq is None:
        # default 3-sigma confidence level
        cut_chisq = _chisq_cvf(1.0 - 0.9973, meas_size - 5)

    # output structure
    if full_gauss:
        peaks = np.full((nx, nt, 6), np.min(td) - 10.0, dtype=np.float64)
        errs = np.zeros((nx, nt, 6), dtype=np.float64)
        end_pk_ind = 5
        end_err_ind = 5
        nearest_end_pk_ind = 2
        nearest_end_err_ind = 2
    else:
        peaks = np.full((nx, nt, 2), np.min(td) - 10.0, dtype=np.float64)
        errs = np.zeros((nx, nt), dtype=np.float64)
        end_pk_ind = 1
        end_err_ind = 0
        nearest_end_pk_ind = 1
        nearest_end_err_ind = 0

    allpeaks = np.ones((nx, nt), dtype=np.int16)
    prominence = np.zeros((nx, nt), dtype=np.float64)
    local_noise = np.zeros((nx, nt), dtype=np.float64)
    quality = np.zeros((nx, nt), dtype=np.float64)

    # find local maxima using shift comparisons
    for bin_ in range(1, num_search_bins + 1):
        compare_right = td - np.roll(td, -bin_, axis=0)
        compare_left = td - np.roll(td, bin_, axis=0)
        not_pk = (allpeaks > 0) & ((compare_left <= 0) | (compare_right < 0))
        allpeaks[not_pk] = 0

    # clean edges
    allpeaks[: num_search_bins + 1, :] = 0
    allpeaks[-num_search_bins :, :] = 0

    pk_locs = np.argwhere(allpeaks > 0)
    num_peaks_to_test = pk_locs.shape[0]

    grad_left = np.zeros((nx, nt), dtype=np.float64)
    grad_right = np.zeros((nx, nt), dtype=np.float64)
    ind_grad = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    ind_fit = np.arange(-half_meas, half_meas + 1, dtype=np.float64)

    # compute gradients
    last_j = -1
    for i, j in pk_locs:
        if j != last_j:
            img_slice = td[:, j]
            err_slice = td_errs[:, j]
            last_j = j
        if i - 4 < 0 or i + 4 >= nx:
            continue
        if not simp_grad:
            # linear fits for slopes
            left_y = img_slice[i - 4 : i + 1]
            right_y = img_slice[i : i + 5]
            try:
                left_coef = np.polyfit(ind_grad, left_y, 1, w=1.0 / np.maximum(err_slice[i - 4 : i + 1], 1e-12))
                right_coef = np.polyfit(ind_grad, right_y, 1, w=1.0 / np.maximum(err_slice[i : i + 5], 1e-12))
                grad_left[i, j] = left_coef[0]
                grad_right[i, j] = right_coef[0]
            except Exception:
                grad_left[i, j] = 0.0
                grad_right[i, j] = 0.0
        else:
            grad_left[i, j] = np.sum(ind_grad * (img_slice[i - 4 : i + 1] - np.mean(img_slice[i - 4 : i + 1]))) / 10.0
            grad_right[i, j] = np.sum(ind_grad * (img_slice[i : i + 5] - np.mean(img_slice[i : i + 5]))) / 10.0

    # auto-scale grad threshold
    if percent_grad is not None and num_peaks_to_test > 0:
        percent_grad = np.clip(percent_grad, 0.0, 100.0)
        grad_left_arr = grad_left[allpeaks > 0]
        grad_right_arr = grad_right[allpeaks > 0]
        abs_largest = np.max([np.max(np.abs(grad_left_arr)), np.max(np.abs(grad_right_arr))])
        num_test = int(abs_largest / 0.01)
        test_grad_arr = np.arange(num_test, dtype=np.float64) * 0.01
        temp_percent = 0.0
        pi = 1
        while temp_percent <= percent_grad and pi < len(test_grad_arr):
            pi += 1
            rejected = (grad_left_arr <= test_grad_arr[pi]) | (grad_right_arr >= -test_grad_arr[pi])
            temp_percent = 100.0 * np.sum(rejected) / max(1, num_peaks_to_test)
        grad = test_grad_arr[pi - 1] if pi > 0 else grad

    # fit peaks
    mini = np.min(peaks)
    last_j = -1
    for i, j in pk_locs:
        if j != last_j:
            img_slice = td[:, j]
            err_slice = td_errs[:, j]
            last_j = j

        if not (grad_left[i, j] > grad and grad_right[i, j] < -grad):
            continue

        if i - half_meas < 0 or i + half_meas >= nx:
            continue

        if nearest_pixel:
            coeff = np.array([mini, 0.0, 0.0, 0.0, 0.0])
            sigma = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            chisq = -1.0
        elif weighted_mean:
            window = img_slice[i - half_meas : i + half_meas + 1]
            sum_int = np.sum(window)
            weights = window / sum_int if sum_int != 0 else np.ones_like(window) / len(window)
            wg_mean = np.sum(weights * ind_fit)
            wg_std = np.sum(weights * (ind_fit - wg_mean) ** 2)
            coeff = np.array([1.0, wg_mean, wg_std, 0.0, 0.0])
            sigma = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
            chisq = 0.0
        else:
            y = img_slice[i - half_meas : i + half_meas + 1]
            x = ind_fit
            yerr = err_slice[i - half_meas : i + half_meas + 1]
            p0 = [img_slice[i], 0.0, 2.0, np.min(y), 0.1]
            try:
                popt, pcov = curve_fit(
                    _gauss_plus_linear,
                    x,
                    y,
                    p0=p0,
                    sigma=np.maximum(yerr, 1e-12),
                    absolute_sigma=True,
                    maxfev=2000,
                )
                coeff = popt
                sigma = np.sqrt(np.diag(pcov))
                residuals = (y - _gauss_plus_linear(x, *popt)) / np.maximum(yerr, 1e-12)
                chisq = np.sum(residuals**2)
            except Exception:
                coeff = np.array([mini, 0.0, 0.0, 0.0, 0.0])
                sigma = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                chisq = np.inf

        if (
            np.abs(coeff[1]) < shift_cut
            and chisq < cut_chisq
            and sigma[1] > 0.0
            and sigma[1] < 1.5
            and coeff[2] < meas_size
            and coeff[0] > mini
        ):
            if weighted_mean:
                peak_val = img_slice[i]
                sig_peak = err_slice[i]
            else:
                peak_val = _gauss_plus_linear(coeff[1], *coeff)
                errpeak = _gauss_plus_linear(coeff[1] + sigma[1], *coeff)
                sig_peak = peak_val - errpeak
            new_i = int(np.round(i + coeff[1]))
            if new_i < 0 or new_i >= nx:
                new_i = i
            peaks[new_i, j, 0 : end_pk_ind + 1] = np.array(
                [i + coeff[1], peak_val, abs(coeff[2]), coeff[3], coeff[4], i]
            )[0 : end_pk_ind + 1]
            if full_gauss:
                errs[new_i, j, 0 : end_err_ind + 1] = np.array(
                    [sigma[1], sig_peak, sigma[2], sigma[3], sigma[4], 1.0]
                )[0 : end_err_ind + 1]
            else:
                errs[new_i, j] = sigma[1] if sigma.size else 0.5
            allpeaks[i, j] = 0
            allpeaks[new_i, j] = 3
        else:
            peaks[i, j, 0 : nearest_end_pk_ind + 1] = np.array([i, img_slice[i], 0.0])[0 : nearest_end_pk_ind + 1]
            if full_gauss:
                errs[i, j, 0 : nearest_end_err_ind + 1] = np.array([0.5, err_slice[i], 0.0])[0 : nearest_end_err_ind + 1]
            else:
                errs[i, j] = 0.5
            allpeaks[i, j] = 2

    located = {
        "peaks": peaks,
        "errs": errs,
        "allpeaks": allpeaks,
        "prominence": prominence,
        "local_noise": local_noise,
        "quality": quality,
        "grad_left": grad_left,
        "grad_right": grad_right,
        "td_img": td,
        "inverted": int(invert),
        "despiked": int(despike),
        "spike_sigma": spike_sigma,
        "smoothed": int(smooth),
        "sm_width": np.array(sm_width),
        "dx": dx,
        "units_dx": set_dx_units,
        "dt": dt,
        "units_dt": set_dt_units,
        "res": res,
        "cad": cad,
        "km_per_arcsec": km_per_arcsec,
        "out_grad": grad,
        "min_prominence": float(min_prominence),
        "min_snr": float(min_snr),
    }

    return located
