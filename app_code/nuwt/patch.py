import numpy as np


def _safe_err(value, floor):
    if np.isfinite(value) and value > 0.0:
        return float(value)
    return float(floor)


def _interp_error_profile(start_val, end_val, start_err, end_err, num_fill):
    if num_fill <= 0:
        return np.array([], dtype=np.float64)

    step_floor = max(abs(float(end_val) - float(start_val)) / float(num_fill + 1) * 0.25, 1e-3)
    start_err = _safe_err(start_err, step_floor)
    end_err = _safe_err(end_err, step_floor)

    frac = np.arange(1, num_fill + 1, dtype=np.float64) / float(num_fill + 1)
    interp_err = np.sqrt(((1.0 - frac) * start_err) ** 2 + (frac * end_err) ** 2 + step_floor**2)
    return interp_err * (1.0 + 0.1 * max(num_fill - 1, 0))


def patch_up_threads(threads, fit_flag=0, simp_fill=False, debug=False):
    """Port of NUWT_PATCH_UP_THREADS.pro (fills gaps)."""
    if not threads:
        return threads

    # Extract values based on fit_flag
    def get_arrays(th):
        if fit_flag in (0, 1):
            return th["pos"], th["err_pos"]
        if fit_flag == 2:
            return th["inten"], th["err_inten"]
        if fit_flag == 3:
            return th["wid"], th["err_wid"]
        raise ValueError("fit_flag must be 0..3")

    def set_arrays(th, tval, terr):
        if fit_flag in (0, 1):
            th["pos"] = tval
            th["err_pos"] = terr
        elif fit_flag == 2:
            th["inten"] = tval
            th["err_inten"] = terr
        elif fit_flag == 3:
            th["wid"] = tval
            th["err_wid"] = terr

    for idx, th in enumerate(threads):
        tval, terr = get_arrays(th)
        tval = tval.copy()
        terr = terr.copy()
        bin_flags = np.asarray(th.get("bin_flags", np.zeros(tval.shape, dtype=int)), dtype=int).copy()
        if bin_flags.shape != tval.shape:
            bin_flags = np.zeros(tval.shape, dtype=int)
        bin_flags[(tval >= 0.0) & (bin_flags == 0)] = 2
        bin_flags[tval < 0.0] = -1

        loc_data = np.where(tval > 0.0)[0]
        loc_zeroes = np.where(tval == 0.0)[0]
        len_thread = np.sum(tval >= 0.0)

        if loc_data.size <= 2:
            tval[:] = -1.0
            terr[:] = -1.0
            bin_flags[:] = -1
            set_arrays(th, tval, terr)
            th["bin_flags"] = bin_flags
            th["interp_mask"] = np.zeros(tval.shape, dtype=bool)
            th["interp_fraction"] = 0.0
            continue

        if loc_zeroes.size >= 0.5 * len_thread:
            tval[:] = -1.0
            terr[:] = -1.0
            bin_flags[:] = -1
            set_arrays(th, tval, terr)
            th["bin_flags"] = bin_flags
            th["interp_mask"] = np.zeros(tval.shape, dtype=bool)
            th["interp_fraction"] = 0.0
            continue

        if not simp_fill:
            index_diff = np.diff(loc_data, prepend=loc_data[0])
            gap_ends = np.where(index_diff > 1)[0]
            for gi in gap_ends:
                if gi == 0:
                    continue
                start_ind = loc_data[gi - 1]
                end_ind = loc_data[gi]
                start_val = tval[start_ind]
                end_val = tval[end_ind]
                num_fill = end_ind - start_ind - 1
                if num_fill <= 0:
                    continue
                fill_inds = np.arange(1, num_fill + 1) + start_ind
                m = (end_val - start_val) / (end_ind - start_ind)
                tval[start_ind + 1 : end_ind] = start_val + m * (fill_inds - start_ind)
                terr[start_ind + 1 : end_ind] = _interp_error_profile(
                    start_val,
                    end_val,
                    terr[start_ind],
                    terr[end_ind],
                    num_fill,
                )
                bin_flags[start_ind + 1 : end_ind] = 1
            if debug:
                if np.any(tval == 0.0):
                    print(f"thread index {idx} still has zeroes")
        else:
            for zi in loc_zeroes:
                if zi == 0:
                    continue
                tval[zi] = tval[zi - 1]
                terr[zi] = _safe_err(terr[zi - 1], 1e-3) * 1.1
                bin_flags[zi] = 1

        set_arrays(th, tval, terr)
        th["bin_flags"] = bin_flags
        th["interp_mask"] = bin_flags == 1
        th["interp_fraction"] = (
            float(np.count_nonzero(bin_flags == 1)) / float(max(len_thread, 1))
        )

    return threads
