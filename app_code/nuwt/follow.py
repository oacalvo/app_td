import numpy as np


def _first_peak_offsets_idl_order(a, mini, dist_pxl_shift, time_pxl_shift):
    """Replicate IDL WHERE() traversal on a [space, time] search box."""
    mask = a > mini
    if not np.any(mask):
        return None

    first_flat = int(np.flatnonzero(mask.ravel(order="F"))[0])
    xm = int(dist_pxl_shift.ravel(order="F")[first_flat])
    ym = int(time_pxl_shift.ravel(order="F")[first_flat])
    return xm, ym


def _extract_search_box(
    arr,
    *,
    left_ind,
    right_ind,
    sooner_ind,
    later_ind,
    nx,
    max_dist_jump,
    fill_value,
    reorder_cols,
):
    search_box_width = 2 * max_dist_jump + 1
    height = later_ind - sooner_ind + 1

    if left_ind < 0:
        first_real_col = abs(left_ind)
        box = np.full((search_box_width, height), fill_value, dtype=float)
        box[first_real_col:, :] = arr[0 : right_ind + 1, sooner_ind : later_ind + 1]
    elif right_ind >= nx:
        last_real_col = 2 * max_dist_jump - (right_ind - nx + 1)
        box = np.full((search_box_width, height), fill_value, dtype=float)
        box[0 : last_real_col + 1, :] = arr[left_ind:nx, sooner_ind : later_ind + 1]
    else:
        box = arr[left_ind : right_ind + 1, sooner_ind : later_ind + 1]

    return box[reorder_cols, :]


def follow_threads(
    located,
    min_tlen=20,
    max_dist_jump=3,
    max_time_skip=4,
    scan_dir="outward",
    continuity_weight=1.5,
    time_weight=2.0,
    quality_weight=0.35,
    error_weight=0.15,
    debug=False,
):
    """Port of NUWT_FOLLOW_THREADS.pro (core tracking).

    The weighting parameters are accepted for local app compatibility but the
    tracked crest selection follows the original IDL rule: choose the first
    valid peak in the reordered search box.
    """
    peaks = located["peaks"]
    errs = located["errs"]
    _ = continuity_weight, time_weight, quality_weight, error_weight

    if max_time_skip < 1:
        max_time_skip = 1
    max_dist_jump = int(max_dist_jump)
    max_time_skip = int(max_time_skip)

    image = peaks[:, :, 0].copy()
    if errs.ndim == 3:
        im_err = errs[:, :, 0].copy()
    else:
        im_err = errs.copy()

    nx, nt = image.shape
    num_pk_layers = peaks.shape[2]
    full_gauss = 1 if num_pk_layers == 6 else 0

    # scan direction
    scan_dir = scan_dir.lower()
    if scan_dir not in ("outward", "inward", "right", "left"):
        scan_dir = "outward"

    search_box_width = 2 * max_dist_jump + 1
    search_box_height = max_time_skip

    if scan_dir == "outward":
        reorder_cols = np.zeros(search_box_width, dtype=int)
        sub_ind = np.arange(max_dist_jump) * 2 + 1
        abs_col_shift = np.arange(max_dist_jump) + 1
        reorder_cols[0] = max_dist_jump
        reorder_cols[sub_ind] = max_dist_jump - abs_col_shift
        reorder_cols[sub_ind + 1] = max_dist_jump + abs_col_shift
    elif scan_dir == "inward":
        reorder_cols = np.zeros(search_box_width, dtype=int)
        sub_ind = np.arange(max_dist_jump) * 2
        abs_col_shift = np.arange(max_dist_jump) + 1
        reorder_cols[-1] = max_dist_jump
        reorder_cols[sub_ind] = (max_dist_jump - abs_col_shift)[::-1]
        reorder_cols[sub_ind + 1] = (max_dist_jump + abs_col_shift)[::-1]
    elif scan_dir == "right":
        reorder_cols = np.arange(search_box_width)
    else:
        reorder_cols = np.arange(search_box_width)[::-1]

    dist_pxl_shift = np.arange(search_box_width) - max_dist_jump
    dist_pxl_shift = np.repeat(dist_pxl_shift[:, None], search_box_height, axis=1)
    dist_pxl_shift = dist_pxl_shift[reorder_cols, :]

    time_pxl_shift = np.arange(search_box_height) + 1
    time_pxl_shift = np.repeat(time_pxl_shift[None, :], search_box_width, axis=0)

    mini = np.min(image)
    threads = []

    if debug:
        th_debug = np.zeros((nx, nt), dtype=float)
    else:
        th_debug = None

    last_timestep = max(max_time_skip, min_tlen)
    raw_th_count = 0

    # IDL FOR loops include the upper bound.
    for j in range(0, nt - last_timestep):
        for i in range(max_dist_jump, nx - max_dist_jump):
            if image[i, j] <= mini:
                continue

            # initialize temp thread
            pos = np.zeros(nt, dtype=float)
            err_pos = np.zeros(nt, dtype=float)
            bin_flags = np.zeros(nt, dtype=int)
            pos[: j + 1] = -1.0
            err_pos[: j + 1] = -1.0
            bin_flags[: j + 1] = -1

            start_bin = j
            end_bin = j
            length = 1

            pos[j] = image[i, j]
            err_pos[j] = im_err[i, j]
            bin_flags[j] = 2

            if full_gauss:
                inten = np.zeros(nt, dtype=float)
                err_inten = np.zeros(nt, dtype=float)
                wid = np.zeros(nt, dtype=float)
                err_wid = np.zeros(nt, dtype=float)
                inten[: j + 1] = -1.0
                err_inten[: j + 1] = -1.0
                wid[: j + 1] = -1.0
                err_wid[: j + 1] = -1.0

                inten[j] = peaks[i, j, 1]
                err_inten[j] = errs[i, j, 2] if errs.ndim == 3 else errs[i, j]
                wid[j] = peaks[i, j, 2]
                err_wid[j] = errs[i, j, 2] if errs.ndim == 3 else errs[i, j]

            # first search box
            a = _extract_search_box(
                image,
                left_ind=i - max_dist_jump,
                right_ind=i + max_dist_jump,
                sooner_ind=j + 1,
                later_ind=j + max_time_skip,
                nx=nx,
                max_dist_jump=max_dist_jump,
                fill_value=mini,
                reorder_cols=reorder_cols,
            )

            h = j
            k = i

            while np.max(a) > mini:
                offsets = _first_peak_offsets_idl_order(
                    a,
                    mini,
                    dist_pxl_shift,
                    time_pxl_shift,
                )
                if offsets is None:
                    break

                xm, ym = offsets

                k = k + xm
                h = h + ym

                pos[h] = image[k, h]
                err_pos[h] = im_err[k, h]
                bin_flags[h] = 2
                image[k, h] = mini
                im_err[k, h] = mini

                if full_gauss:
                    inten[h] = peaks[k, h, 1]
                    err_inten[h] = errs[k, h, 2] if errs.ndim == 3 else errs[k, h]
                    wid[h] = peaks[k, h, 2]
                    err_wid[h] = errs[k, h, 2] if errs.ndim == 3 else errs[k, h]

                if debug:
                    th_debug[k, h] = raw_th_count

                left_ind = k - max_dist_jump
                right_ind = k + max_dist_jump
                sooner_ind = h + 1
                later_ind = min(h + max_time_skip, nt - 1)

                if k <= 0 or k >= nx - 1 or h >= nt - 1:
                    break

                a = _extract_search_box(
                    image,
                    left_ind=left_ind,
                    right_ind=right_ind,
                    sooner_ind=sooner_ind,
                    later_ind=later_ind,
                    nx=nx,
                    max_dist_jump=max_dist_jump,
                    fill_value=mini,
                    reorder_cols=reorder_cols,
                )

                end_bin = h
                length = 1 + h - j

            raw_th_count += 1

            if h < nt - 1:
                pos[h + 1 :] = -1.0
                err_pos[h + 1 :] = -1.0
                bin_flags[h + 1 :] = -1
                if full_gauss:
                    inten[h + 1 :] = -1.0
                    err_inten[h + 1 :] = -1.0
                    wid[h + 1 :] = -1.0
                    err_wid[h + 1 :] = -1.0

            if length >= min_tlen:
                num_nonzero = np.sum(pos > 0.0)
                num_zero = np.sum(pos == 0.0)
                if num_nonzero > 2 and num_zero < 0.5 * length:
                    th = {
                        "pos": pos,
                        "err_pos": err_pos,
                        "bin_flags": bin_flags,
                        "start_bin": start_bin,
                        "end_bin": end_bin,
                        "length": length,
                    }
                    if full_gauss:
                        th.update(
                            {
                                "inten": inten,
                                "err_inten": err_inten,
                                "wid": wid,
                                "err_wid": err_wid,
                            }
                        )
                    threads.append(th)

    return threads, th_debug
