import numpy as np

from .locate import locate_things
from .follow import follow_threads
from .patch import patch_up_threads
from .fft import apply_fft


def run_nuwt(
    td,
    errors=None,
    invert=False,
    grad=0.5,
    smooth=False,
    sm_width=(1, 3),
    min_tlen=20,
    max_dist_jump=3,
    max_time_skip=4,
    res=0.03,
    cad=1.35,
    km_per_arcsec=725.27,
    gauss=True,
    full_gauss=False,
    pad_fft=False,
    fill_pad=False,
    pad_length=1000,
    bootstrap=False,
    num_bootstrap=1000,
    vel_amp_mode=False,
    scan_dir="outward",
    window_func="split_cosine_bell",
    window_param=0.4,
    detrend=False,
):
    """Run the core NUWT pipeline on a time-distance diagram.

    Parameters mirror run_nuwt.pro (subset). TD must be shaped (nx, nt).
    """
    locate_errors = np.abs(np.asarray(td, dtype=np.float64)) * 0.1 if errors is None else errors

    located = locate_things(
        td,
        errors=locate_errors,
        res=res,
        cad=cad,
        km_per_arcsec=km_per_arcsec,
        invert=invert,
        despike=True,
        smooth=smooth,
        sm_width=sm_width,
        grad=grad,
        cut_chisq=1e12,
        nearest_pixel=not gauss,
        full_gauss=full_gauss,
    )

    threads, th_debug = follow_threads(
        located,
        min_tlen=min_tlen,
        max_dist_jump=max_dist_jump,
        max_time_skip=max_time_skip,
        scan_dir=scan_dir,
    )

    # patch gaps
    threads = patch_up_threads(threads, fit_flag=0, simp_fill=False, debug=False)

    fft_spec, fft_peaks = apply_fft(
        threads,
        res=res,
        cad=cad,
        km_per_arcsec=km_per_arcsec,
        detrend=detrend,
        pad_fft=pad_fft,
        fill_pad=fill_pad,
        pad_length=pad_length,
        bootstrap=bootstrap,
        num_bootstrap=num_bootstrap,
        vel_amp_mode=vel_amp_mode,
        window_func=window_func,
        window_param=window_param,
    )

    meta = {
        "grad": grad,
        "smooth": int(smooth),
        "sm_width": tuple(sm_width),
        "min_tlen": min_tlen,
        "max_dist_jump": max_dist_jump,
        "max_time_skip": max_time_skip,
        "res": res,
        "cad": cad,
        "km_per_arcsec": km_per_arcsec,
        "gauss": int(gauss),
        "full_gauss": int(full_gauss),
        "pad_fft": int(pad_fft),
        "fill_pad": int(fill_pad),
        "pad_length": pad_length,
        "bootstrap": int(bootstrap),
        "num_bootstrap": num_bootstrap,
        "vel_amp_mode": int(vel_amp_mode),
        "scan_dir": scan_dir,
        "window_func": window_func,
        "window_param": window_param,
        "detrend": int(detrend),
    }

    return located, threads, fft_spec, fft_peaks, meta
