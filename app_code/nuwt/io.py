import numpy as np
from astropy.io import fits
from astropy.table import Table


def load_td_fits(path, time_axis=0):
    """Load a time-distance FITS file.

    Parameters
    ----------
    path : str
        FITS file path.
    time_axis : int
        Axis index in FITS data that corresponds to time.
        The other axis is distance (x).

    Returns
    -------
    td : np.ndarray
        2D array shaped (nx, nt) as expected by NUWT core.
    header : fits.Header
        FITS header from primary HDU.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError(f"No data in FITS: {path}")

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D TD data. Got shape {data.shape}. "
            "Provide a precomputed time-distance diagram."
        )

    # Ensure time is the second axis in output (nx, nt)
    if time_axis == 0:
        # data shape (nt, nx) -> transpose to (nx, nt)
        td = data.T
    elif time_axis == 1:
        td = data
    else:
        raise ValueError("time_axis must be 0 or 1 for 2D TD data")

    return td.astype(np.float64, copy=False), header


def _table_from_list_of_dicts(rows, name):
    if not rows:
        return fits.BinTableHDU(Table(), name=name)
    return fits.BinTableHDU(Table(rows), name=name)


def save_results_fits(path, located, threads, fft_spec, fft_peaks, meta):
    """Save NUWT results to a FITS file with BINTABLE HDUs.

    Tables:
    - META: single-row metadata
    - LOCATED: per-peak arrays packed as variable-length columns
    - THREADS: per-thread arrays
    - FFT_SPEC: per-thread FFT spectra
    - FFT_PEAKS: per-thread peak parameters
    """
    hdus = [fits.PrimaryHDU()]

    # META
    meta_row = {k: meta.get(k) for k in sorted(meta.keys())}
    meta_table = Table([meta_row])
    hdus.append(fits.BinTableHDU(meta_table, name="META"))

    # LOCATED (store arrays as variable-length columns)
    loc_rows = []
    if located is not None:
        loc_rows.append(
            {
                "peaks": np.asarray(located["peaks"], dtype=np.float32).ravel(),
                "errs": np.asarray(located["errs"], dtype=np.float32).ravel(),
                "allpeaks": np.asarray(located["allpeaks"], dtype=np.int16).ravel(),
                "grad_left": np.asarray(located["grad_left"], dtype=np.float32).ravel(),
                "grad_right": np.asarray(located["grad_right"], dtype=np.float32).ravel(),
                "shape_peaks": np.array(located["peaks"].shape, dtype=np.int16),
                "shape_errs": np.array(located["errs"].shape, dtype=np.int16),
                "shape_td": np.array(located["td_img"].shape, dtype=np.int16),
            }
        )
    hdus.append(_table_from_list_of_dicts(loc_rows, "LOCATED"))

    # THREADS
    th_rows = []
    for th in threads:
        row = {k: np.asarray(v) for k, v in th.items()}
        th_rows.append(row)
    hdus.append(_table_from_list_of_dicts(th_rows, "THREADS"))

    # FFT_SPEC
    spec_rows = []
    for spec in fft_spec:
        row = {k: np.asarray(v) for k, v in spec.items()}
        spec_rows.append(row)
    hdus.append(_table_from_list_of_dicts(spec_rows, "FFT_SPEC"))

    # FFT_PEAKS
    peaks_rows = []
    for pk in fft_peaks:
        row = {k: np.asarray(v) for k, v in pk.items()}
        peaks_rows.append(row)
    hdus.append(_table_from_list_of_dicts(peaks_rows, "FFT_PEAKS"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
