"""Python port of core Auto-NUWT tracking routines (IDL)."""

from .locate import locate_things
from .follow import follow_threads
from .patch import patch_up_threads

__all__ = [
    "load_td_fits",
    "save_results_fits",
    "locate_things",
    "follow_threads",
    "patch_up_threads",
    "apply_fft",
    "run_nuwt",
]


def load_td_fits(*args, **kwargs):
    from .io import load_td_fits as _load_td_fits

    return _load_td_fits(*args, **kwargs)


def save_results_fits(*args, **kwargs):
    from .io import save_results_fits as _save_results_fits

    return _save_results_fits(*args, **kwargs)


def apply_fft(*args, **kwargs):
    from .fft import apply_fft as _apply_fft

    return _apply_fft(*args, **kwargs)


def run_nuwt(*args, **kwargs):
    from .run import run_nuwt as _run_nuwt

    return _run_nuwt(*args, **kwargs)
