"""Pre-calibration sanity checks on a Recording.

Closes section 0.3's second half: cheap checks, run before `calibrate`'s up-to-1000-iteration fit.
Pure; raises nothing -- every outcome is a `Finding`.
"""

import numpy as np

from ax6apple.signals import Recording

from .findings import Finding, breach
from .limits import Limits


def check_rate(rec: Recording, lim: Limits) -> tuple[Finding, ...]:
    """Nominal vs median-interval rate, and interval jitter.

    `signals.nominal_rate_hz` rounds the median interval, so Apple's 50.24 Hz export silently
    becomes fs=50 and that fs drives `build_grid`'s decimation ratio. This records the discrepancy
    (code "rate.rounded") instead of losing it (F10): always recorded, at `warn` once the relative
    error passes `rate_error_warn`, at `info` below it.
    """
    d = np.diff(rec.t_s)
    fs_median = 1.0 / float(np.median(d[d > 0])) if (d > 0).any() else float(rec.fs)
    rate_err = abs(rec.fs - fs_median) / rec.fs if rec.fs else 0.0
    level = "warn" if rate_err > lim.rate_error_warn else "info"
    out = [breach("rate.rounded", level, "device", rate_err, lim.rate_error_warn,
                  f"nominal {rec.fs} Hz vs median-interval {fs_median:.3f} Hz, relative error")]
    jitter = float(d.std() / d.mean()) if len(d) and d.mean() else 0.0
    if jitter > lim.jitter_warn:
        out.append(breach("rate.jitter", "warn", "device", jitter, lim.jitter_warn, "sample interval jitter (SD/mean)"))
    return tuple(out)


def check_timing(rec: Recording, lim: Limits) -> tuple[Finding, ...]:
    """Judges the already-computed Timing: `reordered` and `duplicates_dropped` as fractions of `n_rows`."""
    n = rec.timing.n_rows
    out = []
    for code, count, warn in (("timing.reordered", rec.timing.reordered, lim.reordered_frac_warn),
                              ("timing.duplicates", rec.timing.duplicates_dropped, lim.duplicate_frac_warn)):
        frac = count / n if n else 0.0
        level = "warn" if frac > warn else "info"
        out.append(breach(code, level, "device", frac, warn, f"{code.split('.')[1]} fraction of rows"))
    return tuple(out)


def check_amplitude(xyz: np.ndarray, lim: Limits) -> tuple[Finding, ...]:
    """Gravity present (`norm(mean)` in `mean_mag_g`), no constant or clipped axis, no long zero runs.

    A zero-filled dropout passes `align._still`'s stillness test (std == 0) but fails
    `calibration.still_windows` (norm(mean) < MIN_MEAN_G) -- see F4. This check names it directly.
    """
    out = []
    mean_mag = float(np.linalg.norm(xyz.mean(axis=0)))
    lo, hi = lim.mean_mag_g
    if not (lo <= mean_mag <= hi):
        out.append(breach("amplitude.no_gravity", "fail", "device", mean_mag, lo if mean_mag < lo else hi,
                          "the mean acceleration magnitude", " g"))
    sd_lo, sd_hi = lim.axis_sd_g
    axis_sd = xyz.std(axis=0, ddof=1) if len(xyz) > 1 else np.zeros(3)
    if (axis_sd < sd_lo).any():
        out.append(breach("amplitude.constant_axis", "fail", "device", float(axis_sd.min()), sd_lo,
                          "the smallest per-axis SD", " g"))
    if (axis_sd > sd_hi).any():
        out.append(breach("amplitude.clipped_axis", "fail", "device", float(axis_sd.max()), sd_hi,
                          "the largest per-axis SD", " g"))
    zero_run = float((xyz == 0).all(axis=1).mean()) if len(xyz) else 0.0
    if zero_run > lim.zero_run_frac_warn:
        out.append(breach("amplitude.zero_run", "warn", "device", zero_run, lim.zero_run_frac_warn,
                          "the fraction of all-zero samples"))
    return tuple(out)


def preflight(rec: Recording, xyz: np.ndarray, subject: str, lim: Limits) -> tuple[Finding, ...]:
    """All of the above. Pure; raises nothing."""
    findings = check_rate(rec, lim) + check_timing(rec, lim) + check_amplitude(xyz, lim)
    return tuple(Finding(f.code, f.level, subject, f.message, f.value, f.threshold) for f in findings)
