"""In-situ accelerometer calibration: a per-axis gain and offset from still windows (port of `calibrate_acc.m`).

A device at rest measures gravity, so the mean of a still window must lie on the unit sphere. The fit moves each
axis by `g * x + d` until the still-window means do. The AX6 of the pilot night carries offsets of -37.5 and +52.4 mg on
two axes, which is 25 times its ENMO at rest: without this step the 30 s ENMO correlation with the Apple Watch is
0.18 instead of 0.55.

This is the MATLAB gain+offset fit, not the pipeline's `accelerometer_calibration.m` (ICP with a rotation): the ABK
features were computed on the ICP-rotated signal, so the viewer's calibrated signal is not the features' input.

The loop follows the MATLAB source line by line, including its update `d += d_inc / (g * g_inc)`. Windows are counted
in samples at the nominal rate, as MATLAB does. A window whose mean is shorter than `MIN_MEAN_G` cannot be a device at
rest in gravity (a dropout filled with zeros is "still", with SD 0) and is not used.

One deliberate difference: MATLAB stops when the residual changes by less than 1e-10 (`MATLAB_TOL`), but the residual of a
well fitted still night is itself 1e-8 to 1e-10, so the loop stops while the offsets are still moving (AX6: 3 to 5 mg away,
Apple after 5 iterations). The residual is also not scale-free: `3 * mean(w e^2) / sum(w)` divides by sum(w) a second time,
so it is not a mean squared distance to the sphere (7.5e-9 reported, 8.4e-6 g^2 actual on the AX6) and its tolerance
depends on N and the weights. By default the loop therefore stops when no parameter moves by more than `PARAM_TOL`
(1e-8 g), which lands within 0.0004 mg of the fixed point; `tol=MATLAB_TOL` (or any number) gives MATLAB's residual stop.
`residual` keeps MATLAB's value for fidelity; `rms_g`, the RMS distance of the calibrated still-window means to the unit
sphere, is the number to judge a fit by.

The fixed point is not exactly the least-squares sphere fit: the weights `1/|1-EN|` are capped at 100, and it is least
squares only where they all saturate (|1-EN| < 0.01, as for the Apple: 3e-5 mg from `scipy.least_squares`). On the AX6
night, where they do not, it is 0.03 mg away. The published pilot-night numbers are reproduced.

The fit is only as good as the orientations it saw: `coverage` (mean over the three axes of the span of the
still-window gravity direction, 2.0 for the whole sphere, 0 for a device that never turned) says how far to trust it.
`min_coverage` refuses a fit below it and `low_coverage` sets the `low_orientation_coverage` flag. The 1.74 / 1.60
published in the plan comes from 30 s uncalibrated still windows; the 10 s rule used here, measured after the fit,
gives 1.79 / 1.60 (both pinned in tests/test_calibration.py). The definition is kept.
"""

from typing import NamedTuple

import numpy as np

MAX_ITER = 1000
MATLAB_TOL = 1e-10      # calibrate_acc.m: stops early on quiet nights, see module docstring
PARAM_TOL = 1e-8        # default stop: largest step of any gain or offset, in g
MAX_WEIGHT = 100.0
MIN_MEAN_G = 0.5        # a still window at rest measures gravity: a shorter mean is a dropout, not a device orientation


class Calibration(NamedTuple):
    gain: np.ndarray        # per axis, applied as gain * x + offset
    offset: np.ndarray      # g
    n_windows: int          # complete 10 s windows in the recording
    n_retained: int         # of those, the still ones the fit used
    coverage: float         # orientation coverage of the retained windows (see module docstring)
    residual: float         # MATLAB's `3*mean(w*e^2)/sum(w)`: divides by sum(w) twice, NOT a mean squared distance
    iterations: int
    rms_g: float            # RMS distance of the calibrated still-window means to the unit sphere, g
    converged: bool         # the stop rule was met before MAX_ITER (False: the fit is whatever the last iteration gave)
    low_orientation_coverage: bool   # coverage < low_coverage: the sphere fit rests on few orientations


def still_windows(xyz: np.ndarray, fs: float, window_s: float, std_g: float) -> tuple[np.ndarray, int]:
    """The still `window_s` windows (k x n x 3) of an n x 3 recording, and the number of complete windows.

    Still: the sample SD of every axis is below `std_g` (MATLAB `std`) and the mean is at least `MIN_MEAN_G` long.
    """
    n = int(window_s * fs)
    windows = xyz[: len(xyz) // n * n].reshape(-1, n, 3)
    still = (windows.std(axis=1, ddof=1) < std_g).all(axis=1) & (np.linalg.norm(windows.mean(axis=1), axis=1) >= MIN_MEAN_G)
    return windows[still], len(windows)


def _still_window_means(xyz: np.ndarray, fs: float, window_s: float, std_g: float) -> tuple[np.ndarray, int]:
    windows, n_windows = still_windows(xyz, fs, window_s, std_g)
    return windows.mean(axis=1), n_windows


def _weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Intercept and slope of the weighted least squares line y = b0 + b1 x (what `fitlm(..., 'Weights', w)` returns)."""
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    b1 = (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum()
    return ym - b1 * xm, b1


def _coverage(means: np.ndarray) -> float:
    unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    return float((unit.max(axis=0) - unit.min(axis=0)).mean())


def calibrate(xyz: np.ndarray, fs: float, std_g: float = 0.013, window_s: float = 10, tol: float | None = None,
              min_coverage: float = 0.0, low_coverage: float = 0.0) -> Calibration:
    """Fit gain and offset on the still `window_s` windows of an n x 3 recording in g sampled at `fs` Hz.

    `tol` None stops on the parameters (`PARAM_TOL`); a number stops when the MATLAB residual changes by less than it.
    Raises if there is no usable still window, if their orientation coverage is below `min_coverage`, or if the fit does
    not stay finite. `converged` is False when `MAX_ITER` ran out first.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be n x 3, got shape {xyz.shape}")
    means, n_windows = _still_window_means(xyz, fs, window_s, std_g)
    if len(means) == 0:
        raise ValueError(f"no still window (SD < {std_g} g on all axes, mean >= {MIN_MEAN_G} g) among {n_windows} windows of {window_s} s")
    if np.any(np.ptp(means, axis=0) == 0):
        raise ValueError("orientation coverage is zero on an axis: the device never turned, the sphere fit is undetermined")
    if _coverage(means) < min_coverage:
        raise ValueError(f"orientation coverage {_coverage(means):.3f} of the still windows is below the minimum {min_coverage}: "
                         "the sphere fit is poorly determined")
    d, g = np.zeros(3), np.ones(3)
    c1, weights, res, converged = means, np.ones(len(means)), np.inf, False
    with np.errstate(all="ignore"):   # a diverging fit is reported below, not as a stream of overflow warnings
        for iteration in range(1, MAX_ITER + 1):
            cp = means / np.linalg.norm(c1, axis=1, keepdims=True)   # closest point on the sphere
            fits = [_weighted_line(means[:, k], cp[:, k], weights) for k in range(3)]
            d_inc, g_inc = np.array([f[0] for f in fits]), np.array([f[1] for f in fits])
            step = d_inc / (g * g_inc)
            d = d + step
            g = g * g_inc
            c1 = g * means + d
            en = np.linalg.norm(c1, axis=1)
            if not (np.isfinite(g).all() and np.isfinite(d).all() and np.isfinite(en).all()):
                raise ValueError(f"the calibration fit is not finite after {iteration} iterations (orientation coverage "
                                 f"{_coverage(means):.3f}): the still windows do not determine a sphere fit")
            res_old, res = res, 3 * np.mean(weights * (1 - en) ** 2) / weights.sum()
            if tol is None:
                converged = max(np.abs(step).max(), np.abs(g_inc - 1).max()) < PARAM_TOL
            else:
                converged = abs(res - res_old) < tol
            if converged:
                break
            weights = np.minimum(1 / np.maximum(np.abs(1 - en), 1 / MAX_WEIGHT), MAX_WEIGHT)   # 1/|1-EN| capped at 100
    coverage = _coverage(c1)
    return Calibration(g, d, n_windows, len(means), coverage, float(res), iteration, float(np.sqrt(np.mean((1 - en) ** 2))),
                       bool(converged), bool(coverage < low_coverage))


def apply(xyz: np.ndarray, cal: Calibration) -> np.ndarray:
    """The calibrated signal `gain * x + offset` as a new array."""
    return cal.gain * xyz + cal.offset
