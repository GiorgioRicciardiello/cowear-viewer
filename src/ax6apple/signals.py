"""The signal chain between a device's Parquet and the agreement statistics: load, VM/ENMO, 30 Hz grid, epochs, MA sweep.

Times are the naive New York local times used across the project. Everything is counted from one anchor, the
recording start (`recording_window`), never from a round clock time: MATLAB's `auto_sleep/abk` bouts fall on :07 and :37
seconds because they are cut from the recording start, so a round-time epoch grid would move every sleep flag by up to
one epoch (`rbd_sleep.epoch_vector` assumes this anchor).

Two choices decide whether the numbers match the published ones, and both are deliberate:

- VM and ENMO are computed at each device's native rate and then put on the grid together with x, y, z. ENMO is a
  rectified quantity, so it is not preserved by filtering: ENMO of the anti-alias-filtered 30 Hz axes is 1.81 / 1.16 mg
  (AX6 / Apple) instead of 2.09 / 1.29, and the 30 s ENMO correlation falls from 0.55 to 0.51. Filtering the ENMO
  stream itself keeps every epoch mean, so the epoch correlation stays 0.55. The price is that the grid ENMO rings
  well below 0 where the filter meets a rectified edge (minimum -186 mg AX6, -101 mg Apple on the pilot night): epoch means and
  the night-level r are unaffected, but the w = 0 correlation and the within-epoch r are computed on that ringing stream.
- The moving average runs on the continuous night stream and is cut into epochs afterwards. Smoothing inside each epoch
  would lose w/2 samples at each edge and put a discontinuity on the 30 s period under study.

The common grid is the only place where the two devices are paired: native-rate traces are for one device at a time.
"""

from math import gcd
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from ax6apple.agreement import effective_n_from_ci, epoch_pearson, pearson_block_sensitivity, phase_randomised
from ax6apple.align import peak_lag
from ax6apple.calibration import still_windows
from ax6apple.rbd_sleep import EPOCH
from ax6apple.config import AX6APPLE

VIEWER = AX6APPLE["viewer"]
NULL_PERCENTILES = (2.5, 50, 97.5)
N_EFF_BASIS = "bootstrap_half_width"     # n_eff = (1.96 (1 - r^2) / half_width)^2 of the interval beside it
WITHIN_EPOCH_QUANTILES = (5, 25, 50, 75, 95)


class Timing(NamedTuple):
    n_rows: int
    reordered: int            # steps where the stored time went backwards or repeated (rows sorted into time order)
    duplicates_dropped: int   # exact repeated timestamps, first row kept
    max_gap_s: float


class Recording(NamedTuple):
    t_s: np.ndarray           # seconds after the anchor (negative in a lag margin), strictly increasing
    xyz: np.ndarray           # n x 3, g, as stored
    fs: int                   # nominal rate, Hz
    timing: Timing


class GridSignals(NamedTuple):
    xyz: np.ndarray           # n x 3
    vm: np.ndarray
    enmo: np.ndarray


def recording_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """(anchor, end): the recording start and last compared sample, from the configured variant."""
    name = AX6APPLE["viewer"]["window_variant"]
    variant = next((v for v in AX6APPLE["variants"] if v["name"] == name), None)
    if variant is None:
        raise ValueError(f"config ax6apple.viewer.window_variant {name!r} is not one of ax6apple.variants")
    return pd.Timestamp(variant["start"]), pd.Timestamp(variant["end"])


def epoch_starts(anchor: pd.Timestamp, n_epochs: int) -> pd.DatetimeIndex:
    """Start time of each 30 s epoch, counted from the anchor."""
    return pd.date_range(anchor, periods=n_epochs, freq=EPOCH)


def vm(xyz: np.ndarray) -> np.ndarray:
    """Vector magnitude (Euclidean norm of each row) of an n x 3 acceleration array, in the unit of the input (g)."""
    return np.linalg.norm(xyz, axis=1)


def enmo(xyz: np.ndarray) -> np.ndarray:
    """Euclidean norm minus one, floored at zero (g)."""
    return np.maximum(vm(xyz) - 1.0, 0.0)


def nominal_rate_hz(t_s: np.ndarray) -> int:
    """The rate the device is set to (100, 50 Hz), from the median interval: the Apple export averages 50.24 Hz."""
    return int(round(1.0 / np.median(np.diff(t_s))))


def load_recording(parquet: str | Path, start: pd.Timestamp, end: pd.Timestamp, margin_s: float = 0.0) -> Recording:
    """[start, end] of a `time,x,y,z` Parquet in time order, with the timing counts that were corrected.

    `margin_s` also loads that many seconds either side of the window (`t_s` is then negative before the anchor), so a clock
    lag of either sign, up to the margin, can be corrected without the grid running out of data at its start or end.
    The Apple export steps backwards at batch boundaries (`rbd_inputs.export_raw_csv`): rows are stable-sorted and exact
    repeated timestamps dropped, first kept. The counts are returned, not hidden.
    """
    margin = pd.Timedelta(seconds=margin_s)
    raw = pd.read_parquet(parquet, columns=["time", "x", "y", "z"],
                          filters=[("time", ">=", start - margin), ("time", "<=", end + margin)])
    if raw.empty:
        raise ValueError(f"no samples of {parquet} between {start} and {end}")
    reordered = int((raw.time.diff() <= pd.Timedelta(0)).sum())
    df = raw.sort_values("time", kind="stable").drop_duplicates("time", keep="first")
    t_s = (df.time - start).dt.total_seconds().to_numpy()
    timing = Timing(len(df), reordered, len(raw) - len(df), float(np.diff(t_s).max()) if len(t_s) > 1 else 0.0)
    return Recording(t_s, df[["x", "y", "z"]].to_numpy(dtype=float), nominal_rate_hz(t_s), timing)


def gap_mask(t: np.ndarray, tq: np.ndarray, max_gap_s: float) -> np.ndarray:
    """True where a query time lies between two consecutive samples of `t` more than `max_gap_s` apart."""
    i = np.searchsorted(t, tq)                       # first sample at or after the query
    after = t[np.minimum(i, len(t) - 1)]
    inside = (i > 0) & (i < len(t)) & (after > tq)   # strictly between two samples: a query on a sample is data
    return inside & (after - t[np.maximum(i - 1, 0)] > max_gap_s)


def build_grid(t_s: np.ndarray, xyz: np.ndarray, fs: int, grid_hz: int, n_out: int, lag_s: float = 0.0,
               start_tol_s: float = VIEWER["grid_start_tol_s"], max_gap_s: float = VIEWER["max_gap_s"]) -> GridSignals:
    """x, y, z, VM and ENMO on the common grid: sample k is at anchor + k / grid_hz.

    The stored timestamps are corrected by `lag_s` (see align.py), the samples interpolated onto a uniform grid at the
    native rate (removes the 2.6 % interval jitter) and then low-passed and decimated by `resample_poly` (zero phase,
    so no delay is added). A grid that ends outside the recording, or begins more than `start_tol_s` after it, is refused
    rather than padded with edge values; a start before the anchor is fine (load with a lag margin, `load_recording`),
    so a device whose clock is early is as usable as one that is late.

    Where the device has a gap longer than `max_gap_s` the grid is NaN, not interpolated: a straight line across a 30 s
    dropout would draw as a smooth ramp in x, y, z, VM and ENMO. Shorter gaps are bridged linearly. The filter reaches
    about 0.3 s past a gap edge, so the samples next to a NaN stretch carry a little of the ramp.
    The filter's edge effect (about 0.5 s, 0.01 g on a 1 g level) sits in the first and last samples of the grid.
    """
    if n_out * fs % grid_hz:
        raise ValueError(f"{n_out} samples at {grid_hz} Hz is not a whole number of samples at {fs} Hz")
    n_native = n_out * fs // grid_hz
    t = t_s - lag_s
    if t[-1] < (n_native - 1) / fs:
        raise ValueError(f"the recording ends at {t[-1]:.1f} s after the anchor (lag corrected), "
                         f"before the grid end at {n_out / grid_hz:.1f} s")
    if t[0] > start_tol_s:
        raise ValueError(f"the recording starts {t[0]:.1f} s after the anchor (lag corrected): the grid would begin on edge values")
    scalars = np.column_stack([xyz, vm(xyz), enmo(xyz)])
    tu = np.arange(n_native) / fs
    uniform = np.column_stack([np.interp(tu, t, scalars[:, k]) for k in range(scalars.shape[1])])
    g = gcd(grid_hz, fs)
    out = resample_poly(uniform, grid_hz // g, fs // g, axis=0, padtype="line")   # 'line': no zero-padding transient on the 1 g level
    out[gap_mask(t, np.arange(n_out) / grid_hz, max_gap_s)] = np.nan
    return GridSignals(np.ascontiguousarray(out[:, :3]), np.ascontiguousarray(out[:, 3]), np.ascontiguousarray(out[:, 4]))


def sensor_noise_floor(xyz: np.ndarray, fs: float, window_s: float, std_g: float) -> dict:
    """A device's own ENMO noise floor, in mg, measured on its still native-rate windows (raw, uncalibrated).

    A still device re-centred on 1 g has VM = 1 + noise, so its ENMO is max(VM - mean(VM), 0) and the floor is the mean of
    that over the still windows: what ENMO reads on a device that is not moving. An epoch below it says nothing about
    movement. For Gaussian noise of SD sigma this is sigma / sqrt(2 pi); it is measured directly because the noise is not
    Gaussian (the Apple's sigma / sqrt(2 pi) would be 0.73 mg against the measured 0.65). Pilot night: AX6 1.18 mg (median
    sigma 2.85 mg), Apple 0.65 mg (1.70 mg), about half of each device's mean calibrated ENMO (2.09 / 1.29 mg).
    """
    windows, _ = still_windows(xyz, fs, window_s, std_g)
    v = vm(windows.reshape(-1, 3)).reshape(len(windows), -1)
    sigma = v.std(axis=1, ddof=1) * 1000
    return {"value_mg": float(np.maximum(v - v.mean(axis=1, keepdims=True), 0).mean() * 1000),
            "sigma_vm_mg_mean": float(sigma.mean()), "sigma_vm_mg_median": float(np.median(sigma)), "n_windows": int(len(v)),
            "derivation": f"mean ENMO of the device's own noise: VM re-centred on 1 g in {len(v)} still {window_s:g} s native-rate "
                          f"windows (SD of every axis < {std_g} g), max(VM - window mean, 0) averaged; for Gaussian noise this is "
                          "sigma_VM / sqrt(2 pi)"}


def epoch_matrix(x: np.ndarray, epoch_n: int) -> np.ndarray:
    """A stream as epochs x samples (a view: it does not copy)."""
    if len(x) % epoch_n:
        raise ValueError(f"{len(x)} samples is not a multiple of the {epoch_n}-sample epoch: cut the stream to whole epochs first")
    return x.reshape(-1, epoch_n)


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred `n`-sample moving average. Near the ends the window shrinks to the samples that exist (no zero padding)."""
    if n <= 1:
        return np.array(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    i = np.arange(len(x)) - n // 2
    lo, hi = np.clip(i, 0, len(x)), np.clip(i + n, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def smooth_then_epoch(x: np.ndarray, n: int, epoch_n: int) -> np.ndarray:
    """Moving average over the continuous stream, then cut into epochs (never the other way round)."""
    return epoch_matrix(moving_average(x, n), epoch_n)


def _within_epoch(sa: np.ndarray, sb: np.ndarray, epoch_n: int) -> dict[str, float | int]:
    r = epoch_pearson(epoch_matrix(sa, epoch_n), epoch_matrix(sb, epoch_n))
    ok = r[~np.isnan(r)]
    q = np.percentile(ok, WITHIN_EPOCH_QUANTILES) if len(ok) else np.full(len(WITHIN_EPOCH_QUANTILES), np.nan)
    return {**{f"q{p:02d}": float(v) for p, v in zip(WITHIN_EPOCH_QUANTILES, q, strict=True)}, "n_valid": int(len(ok))}


def _null_floor(sa: list[np.ndarray], b: np.ndarray, ns: list[int], draws: int, rng: np.random.Generator) -> np.ndarray:
    """Pearson r between the smoothed a and phase-randomised b, per draw and window: draws x windows."""
    out = np.empty((draws, len(ns)))
    for d in range(draws):
        surrogate = phase_randomised(b, rng)
        out[d] = [np.corrcoef(sa[j], moving_average(surrogate, n))[0, 1] for j, n in enumerate(ns)]
    return out


def ma_sweep(a: np.ndarray, b: np.ndarray, fs: float, epoch_n: int, cfg: dict) -> list[dict]:
    """Correlation of two 30 Hz streams (reference a, test b) against the width of a moving average, one row per window.

    `cfg` is `config ax6apple.viewer`: `ma_windows_s`, `lag_search_s` and `bootstrap` (block_min, block_sensitivity_min,
    resamples, seed, null_draws). Each row holds the night-level r over all samples with its block-bootstrap interval, the
    distribution of the within-epoch r (smoothing done before epoching), the 2.5 / 50 / 97.5 percentiles of the
    phase-randomised null, the effective sample size, and the lag of the cross-correlation peak. The honest signal is the
    gap between r and the null, not r itself: smoothing raises r on any pair of slow series.

    `r_lo`, `r_hi` are conditional on the block length `block_min`; `block_sensitivity` gives the interval at every block in
    `block_sensitivity_min` (and `block_min`), and any one of them understates the uncertainty if the dependence is longer
    than its block. `n_eff` is derived from that headline interval's half-width (`n_eff_basis`), so it is conditional on
    the same block and never contradicts it. A block longer than the series is an error.

    The window is rounded to whole samples (0.25 s at 30 Hz is 8 samples); `w_eff_s` says what was applied.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"a and b must be 1-D and of the same length, got {a.shape} and {b.shape}")
    windows, boot = cfg["ma_windows_s"], cfg["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    blocks = sorted({*boot["block_sensitivity_min"], boot["block_min"]})
    ns = [max(1, int(round(w * fs))) for w in windows]
    smoothed = [(moving_average(a, n), moving_average(b, n)) for n in ns]
    null = _null_floor([s[0] for s in smoothed], b, ns, boot["null_draws"], rng)
    rows = []
    for j, (w, n, (sa, sb)) in enumerate(zip(windows, ns, smoothed, strict=True)):
        sens = pearson_block_sensitivity(sa, sb, blocks, fs, boot["resamples"], boot["seed"])
        head = next(x for x in sens if x["block_min"] == boot["block_min"])
        r = float(np.corrcoef(sa, sb)[0, 1])
        lag = peak_lag(sa, sb, fs, cfg["lag_search_s"])
        nlo, nmed, nhi = np.percentile(null[:, j], NULL_PERCENTILES)
        rows.append({"w_s": w, "n_samples": n, "w_eff_s": n / fs, "r": r, "r_lo": head["lo"], "r_hi": head["hi"],
                     "block_sensitivity": sens, "within_epoch_r": _within_epoch(sa, sb, epoch_n), "null_lo": float(nlo),
                     "null_med": float(nmed), "null_hi": float(nhi),
                     "n_eff": effective_n_from_ci(r, head["lo"], head["hi"], len(a)), "n_eff_basis": N_EFF_BASIS,
                     "lag_s": lag.lag_s, "lag_r": lag.r})
    return rows
