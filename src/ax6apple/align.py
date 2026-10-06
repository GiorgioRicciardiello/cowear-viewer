"""Aligning the two devices: the clock lag and the rotation between their sensor frames.

Both are estimated from the data because neither is known. The two clocks are stored as the same New York local time
but differ by about 3 s (the movement cross-correlation peak is weak, r about 0.45, because a sleeping wrist barely
moves). The wrist and the sensor frame of the Apple Watch are unrecorded, so its axes cannot be compared with the AX6's
until one frame is rotated onto the other.

Sign convention: a positive `lag_s` means the test device is late, i.e. its event at time t shows up at t + lag on the
test clock. The correction subtracts it from the test timestamps (`signals.build_grid(lag_s=...)`).

The rotation is a Kabsch fit on the gravity vectors of windows where both devices are still, with the AX6 as the
reference frame (`ref = R @ test`). Gravity fixes two of the three rotation angles; the roll about gravity is only
identified by how much the wrist tilted overnight (orientation coverage 1.79 for the AX6, 1.60 for the Apple with the 10 s
calibrated still windows used here; the 1.74 / 1.60 of the plan text are 30 s uncalibrated windows. Either is far from a
full sphere). `fit_rotation` therefore reports a block-bootstrap uncertainty next to the residual, and any view in the
aligned frame should carry it.
"""

from typing import NamedTuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ax6apple.agreement import CONSTANT_REL, block_indices, epoch_pearson, phase_randomised
from ax6apple.calibration import MIN_MEAN_G

MIN_SINGULAR_RATIO = 1e-9
UNCERTAINTY_PERCENTILE = 95
EPOCH_NULL_PERCENTILE = 95
WINDOW_VARIANCE_REL = 1e-12   # an overlap whose variance is below this fraction of its epoch's is constant: no r at that lag


class Lag(NamedTuple):
    lag_s: float
    r: float             # cross-correlation at the peak
    at_boundary: bool    # the peak sits at the edge of the search range: the true lag may lie outside it


class EpochLags(NamedTuple):
    lag_s: np.ndarray        # per epoch: lag at the peak of r over the search range (NaN for a constant epoch)
    r_peak: np.ndarray       # r at that lag: a MAXIMUM over the lags, biased upwards (see `epoch_lags`)
    r_lag0: np.ndarray       # r at lag 0: no selection, the number that can be read as agreement
    r_null: np.ndarray       # per epoch, the 95th percentile of the same peak r against a phase-randomised test epoch


class Rotation(NamedTuple):
    matrix: np.ndarray       # 3 x 3 proper rotation, ref = matrix @ test
    n_windows: int           # jointly still windows the fit used
    residual_deg: float      # RMS angle between rotated test gravity and reference gravity
    uncertainty_deg: float   # 95th percentile of the angle between a block-bootstrap rotation and `matrix`


def band_vm(vm: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase band-passed vector magnitude: gravity and slow drift removed, movement kept."""
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, vm)


def movement_envelope(vm: np.ndarray, fs: float, band: tuple[float, float], bin_s: float) -> np.ndarray:
    """Mean of |band-passed VM| per `bin_s` seconds: how much the wrist moved, not the waveform it moved with."""
    n = int(round(bin_s * fs))
    x = np.abs(band_vm(vm, fs, band))
    return x[: len(x) // n * n].reshape(-1, n).mean(axis=1)


def _lag_range(fs: float, max_lag_s: float) -> int:
    return int(round(max_lag_s * fs))


def peak_lag(ref: np.ndarray, test: np.ndarray, fs: float, max_lag_s: float) -> Lag:
    """Lag of `test` behind `ref` at the peak of their normalised cross-correlation, refined below one sample."""
    if ref.shape != test.shape or ref.ndim != 1:
        raise ValueError(f"ref and test must be 1-D and of the same length, got {ref.shape} and {test.shape}")
    n, span = len(ref), _lag_range(fs, max_lag_s)
    if span >= n:
        raise ValueError(f"the series is shorter ({n} samples) than the lag search range of {span} samples")
    if not (np.isfinite(ref).all() and np.isfinite(test).all()):
        bad, which = (ref, "ref") if not np.isfinite(ref).all() else (test, "test")
        idx = int(np.argmax(~np.isfinite(bad)))
        raise ValueError(f"NaN or infinite value in the series ({which}[{idx}], t={idx / fs:.3f}s): a cross-correlation "
                         "peak would be meaningless -- a gap is the usual cause (F8)")
    if ref.std() == 0 or test.std() == 0:
        raise ValueError("a constant series has no cross-correlation peak")

    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std()

    c = np.fft.irfft(np.conj(np.fft.rfft(z(ref), 2 * n)) * np.fft.rfft(z(test), 2 * n))   # c[k] = sum_t ref[t] test[t+k]
    ks = np.arange(-span, span + 1)
    r = np.concatenate([c[len(c) - span:], c[:span + 1]]) / (n - np.abs(ks))   # lags -span..span, overlap-normalised
    i = int(np.argmax(r))
    at_boundary = i in (0, len(r) - 1)
    shift = 0.0 if at_boundary else 0.5 * (r[i - 1] - r[i + 1]) / (r[i - 1] - 2 * r[i] + r[i + 1])   # parabola through the peak
    return Lag(float((ks[i] + shift) / fs), float(r[i]), at_boundary)


def estimate_lag(ref_vm: np.ndarray, test_vm: np.ndarray, fs: float, band: tuple[float, float], bin_s: float,
                 max_lag_s: float) -> Lag:
    """Clock lag of the test device from the cross-correlation of the two movement envelopes.

    The signed 30 Hz waveforms of the two wrists barely correlate (r 0.03 on the pilot night: different placement and on-device
    filtering), and the peak they give is noise. The envelope of the movement does correlate (r 0.45, lag 3.2 s), at the
    price of a resolution of about one bin, so `lag_s` is good to a fraction of `bin_s`, not to a sample.
    """
    def env(vm: np.ndarray) -> np.ndarray:
        return movement_envelope(vm, fs, band, bin_s)

    return peak_lag(env(ref_vm), env(test_vm), 1.0 / bin_s, max_lag_s)


def _lagged_r(ref: np.ndarray, test: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Pearson r over the overlap of `ref` and `test` shifted by each lag k in `ks` (test late by k), per row: len(ks) x rows.

    For k >= 0 the pairs are ref[t], test[t + k]. The cross products come from one FFT per row and the overlap sums from
    cumulative sums, so all lags cost about one lag's worth of work. NaN where a row is constant or an overlap is.
    """
    n = ref.shape[1]
    a, b = ref - ref.mean(axis=1, keepdims=True), test - test.mean(axis=1, keepdims=True)
    cross = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n, axis=1)) * np.fft.rfft(b, 2 * n, axis=1), 2 * n, axis=1)[:, ks % (2 * n)]
    cum = {k: np.concatenate([np.zeros((len(v), 1)), np.cumsum(v, axis=1)], axis=1) for k, v in
           {"a": a, "aa": a * a, "b": b, "bb": b * b}.items()}
    r0, r1, t0, t1 = np.maximum(-ks, 0), n - np.maximum(ks, 0), np.maximum(ks, 0), n - np.maximum(-ks, 0)   # overlap slices
    m = n - np.abs(ks)
    sa, saa = cum["a"][:, r1] - cum["a"][:, r0], cum["aa"][:, r1] - cum["aa"][:, r0]
    sb, sbb = cum["b"][:, t1] - cum["b"][:, t0], cum["bb"][:, t1] - cum["bb"][:, t0]
    va, vb = saa - sa ** 2 / m, sbb - sb ** 2 / m
    full_a, full_b = cum["aa"][:, -1:], cum["bb"][:, -1:]
    constant = (full_a <= CONSTANT_REL * (ref * ref).sum(axis=1, keepdims=True)) | \
               (full_b <= CONSTANT_REL * (test * test).sum(axis=1, keepdims=True))
    varying = (va > WINDOW_VARIANCE_REL * full_a) & (vb > WINDOW_VARIANCE_REL * full_b) & ~constant
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(varying, (cross - sa * sb / m) / np.sqrt(va * vb), np.nan)
    return np.clip(r, -1.0, 1.0).T


def _peak(r: np.ndarray, ks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(lag in s, r) at the maximum of r over the lags (rows of `r`); NaN for a row with no valid lag."""
    valid = ~np.isnan(r).all(axis=0)
    best = np.argmax(np.where(np.isnan(r), -np.inf, r), axis=0)
    return np.where(valid, ks[best] / fs, np.nan), np.where(valid, r[best, np.arange(r.shape[1])], np.nan)


def epoch_lags(ref: np.ndarray, test: np.ndarray, fs: float, max_lag_s: float, null_draws: int = 0,
               rng: np.random.Generator | None = None) -> EpochLags:
    """Per epoch (rows of epochs x samples): the lag of `test` behind `ref` at the peak Pearson r over the overlap, that
    peak r, r at lag 0, and the peak r that a phase-randomised `test` reaches.

    The peak r is a maximum over 2 * `max_lag_s` * `fs` + 1 lags, so it is biased upwards and its lag is largely
    noise-driven: on independent AR(0.9) epochs (900 samples, +/-5 s) the mean peak r is 0.22 against 0.08 at lag 0, and on
    the pilot-night VM the median peak r is 0.279 at a median lag of 3.6 s, inside what noise alone produces. Do not present
    the peak r as agreement or its lag as a measured offset: `r_lag0` is the unselected r and `r_null`, the
    `EPOCH_NULL_PERCENTILE` of the same maximisation on `null_draws` phase-randomised surrogates of `test` (its spectrum
    and autocorrelation kept, its timing against `ref` destroyed), is what the peak must clear. By construction a null-only
    epoch exceeds it 5 % of the time. With `null_draws` = 0 `r_null` is NaN.

    Whole-sample resolution. NaN for an epoch that is constant on either device: it has no peak to report.
    """
    n, span = ref.shape[1], _lag_range(fs, max_lag_s)
    if span >= n:
        raise ValueError(f"an epoch of {n} samples is shorter than the lag search range of {span} samples")
    if null_draws and rng is None:
        raise ValueError("a null needs an rng")
    ks = np.arange(-span, span + 1)
    lags, peak = _peak(_lagged_r(ref, test, ks), ks, fs)
    null = np.full(len(ref), np.nan)
    if null_draws:
        draws = np.stack([_peak(_lagged_r(ref, phase_randomised(test, rng), ks), ks, fs)[1] for _ in range(null_draws)])
        null = np.where(np.isnan(peak), np.nan, np.percentile(draws, EPOCH_NULL_PERCENTILE, axis=0))
    return EpochLags(lags, peak, epoch_pearson(ref, test), null)


def kabsch(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """The proper rotation R minimising sum |ref_i - R test_i|^2 for n x 3 row vectors (`ref = R @ test` per vector).

    The SVD of `test.T @ ref` gives the optimum; if it is a reflection (det -1) the weakest singular direction is flipped
    so the result is always a rotation. Raises when the vectors span a single direction: the roll about it is then undetermined.
    """
    if ref.shape != test.shape or ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"ref and test must be n x 3 of the same shape, got {ref.shape} and {test.shape}")
    u, s, vt = np.linalg.svd(test.T @ ref)
    if s[1] <= MIN_SINGULAR_RATIO * s[0]:
        raise ValueError("the vectors span a single orientation: the rotation is undetermined")
    d = np.sign(np.linalg.det(vt.T @ u.T))    # -1 would be a reflection: flip the weakest direction
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def rotate(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """n x 3 test-frame samples in the reference frame, as a new array."""
    return xyz @ matrix.T


def _still(xyz: np.ndarray, n: int, std_g: float) -> tuple[np.ndarray, np.ndarray]:
    """F4: uses `calibration.still_windows`' predicate (SD below `std_g` **and** mean at least `MIN_MEAN_G`), not just
    the SD test. Before this fix the rotation fit accepted zero-filled dropout windows the calibration fit rejected:
    `calibration.still_windows` and `align._still` were handed the same
    `(std_g, window_s)` but disagreed on what "still" means.
    """
    windows = xyz[: len(xyz) // n * n].reshape(-1, n, 3)
    means = windows.mean(axis=1)
    still = (windows.std(axis=1, ddof=1) < std_g).all(axis=1) & (np.linalg.norm(means, axis=1) >= MIN_MEAN_G)
    return means, still


def gravity_pairs(ref_xyz: np.ndarray, test_xyz: np.ndarray, fs: float, window_s: float,
                  std_g: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Unit gravity vectors of the windows in which both devices are still (same clock: use the common grid), and the
    number of windows examined (F11: `m`, the denominator `Rotation.n_windows` -- the *kept* count -- has no other way
    to be read as a rejection rate)."""
    n = int(window_s * fs)
    (ref, ref_ok), (test, test_ok) = _still(ref_xyz, n, std_g), _still(test_xyz, n, std_g)
    m = min(len(ref), len(test))
    keep = ref_ok[:m] & test_ok[:m]
    def unit(v: np.ndarray) -> np.ndarray:
        return v[:m][keep] / np.linalg.norm(v[:m][keep], axis=1, keepdims=True)

    return unit(ref), unit(test), m


def _angle_deg(cos: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def fit_rotation(ref_xyz: np.ndarray, test_xyz: np.ndarray, fs: float, window_s: float, std_g: float,
                 block_windows: int, resamples: int, rng: np.random.Generator) -> Rotation:
    """Rotation of the test frame onto the reference frame with its residual and block-bootstrap uncertainty."""
    ref, test, _ = gravity_pairs(ref_xyz, test_xyz, fs, window_s, std_g)
    if len(ref) == 0:
        raise ValueError(f"no window of {window_s} s is still on both devices (SD < {std_g} g)")
    matrix = kabsch(ref, test)
    residual = float(np.sqrt(np.mean(_angle_deg((test @ matrix.T * ref).sum(axis=1)) ** 2)))
    idx = block_indices(len(ref), min(block_windows, len(ref)), resamples, rng)
    boot = np.array([_angle_deg((np.trace(matrix.T @ kabsch(ref[i], test[i])) - 1) / 2) for i in idx])
    return Rotation(matrix, len(ref), residual, float(np.percentile(boot, UNCERTAINTY_PERCENTILE)))
