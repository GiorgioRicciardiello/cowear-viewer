"""Frequency-domain comparison of the two devices.

Runs on the `grid50` tier (`signals.build_grid(..., grid_hz=50)`), never on the viewer's `native` chunks: those are
linearly interpolated onto a uniform axis (`viewer_export.uniform_native`), a mild rate-dependent low-pass that would
bias exactly the band under study.

A sleeping wrist is mostly still, and each device's own sensor noise floor rises with frequency at a different rate
(measured on the pilot night: AX6 flat around -69 dB g2/Hz from 12-48 Hz; Apple rolling off to -81 dB by 24 Hz). An
uncorrected spectral comparison above about 10 Hz therefore mostly compares noise floors, not movement -- every
band/scalar function here is meant to be read beside `noise_psd`/`noise_correct`, not alone.

Comparisons stop at 20 Hz (`BANDS["margin"]` and above are reported per device only): Apple's anti-alias response
near its own 25 Hz Nyquist is unknown, and above 25 Hz it has no samples at all.

The wavelet decomposition (`modwt_epoch_energy`) uses the SWT/MODWT (`pywt.swt`), not a decimated DWT: a decimated
DWT is not shift-invariant, and the two devices carry a residual clock lag (`manifest.lag_s`, about 3.18 s on
the pilot night, plus a per-epoch residual), so a decimated transform's coefficients would differ between devices for
reasons that are pure misalignment, not signal.
"""

from math import ceil

import numpy as np
import pywt
from scipy.signal import coherence as _coherence
from scipy.signal import welch

from ax6apple.align import band_vm
from ax6apple.calibration import still_windows
from ax6apple.signals import epoch_matrix, vm

# Band edges (Hz), named by what they are for rather than by number alone.
# B5 and B6 are diagnostic: never entered into a cross-device agreement statistic.
BANDS = {
    "posture": (0.1, 0.5),
    "gross_movement": (0.5, 3.0),
    "jerk_twitch": (3.0, 8.0),          # Apple carries an unexplained, unfixable excess here -- see the module docstring's
    "tremor": (8.0, 15.0),              # physiological tremor range
    "diagnostic": (15.0, 20.0),
    "margin": (20.0, 24.9),              # per device only: Apple's near-Nyquist response is unknown; 24.9 not 25.0 so a
                                          # Butterworth bandpass on exactly-50 Hz grid50 has a Wn strictly below Nyquist
}
COMPARABLE_BANDS = ("posture", "gross_movement", "jerk_twitch", "tremor", "diagnostic")
SCALAR_BAND = (0.1, 20.0)   # the range spectral_scalars() is computed over
AX6_ONLY_BAND = (25.0, 49.0)   # Apple has no samples up here at all (F6: the single definition; `BANDS` used to carry a
                               # second, unused (25, 50) entry under the same name)


def epoch_psd(x: np.ndarray, fs: float, epoch_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of each epoch as its own single Hann-windowed segment: f (n_freq,), P (n_epochs, n_freq)."""
    f, P = welch(epoch_matrix(x, epoch_n), fs=fs, nperseg=epoch_n, axis=1, detrend="constant")
    return f, P


def noise_psd(xyz: np.ndarray, fs: float, window_s: float, std_g: float) -> tuple[np.ndarray, np.ndarray]:
    """A device's noise floor N(f): VM Welch PSD averaged over its own still windows (`calibration.still_windows`),
    the same still-window selection `signals.sensor_noise_floor` uses, so the two stay consistent.
    """
    windows, _ = still_windows(xyz, fs, window_s, std_g)
    if len(windows) == 0:
        raise ValueError("no still windows at this std_g: cannot estimate a noise floor")
    v = vm(windows.reshape(-1, 3)).reshape(len(windows), -1)
    f, P = welch(v - v.mean(axis=1, keepdims=True), fs=fs, nperseg=v.shape[1], axis=1, detrend="constant")
    return f, P.mean(axis=0)


def noise_correct(f: np.ndarray, P: np.ndarray, f_noise: np.ndarray, N: np.ndarray) -> np.ndarray:
    """max(P - N, 0); `N` is linearly interpolated onto `f` first since the two may come from different segment lengths."""
    return np.maximum(P - np.interp(f, f_noise, N), 0.0)


def band_epoch_rms(x: np.ndarray, fs: float, band: tuple[float, float], epoch_n: int) -> np.ndarray:
    """Per-epoch RMS (unit of `x`) inside `band`: `align.band_vm`'s zero-phase Butterworth bandpass, then epoch RMS.

    F9 (D9): a NaN gap is bridged (linearly interpolated) before the whole-stream filter runs, as `build_grid` already
    bridges short gaps -- `sosfiltfilt` otherwise NaNs every output sample, costing all epochs for one dropout, unlike
    `epoch_psd` beside it (per-epoch Welch, so only the affected epoch is lost). Only the epochs that actually overlapped
    a gap are masked afterwards; filtering per epoch instead would add edge effects at every epoch boundary (D9).
    """
    x = np.asarray(x, dtype=float)
    nan_mask = ~np.isfinite(x)
    bridged = x
    if nan_mask.any():
        idx = np.arange(len(x))
        bridged = x.copy()
        bridged[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    rms = np.sqrt(epoch_matrix(band_vm(bridged, fs, band) ** 2, epoch_n).mean(axis=1))
    if nan_mask.any():
        rms = np.where(epoch_matrix(nan_mask, epoch_n).any(axis=1), np.nan, rms)
    return rms


def spectral_scalars(f: np.ndarray, P: np.ndarray, f_lo: float = SCALAR_BAND[0], f_hi: float = SCALAR_BAND[1]) -> dict[str, np.ndarray]:
    """Per-epoch centroid, 95th-percentile spectral edge frequency, dominant frequency and normalised spectral entropy,
    over `[f_lo, f_hi]`. `P` is `epoch_psd`'s (n_epochs, n_freq), raw or noise-corrected. NaN where the band has no power.
    """
    m = (f >= f_lo) & (f <= f_hi)
    fb, Pb = f[m], np.atleast_2d(P)[:, m]
    tot = Pb.sum(axis=1)
    safe = np.where(tot > 0, tot, 1.0)
    centroid = np.where(tot > 0, (Pb * fb).sum(axis=1) / safe, np.nan)
    cum = np.cumsum(Pb, axis=1)
    edge_i = np.array([np.searchsorted(c, 0.95 * t) for c, t in zip(cum, tot, strict=True)])
    sef95 = np.where(tot > 0, fb[np.clip(edge_i, 0, len(fb) - 1)], np.nan)
    dominant = np.where(tot > 0, fb[np.argmax(Pb, axis=1)], np.nan)
    p = Pb / safe[:, None]
    entropy = np.where(tot > 0, -(p * np.log(p + 1e-30)).sum(axis=1) / np.log(len(fb)), np.nan)
    return {"centroid_hz": centroid, "sef95_hz": sef95, "dominant_hz": dominant, "spectral_entropy": entropy}


def coherence(a: np.ndarray, b: np.ndarray, fs: float, seg_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Magnitude-squared coherence of two equal-length, time-aligned `grid50` streams, Hann `seg_s` windows, 50% overlap.

    Returns (f, coherence, bias_floor): an independent pair averages to about `bias_floor` = 1 / n_segments, not 0, so
    that is the null to compare against, not zero.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must be the same length, got {len(a)} and {len(b)}")
    nperseg = int(seg_s * fs)
    f, C = _coherence(a, b, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    n_segments = max(1, ceil(len(a) / (nperseg - nperseg // 2)) - 1)
    return f, C, 1.0 / n_segments


def spectrogram_db(x: np.ndarray, fs: float, seg_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dB(g2/Hz) spectrogram for the viewer's Spectrum page: f, t (s, segment centres), S (n_freq, n_time)."""
    from scipy.signal import spectrogram as _spectrogram
    nperseg = int(seg_s * fs)
    f, t, S = _spectrogram(x, fs=fs, nperseg=nperseg, noverlap=nperseg - int(hop_s * fs), detrend="constant")
    return f, t, 10 * np.log10(S + 1e-30)


def dyadic_bands(fs: float, level: int) -> list[tuple[float, float]]:
    """(lo, hi) Hz of each SWT detail level D1 (finest, nearest Nyquist) .. D`level` (coarsest), from `modwt_epoch_energy`."""
    nyq = fs / 2
    return [(nyq / 2 ** k, nyq / 2 ** (k - 1)) for k in range(1, level + 1)]


def modwt_epoch_energy(x: np.ndarray, epoch_n: int, wavelet: str = "sym8", level: int = 6) -> np.ndarray:
    """Per-epoch mean-square energy at each SWT detail level, D1 (highest frequency) first: an (n_epochs, level) array.

    The signal is edge-padded to a multiple of 2**level (pywt.swt's own requirement), decomposed once over its full
    length, then the pad is dropped and the per-sample squared detail coefficients are averaged inside each epoch --
    the padding only touches the last partial block, never an epoch boundary inside the real data.
    """
    x = np.asarray(x, dtype=float)
    pad = (-len(x)) % (2 ** level)
    xp = np.pad(x, (0, pad), mode="edge") if pad else x
    coeffs = pywt.swt(xp, wavelet, level=level, trim_approx=True)   # [cA_level, cD_level, ..., cD_1]
    details = coeffs[1:][::-1]                                      # -> [cD_1 (highest freq), ..., cD_level]
    energy = np.stack([d[:len(x)] ** 2 for d in details], axis=1)    # (n, level), pad dropped
    n_epochs = len(x) // epoch_n
    return energy[:n_epochs * epoch_n].reshape(n_epochs, epoch_n, level).mean(axis=1)
