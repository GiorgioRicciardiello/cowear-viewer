"""Frequency-domain comparison functions, tested against synthetic signals with a
known band content -- there is no published frequency-domain regression anchor to check against yet.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pywt

from ax6apple.spectral import (BANDS, band_epoch_rms, coherence, dyadic_bands, epoch_psd, modwt_epoch_energy, noise_correct,
                               noise_psd, spectral_scalars, spectrogram_db)

FS = 50.0
RNG = np.random.default_rng(0)


def _tone(freq, seconds, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase), t


# --- band_epoch_rms ---

def test_band_epoch_rms_recovers_a_sine_amplitude_inside_its_own_band():
    x, _ = _tone(2.0, seconds=60, amp=0.5)                       # 2 Hz sine, amplitude 0.5
    epoch_n = int(30 * FS)
    rms = band_epoch_rms(x, FS, BANDS["gross_movement"], epoch_n)   # 0.5-3 Hz contains 2 Hz
    assert rms == pytest.approx(0.5 / np.sqrt(2), rel=0.02)          # RMS of a sine = amplitude / sqrt(2)


def test_band_epoch_rms_rejects_a_tone_outside_the_band():
    x, _ = _tone(2.0, seconds=60, amp=0.5)
    epoch_n = int(30 * FS)
    rms = band_epoch_rms(x, FS, BANDS["tremor"], epoch_n)          # 8-15 Hz: a 2 Hz tone should be mostly gone
    assert (rms < 0.05).all()


# --- F9: a NaN gap costs only its own epochs, not every epoch (D9) ---

def test_band_epoch_rms_bridges_a_gap_and_masks_only_its_own_epochs():
    epoch_n = int(30 * FS)
    x, _ = _tone(2.0, seconds=180, amp=0.5)
    gapped = x.copy()
    gapped[epoch_n * 2: epoch_n * 2 + 10] = np.nan           # a short gap inside epoch 2 only
    rms = band_epoch_rms(gapped, FS, BANDS["gross_movement"], epoch_n)
    clean = band_epoch_rms(x, FS, BANDS["gross_movement"], epoch_n)
    assert np.isnan(rms[2]) and np.isfinite(rms).sum() == len(rms) - 1
    ok = np.isfinite(rms)
    assert rms[ok] == pytest.approx(clean[ok], rel=0.05)


def test_band_epoch_rms_is_unchanged_when_there_is_no_gap():
    x, _ = _tone(2.0, seconds=60, amp=0.5)
    epoch_n = int(30 * FS)
    assert np.array_equal(band_epoch_rms(x, FS, BANDS["gross_movement"], epoch_n),
                          band_epoch_rms(x.copy(), FS, BANDS["gross_movement"], epoch_n))


@pytest.mark.slow
def test_band_epoch_rms_on_the_gappy_fixture_costs_only_its_own_epochs(gappy_night):
    """Contract test (section 8 #7): the gappy fixture's NaN count in band_epoch_rms equals that in epoch_psd, not 1200."""
    from ax6apple.signals import build_grid, vm as vm_fn

    rec = gappy_night
    grid_hz = 30
    n_out = int(rec.t_s[-1] * grid_hz)
    n_out -= n_out % (30 * grid_hz)
    grid = build_grid(rec.t_s, rec.xyz, rec.fs, grid_hz, n_out, max_gap_s=1.0)
    epoch_n = 30 * grid_hz
    rms = band_epoch_rms(grid.vm, grid_hz, BANDS["gross_movement"], epoch_n)
    f, P = epoch_psd(grid.vm, grid_hz, epoch_n)
    psd_nan_epochs = np.isnan(P).all(axis=1)
    rms_nan_epochs = np.isnan(rms)
    assert rms_nan_epochs.sum() <= psd_nan_epochs.sum() + 1          # the filter's edge reach is at most one neighbouring epoch
    assert rms_nan_epochs.sum() < len(rms)                           # not every epoch, unlike the pre-F9 behaviour


# --- epoch_psd / spectral_scalars ---

def test_spectral_scalars_centroid_and_dominant_track_a_pure_tone():
    x, _ = _tone(5.0, seconds=30, amp=1.0)
    f, P = epoch_psd(x, FS, len(x))
    s = spectral_scalars(f, P)
    assert s["dominant_hz"][0] == pytest.approx(5.0, abs=f[1] - f[0])
    assert s["centroid_hz"][0] == pytest.approx(5.0, abs=0.5)
    assert s["sef95_hz"][0] >= s["dominant_hz"][0] - 0.1            # 95 % edge is at or above a single tone's own frequency


def test_spectral_entropy_is_higher_for_broadband_than_for_a_pure_tone():
    tone, _ = _tone(5.0, seconds=30, amp=1.0)
    noise = RNG.normal(size=len(tone))
    f, Pt = epoch_psd(tone, FS, len(tone))
    _, Pn = epoch_psd(noise, FS, len(noise))
    assert spectral_scalars(f, Pn)["spectral_entropy"][0] > spectral_scalars(f, Pt)["spectral_entropy"][0]


def test_spectral_scalars_are_nan_when_the_band_has_no_power():
    x = np.zeros(int(30 * FS))
    f, P = epoch_psd(x, FS, len(x))
    s = spectral_scalars(f, P)
    assert all(np.isnan(v[0]) for v in s.values())


# --- noise_psd / noise_correct ---

def test_noise_psd_is_flat_for_white_noise_and_noise_correct_removes_it():
    from ax6apple.calibration import MIN_MEAN_G
    win_s, n_windows = 10.0, 40
    n = int(win_s * FS)
    still = RNG.normal(scale=0.002, size=(n_windows, n, 3)) + [0, 0, 1.0]
    assert still.std(axis=1).max() < 0.013 and np.linalg.norm(still.mean(axis=1), axis=1).min() >= MIN_MEAN_G
    xyz = still.reshape(-1, 3)
    f_n, N = noise_psd(xyz, FS, win_s, std_g=0.013)
    assert N.std() / N.mean() < 0.5                                  # roughly flat (white): no strong frequency dependence

    tone, _ = _tone(5.0, seconds=win_s, amp=0.5)
    signal_plus_noise = tone + RNG.normal(scale=0.002, size=len(tone))
    f, P = epoch_psd(signal_plus_noise, FS, len(signal_plus_noise))
    corrected = noise_correct(f, P, f_n, N)
    assert (corrected >= 0).all()
    tone_bin = np.argmin(abs(f - 5.0))
    away = abs(f - 5.0) > 3
    assert corrected[0, tone_bin] > corrected[0, away].max()          # the tone survives; the noise-dominated region drops


def test_noise_psd_rejects_a_device_with_no_still_windows():
    moving = RNG.normal(scale=0.05, size=(300, 3)) + [0, 0, 1.0]     # SD well above std_g: never still
    with pytest.raises(ValueError, match="still"):
        noise_psd(moving, FS, window_s=10.0, std_g=0.013)


# --- coherence ---

def test_coherence_is_high_for_correlated_signals_and_near_its_bias_floor_for_independent_ones():
    tone, _ = _tone(2.0, seconds=120, amp=1.0)
    a = tone + RNG.normal(scale=0.01, size=len(tone))
    b = tone + RNG.normal(scale=0.01, size=len(tone))
    f, C, floor = coherence(a, b, FS, seg_s=10.0)
    i2 = np.argmin(abs(f - 2.0))
    assert C[i2] > 0.9

    ind_a = RNG.normal(size=len(tone))
    ind_b = RNG.normal(size=len(tone))
    _, C_ind, floor_ind = coherence(ind_a, ind_b, FS, seg_s=10.0)
    assert floor_ind == pytest.approx(floor, rel=1e-9)
    assert abs(np.median(C_ind) - floor_ind) < 0.15                   # independent series average near the bias floor, not 0


def test_coherence_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        coherence(np.zeros(100), np.zeros(90), FS, seg_s=1.0)


# --- spectrogram_db ---

def test_spectrogram_db_places_a_tone_at_the_right_time_and_frequency():
    quiet = np.zeros(int(10 * FS))
    tone, _ = _tone(10.0, seconds=10, amp=1.0)
    x = np.concatenate([quiet, tone, quiet])
    f, t, S = spectrogram_db(x, FS, seg_s=2.0, hop_s=1.0)
    fi = np.argmin(abs(f - 10.0))
    early = t < 9
    mid = (t > 11) & (t < 19)
    assert S[fi, mid].mean() > S[fi, early].mean() + 10               # >10 dB louder once the tone is present


# --- dyadic_bands / modwt_epoch_energy ---

def test_dyadic_bands_halve_from_nyquist_and_are_contiguous():
    bands = dyadic_bands(FS, level=4)                                 # fs=50 -> Nyquist 25
    assert bands[0] == pytest.approx((12.5, 25.0))                    # D1: nearest Nyquist
    assert bands[-1] == pytest.approx((25 / 16, 25 / 8))               # D4: coarsest
    for (lo, hi), (lo2, _) in zip(bands, bands[1:], strict=False):
        assert lo == pytest.approx(lo2 * 2)                            # each level is half the one before


def test_modwt_epoch_energy_places_a_tone_in_the_band_the_frequency_says():
    epoch_n = int(30 * FS)
    level = 4
    bands = dyadic_bands(FS, level)
    hi_tone, _ = _tone(20.0, seconds=60, amp=1.0)                     # in D1's band (12.5-25 Hz)
    lo_tone, _ = _tone(2.0, seconds=60, amp=1.0)                       # in D4's band (1.56-3.125 Hz) at level 4
    e_hi = modwt_epoch_energy(hi_tone, epoch_n, level=level)
    e_lo = modwt_epoch_energy(lo_tone, epoch_n, level=level)
    assert (bands[0][0] <= 20.0 <= bands[0][1])
    assert e_hi[:, 0].mean() > e_hi[:, 1:].mean(axis=1).mean() * 5     # the 20 Hz tone dominates D1
    lo_level = next(i for i, (lo, hi) in enumerate(bands) if lo <= 2.0 <= hi)
    others = [i for i in range(level) if i != lo_level]
    assert e_lo[:, lo_level].mean() > e_lo[:, others].mean(axis=1).mean() * 5


def test_modwt_is_shift_invariant_unlike_a_decimated_dwt():
    rng = np.random.default_rng(3)
    x = rng.normal(size=2048)
    shift = 37                                                        # not a multiple of any 2**k here
    xs = np.roll(x, shift)

    swt_x = pywt.swt(x, "sym8", level=4, trim_approx=True)
    swt_xs = pywt.swt(xs, "sym8", level=4, trim_approx=True)
    for cx, cxs in zip(swt_x, swt_xs, strict=True):
        assert np.allclose(np.roll(cx, shift), cxs, atol=1e-8)         # SWT: shifting the input just shifts the coefficients

    _, cD = pywt.dwt(x, "sym8")
    _, cD_s = pywt.dwt(xs, "sym8")
    predicted = np.roll(cD, shift // 2)                                 # the nearest a downsampled transform could match a shift
    assert np.linalg.norm(predicted - cD_s) / np.linalg.norm(predicted) > 0.5   # DWT: an odd shift scrambles the decimation phase


def test_modwt_epoch_energy_is_non_negative_and_drops_the_pad():
    x = np.random.default_rng(4).normal(size=int(45 * FS))             # not a multiple of 2**level
    e = modwt_epoch_energy(x, epoch_n=int(30 * FS), level=6)
    assert e.shape == (1, 6) and (e >= 0).all()                        # one whole 30 s epoch out of 45 s of data
