"""Clock lag and gravity-frame rotation: synthetic recovery of a known lag and rotation, then the real pilot night."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.agreement import epoch_pearson
from ax6apple.align import (_lagged_r, band_vm, epoch_lags, estimate_lag, fit_rotation, gravity_pairs, kabsch, movement_envelope,
                            peak_lag, rotate)
from ax6apple.config import AX6APPLE

VIEWER = AX6APPLE["viewer"]
FS = 30


def _movement(n=FS * 600, seed=0):
    return band_vm(np.random.default_rng(seed).normal(size=n), FS, (0.5, 10))


def _delay(x, seconds):
    """x delayed by a possibly fractional number of seconds (linear interpolation)."""
    t = np.arange(len(x))
    return np.interp(t - seconds * FS, t, x)


def _rotation(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


# --- band-passed VM ---

def test_band_vm_removes_gravity_and_slow_drift_and_keeps_movement():
    t = np.arange(FS * 120) / FS
    vm = 1.0 + 0.2 * np.sin(2 * np.pi * 0.02 * t) + 0.05 * np.sin(2 * np.pi * 3 * t)
    out = band_vm(vm, FS, (0.5, 10))
    assert abs(out[FS * 20:-FS * 20].mean()) < 1e-3
    assert np.abs(out[FS * 20:-FS * 20]).max() == pytest.approx(0.05, rel=0.1)


def test_band_vm_does_not_change_its_input():
    x = np.random.default_rng(0).normal(size=1000)
    before = x.copy()
    band_vm(x, FS, (0.5, 10))
    assert (x == before).all()


# --- clock lag ---

def test_peak_lag_is_positive_when_the_test_device_is_late():
    ref = _movement()
    got = peak_lag(ref, _delay(ref, 3.0), FS, 5)
    assert got.lag_s == pytest.approx(3.0, abs=0.02)
    assert got.r > 0.95 and not got.at_boundary


def test_peak_lag_is_negative_when_the_test_device_is_early():
    ref = _movement()
    assert peak_lag(ref, _delay(ref, -1.5), FS, 5).lag_s == pytest.approx(-1.5, abs=0.02)


def test_peak_lag_resolves_less_than_one_sample():
    ref = _movement()
    assert peak_lag(ref, _delay(ref, 0.2), FS, 5).lag_s == pytest.approx(0.2, abs=0.01)   # one sample is 0.033 s


def test_peak_lag_flags_a_peak_at_the_edge_of_the_search_range():
    ref = band_vm(np.random.default_rng(0).normal(size=FS * 3000), FS, (0.05, 0.3))   # slow: correlated over seconds
    got = peak_lag(ref, _delay(ref, 6.0), FS, 5)
    assert got.at_boundary and got.lag_s == pytest.approx(5.0)


def test_peak_lag_of_unrelated_signals_has_a_weak_peak():
    assert peak_lag(_movement(seed=1), _movement(seed=2), FS, 5).r < 0.1


def test_peak_lag_refuses_nan_instead_of_returning_a_boundary_peak():
    """A NaN passed the `std() == 0` guard (std is NaN), argmax returned 0 and the result was Lag(-1.0, nan, at_boundary=True)."""
    ref = _movement()
    test = _delay(ref, 1.0)
    test[100] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        peak_lag(ref, test, FS, 5)
    test[100] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        peak_lag(ref, test, FS, 5)


def test_peak_lag_nan_error_names_the_first_bad_index_and_its_time():
    """F8: the raise used to never mention where the NaN was, so an operator had no path from it to a gap."""
    ref = _movement()
    test = _delay(ref, 1.0)
    test[100] = np.nan
    with pytest.raises(ValueError, match=r"test\[100\], t=3\.333s"):
        peak_lag(ref, test, FS, 5)
    ref2 = _movement()
    ref2[50] = np.nan
    with pytest.raises(ValueError, match=r"ref\[50\]"):
        peak_lag(ref2, _delay(ref2, 1.0), FS, 5)


def test_peak_lag_validates_input():
    with pytest.raises(ValueError, match="same length"):
        peak_lag(np.zeros(100), np.zeros(90), FS, 1)
    with pytest.raises(ValueError, match="constant"):
        peak_lag(np.zeros(100), np.arange(100.0), FS, 1)
    with pytest.raises(ValueError, match="NaN"):
        peak_lag(np.arange(100.0), np.r_[np.arange(99.0), np.nan], FS, 1)
    with pytest.raises(ValueError, match="shorter"):
        peak_lag(np.arange(100.0), np.arange(100.0), FS, 10)


def _bursts(n=FS * 900, delay_s=0.0, seed=0):
    """A device-specific fast waveform (independent per device) under a shared slow movement envelope, delayed by delay_s."""
    rng = np.random.default_rng(99)
    slow = np.abs(np.convolve(rng.normal(size=n), np.ones(FS * 10) / (FS * 10), mode="same")) * 8 + 0.05   # shared envelope
    env = _delay(slow, delay_s)
    return 1.0 + env * np.random.default_rng(seed).normal(size=n) * 0.3


def test_movement_envelope_is_the_mean_absolute_band_signal_per_bin():
    vm = 1.0 + 0.1 * np.sin(2 * np.pi * 3 * np.arange(FS * 60) / FS)
    env = movement_envelope(vm, FS, (0.5, 10), 1.0)
    assert len(env) == 60
    assert env[10:-10].mean() == pytest.approx(0.1 * 2 / np.pi, rel=0.1)


def test_estimate_lag_uses_the_envelope_where_the_waveforms_do_not_correlate():
    a, b = _bursts(seed=1), _bursts(delay_s=3.0, seed=2)
    signed = peak_lag(band_vm(a, FS, (0.5, 10)), band_vm(b, FS, (0.5, 10)), FS, 5)
    got = estimate_lag(a, b, FS, (0.5, 10), 1.0, 5)
    assert signed.r < 0.1
    assert got.lag_s == pytest.approx(3.0, abs=0.3)
    assert got.r > 0.5


def test_epoch_lags_recovers_a_shift_epoch_by_epoch_and_nan_for_a_flat_epoch():
    ref = _movement(n=FS * 30 * 6)
    ref[FS * 30 * 2:FS * 30 * 3] = 0.0                 # one silent epoch
    test = np.roll(ref, 2 * FS // 3)                   # 0.667 s late (20 samples)
    got = epoch_lags(ref.reshape(6, -1), test.reshape(6, -1), FS, 5)
    good = [0, 1, 4, 5]
    assert got.lag_s[good] == pytest.approx(20 / FS, abs=1 / FS)
    assert got.r_peak[good].min() > 0.8
    assert all(np.isnan(x[2]) for x in got[:3]) and np.isnan(got.r_null).all()      # a flat epoch: nothing; no draws: no null


def _ar_epochs(n_epochs, n, phi, seed):
    x = np.random.default_rng(seed).normal(size=(n_epochs, n))
    for i in range(1, n):
        x[:, i] += phi * x[:, i - 1]
    return x


def test_epoch_lags_r_lag0_is_the_plain_epoch_correlation_and_the_peak_is_never_below_it():
    ref, test = _ar_epochs(60, 900, 0.9, 1), _ar_epochs(60, 900, 0.9, 2)
    got = epoch_lags(ref, test, FS, 5)
    assert got.r_lag0 == pytest.approx(epoch_pearson(ref, test))
    assert (got.r_peak >= got.r_lag0 - 1e-12).all()


def test_peak_r_over_the_lag_search_is_biased_upwards_on_independent_epochs():
    """The selection bias: independent AR(0.9) epochs (900 samples, +/-5 s) have mean peak r about 0.22 and mean |r| at lag 0
    about 0.08, and the peak sits at some arbitrary lag. Peak r alone is not evidence of agreement."""
    ref, test = _ar_epochs(300, 900, 0.9, 3), _ar_epochs(300, 900, 0.9, 4)
    got = epoch_lags(ref, test, FS, 5)
    assert np.mean(got.r_peak) > 0.15 and np.mean(got.r_peak) > 2 * np.mean(np.abs(got.r_lag0))
    assert np.abs(got.lag_s).max() > 3.0 and np.std(got.lag_s) > 2.0          # lags scattered over the search range


def test_epoch_null_is_what_independent_epochs_reach_and_a_shared_signal_clears_it():
    rng = np.random.default_rng(5)
    ref, test = _ar_epochs(300, 900, 0.9, 6), _ar_epochs(300, 900, 0.9, 7)
    noise = epoch_lags(ref, test, FS, 5, null_draws=40, rng=rng)
    assert np.mean(noise.r_peak > noise.r_null) < 0.15                        # about 5 % of null-only epochs exceed their null
    assert noise.r_null.min() > 0.05
    shared = epoch_lags(ref, 0.7 * ref + 0.7 * test, FS, 5, null_draws=40, rng=rng)
    assert np.mean(shared.r_peak > shared.r_null) > 0.95


def test_epoch_null_is_reproducible_needs_an_rng_and_is_nan_for_constant_epochs():
    ref, test = _ar_epochs(10, 900, 0.9, 8), _ar_epochs(10, 900, 0.9, 9)
    test[3] = 2.0
    a = epoch_lags(ref, test, FS, 5, null_draws=5, rng=np.random.default_rng(1))
    b = epoch_lags(ref, test, FS, 5, null_draws=5, rng=np.random.default_rng(1))
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b, strict=True))
    assert all(np.isnan(x[3]) for x in a) and np.isfinite(a.r_null[[0, 1, 2, 4]]).all()
    with pytest.raises(ValueError, match="rng"):
        epoch_lags(ref, test, FS, 5, null_draws=5)


def test_lagged_r_equals_the_pearson_r_of_each_overlap():
    """The FFT + cumulative-sum implementation against a direct Pearson r on the shifted slices (the definition)."""
    ref, test = _ar_epochs(8, 200, 0.8, 10), _ar_epochs(8, 200, 0.8, 11)
    ks = np.arange(-40, 41)
    got = _lagged_r(ref, test, ks)
    for i, k in enumerate(ks):
        n = ref.shape[1]
        direct = epoch_pearson(ref[:, max(-k, 0):n - max(k, 0)], test[:, max(k, 0):n - max(-k, 0)])
        assert got[i] == pytest.approx(direct, abs=1e-9)


# --- Kabsch ---

def _cloud(n=200, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_kabsch_recovers_a_known_rotation_exactly_without_noise():
    R = _rotation([1, 2, 3], 40)
    test = _cloud()
    got = kabsch(test @ R.T, test)             # ref = R test
    assert got == pytest.approx(R, abs=1e-10)
    assert np.linalg.det(got) == pytest.approx(1.0)


def test_kabsch_with_noise_is_close_and_proper():
    R = _rotation([0, 1, 1], -25)
    test = _cloud(seed=1)
    ref = test @ R.T + np.random.default_rng(2).normal(scale=0.01, size=test.shape)
    got = kabsch(ref, test)
    assert got == pytest.approx(R, abs=0.02)
    assert got @ got.T == pytest.approx(np.eye(3), abs=1e-10)


def test_kabsch_returns_a_rotation_not_a_reflection_for_mirrored_data():
    test = _cloud(seed=3)
    got = kabsch(test * np.array([1, 1, -1]), test)
    assert np.linalg.det(got) == pytest.approx(1.0)


def test_kabsch_needs_at_least_two_directions():
    same = np.tile([0.0, 0.0, 1.0], (50, 1))
    with pytest.raises(ValueError, match="orientation"):
        kabsch(same, same)
    with pytest.raises(ValueError, match="same shape"):
        kabsch(np.zeros((5, 3)), np.zeros((4, 3)))


def test_rotate_applies_the_matrix_to_row_vectors_and_keeps_the_input():
    R = _rotation([0, 0, 1], 90)
    xyz = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    before = xyz.copy()
    assert rotate(xyz, R) == pytest.approx(np.array([[0.0, 1, 0], [-1.0, 0, 0]]), abs=1e-12)
    assert (xyz == before).all()


# --- still windows and the fitted rotation ---

def _still_recording(vectors, window_n, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    return np.repeat(vectors, window_n, axis=0) + rng.normal(scale=noise, size=(len(vectors) * window_n, 3))


WINDOW_N = 10 * FS


def test_gravity_pairs_keeps_only_windows_that_are_still_on_both_devices():
    v = _cloud(40, seed=4)
    a, b = _still_recording(v, WINDOW_N), _still_recording(v, WINDOW_N, seed=1)
    a[5 * WINDOW_N:6 * WINDOW_N] += np.random.default_rng(0).normal(scale=0.2, size=(WINDOW_N, 3))    # a moves in window 5
    b[9 * WINDOW_N:10 * WINDOW_N] += np.random.default_rng(1).normal(scale=0.2, size=(WINDOW_N, 3))   # b moves in window 9
    ref, test, n_examined = gravity_pairs(a, b, FS, 10, 0.013)
    assert ref.shape == test.shape == (38, 3)
    assert n_examined == 40            # F11: windows examined, before the stillness filter keeps 38 of them
    assert np.linalg.norm(ref, axis=1) == pytest.approx(np.ones(38))


def test_fit_rotation_recovers_rotation_with_small_residual_and_uncertainty_on_full_coverage():
    R = _rotation([1, 1, 0.5], 35)
    v = _cloud(300, seed=5)
    got = fit_rotation(_still_recording(v, WINDOW_N), _still_recording(v @ R.T, WINDOW_N, seed=1), FS, 10, 0.013, 30, 300,
                       np.random.default_rng(0))
    # ref = matrix @ test, and test = R^-1 ref here
    assert got.matrix == pytest.approx(R.T, abs=0.02)
    assert got.residual_deg < 0.5
    assert got.uncertainty_deg < 1.0
    assert got.n_windows == 300


def test_fit_rotation_uncertainty_is_larger_when_orientations_barely_vary():
    """The roll about gravity is weakly identified when the wrist only ever tilts a little."""
    rng = np.random.default_rng(6)
    R = _rotation([1, 0, 0], 20)
    full = _cloud(200, seed=7)
    narrow = np.array([0.0, 0.0, 1.0]) + rng.normal(scale=0.15, size=(200, 3))
    narrow /= np.linalg.norm(narrow, axis=1, keepdims=True)
    def unc(v):
        return fit_rotation(_still_recording(v, WINDOW_N), _still_recording(v @ R.T, WINDOW_N, noise=0.02, seed=1), FS, 10,
                            0.05, 20, 200, np.random.default_rng(1)).uncertainty_deg
    assert unc(narrow) > 3 * unc(full)


def test_fit_rotation_with_no_jointly_still_window_raises():
    moving = np.random.default_rng(0).normal(scale=0.3, size=(WINDOW_N * 5, 3))
    with pytest.raises(ValueError, match="still"):
        fit_rotation(moving, moving, FS, 10, 0.013, 5, 100, np.random.default_rng(0))


# --- real night ---

@pytest.mark.slow
def test_real_night_clock_lag_is_about_three_seconds_with_a_weak_peak(real_night):
    lag = estimate_lag(real_night["ax6"].grid_cal.vm, real_night["apple"].grid_cal.vm, FS, tuple(VIEWER["lag_band_hz"]),
                       VIEWER["lag_bin_s"], VIEWER["lag_search_s"])
    assert lag.lag_s == pytest.approx(3.0, abs=0.5)    # published: about 3 s
    assert lag.r == pytest.approx(0.45, abs=0.03)      # published: r = 0.45
    assert not lag.at_boundary


@pytest.mark.slow
def test_real_night_rotation_is_proper_and_reports_that_no_fixed_rotation_fits(real_night):
    """The RMS gravity residual is 57 degrees: the two frames are not related by one rotation (different wrists, or the
    watch moving on the arm). The viewer must say so next to the aligned view; a small residual here would be a bug."""
    rot = real_night["rotation"]
    assert np.linalg.det(rot.matrix) == pytest.approx(1.0)
    assert rot.n_windows > 1000
    assert rot.residual_deg > 30
    assert rot.uncertainty_deg > 0


def test_epoch_lags_rejects_an_epoch_shorter_than_the_search_range():
    with pytest.raises(ValueError, match="shorter"):
        epoch_lags(np.zeros((2, 100)), np.zeros((2, 100)), FS, 10)


@pytest.mark.slow
def test_real_night_signed_vm_waveforms_barely_correlate_and_their_peak_is_noise(real_night):
    """estimate_lag's docstring: the signed 30 Hz waveforms give r 0.03 at their peak and -0.003 at zero lag; only the
    movement envelope carries the clock lag (r 0.45)."""
    ax6, apple = (band_vm(real_night[d].grid_cal.vm, FS, tuple(VIEWER["lag_band_hz"])) for d in ("ax6", "apple"))
    assert peak_lag(ax6, apple, FS, VIEWER["lag_search_s"]).r == pytest.approx(0.03, abs=0.01)
    assert np.corrcoef(ax6, apple)[0, 1] == pytest.approx(-0.003, abs=0.005)
