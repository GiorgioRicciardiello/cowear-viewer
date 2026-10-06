"""calibrate_acc.m port: synthetic recovery of a known gain and offset, window screening, and the published pilot-night fits."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.calibration import MATLAB_TOL, MAX_ITER, Calibration, _coverage, _still_window_means, apply, calibrate
from ax6apple.config import AX6APPLE

FS = 10
WINDOW = 10 * FS   # samples per 10 s window


def _still_night(gain, offset, n_windows=120, seed=0, noise=0.002):
    """Still 10 s windows at random orientations, as a device with the given gain and offset would record them."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n_windows, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    raw = (g - offset) / gain
    return (np.repeat(raw, WINDOW, axis=0) + rng.normal(scale=noise, size=(n_windows * WINDOW, 3)))


def test_recovers_known_gain_and_offset():
    gain, offset = np.array([1.02, 0.97, 1.01]), np.array([0.04, -0.05, 0.01])
    cal = calibrate(_still_night(gain, offset), FS)
    assert cal.gain == pytest.approx(gain, abs=0.01)
    assert cal.offset == pytest.approx(offset, abs=0.01)


def test_calibrated_still_windows_land_on_the_unit_sphere():
    raw = _still_night(np.array([1.03, 0.96, 1.0]), np.array([0.05, 0.03, -0.04]))
    cal = calibrate(raw, FS)
    vm = np.linalg.norm(apply(raw, cal), axis=1)
    assert abs(vm.mean() - 1) < 0.003
    assert cal.rms_g < 0.005        # a scale-free bound: the old `residual < 1e-4` allowed |e| up to 0.06 g at N = 120


def test_rms_g_is_the_true_rms_distance_and_the_matlab_residual_is_not_a_mean_squared_distance():
    """MATLAB's `3*mean(w e^2)/sum(w)` divides by sum(w) a second time: at N = 120 it is about 3/120 of the real MSD."""
    raw = _still_night(np.array([1.03, 0.96, 1.0]), np.array([0.05, 0.03, -0.04]))
    raw = raw + np.repeat(np.random.default_rng(5).normal(scale=0.02, size=(120, 3)), WINDOW, axis=0)   # windows off the sphere
    cal = calibrate(raw, FS)
    means, _ = _still_window_means(raw, FS, 10, 0.013)
    en = np.linalg.norm(cal.gain * means + cal.offset, axis=1)
    assert cal.rms_g == pytest.approx(np.sqrt(np.mean((en - 1) ** 2)))
    assert cal.rms_g > 0.005 and cal.residual < 0.1 * cal.rms_g ** 2


def test_a_poor_fit_fails_the_rms_bound_that_the_old_residual_bound_let_through():
    """120 windows, w = 100: residual < 1e-4 only needed |e| < 0.06 g. rms_g sees a 20 mg misfit as 20 mg."""
    raw = _still_night(np.ones(3), np.zeros(3))
    poor = Calibration(np.ones(3), np.array([0.0, 0.0, 0.02]), 0, 0, 0.0, 0.0, 0, 0.0, True, False)
    assert np.sqrt(np.mean((np.linalg.norm(apply(raw, poor), axis=1) - 1) ** 2)) > 0.005
    assert calibrate(raw, FS).rms_g < 0.005


def test_moving_windows_are_not_retained():
    still = _still_night(np.ones(3), np.zeros(3), n_windows=60)
    moving = np.random.default_rng(1).normal(scale=0.3, size=(20 * WINDOW, 3))
    cal = calibrate(np.vstack([still, moving]), FS)
    assert (cal.n_windows, cal.n_retained) == (80, 60)


def test_incomplete_trailing_window_is_ignored():
    raw = _still_night(np.ones(3), np.zeros(3), n_windows=50)
    cal = calibrate(np.vstack([raw, raw[:WINDOW - 1]]), FS)
    assert cal.n_windows == 50


def test_std_threshold_uses_sample_std_ddof1_on_every_axis():
    still = _still_night(np.ones(3), np.zeros(3), n_windows=50)
    noisy = still[:WINDOW].copy()
    noisy[:, 2] += np.random.default_rng(2).normal(scale=0.02, size=WINDOW)   # one axis over 0.013 rejects the window
    cal = calibrate(np.vstack([still, noisy]), FS)
    assert cal.n_retained == 50


def test_apply_returns_new_array_and_leaves_input_alone():
    raw = _still_night(np.ones(3), np.zeros(3), n_windows=30)
    before = raw.copy()
    out = apply(raw, Calibration(gain=np.array([2.0, 2, 2]), offset=np.array([1.0, 1, 1]), n_windows=0, n_retained=0,
                                 coverage=0.0, residual=0.0, iterations=0, rms_g=0.0, converged=True,
                                 low_orientation_coverage=False))
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(out, 2 * raw + 1)


def test_coverage_is_mean_axis_span_of_still_unit_vectors():
    cal = calibrate(_still_night(np.ones(3), np.zeros(3)), FS)
    assert 1.5 < cal.coverage <= 2.0            # random orientations: nearly the whole sphere


def test_converges_before_the_iteration_cap():
    cal = calibrate(_still_night(np.array([1.02, 0.98, 1.0]), np.array([0.03, 0.02, -0.01])), FS)
    assert 1 <= cal.iterations < 1000


def test_no_still_window_raises():
    moving = np.random.default_rng(3).normal(scale=0.3, size=(20 * WINDOW, 3))
    with pytest.raises(ValueError, match="no still window"):
        calibrate(moving, FS)


def test_single_orientation_raises_because_the_fit_is_undetermined():
    one = np.tile([0.0, 0.0, 1.0], (30 * WINDOW, 1))
    with pytest.raises(ValueError, match="orientation"):
        calibrate(one, FS)


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="n x 3"):
        calibrate(np.zeros((100, 2)), FS)


# --- validity guards: dropouts, divergence, convergence, coverage ---

def test_a_zero_vector_still_window_is_not_used_and_does_not_nan_the_night():
    """A dropout filled with zeros has SD 0, so it is 'still', with mean (0, 0, 0): EN = 0 and 0/0 = NaN for the whole fit."""
    good = _still_night(np.array([1.02, 0.97, 1.01]), np.array([0.04, -0.05, 0.01]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # the old failure was a bare RuntimeWarning
        cal = calibrate(np.vstack([good, np.zeros((WINDOW, 3))]), FS)
    assert (cal.n_windows, cal.n_retained) == (121, 120)
    assert np.isfinite(cal.gain).all() and np.isfinite(cal.offset).all() and cal.converged
    assert cal.gain == pytest.approx(calibrate(good, FS).gain)
    assert np.isfinite(apply(good, cal)).all()


def test_a_window_shorter_than_gravity_is_a_dropout_even_if_it_is_not_zero():
    good = _still_night(np.ones(3), np.zeros(3))
    dropout = np.tile([0.1, 0.1, 0.1], (WINDOW, 1))          # constant, |mean| = 0.17 g
    assert calibrate(np.vstack([good, dropout]), FS).n_retained == 120


def _cone_night(spread, n=120, seed=0):
    """Still windows whose gravity direction stays within a cone about z: little orientation coverage."""
    rng = np.random.default_rng(seed)
    g = np.array([0, 0, 1.0]) + rng.normal(scale=spread, size=(n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    raw = (g - np.array([0.04, -0.05, 0.01])) / np.array([1.02, 0.97, 1.01])
    return np.repeat(raw, WINDOW, axis=0) + rng.normal(scale=0.002, size=(n * WINDOW, 3))


def test_a_fit_that_diverges_raises_an_explicit_error_instead_of_nans_or_weights_sum_to_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="not finite"):
            calibrate(_cone_night(0.05), FS)


def test_min_coverage_refuses_a_fit_the_orientations_cannot_determine():
    raw = _cone_night(0.05)
    assert _coverage(_still_window_means(raw, FS, 10, 0.013)[0]) < 0.5
    with pytest.raises(ValueError, match="below the minimum"):
        calibrate(raw, FS, min_coverage=0.5)


def test_low_orientation_coverage_is_flagged_but_a_wide_fit_is_not():
    narrow = calibrate(_cone_night(0.4), FS, min_coverage=0.5, low_coverage=1.9)
    wide = calibrate(_still_night(np.ones(3), np.zeros(3)), FS, min_coverage=0.5, low_coverage=1.0)
    assert narrow.low_orientation_coverage and not wide.low_orientation_coverage


def test_converged_flag_is_true_when_the_stop_rule_is_met_and_false_when_the_iteration_cap_ran_out():
    raw = _still_night(np.array([1.02, 0.98, 1.0]), np.array([0.03, 0.02, -0.01]))
    assert calibrate(raw, FS).converged
    assert calibrate(raw, FS, tol=MATLAB_TOL).converged
    never = calibrate(raw, FS, tol=0.0)                      # a residual change is never < 0: runs to the cap
    assert never.iterations == MAX_ITER and not never.converged


def test_the_default_stop_lands_on_the_least_squares_fit_where_the_weights_saturate():
    raw = _still_night(np.array([1.02, 0.97, 1.01]), np.array([0.04, -0.05, 0.01]))
    means, _ = _still_window_means(raw, FS, 10, 0.013)
    ls = least_squares(lambda p: np.linalg.norm(p[:3] * means + p[3:], axis=1) - 1, np.r_[np.ones(3), np.zeros(3)],
                       xtol=1e-15, ftol=1e-15, gtol=1e-15)
    cal = calibrate(raw, FS)
    assert cal.gain == pytest.approx(ls.x[:3], abs=1e-6) and cal.offset == pytest.approx(ls.x[3:], abs=1e-6)


def _literal_matlab_loop(means, tol):
    """calibrate_acc.m written out with polyfit: the reference the port must reproduce on the weighted path."""
    c1, d, g, w, res = means.copy(), np.zeros(3), np.ones(3), np.ones(len(means)), np.inf
    for it in range(1000):
        cp = means / np.sqrt((c1 ** 2).sum(1))[:, None]
        d_inc, g_inc = np.zeros(3), np.ones(3)
        for k in range(3):
            g_inc[k], d_inc[k] = np.polyfit(means[:, k], cp[:, k], 1, w=np.sqrt(w))
        d, g = d + d_inc / (g * g_inc), g * g_inc
        en = np.sqrt(((g * means + d) ** 2).sum(1))
        c1, res_old, res = g * means + d, res, 3 * np.mean(w * (1 - en) ** 2) / w.sum()
        if abs(res - res_old) < tol:
            break
        w = np.minimum(1 / np.abs(1 - en), 100)
    return g, d, it + 1


@pytest.mark.parametrize("tol", [MATLAB_TOL, 1e-14])
def test_weighted_path_reproduces_the_literal_matlab_loop(tol):
    """Misfit large enough that the weights are not all 100, so the weighting itself is tested."""
    rng = np.random.default_rng(3)
    u = rng.normal(size=(150, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    means = (u - np.array([0.12, -0.1, 0.05])) / np.array([1.08, 0.9, 1.05]) + rng.normal(scale=0.03, size=u.shape)
    en = np.linalg.norm(np.array([1.08, 0.9, 1.05]) * means + np.array([0.12, -0.1, 0.05]), axis=1)
    assert 0.5 < np.mean(np.abs(1 - en) > 0.01)              # most windows are off the sphere: weights are not saturated
    g, d, iterations = _literal_matlab_loop(means, tol)
    cal = calibrate(np.repeat(means, WINDOW, axis=0), FS, std_g=1.0, tol=tol)
    assert cal.gain == pytest.approx(g, abs=1e-9) and cal.offset == pytest.approx(d, abs=1e-9)
    assert cal.iterations == iterations


# --- regression anchors: the published numbers of the real pilot night ---

@pytest.mark.slow
def test_ax6_fit_reproduces_published_gain_and_offset(real_night):
    cal = real_night["ax6"].calibration
    assert cal.gain == pytest.approx([1.001, 0.989, 0.999], abs=0.002)
    assert cal.offset * 1000 == pytest.approx([-37.5, 52.4, -5.1], abs=2.0)
    # The plan's 1.74 comes from 30 s uncalibrated still windows (see the test at the end of this file); this 10 s rule gives
    # 1.786 and 1.599 vs 1.60 for the Apple. The span of the extreme still windows is sensitive to a handful of them, so this
    # is pinned to notice a change of the window rule.
    assert cal.coverage == pytest.approx(1.786, abs=0.005)
    assert cal.converged and not cal.low_orientation_coverage
    assert cal.rms_g * 1000 == pytest.approx(2.90, abs=0.02)             # the true RMS distance ...
    assert cal.residual == pytest.approx(7.5e-9, rel=0.02)               # ... against MATLAB's, 1e-3 of its square


@pytest.mark.slow
def test_apple_fit_reproduces_published_gain_and_offset(real_night):
    cal = real_night["apple"].calibration
    assert cal.gain[0] == pytest.approx(1.001, abs=0.002)
    assert cal.offset * 1000 == pytest.approx([-0.5, 2.5, -4.0], abs=2.0)
    assert cal.coverage == pytest.approx(1.60, abs=0.02)
    assert cal.rms_g * 1000 == pytest.approx(0.19, abs=0.01)
    assert cal.offset[2] * 1000 == pytest.approx(-3.96, abs=0.02)         # the 1e-14 residual stop gave -3.82


@pytest.mark.slow
def test_matlab_tolerance_stops_early_and_lands_on_the_wrong_offsets(real_night):
    """Characterises why `tol` defaults to CONVERGED_TOL: MATLAB's 1e-10 stop is 3 to 5 mg off the published fit."""
    ax6 = real_night["ax6"]
    early = calibrate(ax6.xyz, 100, tol=MATLAB_TOL)
    assert early.iterations < ax6.calibration.iterations
    assert np.abs(early.offset - ax6.calibration.offset).max() * 1000 > 2.5


def test_matlab_tolerance_stops_no_later_than_converged_tolerance():
    raw = _still_night(np.array([1.02, 0.98, 1.0]), np.array([0.03, 0.02, -0.01]))
    assert calibrate(raw, FS, tol=MATLAB_TOL).iterations <= calibrate(raw, FS).iterations


@pytest.mark.slow
def test_published_orientation_coverage_comes_from_30_s_uncalibrated_still_windows(real_night):
    """The plan's 1.74 / 1.60 is reproduced by 30 s uncalibrated windows (1.736 / 1.594); the 10 s calibrated rule used
    here gives 1.786 / 1.599. The current definition is kept and both are pinned so that neither drifts unnoticed."""
    v = AX6APPLE["viewer"]
    for device, published, ten_s in (("ax6", 1.736, 1.786), ("apple", 1.594, 1.599)):
        night = real_night[device]
        means30, _ = _still_window_means(night.xyz, night.rec.fs, 30, v["calibration_std_g"])
        assert _coverage(means30) == pytest.approx(published, abs=0.003)
        assert night.calibration.coverage == pytest.approx(ten_s, abs=0.003)
