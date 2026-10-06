"""`ax6apple.preprocess.verdict`: device and alignment quality assembly, and capability gating."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.align import Lag, Rotation
from ax6apple.calibration import Calibration
from ax6apple.preprocess.limits import Limits
from ax6apple.preprocess.verdict import alignment_quality, capabilities, device_quality
from ax6apple.signals import Recording, Timing

LIM = Limits.from_config()


def _good_cal(coverage=1.8, rms_mg=2.0, converged=True, low_cov=False):
    return Calibration(gain=np.ones(3), offset=np.zeros(3), n_windows=100, n_retained=95, coverage=coverage,
                       residual=1e-9, iterations=10, rms_g=rms_mg / 1000, converged=converged,
                       low_orientation_coverage=low_cov)


def _rec(n=36000, fs=50):
    t = np.arange(n) / fs
    xyz = np.random.default_rng(0).normal(0, 0.01, (n, 3)) + [0, 0, 1.0]
    return Recording(t, xyz, fs, Timing(n, 0, 0, 1 / fs)), xyz


def test_device_quality_is_clean_for_a_good_recording():
    rec, xyz = _rec()
    dq = device_quality("ax6", rec, xyz, _good_cal(), {"value_mg": 1.0, "n_windows": 50}, 30, len(rec.t_s) * 30 // 50,
                        1.0, LIM)
    assert dq.dev == "ax6" and dq.fs_median == pytest.approx(50.0, abs=0.1)
    assert dq.gaps.completeness == 1.0
    assert all(f.level != "fail" for f in dq.findings)


def test_device_quality_flags_a_non_converged_or_low_coverage_calibration():
    rec, xyz = _rec()
    dq = device_quality("apple", rec, xyz, _good_cal(converged=False, low_cov=True), {"value_mg": 1.0, "n_windows": 50},
                        30, len(rec.t_s) * 30 // 50, 1.0, LIM)
    codes = {f.code for f in dq.findings}
    assert "calibration.not_converged" in codes and "calibration.low_coverage" in codes


def test_device_quality_warns_on_low_coverage_number_and_high_rms():
    rec, xyz = _rec()
    dq = device_quality("ax6", rec, xyz, _good_cal(coverage=0.9, rms_mg=8.0), {"value_mg": 1.0, "n_windows": 50},
                        30, len(rec.t_s) * 30 // 50, 1.0, LIM)
    codes = {f.code for f in dq.findings}
    assert "calibration.coverage" in codes and "calibration.rms" in codes


# --- alignment_quality ---

def test_alignment_quality_is_clean_for_a_good_lag_and_rotation():
    lag, rot = Lag(3.0, 0.5, False), Rotation(np.eye(3), 500, 5.0, 2.0)
    al = alignment_quality(lag, rot, n_examined=520, lim=LIM)
    assert al.findings == () and al.rotation_n_examined == 520


def test_alignment_quality_warns_on_weak_lag_and_boundary():
    lag = Lag(5.0, 0.1, True)
    al = alignment_quality(lag, Rotation(np.eye(3), 500, 5.0, 2.0), 520, LIM)
    codes = {f.code for f in al.findings}
    assert {"lag.weak_r", "lag.at_boundary"} <= codes


def test_alignment_quality_warns_on_both_rotation_limits_like_the_pilot_night():
    """Section 3: the measured 56.8 / 14.9 both fail their limits (20 / 10 deg)."""
    al = alignment_quality(Lag(3.18, 0.45, False), Rotation(np.eye(3), 3279, 56.8, 14.9), 3300, LIM)
    codes = {f.code for f in al.findings}
    assert {"rotation.residual", "rotation.uncertainty"} <= codes
    assert "lag.weak_r" not in codes and "lag.at_boundary" not in codes


# --- capabilities ---

def test_capabilities_on_the_pilot_night_gives_axis_comparison_false_others_true():
    """Contract test (section 8 #8)."""
    rec, xyz = _rec()
    good = device_quality("ax6", rec, xyz, _good_cal(), {"value_mg": 1.0, "n_windows": 50}, 30,
                          len(rec.t_s) * 30 // 50, 1.0, LIM)
    devices = {"ax6": good, "apple": good}
    al = alignment_quality(Lag(3.18, 0.45, False), Rotation(np.eye(3), 3279, 56.8, 14.9), 3300, LIM)
    caps = capabilities(devices, al, LIM)
    assert caps.axis_comparison is False
    assert caps.enmo_agreement is True and caps.spectral_comparison is True and caps.per_epoch_lag is True


def test_capabilities_are_false_when_a_device_has_a_fail_finding():
    rec, xyz = _rec()
    bad_xyz = np.zeros_like(xyz)                      # no gravity: a fail finding
    bad = device_quality("apple", rec, bad_xyz, _good_cal(), {"value_mg": 1.0, "n_windows": 50}, 30,
                         len(rec.t_s) * 30 // 50, 1.0, LIM)
    good = device_quality("ax6", rec, xyz, _good_cal(), {"value_mg": 1.0, "n_windows": 50}, 30,
                          len(rec.t_s) * 30 // 50, 1.0, LIM)
    al = alignment_quality(Lag(3.0, 0.5, False), Rotation(np.eye(3), 500, 5.0, 2.0), 520, LIM)
    caps = capabilities({"ax6": good, "apple": bad}, al, LIM)
    assert not caps.enmo_agreement and not caps.axis_comparison and not caps.spectral_comparison
