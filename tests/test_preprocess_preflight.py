"""`ax6apple.preprocess.preflight`: cheap sanity checks on a Recording, before any expensive fit."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.preprocess.limits import Limits
from ax6apple.preprocess.preflight import check_amplitude, check_rate, check_timing, preflight
from ax6apple.signals import Recording, Timing

LIM = Limits.from_config()


def _rec(t_s, fs, timing=None):
    xyz = np.column_stack([np.zeros(len(t_s)), np.zeros(len(t_s)), np.ones(len(t_s))])
    return Recording(t_s, xyz, fs, timing or Timing(len(t_s), 0, 0, float(np.diff(t_s).max()) if len(t_s) > 1 else 0.0))


# --- check_rate ---

def test_check_rate_is_info_when_the_rate_matches_nominal():
    t = np.arange(1000) / 50.0
    out = check_rate(_rec(t, 50), LIM)
    assert any(f.code == "rate.rounded" and f.level == "info" for f in out)


def test_check_rate_warns_when_the_true_rate_diverges_from_nominal():
    t = np.arange(1000) / 44.0        # nominal 50 vs true 44: 12 % error
    out = check_rate(_rec(t, 50), LIM)
    assert any(f.code == "rate.rounded" and f.level == "warn" for f in out)


def test_check_rate_warns_on_high_jitter():
    rng = np.random.default_rng(0)
    t = np.sort(np.cumsum(np.r_[0, rng.uniform(0.005, 0.035, 999)]))
    out = check_rate(_rec(t, 50), LIM)
    assert any(f.code == "rate.jitter" and f.level == "warn" for f in out)


# --- check_timing ---

def test_check_timing_is_info_for_a_tiny_fraction_and_warn_for_a_large_one():
    t = np.arange(1_000_000) / 50.0
    small = check_timing(_rec(t, 50, Timing(len(t), 50, 3, 0.02)), LIM)
    assert all(f.level == "info" for f in small)
    big = check_timing(_rec(t, 50, Timing(len(t), 5000, 3000, 0.02)), LIM)
    assert any(f.level == "warn" for f in big)


# --- check_amplitude ---

def test_check_amplitude_fails_when_gravity_is_absent():
    xyz = np.random.default_rng(0).normal(scale=0.01, size=(1000, 3))     # no gravity: mean near 0
    out = check_amplitude(xyz, LIM)
    assert any(f.code == "amplitude.no_gravity" and f.level == "fail" for f in out)


def test_check_amplitude_fails_on_a_constant_axis():
    rng = np.random.default_rng(1)
    xyz = np.column_stack([np.full(1000, 0.0), rng.normal(0, 0.02, 1000), np.full(1000, 1.0) + rng.normal(0, 0.02, 1000)])
    out = check_amplitude(xyz, LIM)
    assert any(f.code == "amplitude.constant_axis" for f in out)


def test_check_amplitude_warns_on_a_long_zero_run():
    rng = np.random.default_rng(2)
    xyz = rng.normal(0, 0.01, (1000, 3)) + [0, 0, 1.0]
    xyz[:300] = 0.0
    out = check_amplitude(xyz, LIM)
    assert any(f.code == "amplitude.zero_run" and f.level == "warn" for f in out)


def test_check_amplitude_is_clean_for_an_ordinary_still_device():
    rng = np.random.default_rng(3)
    xyz = rng.normal(0, 0.01, (1000, 3)) + [0, 0, 1.0]
    assert check_amplitude(xyz, LIM) == ()


# --- preflight() assembly ---

def test_preflight_tags_every_finding_with_the_subject():
    t = np.arange(1000) / 50.0
    xyz = np.zeros((1000, 3))                 # no gravity: at least one fail finding
    out = preflight(_rec(t, 50), xyz, "apple", LIM)
    assert len(out) > 0 and all(f.subject == "apple" for f in out)


@pytest.mark.parametrize("case,expected_code", [
    ("constant", "amplitude.constant_axis"),
    ("clipped", "amplitude.clipped_axis"),
    ("zero_run", "amplitude.zero_run"),
    ("no_gravity", "amplitude.no_gravity"),
])
def test_bad_recordings_each_produce_their_own_finding_code(case, expected_code):
    """section 8's `bad_recordings` fixture, parametrised here directly."""
    rng = np.random.default_rng(hash(case) % 2**32)
    n = 1000
    if case == "constant":
        xyz = np.column_stack([np.zeros(n), rng.normal(0, 0.02, n), np.ones(n) + rng.normal(0, 0.02, n)])
    elif case == "clipped":
        xyz = np.column_stack([rng.normal(0, 15.0, n), rng.normal(0, 0.02, n), np.ones(n)])
    elif case == "zero_run":
        xyz = rng.normal(0, 0.01, (n, 3)) + [0, 0, 1.0]
        xyz[: n // 2] = 0.0
    else:
        xyz = rng.normal(scale=0.01, size=(n, 3))
    assert any(f.code == expected_code for f in check_amplitude(xyz, LIM))
