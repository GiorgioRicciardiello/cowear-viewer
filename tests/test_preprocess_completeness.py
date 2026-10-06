"""`ax6apple.preprocess.completeness`: gap counting that `signals.build_grid` already implies but never reports."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.preprocess.completeness import gap_runs, gap_stats, judge
from ax6apple.preprocess.limits import Limits
from ax6apple.signals import build_grid
from ax6apple.config import AX6APPLE

VIEWER = AX6APPLE["viewer"]
LIM = Limits.from_config()


def _gapped(gap_from, gap_to, seconds=90, fs=50):
    t = np.arange(-5 * fs, (seconds + 5) * fs) / fs
    keep = (t < gap_from) | (t >= gap_to)
    return t[keep], np.column_stack([np.sin(2 * np.pi * 0.1 * t[keep])] * 3) + np.array([0.0, 0.0, 1.0])


def test_gap_runs_finds_contiguous_true_stretches():
    mask = np.array([False, True, True, False, False, True, False])
    assert gap_runs(mask, 30) == ((1, 3), (5, 6))


def test_gap_runs_handles_no_runs_and_a_run_touching_the_edges():
    assert gap_runs(np.zeros(5, dtype=bool), 30) == ()
    assert gap_runs(np.array([True, True, False]), 30) == ((0, 2),)


def test_gap_stats_on_a_clean_signal_is_fully_complete():
    t = np.arange(-5 * 50, 95 * 50) / 50
    xyz = np.column_stack([np.sin(2 * np.pi * 0.1 * t)] * 3) + np.array([0.0, 0.0, 1.0])
    g = gap_stats(t, 30, 90 * 30, max_gap_s=1.0)
    assert g.completeness == 1.0 and g.n_gap_runs == 0 and g.gapped_samples == 0 and g.longest_gap_s == 0


def test_gap_stats_counts_match_build_grid_nans_exactly():
    """Contract test (section 8 #1): gap_stats' gapped-sample count equals the NaNs build_grid actually writes."""
    t, xyz = _gapped(30.0, 60.0)
    n_out = 90 * 30
    grid = build_grid(t, xyz, 50, 30, n_out, max_gap_s=1.0)
    g = gap_stats(t, 30, n_out, max_gap_s=1.0)
    assert g.gapped_samples == int(np.isnan(grid.vm).sum())
    assert g.n_gap_runs == 1
    assert g.longest_gap_s == pytest.approx(30.0, abs=1 / 30)
    assert g.completeness == pytest.approx(1 - 30.0 * 30 / n_out, abs=1e-3)


def test_gap_stats_bridges_a_short_gap_without_counting_it_as_gapped():
    t, xyz = _gapped(30.0, 30.5)
    n_out = 90 * 30
    g = gap_stats(t, 30, n_out, max_gap_s=1.0)
    assert g.gapped_samples == 0 and g.completeness == 1.0
    assert g.bridged_runs >= 1


def test_judge_fails_below_completeness_fail_and_warns_below_warn():
    full = gap_stats(np.arange(-5 * 50, 95 * 50) / 50, 30, 90 * 30, 1.0)
    assert judge(full, "ax6", LIM) == ()
    t, _ = _gapped(10.0, 70.0)
    mostly_gapped = gap_stats(t, 30, 90 * 30, 1.0)
    findings = judge(mostly_gapped, "apple", LIM)
    assert any(f.code == "completeness.low" and f.level == "fail" for f in findings)


def test_judge_warns_on_many_gap_runs():
    rng = np.random.default_rng(0)
    fs = 50
    t = np.arange(-5 * fs, 95 * fs) / fs
    for k in range(15):
        start = 1.0 + k * 5.0
        t = t[(t < start) | (t >= start + 1.2)]
    g = gap_stats(t, 30, 90 * 30, 1.0)
    findings = judge(g, "ax6", LIM)
    assert any(f.code == "completeness.gap_runs" for f in findings)


@pytest.mark.slow
def test_gap_stats_on_the_real_night_is_fully_complete(real_night):
    """Pilot-night real max gaps are 0.011 / 0.095 s: well under max_gap_s = 1.0 (phase 0 Done-when, section 0.5)."""
    n_out = int((real_night["ax6"].grid_cal.xyz.shape[0]))
    g = gap_stats(real_night["ax6"].rec.t_s, VIEWER["grid_hz"], n_out, VIEWER["max_gap_s"])
    assert g.completeness == 1.0 and g.n_gap_runs == 0


@pytest.mark.slow
def test_gap_stats_on_the_gappy_fixture_finds_the_excised_window(gappy_night):
    rec = gappy_night
    n_out = int(rec.t_s[-1] * VIEWER["grid_hz"])
    g = gap_stats(rec.t_s, VIEWER["grid_hz"], n_out, VIEWER["max_gap_s"])
    assert g.n_gap_runs == 1
    assert g.longest_gap_s == pytest.approx(1.1, abs=0.05)
