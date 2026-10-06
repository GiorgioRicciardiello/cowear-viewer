"""Signal chain: loading, VM/ENMO, the common 30 Hz grid, epoching and the moving-average sweep.

The fast tests use synthetic signals with known answers. The slow ones run the real pilot night and check its
published numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.agreement import effective_n_from_ci
from ax6apple.calibration import apply
from ax6apple.rbd_sleep import EPOCH, epoch_vector
from ax6apple.signals import (build_grid, enmo, epoch_matrix, epoch_starts, gap_mask, load_recording, ma_sweep, moving_average,
                              nominal_rate_hz, recording_window, sensor_noise_floor, smooth_then_epoch, vm)
from ax6apple.config import AX6APPLE

VIEWER = AX6APPLE["viewer"]
ANCHOR = pd.Timestamp("2026-07-13 21:53:37.540")


# --- VM and ENMO ---

def test_vm_is_the_euclidean_norm():
    assert vm(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])) == pytest.approx([5.0, 1.0])


def test_enmo_is_vm_minus_one_floored_at_zero():
    got = enmo(np.array([[0.0, 0.0, 1.05], [0.0, 0.0, 0.97], [0.0, 0.0, 1.0]]))
    assert got == pytest.approx([0.05, 0.0, 0.0])


# --- the epoch anchor ---

def test_anchor_comes_from_the_window_variant_and_is_not_a_round_time():
    start, end = recording_window()
    variant = next(v for v in AX6APPLE["variants"] if v["name"] == VIEWER["window_variant"])
    assert (start, end) == (pd.Timestamp(variant["start"]), pd.Timestamp(variant["end"]))
    assert start.second % 30 or start.microsecond              # a round-clock anchor would have second % 30 == 0


def test_epoch_starts_are_30_s_steps_from_the_anchor():
    t = epoch_starts(ANCHOR, 1200)
    assert len(t) == 1200 and t[0] == ANCHOR
    assert (t[1:] - t[:-1] == EPOCH).all()
    assert t[-1] == ANCHOR + 1199 * EPOCH


def test_round_time_anchor_shifts_the_sleep_flags_by_one_epoch():
    """MATLAB writes bout times at :07 and :37 s; only the recording-start anchor puts them on the epoch they belong to."""
    bouts = pd.DataFrame({"start": [pd.Timestamp("2026-07-13 21:54:07")], "end": [pd.Timestamp("2026-07-13 21:54:07")]})
    round_start = pd.Timestamp("2026-07-13 21:54:00")
    true_epoch = np.flatnonzero(epoch_vector(bouts, ANCHOR, ANCHOR + 10 * EPOCH))
    round_epoch = np.flatnonzero(epoch_vector(bouts, round_start, round_start + 10 * EPOCH))
    assert true_epoch.tolist() == [1]
    assert round_epoch.tolist() == [0]


# --- loading: order, duplicates, window, counts ---

def _write(tmp_path, times):
    df = pd.DataFrame({"time": pd.to_datetime(times), "x": 0.0, "y": 0.0, "z": np.arange(len(times), dtype=float), "extra": 1})
    path = tmp_path / "dev.parquet"
    df.to_parquet(path)
    return path


def test_load_sorts_drops_duplicates_and_reports_counts(tmp_path):
    t0 = ANCHOR
    step = pd.Timedelta(milliseconds=20)
    times = [t0 + i * step for i in range(10)]
    times = times[:5] + [times[3]] + times[5:8] + [times[4]] + times[8:]   # two exact repeats, both stepping backwards
    rec = load_recording(_write(tmp_path, times), t0, t0 + pd.Timedelta(seconds=1))
    assert rec.t_s[0] == 0.0
    assert (np.diff(rec.t_s) > 0).all()
    assert rec.timing.n_rows == 10
    assert rec.timing.duplicates_dropped == 2
    assert rec.timing.reordered == 2
    assert rec.timing.max_gap_s == pytest.approx(0.02)
    assert rec.fs == 50


def test_load_keeps_only_the_requested_window(tmp_path):
    times = [ANCHOR + pd.Timedelta(seconds=s) for s in range(-5, 20)]
    rec = load_recording(_write(tmp_path, times), ANCHOR, ANCHOR + pd.Timedelta(seconds=9))
    assert rec.t_s.min() == 0.0 and rec.t_s.max() == 9.0


def test_load_empty_window_raises(tmp_path):
    path = _write(tmp_path, [ANCHOR + pd.Timedelta(seconds=s) for s in range(5)])
    with pytest.raises(ValueError, match="no samples"):
        load_recording(path, ANCHOR + pd.Timedelta(hours=1), ANCHOR + pd.Timedelta(hours=2))


def test_nominal_rate_rounds_the_median_interval():
    jittered = np.cumsum(np.r_[0, np.full(1000, 1 / 50.24)])
    assert nominal_rate_hz(jittered) == 50


# --- the common grid ---

def _native(fs, seconds, fn, lag=0.0, jitter=0.0, pad=0, seed=0):
    """Native samples of fn(true time) over [-pad, seconds + pad), stamped `lag` late; the grid covers [0, seconds)."""
    t = np.arange(-pad * fs, int((seconds + pad) * fs)) / fs
    t = np.sort(t + np.random.default_rng(seed).uniform(-jitter, jitter, len(t)))
    return t + lag, np.column_stack([fn(t)] * 3)


def test_grid_reproduces_a_slow_sine_in_amplitude_and_phase():
    t, xyz = _native(100, 60, lambda s: np.sin(2 * np.pi * 0.5 * s), pad=5)
    g = build_grid(t, xyz, 100, 30, 60 * 30)
    tg = np.arange(60 * 30) / 30
    assert g.xyz[30:-30, 0] == pytest.approx(np.sin(2 * np.pi * 0.5 * tg[30:-30]), abs=2e-3)   # sample k sits at the anchor + k/30 s


def test_grid_interpolates_over_timestamp_jitter():
    t, xyz = _native(50, 60, lambda s: np.sin(2 * np.pi * 0.25 * s), jitter=0.004, pad=5)
    g = build_grid(t, xyz, 50, 30, 60 * 30)
    tg = np.arange(60 * 30) / 30
    assert np.abs(g.xyz[30:-30, 0] - np.sin(2 * np.pi * 0.25 * tg[30:-30])).max() < 0.01


def test_grid_attenuates_content_above_the_new_nyquist():
    t, xyz = _native(100, 60, lambda s: np.sin(2 * np.pi * 20 * s), pad=5)   # 20 Hz would alias to 10 Hz at 30 Hz
    g = build_grid(t, xyz, 100, 30, 60 * 30)
    assert np.abs(g.xyz[60:-60, 0]).max() < 0.02


def test_grid_lag_correction_moves_the_clock_back():
    fn = lambda s: np.sin(2 * np.pi * 0.2 * s)
    t, xyz = _native(50, 60, fn, lag=2.0, pad=5)     # this device stamps every sample 2 s late
    late = build_grid(t, xyz, 50, 30, 60 * 30)
    fixed = build_grid(t, xyz, 50, 30, 60 * 30, lag_s=2.0)
    tg = np.arange(60 * 30) / 30
    assert np.abs(fixed.xyz[30:-30, 0] - fn(tg[30:-30])).max() < 0.01
    assert np.abs(late.xyz[30:-30, 0] - fn(tg[30:-30])).max() > 0.5


def test_grid_scalar_streams_are_computed_before_decimation():
    """ENMO is rectified noise: computed on the filtered 30 Hz axes it loses the noise above 15 Hz (0.51 vs 0.55 epoch r)."""
    rng = np.random.default_rng(1)
    t = np.arange(-500, 6500) / 100
    xyz = np.column_stack([rng.normal(0, 0.02, len(t)), rng.normal(0, 0.02, len(t)), 1 + rng.normal(0, 0.02, len(t))])
    g = build_grid(t, xyz, 100, 30, 60 * 30)
    assert enmo(xyz).mean() == pytest.approx(g.enmo[60:-60].mean(), rel=0.02)
    assert enmo(g.xyz).mean() < 0.8 * g.enmo[60:-60].mean()


def test_grid_accepts_an_early_device_when_loaded_with_a_lag_margin_and_refuses_it_without(tmp_path):
    """A device whose clock is EARLY (lag <= -1 s) was refused: its corrected start is after the anchor unless the
    recording was loaded with a margin. Its data were valid."""
    def fn(s):
        return np.sin(2 * np.pi * 0.2 * s)

    t = np.arange(-6 * 50, 70 * 50) / 50
    stamped = t - 3.0                                                 # every sample stamped 3 s early
    df = pd.DataFrame({"time": ANCHOR + pd.to_timedelta(stamped, unit="s"), "x": fn(t), "y": fn(t), "z": fn(t)})
    path = tmp_path / "early.parquet"
    df.to_parquet(path)
    end = ANCHOR + pd.Timedelta(seconds=60)
    plain = load_recording(path, ANCHOR, end)
    margin = load_recording(path, ANCHOR, end, margin_s=5)
    assert plain.t_s[0] == pytest.approx(0.0, abs=0.02) and margin.t_s[0] == pytest.approx(-5.0, abs=0.02)
    with pytest.raises(ValueError, match="starts"):
        build_grid(plain.t_s, plain.xyz, 50, 30, 60 * 30, lag_s=-3.0)
    g = build_grid(margin.t_s, margin.xyz, 50, 30, 60 * 30, lag_s=-3.0)
    tg = np.arange(60 * 30) / 30
    assert np.abs(g.xyz[30:-30, 0] - fn(tg[30:-30])).max() < 0.01


def test_grid_start_tolerance_is_a_parameter_not_a_magic_number():
    t, xyz = _native(50, 60, lambda s: s * 0 + 0.5, lag=0.0, pad=5)
    t, xyz = t[t >= 0.6], xyz[t >= 0.6]                              # first sample 0.6 s after the anchor
    with pytest.raises(ValueError, match="starts"):
        build_grid(t, xyz, 50, 30, 60 * 30, start_tol_s=0.5)
    assert np.isfinite(build_grid(t, xyz, 50, 30, 60 * 30, start_tol_s=1.0).xyz).all()


def _gapped(gap_from, gap_to, seconds=90, fs=50):
    """A 0.1 Hz sine sampled at `fs` with every sample in [gap_from, gap_to) missing."""
    t = np.arange(-5 * fs, (seconds + 5) * fs) / fs
    keep = (t < gap_from) | (t >= gap_to)
    return t[keep], np.column_stack([np.sin(2 * np.pi * 0.1 * t[keep])] * 3) + np.array([0.0, 0.0, 1.0])


def test_grid_is_nan_inside_a_long_gap_not_a_ramp_across_it():
    """np.interp bridged a 30 s dropout with a straight line: a smooth ramp in x, y, z, VM and ENMO."""
    t, xyz = _gapped(30.0, 60.0)
    g = build_grid(t, xyz, 50, 30, 90 * 30, max_gap_s=1.0)
    tg = np.arange(90 * 30) / 30
    inside, outside = (tg > 30.0) & (tg < 60.0), (tg < 29.0) | (tg > 61.0)
    for stream in (g.xyz[:, 0], g.xyz[:, 2], g.vm, g.enmo):
        assert np.isnan(stream[inside]).all()
        assert np.isfinite(stream[outside]).all()
    assert np.isfinite(build_grid(t, xyz, 50, 30, 90 * 30, max_gap_s=60.0).vm).all()      # the threshold is the parameter


def test_grid_bridges_a_gap_shorter_than_the_threshold():
    t, xyz = _gapped(30.0, 30.5)
    g = build_grid(t, xyz, 50, 30, 90 * 30, max_gap_s=1.0)
    assert np.isfinite(g.vm).all()
    tg = np.arange(90 * 30) / 30
    assert np.abs(g.xyz[(tg > 20) & (tg < 40), 0] - np.sin(2 * np.pi * 0.1 * tg[(tg > 20) & (tg < 40)])).max() < 0.01


def test_gap_mask_marks_only_query_times_inside_a_longer_gap():
    t = np.array([0.0, 1.0, 2.0, 12.0, 13.0])
    tq = np.array([-1.0, 0.5, 1.5, 2.0, 5.0, 12.0, 12.5, 14.0])
    assert gap_mask(t, tq, 5.0).tolist() == [False, False, False, False, True, False, False, False]
    assert not gap_mask(t, tq, 10.0).any()


def test_grid_rejects_a_recording_that_ends_early():
    t, xyz = _native(50, 30, lambda s: s * 0)
    with pytest.raises(ValueError, match="ends"):
        build_grid(t, xyz, 50, 30, 60 * 30)


def test_grid_does_not_change_its_inputs():
    t, xyz = _native(50, 60, np.sin, pad=5)
    t0, x0 = t.copy(), xyz.copy()
    build_grid(t, xyz, 50, 30, 60 * 30)
    assert (t == t0).all() and (xyz == x0).all()


# --- epoching and the moving average ---

def test_epoch_matrix_is_epochs_by_samples():
    m = epoch_matrix(np.arange(12.0), 4)
    assert m.shape == (3, 4) and m[1].tolist() == [4, 5, 6, 7]


def test_epoch_matrix_rejects_a_partial_epoch():
    with pytest.raises(ValueError, match="multiple"):
        epoch_matrix(np.arange(10.0), 4)


def test_moving_average_of_a_constant_is_that_constant_including_the_edges():
    assert moving_average(np.full(50, 2.5), 9) == pytest.approx(np.full(50, 2.5))


def test_moving_average_known_values_and_identity():
    x = np.array([0.0, 0, 9, 0, 0])
    assert moving_average(x, 3).tolist() == pytest.approx([0, 3, 3, 3, 0])
    assert moving_average(x, 1).tolist() == x.tolist()


def test_moving_average_does_not_change_its_input():
    x = np.arange(20.0)
    before = x.copy()
    moving_average(x, 5)
    assert (x == before).all()


def test_smoothing_runs_on_the_continuous_stream_before_epoching():
    """A burst in the last sample of epoch 0 must leak into epoch 1's first samples; per-epoch smoothing cannot do that."""
    epoch_n, n = 30, 5
    x = np.zeros(4 * epoch_n)
    x[epoch_n - 1] = 1.0
    continuous = smooth_then_epoch(x, n, epoch_n)
    per_epoch = np.vstack([moving_average(row, n) for row in epoch_matrix(x, epoch_n)])
    assert continuous[1, 0] > 0 and per_epoch[1, 0] == 0
    assert continuous.sum() == pytest.approx(1.0)          # continuous smoothing conserves the burst
    assert per_epoch.sum() < 1.0                           # per-epoch smoothing loses part of it at the edge
    assert per_epoch[0, -1] != continuous[0, -1]           # and the two disagree exactly at the epoch boundary


# --- moving-average sweep on synthetic streams ---

def _pair(n=30 * 60 * 40, seed=0):
    """Two streams sharing a slow component and each with independent fast noise."""
    rng = np.random.default_rng(seed)
    slow = moving_average(rng.normal(size=n), 900) * 30
    return slow + rng.normal(size=n), slow + rng.normal(size=n)


CFG = {**VIEWER, "ma_windows_s": [0, 1, 30],      # blocks fit the 40 min test streams: a 60 min block would exceed them
       "bootstrap": {**VIEWER["bootstrap"], "resamples": 200, "null_draws": 20, "block_sensitivity_min": [1, 5, 10]}}


def test_sweep_correlation_rises_with_the_window_and_stays_above_the_null():
    a, b = _pair()
    rows = ma_sweep(a, b, 30, 900, CFG)
    assert [r["w_s"] for r in rows] == [0, 1, 30]
    assert rows[0]["r"] < rows[1]["r"] < rows[2]["r"]
    assert rows[2]["r"] > rows[2]["null_hi"] + 0.2
    assert rows[2]["r_lo"] <= rows[2]["r"] <= rows[2]["r_hi"]
    assert abs(rows[0]["null_med"]) < 0.05


def test_sweep_effective_sample_size_shrinks_as_smoothing_grows():
    a, b = _pair()
    rows = ma_sweep(a, b, 30, 900, CFG)
    assert rows[0]["n_eff"] > rows[1]["n_eff"] > rows[2]["n_eff"] >= 1
    assert rows[0]["n_eff"] <= len(a)


def test_sweep_n_eff_is_derived_from_the_interval_beside_it_and_says_so():
    """n_eff came from the Bartlett formula under r = 0 and was 10 to 24 times too large beside a wide interval."""
    a, b = _pair()
    for row in ma_sweep(a, b, 30, 900, CFG):
        assert row["n_eff_basis"] == "bootstrap_half_width"
        assert row["n_eff"] == pytest.approx(effective_n_from_ci(row["r"], row["r_lo"], row["r_hi"], len(a)))
        if 1 < row["n_eff"] < len(a):                  # not clipped: the interval it implies is the interval shown
            half = (row["r_hi"] - row["r_lo"]) / 2
            assert 1.96 * (1 - row["r"] ** 2) / np.sqrt(row["n_eff"]) == pytest.approx(half, rel=1e-3)


def test_sweep_ships_block_sensitivity_and_the_headline_interval_is_its_configured_block():
    a, b = _pair()
    rows = ma_sweep(a, b, 30, 900, CFG)
    for row in rows:
        sens = row["block_sensitivity"]
        assert [x["block_min"] for x in sens] == [1, 5, 10]
        assert all(x["lo"] <= row["r"] <= x["hi"] for x in sens)
        head = next(x for x in sens if x["block_min"] == VIEWER["bootstrap"]["block_min"])
        assert (row["r_lo"], row["r_hi"]) == (head["lo"], head["hi"])


def test_sweep_includes_the_configured_block_even_if_it_is_not_listed():
    a, b = _pair()
    cfg = {**CFG, "bootstrap": {**CFG["bootstrap"], "block_min": 2, "block_sensitivity_min": [1, 5]}}
    assert [x["block_min"] for x in ma_sweep(a, b, 30, 900, cfg)[0]["block_sensitivity"]] == [1, 2, 5]


def test_sweep_reports_within_epoch_r_distribution_and_lag():
    a, b = _pair()
    row = ma_sweep(a, b, 30, 900, CFG)[1]
    q = row["within_epoch_r"]
    assert q["q05"] <= q["q25"] <= q["q50"] <= q["q75"] <= q["q95"]
    assert q["n_valid"] == len(a) // 900
    assert abs(row["lag_s"]) <= 5 and row["lag_r"] == pytest.approx(row["r"], abs=0.1)


def test_sweep_finds_a_known_lag_and_is_reproducible():
    a, b = _pair()
    shifted = np.roll(b, 60)           # b is 2 s late
    r1 = ma_sweep(a, shifted, 30, 900, CFG)
    r2 = ma_sweep(a, shifted, 30, 900, CFG)
    assert r1[2]["lag_s"] == pytest.approx(2.0, abs=0.25)     # a 30 s window makes the peak broad
    assert r1 == r2


def test_sweep_w_zero_is_the_unsmoothed_correlation():
    a, b = _pair()
    row = ma_sweep(a, b, 30, 900, CFG)[0]
    assert row["n_samples"] == 1
    assert row["r"] == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_sweep_rejects_unequal_streams():
    with pytest.raises(ValueError, match="same length"):
        ma_sweep(np.zeros(900), np.zeros(1800), 30, 900, CFG)


# --- the device's own noise floor ---

def _still_noise(sigma_g, seed, windows=60, fs=50, window_s=10):
    rng = np.random.default_rng(seed)
    return np.array([0.0, 0.0, 1.0]) + rng.normal(scale=sigma_g, size=(windows * window_s * fs, 3))


def test_noise_floor_is_the_mean_enmo_of_gaussian_noise_and_is_per_device():
    quiet, noisy = _still_noise(0.002, 1), _still_noise(0.005, 2)
    a, b = (sensor_noise_floor(x, 50, 10, 0.013) for x in (quiet, noisy))
    assert a["value_mg"] == pytest.approx(0.002 * 1000 / np.sqrt(2 * np.pi), rel=0.03)      # sigma_VM / sqrt(2 pi)
    assert b["value_mg"] == pytest.approx(0.005 * 1000 / np.sqrt(2 * np.pi), rel=0.03)
    assert b["value_mg"] > 2 * a["value_mg"]
    assert a["n_windows"] == 60 and a["sigma_vm_mg_median"] == pytest.approx(2.0, rel=0.05) and "sqrt(2 pi)" in a["derivation"]


def test_noise_floor_uses_only_still_windows_and_skips_dropouts():
    still = _still_noise(0.002, 3, windows=30)
    moving = np.random.default_rng(4).normal(scale=0.3, size=(500 * 4, 3))      # 4 windows far from still
    dropout = np.zeros((500, 3))
    got = sensor_noise_floor(np.vstack([still, moving, dropout]), 50, 10, 0.013)
    assert got["n_windows"] == 30
    assert got["value_mg"] == pytest.approx(sensor_noise_floor(still, 50, 10, 0.013)["value_mg"])


# --- regression anchors on the real night ---

@pytest.mark.slow
def test_real_night_timing_report(real_night):
    assert (real_night["ax6"].rec.timing.reordered, real_night["ax6"].rec.timing.duplicates_dropped) == (0, 0)
    apple = real_night["apple"].rec.timing
    assert (apple.reordered, apple.duplicates_dropped) == (97, 6)     # batch boundaries of the Apple export
    assert (real_night["ax6"].rec.fs, real_night["apple"].rec.fs) == (100, 50)
    assert apple.max_gap_s < 0.2


@pytest.mark.slow
def test_real_night_mean_calibrated_enmo(real_night):
    ax6, apple = real_night["ax6"], real_night["apple"]
    assert enmo(apply(ax6.rec.xyz, ax6.calibration)).mean() * 1000 == pytest.approx(2.09, abs=0.02)
    assert enmo(apply(apple.rec.xyz, apple.calibration)).mean() * 1000 == pytest.approx(1.29, abs=0.02)


@pytest.mark.slow
def test_real_night_30s_enmo_correlation_raw_and_calibrated(real_night):
    def r(key):
        a, b = (epoch_matrix(getattr(real_night[d], key).enmo, 900).mean(axis=1) for d in ("ax6", "apple"))
        return np.corrcoef(a, b)[0, 1]
    assert r("grid_raw") == pytest.approx(0.18, abs=0.02)
    assert r("grid_cal") == pytest.approx(0.55, abs=0.02)


@pytest.mark.slow
def test_real_night_sweep_at_30_s_lands_on_the_published_epoch_correlation(real_night):
    cfg = {**VIEWER, "bootstrap": {**VIEWER["bootstrap"], "null_draws": 10}}
    rows = ma_sweep(real_night["ax6"].grid_cal.enmo, real_night["apple"].grid_cal.enmo, 30, 900, cfg)
    by_w = {r["w_s"]: r for r in rows}
    assert by_w[30]["r"] == pytest.approx(0.55, abs=0.02)
    assert by_w[0]["r"] < by_w[1]["r"] < by_w[30]["r"]
    assert by_w[30]["r_lo"] > by_w[30]["null_hi"]      # the honest signal: the curve clears its null floor


@pytest.mark.slow
def test_real_night_grid_enmo_rings_far_below_zero_and_axes_first_order_gives_the_lower_figures(real_night):
    """The docstring claims: ringing to -186 / -101 mg on the ENMO stream, and 1.81 / 1.16 mg with r 0.51 if ENMO is taken
    of the filtered axes instead of filtering ENMO."""
    ax6, apple = real_night["ax6"].grid_cal, real_night["apple"].grid_cal
    assert ax6.enmo.min() * 1000 == pytest.approx(-186, abs=5) and apple.enmo.min() * 1000 == pytest.approx(-101, abs=5)
    assert enmo(ax6.xyz).mean() * 1000 == pytest.approx(1.81, abs=0.02) and enmo(apple.xyz).mean() * 1000 == pytest.approx(1.16, abs=0.02)
    a, b = (epoch_matrix(enmo(g.xyz), 900).mean(axis=1) for g in (ax6, apple))
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.51, abs=0.01)
    assert epoch_matrix(ax6.enmo, 900).mean() * 1000 == pytest.approx(2.09, abs=0.02)      # the stream itself keeps the epoch means


@pytest.mark.slow
def test_real_night_measured_sensor_noise_floors(real_night):
    floors = {d: sensor_noise_floor(real_night[d].xyz, real_night[d].rec.fs, VIEWER["still_window_s"], VIEWER["calibration_std_g"])
              for d in ("ax6", "apple")}
    assert floors["ax6"]["value_mg"] == pytest.approx(1.18, abs=0.02) and floors["apple"]["value_mg"] == pytest.approx(0.65, abs=0.02)
    assert floors["ax6"]["sigma_vm_mg_median"] == pytest.approx(2.85, abs=0.05)
    assert floors["apple"]["sigma_vm_mg_median"] == pytest.approx(1.70, abs=0.05)


@pytest.mark.slow
def test_real_night_n_eff_is_consistent_with_the_bootstrap_interval(real_night):
    """858 at w = 30 s was reported beside CI [0.352, 0.655], which implies about 83."""
    cfg = {**VIEWER, "ma_windows_s": [0, 30], "bootstrap": {**VIEWER["bootstrap"], "null_draws": 5}}
    rows = ma_sweep(real_night["ax6"].grid_cal.enmo, real_night["apple"].grid_cal.enmo, 30, 900, cfg)
    w0, w30 = rows
    assert w30["n_eff"] == pytest.approx(84, abs=25) and w30["n_eff"] < 858 / 5
    assert w0["n_eff"] < 78533 / 10
    for row in rows:
        assert row["n_eff"] == pytest.approx(effective_n_from_ci(row["r"], row["r_lo"], row["r_hi"], 1_080_000))
    widths = [x["hi"] - x["lo"] for x in w30["block_sensitivity"]]
    assert [x["block_min"] for x in w30["block_sensitivity"]] == [5, 15, 30, 60]
    assert widths[-1] > 1.3 * widths[0]                                        # block 5 understates it


def test_grid_rejects_a_recording_that_starts_late_and_a_rate_that_does_not_divide():
    t, xyz = _native(50, 60, lambda s: s * 0, lag=2.5, pad=5)
    with pytest.raises(ValueError, match="starts"):
        build_grid(t[t > 2.5 + 0.5], xyz[t > 3.0], 50, 30, 60 * 30)     # first sample 0.5 s after 2.5 s: corrected start 3 s
    with pytest.raises(ValueError, match="whole number"):
        build_grid(t, xyz, 50, 30, 1801)


def test_unknown_window_variant_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(VIEWER, "window_variant", "no_such_variant")
    with pytest.raises(ValueError, match="window_variant"):
        recording_window()
