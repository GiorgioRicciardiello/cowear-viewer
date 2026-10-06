"""Agreement statistics: known values on small examples, bootstrap behaviour, and the real-night sleep mask numbers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.agreement import (_interval, agreement_class, bland_altman, block_indices, block_length, bootstrap, ccc,
                                effective_n_from_ci, epoch_pearson, icc21, kappa_ci, log2_ratio, night_stats, night_stats_ci,
                                noise_floor_flag, pearson, pearson_block_ci, pearson_block_sensitivity, phase_randomised,
                                spearman)
from ax6apple.rbd_sleep import epoch_vector, load_bouts, sleep_agreement
from ax6apple.config import AX6APPLE, RBD_ROOT


def _ar1(n, phi, seed):
    x = np.random.default_rng(seed).normal(size=n)
    out = np.empty(n)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + x[i]
    return out


# --- point estimates ---

def test_pearson_and_spearman_on_a_monotone_nonlinear_pair():
    a = np.arange(1.0, 21)
    assert pearson(a, a ** 3) < 1.0
    assert spearman(a, a ** 3) == pytest.approx(1.0)
    assert pearson(a, 2 * a + 5) == pytest.approx(1.0)


def test_ccc_is_one_only_for_identity_and_penalises_bias_and_scale():
    a = np.random.default_rng(0).normal(size=200)
    assert ccc(a, a) == pytest.approx(1.0)
    assert ccc(a, a + 1) < pearson(a, a + 1) == pytest.approx(1.0)
    assert ccc(a, 2 * a) < 1.0


def test_ccc_matches_lins_formula():
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([1.5, 2, 4, 4, 6])
    sxy = ((a - a.mean()) * (b - b.mean())).mean()
    assert ccc(a, b) == pytest.approx(2 * sxy / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def test_icc21_matches_the_anova_definition_on_a_worked_example():
    a, b = np.array([9.0, 6, 8, 7, 10, 6]), np.array([2.0, 1, 4, 1, 5, 2])
    data = np.column_stack([a, b])
    n, k = data.shape
    grand = data.mean()
    msr = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((data - data.mean(axis=1, keepdims=True) - data.mean(axis=0) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    expected = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    assert icc21(a, b) == pytest.approx(expected)
    assert icc21(a, a) == pytest.approx(1.0)


def test_icc21_drops_with_a_constant_offset_while_pearson_does_not():
    a = np.random.default_rng(1).normal(size=300)
    assert icc21(a, a + 3) < 0.5
    assert pearson(a, a + 3) == pytest.approx(1.0)


def test_bland_altman_is_test_minus_reference_with_1_96_sd_limits():
    a = np.array([1.0, 2, 3, 4])
    b = a + np.array([0.5, 1.5, 0.5, 1.5])
    ba = bland_altman(a, b)
    assert ba["bias"] == pytest.approx(1.0)
    assert ba["sd"] == pytest.approx(np.std([0.5, 1.5, 0.5, 1.5], ddof=1))
    assert ba["loa_low"] == pytest.approx(1.0 - 1.96 * ba["sd"])
    assert ba["loa_high"] == pytest.approx(1.0 + 1.96 * ba["sd"])


def test_night_stats_has_every_statistic_of_plan_section_5():
    a = np.random.default_rng(2).normal(size=100)
    got = night_stats(a, a + np.random.default_rng(3).normal(scale=0.3, size=100))
    assert set(got) == {"pearson", "spearman", "ccc", "icc21", "ba_bias", "ba_loa_low", "ba_loa_high"}


@pytest.mark.parametrize("fn", [pearson, spearman, ccc, icc21, bland_altman])
def test_statistics_reject_unequal_or_too_short_input(fn):
    with pytest.raises(ValueError, match="same length"):
        fn(np.zeros(5), np.zeros(6))
    with pytest.raises(ValueError, match="at least 3"):
        fn(np.zeros(2), np.zeros(2))


def test_statistics_reject_nan_instead_of_hiding_it():
    with pytest.raises(ValueError, match="NaN"):
        pearson(np.array([1.0, np.nan, 3, 4]), np.arange(4.0))


# --- moving-block bootstrap ---

def test_block_indices_are_contiguous_blocks_inside_the_series():
    idx = block_indices(100, 10, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() <= 99
    within = np.diff(idx.reshape(50, 10, 10), axis=2)
    assert (within == 1).all()


def test_block_indices_reject_a_block_longer_than_the_series():
    with pytest.raises(ValueError, match="block"):
        block_indices(5, 10, 10, np.random.default_rng(0))


def test_block_length_converts_minutes_to_samples():
    assert block_length(5, 1 / 30) == 10
    assert block_length(5, 30) == 9000


def test_moving_blocks_give_a_wider_interval_than_iid_resampling_on_autocorrelated_data():
    x = _ar1(2000, 0.9, seed=4)
    mean_stat = lambda a, b: np.array([a.mean()])
    iid = bootstrap(mean_stat, x, x, 1, 500, np.random.default_rng(0))
    blocked = bootstrap(mean_stat, x, x, 50, 500, np.random.default_rng(0))
    assert np.ptp(np.percentile(blocked, [2.5, 97.5])) > 1.5 * np.ptp(np.percentile(iid, [2.5, 97.5]))


def test_night_stats_ci_brackets_each_estimate_and_is_reproducible():
    rng = np.random.default_rng(5)
    a = np.abs(rng.normal(size=600)) + 1
    b = a * 0.8 + rng.normal(scale=0.3, size=600)
    r1 = night_stats_ci(a, b, 10, 300, np.random.default_rng(9))
    r2 = night_stats_ci(a, b, 10, 300, np.random.default_rng(9))
    assert r1 == r2
    for name, v in r1.items():
        assert v["lo"] <= v["estimate"] <= v["hi"], name
    assert r1["ba_bias"]["estimate"] == pytest.approx(bland_altman(a, b)["bias"])


def test_pearson_block_ci_agrees_with_the_generic_bootstrap():
    x = _ar1(6000, 0.95, seed=6)
    y = 0.6 * x + _ar1(6000, 0.95, seed=7)
    lo, hi = pearson_block_ci(x, y, 300, 1000, np.random.default_rng(1))
    generic = bootstrap(lambda a, b: np.array([pearson(a, b)]), x, y, 300, 1000, np.random.default_rng(1))
    glo, ghi = np.percentile(generic, [2.5, 97.5])
    assert lo <= pearson(x, y) <= hi
    assert (hi - lo) == pytest.approx(ghi - glo, rel=0.35)


def test_kappa_ci_brackets_the_kappa_of_sleep_agreement():
    rng = np.random.default_rng(8)
    ref = np.repeat(rng.random(120) < 0.6, 10)
    test = ref ^ (rng.random(1200) < 0.1)
    est, lo, hi = kappa_ci(ref, test, 20, 300, np.random.default_rng(2))
    assert est == pytest.approx(sleep_agreement(ref, test)["kappa"])
    assert lo <= est <= hi


# --- null floor and effective sample size ---

def test_phase_randomised_keeps_the_power_spectrum_and_destroys_the_pairing():
    x = _ar1(4096, 0.9, seed=10)
    y = phase_randomised(x, np.random.default_rng(0))
    assert y.shape == x.shape and np.isrealobj(y)
    assert np.abs(np.fft.rfft(y)) == pytest.approx(np.abs(np.fft.rfft(x)), rel=1e-6, abs=1e-6)
    assert abs(pearson(x, y)) < 0.2
    assert not np.array_equal(x, y)


def test_phase_randomised_works_row_wise_on_a_matrix():
    x = np.stack([_ar1(512, 0.9, seed=11), _ar1(512, 0.5, seed=12)])
    y = phase_randomised(x, np.random.default_rng(0))
    assert y.shape == x.shape
    assert np.abs(np.fft.rfft(y, axis=1)) == pytest.approx(np.abs(np.fft.rfft(x, axis=1)), rel=1e-6, abs=1e-6)


def test_effective_n_reproduces_the_review_figure_for_the_30_s_enmo_interval():
    """r = 0.54 with the block-5 interval [0.352, 0.655]: about 83 independent samples, not the Bartlett 858."""
    assert effective_n_from_ci(0.54, 0.352, 0.655, 1_080_000) == pytest.approx(84, abs=3)
    assert effective_n_from_ci(0.54, 0.352, 0.655, 1_080_000) < 858 / 8


def test_effective_n_is_the_length_for_independent_pairs_and_is_clipped_to_the_length():
    rng = np.random.default_rng(0)
    a = rng.normal(size=4000)
    b = 0.6 * a + 0.8 * rng.normal(size=4000)                      # iid pairs with r about 0.6
    lo, hi = pearson_block_ci(a, b, 1, 1000, np.random.default_rng(1))
    assert effective_n_from_ci(pearson(a, b), lo, hi, 4000) == pytest.approx(4000, rel=0.2)
    assert effective_n_from_ci(0.5, 0.4999, 0.5001, 100) == 100        # a tighter interval than n allows: clipped
    assert effective_n_from_ci(0.5, 0.5, 0.5, 100) == 100              # a degenerate interval says nothing
    assert effective_n_from_ci(0.99, -0.5, 0.99, 100) == 1


def test_effective_n_never_contradicts_the_interval_it_sits_beside():
    """On a strongly autocorrelated pair (r near 0.5, dependence far longer than the block) n_eff is a small fraction of n and
    the interval it implies, r +/- z (1 - r^2) / sqrt(n_eff), is the bootstrap half-width."""
    n = 30000
    slow = _ar1(n, 0.995, seed=20)
    a, b = slow + _ar1(n, 0.995, seed=21), slow + _ar1(n, 0.995, seed=22)
    r = pearson(a, b)
    lo, hi = pearson_block_ci(a, b, 300, 1000, np.random.default_rng(2))
    n_eff = effective_n_from_ci(r, lo, hi, n)
    assert n_eff < 0.02 * n
    assert 1.96 * (1 - r ** 2) / np.sqrt(n_eff) == pytest.approx((hi - lo) / 2, rel=1e-4)


def test_block_sensitivity_widens_the_interval_with_block_length_under_long_dependence():
    n = 20000
    slow = _ar1(n, 0.99, seed=30)
    a, b = slow + _ar1(n, 0.99, seed=31), slow + _ar1(n, 0.99, seed=32)
    got = pearson_block_sensitivity(a, b, [5, 15, 30, 60], 1 / 3, 500, seed=1)         # 1 sample per 3 s: blocks of 100 to 1200
    assert [g["block_min"] for g in got] == [5, 15, 30, 60]
    assert all(set(g) == {"block_min", "lo", "hi"} for g in got)
    width = [g["hi"] - g["lo"] for g in got]
    assert width[-1] > 1.3 * width[0]
    assert got == pearson_block_sensitivity(a, b, [5, 15, 30, 60], 1 / 3, 500, seed=1)   # reproducible
    with pytest.raises(ValueError, match="block"):
        pearson_block_sensitivity(a[:300], b[:300], [60], 1 / 3, 100, seed=1)              # 1200-sample block, 300 samples


# --- per-epoch statistics and flags ---

def test_epoch_pearson_per_row_with_nan_for_a_constant_epoch():
    a = np.array([[1.0, 2, 3, 4], [1.0, 2, 3, 4], [5.0, 5, 5, 5]])
    b = np.array([[2.0, 4, 6, 8], [4.0, 3, 2, 1], [1.0, 2, 3, 4]])
    r = epoch_pearson(a, b)
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(-1.0) and np.isnan(r[2])


def test_noise_floor_flags_epochs_where_either_device_is_below_and_ratio_is_masked():
    a, b = np.array([0.0, 2.0, 4.0, 1.0]), np.array([1.0, 1.0, 0.2, 2.0])
    flag = noise_floor_flag(a, b, 0.5, 0.5)
    assert flag.tolist() == [True, False, True, False]
    ratio = log2_ratio(a, b, flag)
    assert np.isnan(ratio[[0, 2]]).all()
    assert ratio[1] == pytest.approx(-1.0) and ratio[3] == pytest.approx(1.0)


def test_noise_floor_is_per_device():
    a, b = np.array([0.8, 0.8, 2.0]), np.array([0.8, 0.4, 2.0])
    assert noise_floor_flag(a, b, 1.2, 0.5).tolist() == [True, True, False]     # the first: a below 1.2; the second: a below 1.2
    assert noise_floor_flag(a, b, 0.5, 0.65).tolist() == [False, True, False]   # 0.8 clears 0.5 and 0.65; b = 0.4 does not clear 0.65


def test_epoch_pearson_constant_test_is_scale_free():
    """The old `den > 1e-12` was in the unit of the data: a real signal of amplitude 1e-9 was NaN, a constant of 1e6 was not."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 200))
    b = 0.5 * a + rng.normal(size=(3, 200))
    ref = epoch_pearson(a, b)
    assert epoch_pearson(a * 1e-9, b * 1e-9) == pytest.approx(ref)
    assert epoch_pearson(a * 1e6, b * 1e6) == pytest.approx(ref)
    big_constant = np.full((1, 200), 1.05e6)
    tiny_constant = np.full((1, 200), 3.7e-9)
    assert np.isnan(epoch_pearson(big_constant, a[:1])).all() and np.isnan(epoch_pearson(a[:1], big_constant)).all()
    assert np.isnan(epoch_pearson(tiny_constant, a[:1] * 1e-9)).all()
    assert np.isnan(epoch_pearson(np.zeros((1, 200)), a[:1])).all()


def test_interval_refuses_to_drop_nan_resamples_silently():
    good = np.random.default_rng(0).normal(size=(200, 3))
    assert _interval(good) == pytest.approx(np.percentile(good, [2.5, 97.5], axis=0))
    bad = good.copy()
    bad[7, 1] = np.nan
    with pytest.raises(ValueError, match="1 of 200 bootstrap resamples gave NaN"):
        _interval(bad)
    with pytest.raises(ValueError, match="NaN"):
        _interval(np.array([1.0, np.nan, 3.0]))


def test_kappa_ci_raises_when_a_resample_has_no_variance():
    ref = np.zeros(40, dtype=bool)
    ref[:2] = True                              # a resample of 4 blocks of 10 can be all-wake: kappa is 0/0 there
    with pytest.raises(ValueError, match="NaN"):
        kappa_ci(ref, ref.copy(), 10, 300, np.random.default_rng(0))


def test_agreement_class_of_the_two_sleep_masks():
    ax6 = np.array([True, False, True, False])
    apple = np.array([True, False, False, True])
    assert agreement_class(ax6, apple).tolist() == ["both_sleep", "both_wake", "ax6_only", "apple_only"]


# --- real night ---

SLEEP = {"ax6": RBD_ROOT / "fit/v2_ax6_trim_csv/auto_sleep/abk", "apple": RBD_ROOT / "fit/v3_apple_csv/auto_sleep/abk"}


def _masks():
    start = pd.Timestamp(next(v for v in AX6APPLE["variants"] if v["name"] == AX6APPLE["viewer"]["window_variant"])["start"])
    end = start + 1200 * pd.Timedelta(seconds=30)
    return [epoch_vector(load_bouts(next(SLEEP[d].glob("*.csv"))), start, end) for d in ("ax6", "apple")]


@pytest.mark.slow
def test_real_sleep_masks_reproduce_the_published_agreement():
    ax6, apple = _masks()
    got = sleep_agreement(ax6, apple)
    assert got["n_epochs"] == 1200
    assert got["agreement"] == pytest.approx(0.9333, abs=5e-5)
    assert got["kappa"] == pytest.approx(0.6605, abs=5e-5)
    assert (got["sleep_min_ref"], got["sleep_min_test"]) == (544, 524)


@pytest.mark.slow
def test_real_night_epoch_enmo_statistics_and_intervals(real_night):
    a, b = (real_night[d].grid_cal.enmo.reshape(1200, 900).mean(axis=1) * 1000 for d in ("ax6", "apple"))
    got = night_stats_ci(a, b, block_length(5, 1 / 30), 2000, np.random.default_rng(AX6APPLE["viewer"]["bootstrap"]["seed"]))
    assert got["pearson"]["estimate"] == pytest.approx(0.55, abs=0.02)
    assert got["pearson"]["lo"] < 0.55 < got["pearson"]["hi"]
    assert got["ccc"]["estimate"] < got["pearson"]["estimate"]        # the level gap costs concordance
    assert got["ba_bias"]["estimate"] < 0                              # Apple below AX6: 1.29 vs 2.09 mg
