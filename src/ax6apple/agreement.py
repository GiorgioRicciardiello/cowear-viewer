"""Agreement between the two devices: r, rho, CCC, ICC(2,1), Bland-Altman, and moving-block bootstrap intervals.

Descriptive only, in the spirit of `rbd_compare.py`: one night and one wearer have no distribution to test, so nothing
here returns a p-value or a pass/fail. The reference device is the first argument (AX6) and the test device the second
(Apple); differences are test - reference.

Epochs and samples are serially dependent, so a plain bootstrap would give intervals that are far too narrow. The
resampling unit is a moving block of consecutive values (configured in minutes, `block_length` turns that into
samples). For the 30 Hz stream, where 1.08 M samples times 2000 resamples would not fit, Pearson r is rebuilt from the
per-block sums (`pearson_block_ci`). `phase_randomised` gives a null floor that keeps one stream's autocorrelation but
breaks its pairing with the other, and `effective_n_from_ci` says how many independent samples a correlation really rests on.

Any single bootstrap interval is conditional on its block length: a block shorter than the dependence in the data
understates the interval (the 30 s ENMO r = 0.551 runs [0.334, 0.689] at 5 min and [0.170, 0.720] at 60 min, and under AR(0.97)
dependence longer than the block the simulated coverage of a nominal 95 % interval is 0.60). `pearson_block_sensitivity`
gives the interval at several block lengths; report those, not one.

Epochs where a device sits below its own ENMO noise floor are flagged (`noise_floor_flag`) and left out of ratio statistics
(`log2_ratio` returns NaN there): flagged, never silently dropped.
"""

from typing import Callable

import numpy as np
from scipy.stats import norm, spearmanr

from ax6apple.rbd_sleep import sleep_agreement

LOA_Z = 1.96
CI_LEVEL = 0.95
CI_Z = float(norm.ppf((1 + CI_LEVEL) / 2))
CONSTANT_REL = 1e-20     # a row is constant when its centred sum of squares is below this fraction of its raw one (scale free)


def _pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"a and b must be 1-D and of the same length, got {a.shape} and {b.shape}")
    if len(a) < 3:
        raise ValueError(f"at least 3 pairs are needed, got {len(a)}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN in input: drop or flag those epochs explicitly before computing agreement")
    return a, b


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _pair(a, b)
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _pair(a, b)
    return float(spearmanr(a, b).statistic)


def ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's concordance: correlation discounted for a difference in level and in scale."""
    a, b = _pair(a, b)
    sxy = ((a - a.mean()) * (b - b.mean())).mean()
    return float(2 * sxy / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def icc21(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (two devices, n epochs)."""
    a, b = _pair(a, b)
    data, k = np.column_stack([a, b]), 2
    n = len(a)
    row, col, grand = data.mean(axis=1), data.mean(axis=0), data.mean()
    msr = k * ((row - grand) ** 2).sum() / (n - 1)
    msc = n * ((col - grand) ** 2).sum() / (k - 1)
    mse = ((data - row[:, None] - col + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Bias (mean of b - a), SD of the differences and the 95 % limits of agreement."""
    a, b = _pair(a, b)
    d = b - a
    bias, sd = d.mean(), d.std(ddof=1)
    return {"bias": float(bias), "sd": float(sd), "loa_low": float(bias - LOA_Z * sd), "loa_high": float(bias + LOA_Z * sd)}


def night_stats(a: np.ndarray, b: np.ndarray) -> dict:
    ba = bland_altman(a, b)
    return {"pearson": pearson(a, b), "spearman": spearman(a, b), "ccc": ccc(a, b), "icc21": icc21(a, b),
            "ba_bias": ba["bias"], "ba_loa_low": ba["loa_low"], "ba_loa_high": ba["loa_high"]}


# --- moving-block bootstrap ---

def block_length(block_min: float, rate_hz: float) -> int:
    """Samples in a block of `block_min` minutes of a series sampled at `rate_hz` (epochs: 1/30 Hz)."""
    return max(1, int(round(block_min * 60 * rate_hz)))


def _starts(n: int, block: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= block <= n:
        raise ValueError(f"block must be between 1 and the series length {n}, got {block}")
    return rng.integers(0, n - block + 1, size=(resamples, -(-n // block)))


def block_indices(n: int, block: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Resample index matrix (resamples x n): consecutive blocks of `block` values starting anywhere, cut to n."""
    starts = _starts(n, block, resamples, rng)
    return (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n]


def bootstrap(stat_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray, block: int,
              resamples: int, rng: np.random.Generator) -> np.ndarray:
    """`stat_fn` (returns a 1-D array) on every moving-block resample of the pair: resamples x len(stat)."""
    a, b = np.asarray(a), np.asarray(b)
    idx = block_indices(len(a), block, resamples, rng)
    return np.array([stat_fn(a[i], b[i]) for i in idx])


def _interval(samples: np.ndarray) -> np.ndarray:
    """Percentile interval of resampled statistics (resamples x k). A NaN resample is an error, never dropped silently."""
    dropped = int(np.isnan(samples).reshape(len(samples), -1).any(axis=1).sum())
    if dropped:
        raise ValueError(f"{dropped} of {len(samples)} bootstrap resamples gave NaN (a constant or degenerate resample): "
                         "the interval would silently exclude them")
    tail = 100 * (1 - CI_LEVEL) / 2
    return np.percentile(samples, [tail, 100 - tail], axis=0)


def night_stats_ci(a: np.ndarray, b: np.ndarray, block: int, resamples: int, rng: np.random.Generator) -> dict:
    """Each night statistic as {estimate, lo, hi}, the interval from a moving-block bootstrap."""
    est = night_stats(a, b)
    boot = bootstrap(lambda x, y: np.array(list(night_stats(x, y).values())), a, b, block, resamples, rng)
    lo, hi = _interval(boot)
    return {k: {"estimate": est[k], "lo": float(lo[i]), "hi": float(hi[i])} for i, k in enumerate(est)}


def kappa_ci(ref: np.ndarray, test: np.ndarray, block: int, resamples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Cohen's kappa of two sleep masks (as `rbd_sleep.sleep_agreement`) with its block bootstrap interval."""
    boot = bootstrap(lambda r, t: np.array([sleep_agreement(r, t)["kappa"]]), ref, test, block, resamples, rng)
    lo, hi = _interval(boot)
    return float(sleep_agreement(ref, test)["kappa"]), float(lo[0]), float(hi[0])


def pearson_block_ci(a: np.ndarray, b: np.ndarray, block: int, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Block bootstrap interval of Pearson r from per-block sums (for streams too long to resample value by value)."""
    a, b = _pair(a, b)
    a, b = a - a.mean(), b - b.mean()
    sums = np.stack([np.cumsum(v) for v in (a, b, a * a, b * b, a * b)], axis=1)
    sums = np.vstack([np.zeros(5), sums])
    per_block = sums[block:] - sums[:-block]                       # sums over every block start
    s = per_block[_starts(len(a), block, resamples, rng)].sum(axis=1)   # resamples x 5, blocks summed
    k = -(-len(a) // block) * block                                 # values in a resample (r is scale free)
    r = (s[:, 4] / k - s[:, 0] * s[:, 1] / k ** 2) / np.sqrt((s[:, 2] / k - (s[:, 0] / k) ** 2) * (s[:, 3] / k - (s[:, 1] / k) ** 2))
    lo, hi = _interval(r)
    return float(lo), float(hi)


def pearson_block_sensitivity(a: np.ndarray, b: np.ndarray, blocks_min: list[float], rate_hz: float, resamples: int,
                              seed: int) -> list[dict]:
    """Pearson r interval at each block length: [{block_min, lo, hi}], from `pearson_block_ci` with the same seed each time.

    Every interval is conditional on its block length, so none of them is "the" interval: a block shorter than the
    dependence in the data understates it. A block longer than the series is an error.
    """
    out = []
    for m in blocks_min:
        lo, hi = pearson_block_ci(a, b, block_length(m, rate_hz), resamples, np.random.default_rng(seed))
        out.append({"block_min": m, "lo": lo, "hi": hi})
    return out


# --- null floor and effective sample size ---

def phase_randomised(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with the same amplitude spectrum (so the same autocorrelation) and random phases; rows of a 2-D array each."""
    x = np.asarray(x, dtype=float)
    f = np.fft.rfft(x, axis=-1)
    phase = rng.uniform(0, 2 * np.pi, f.shape)
    phase[..., 0] = 0.0
    if x.shape[-1] % 2 == 0:
        phase[..., -1] = 0.0                                        # the Nyquist bin must stay real
    return np.fft.irfft(f * np.exp(1j * phase), x.shape[-1], axis=-1)


def effective_n_from_ci(r: float, lo: float, hi: float, n: int) -> float:
    """Independent samples that a correlation r with the bootstrap interval [lo, hi] is worth: (z (1 - r^2) / half_width)^2.

    Inverts the large-sample standard error (1 - r^2) / sqrt(n_eff) of a Pearson r, so it is as conditional on the
    interval's block length as the interval is, and it can never contradict the interval it is shown beside. It replaces
    the Bartlett figure n / (1 + 2 sum rho_a rho_b), which assumes r = 0 and was 10 to 24 times too large on r = 0.54.
    Clipped to [1, n].
    """
    half_width = (hi - lo) / 2
    if not half_width > 0:
        return float(n)
    return float(np.clip((CI_Z * (1 - r ** 2) / half_width) ** 2, 1, n))


# --- per-epoch statistics and flags ---

def epoch_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r of every row of two epochs x samples matrices; NaN for an epoch that is constant on either device.

    Constant is judged relative to the row's own level (`CONSTANT_REL`), so the test does not depend on the unit.
    """
    ca, cb = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    saa, sbb = (ca * ca).sum(axis=1), (cb * cb).sum(axis=1)
    varying = (saa > CONSTANT_REL * (a * a).sum(axis=1)) & (sbb > CONSTANT_REL * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(varying, (ca * cb).sum(axis=1) / np.sqrt(saa * sbb), np.nan)


def noise_floor_flag(enmo_a_mg: np.ndarray, enmo_b_mg: np.ndarray, floor_a_mg: float, floor_b_mg: float) -> np.ndarray:
    """True where either device's epoch ENMO is below that device's own noise floor. All four arguments are in mg."""
    return (np.asarray(enmo_a_mg) < floor_a_mg) | (np.asarray(enmo_b_mg) < floor_b_mg)


def log2_ratio(a: np.ndarray, b: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """log2(b / a), NaN on flagged epochs (a ratio of two noise-floor values says nothing)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    safe = ~flagged & (a > 0) & (b > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(safe, np.log2(np.where(safe, b / np.where(safe, a, 1), 1)), np.nan)


def agreement_class(ax6_sleep: np.ndarray, apple_sleep: np.ndarray) -> np.ndarray:
    """Per epoch, from two boolean sleep masks: both_sleep, both_wake, ax6_only (AX6 asleep, Apple awake) or apple_only."""
    return np.select([ax6_sleep & apple_sleep, ~ax6_sleep & ~apple_sleep, ax6_sleep],
                     ["both_sleep", "both_wake", "ax6_only"], default="apple_only")
