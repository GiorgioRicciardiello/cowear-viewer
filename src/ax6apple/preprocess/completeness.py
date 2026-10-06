"""Data completeness: gap runs, gapped samples, total gap time.

Closes section 0.3: no completeness number exists today. Reuses `signals.gap_mask` rather than
reimplementing the gap rule, so this library and `signals.build_grid` can never disagree about
what a gap is.
"""

from dataclasses import dataclass

import numpy as np

from ax6apple.signals import gap_mask

from .findings import Finding, breach
from .limits import Limits


@dataclass(frozen=True)
class GapStats:
    n_grid_samples: int
    gapped_samples: int          # grid samples falling inside a gap longer than max_gap_s
    n_gap_runs: int
    total_gap_s: float
    longest_gap_s: float
    completeness: float          # 1 - gapped_samples / n_grid_samples, in [0, 1]
    bridged_runs: int            # gaps shorter than max_gap_s, interpolated not NaN'd


def gap_runs(mask: np.ndarray, grid_hz: float) -> tuple[tuple[int, int], ...]:
    """Contiguous True runs of a gap mask as (start, stop) grid indices (stop exclusive)."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return tuple(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def gap_stats(t_s: np.ndarray, grid_hz: float, n_out: int, max_gap_s: float) -> GapStats:
    """Completeness of the grid `signals.build_grid` would produce, without building it.

    Uses `signals.gap_mask` on the same query times `build_grid` uses, so the count is exactly the
    number of samples `build_grid` would set to NaN.
    """
    tq = np.arange(n_out) / grid_hz
    mask = gap_mask(t_s, tq, max_gap_s)
    runs = gap_runs(mask, grid_hz)
    gapped = int(mask.sum())
    longest = max((stop - start for start, stop in runs), default=0) / grid_hz
    d = np.diff(t_s)
    nominal = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    bridged = int(((d > 1.5 * nominal) & (d <= max_gap_s)).sum()) if nominal > 0 else 0
    return GapStats(n_grid_samples=n_out, gapped_samples=gapped, n_gap_runs=len(runs), total_gap_s=gapped / grid_hz,
                     longest_gap_s=longest, completeness=1.0 - gapped / n_out, bridged_runs=bridged)


def judge(g: GapStats, subject: str, lim: Limits) -> tuple[Finding, ...]:
    """`completeness_fail` -> fail, `completeness_warn` -> warn, `gap_run_warn` -> warn."""
    out = []
    if g.completeness < lim.completeness_fail:
        out.append(breach("completeness.low", "fail", subject, g.completeness, lim.completeness_fail, "completeness"))
    elif g.completeness < lim.completeness_warn:
        out.append(breach("completeness.low", "warn", subject, g.completeness, lim.completeness_warn, "completeness"))
    if g.n_gap_runs > lim.gap_run_warn:
        out.append(breach("completeness.gap_runs", "warn", subject, g.n_gap_runs, lim.gap_run_warn, "gap run count"))
    return tuple(out)
