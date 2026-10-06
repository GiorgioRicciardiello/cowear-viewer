"""Agreement of MATLAB's automatic sleep detection (`auto_sleep/abk/<ID>.csv`, sleep bouts) between two runs.

The RBD features are computed on the detected sleep bouts, so a different sleep mask shifts every feature.
Bouts are put on a 30 s epoch grid anchored at the window start; a bout covers its start and end epochs
(MATLAB writes the start time of the last epoch as `annotation_end`: a one-epoch bout has end == start).
"""

from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = pd.Timedelta(seconds=30)


def load_bouts(path: Path) -> pd.DataFrame:
    a = pd.read_csv(path)
    return pd.DataFrame({"start": pd.to_datetime(a.annotation_start), "end": pd.to_datetime(a.annotation_end)})


def epoch_vector(bouts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """True for every 30 s epoch of [start, end) that lies inside a sleep bout."""
    n = int((end - start) / EPOCH)
    asleep = np.zeros(n, dtype=bool)
    first = np.round((bouts.start - start) / EPOCH).astype(int)
    last = np.round((bouts.end - start) / EPOCH).astype(int)
    for i, j in zip(first, last):
        if j >= 0 and i <= n - 1:  # a bout outside the window adds nothing; a partly overlapping one is clipped
            asleep[max(i, 0):min(j, n - 1) + 1] = True
    return asleep


def sleep_agreement(ref: np.ndarray, test: np.ndarray) -> dict:
    po = (ref == test).mean()
    pe = ref.mean() * test.mean() + (1 - ref.mean()) * (1 - test.mean())
    minutes = EPOCH.total_seconds() / 60
    return {"n_epochs": len(ref), "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
            "sleep_min_ref": ref.sum() * minutes, "sleep_min_test": test.sum() * minutes,
            "min_only_ref": (ref & ~test).sum() * minutes, "min_only_test": (~ref & test).sum() * minutes,
            "min_both": (ref & test).sum() * minutes}


def disagreement_by_hour(ref: np.ndarray, test: np.ndarray, start: pd.Timestamp) -> pd.DataFrame:
    """Minutes per clock hour that only the reference, or only the test, calls sleep."""
    minutes = EPOCH.total_seconds() / 60
    t = pd.date_range(start, periods=len(ref), freq=EPOCH)
    df = pd.DataFrame({"hour": t.floor("h"), "min_only_ref": (ref & ~test) * minutes, "min_only_test": (~ref & test) * minutes})
    return df.groupby("hour", as_index=False).sum()
