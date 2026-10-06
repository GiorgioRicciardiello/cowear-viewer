"""Shared fixtures: the real pilot night, loaded and processed once per test session (slow tests only)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.config import AX6APPLE, PREPROCESSED_ROOT

VIEWER = AX6APPLE["viewer"]
PARQUET = {"ax6": PREPROCESSED_ROOT / AX6APPLE["sources"]["ax6"]["parquet"],
           "apple": PREPROCESSED_ROOT / AX6APPLE["sources"]["apple"]["parquet"]}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: regression anchors on the private pilot recordings (not distributed; run with -m slow and COWEAR_CONFIG)")


@pytest.fixture(scope="session")
def real_night():
    from ax6apple.align import fit_rotation
    from ax6apple.calibration import apply, calibrate
    from ax6apple.signals import build_grid, load_recording, recording_window

    if not all(p.exists() for p in PARQUET.values()):
        pytest.skip("source drive unreachable")
    start, end = recording_window()
    n_out = int((end - start).total_seconds() // VIEWER["epoch_s"]) * VIEWER["epoch_s"] * VIEWER["grid_hz"]
    night = {}
    guard = {"min_coverage": VIEWER["calibration_min_coverage"], "low_coverage": VIEWER["calibration_low_coverage"]}
    for device, path in PARQUET.items():
        rec = load_recording(path, start, end)
        cal = calibrate(rec.xyz, rec.fs, VIEWER["calibration_std_g"], VIEWER["still_window_s"], **guard)
        night[device] = SimpleNamespace(
            rec=rec, calibration=cal, xyz=rec.xyz,
            grid_raw=build_grid(rec.t_s, rec.xyz, rec.fs, VIEWER["grid_hz"], n_out),
            grid_cal=build_grid(rec.t_s, apply(rec.xyz, cal), rec.fs, VIEWER["grid_hz"], n_out))
    night["rotation"] = fit_rotation(
        night["ax6"].grid_cal.xyz, night["apple"].grid_cal.xyz, VIEWER["grid_hz"], VIEWER["still_window_s"],
        VIEWER["calibration_std_g"], int(VIEWER["bootstrap"]["block_min"] * 60 / VIEWER["still_window_s"]),
        VIEWER["bootstrap"]["resamples"], np.random.default_rng(VIEWER["bootstrap"]["seed"]))
    return night


@pytest.fixture(scope="session")
def gappy_night():
    """The real Apple timestamp series (the pilot night) with a 1.1 s window excised (D8, R4): a synthetic gap built from a
    real clock, not a synthetic one, so the NaN path (unreachable on this night as recorded) becomes testable."""
    from ax6apple.signals import Recording, Timing, load_recording, recording_window

    if not PARQUET["apple"].exists():
        pytest.skip("source drive unreachable")
    start, end = recording_window()
    rec = load_recording(PARQUET["apple"], start, end)
    mid = rec.t_s[len(rec.t_s) // 2]
    keep = (rec.t_s < mid) | (rec.t_s >= mid + 1.1)
    t_s, xyz = rec.t_s[keep], rec.xyz[keep]
    timing = Timing(len(t_s), rec.timing.reordered, rec.timing.duplicates_dropped, float(np.diff(t_s).max()))
    return Recording(t_s, xyz, rec.fs, timing)
