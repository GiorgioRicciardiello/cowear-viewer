"""`ax6apple.preprocess.buildlog`: the step recorder (the one mutable type, D5)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.preprocess.buildlog import BuildLog


def test_step_records_ok_with_a_positive_duration():
    log = BuildLog()
    with log.step("load:ax6"):
        pass
    rows = log.rows()
    assert len(rows) == 1
    assert rows[0].name == "load:ax6" and rows[0].status == "ok" and rows[0].seconds >= 0 and rows[0].detail is None


def test_step_records_failed_and_re_raises():
    """Contract test (section 8 #4)."""
    log = BuildLog()
    with pytest.raises(ValueError, match="boom"):
        with log.step("fit:rotation"):
            raise ValueError("boom")
    rows = log.rows()
    assert len(rows) == 1
    assert rows[0].status == "failed" and "boom" in rows[0].detail


def test_skip_records_a_skipped_row_with_no_timing():
    log = BuildLog()
    log.skip("spectral", "disabled for this night")
    assert log.rows() == (log.rows()[0],)
    assert log.rows()[0].status == "skipped" and log.rows()[0].seconds == 0.0 and log.rows()[0].detail == "disabled for this night"


def test_rows_accumulate_in_order_and_are_a_tuple():
    log = BuildLog()
    with log.step("a"):
        pass
    log.skip("b", "n/a")
    try:
        with log.step("c"):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    rows = log.rows()
    assert isinstance(rows, tuple)
    assert [r.name for r in rows] == ["a", "b", "c"]
    assert [r.status for r in rows] == ["ok", "skipped", "failed"]


def test_a_clean_build_has_twenty_steps():
    """Phase 2 Done-when: `rows()` has 20 entries on a clean build."""
    log = BuildLog()
    names = ["load:ax6", "load:apple", "sleep_flags", "grid:raw:ax6", "grid:raw:apple", "grid:cal:ax6", "grid:cal:apple",
             "lag", "rotation", "variant:calibrated", "variant:raw", "spectral:vm", "spectral:noise", "spectral:epochs",
             "spectral:summary", "chunks:all", "chunks:spectrogram", "manifest", "epoch_rows", "write_json"]
    assert len(names) == 20
    for name in names:
        with log.step(name):
            pass
    assert len(log.rows()) == 20
    assert all(r.status == "ok" for r in log.rows())
