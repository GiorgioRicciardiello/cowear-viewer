"""`ax6apple.preprocess.report`: QualityReport assembly and its JSON / CSV writers."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.align import Lag, Rotation
from ax6apple.calibration import Calibration
from ax6apple.preprocess.buildlog import Step
from ax6apple.preprocess.limits import Limits
from ax6apple.preprocess.report import QualityReport, status_of, to_jsonable, write
from ax6apple.preprocess.verdict import alignment_quality, capabilities, device_quality
from ax6apple.signals import Recording, Timing

LIM = Limits.from_config()


def _report() -> QualityReport:
    n = 3600
    t = np.arange(n) / 50.0
    xyz = np.random.default_rng(0).normal(0, 0.01, (n, 3)) + [0, 0, 1.0]
    rec = Recording(t, xyz, 50, Timing(n, 3, 1, 0.02))
    cal = Calibration(gain=np.array([1.001, 0.989, 0.999]), offset=np.array([-0.0375, 0.0524, -0.0051]), n_windows=100,
                      n_retained=95, coverage=1.786, residual=7.5e-9, iterations=42, rms_g=0.0029, converged=True,
                      low_orientation_coverage=False)
    devices = {d: device_quality(d, rec, xyz, cal, {"value_mg": 1.18, "n_windows": 40}, 30, n * 30 // 50, 1.0, LIM)
              for d in ("ax6", "apple")}
    al = alignment_quality(Lag(3.18, 0.45, False), Rotation(np.eye(3), 3279, 56.8, 14.9), 3300, LIM)
    caps = capabilities(devices, al, LIM)
    steps = (Step("load:ax6", "ok", 0.1), Step("load:apple", "ok", 0.2))
    findings = tuple(f for dq in devices.values() for f in dq.findings) + al.findings
    return QualityReport(built_at="2026-07-14T00:00:00+00:00", window=("2026-07-13T21:53:37.540", "2026-07-14T07:53:57.562"),
                        window_variant="v2_ax6_trim_csv", devices=devices, alignment=al, capabilities=caps, limits=LIM,
                        steps=steps, status=status_of(findings))


def test_quality_report_findings_combines_devices_then_alignment_worst_first():
    report = _report()
    findings = report.findings
    assert len(findings) > 0
    levels = [f.level for f in findings]
    rank = {"fail": 0, "warn": 1, "info": 2}
    assert levels == sorted(levels, key=lambda l: rank[l])


def test_status_is_marginal_when_only_warnings_are_present():
    report = _report()
    assert report.status in ("ok", "marginal", "unusable")
    assert report.status == "marginal"      # the rotation residual/uncertainty both warn on this fixture


def test_quality_report_json_round_trips_structurally():
    """Contract test (section 8 #3): QualityReport -> JSON -> parsed back is equal."""
    report = _report()
    before = to_jsonable(report)
    after = json.loads(json.dumps(before))
    assert before == after


def test_write_produces_a_findings_csv_with_one_row_per_finding(tmp_path):
    report = _report()
    write(report, tmp_path)
    assert (tmp_path / "quality.json").exists() and (tmp_path / "findings.csv").exists()
    with (tmp_path / "findings.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(report.findings)
    assert {"code", "level", "subject", "message", "value", "threshold"} <= set(rows[0]) if rows else True
    loaded = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert loaded["status"] == report.status
    assert loaded["limits"]["rotation_residual_warn_deg"] == LIM.rotation_residual_warn_deg


def test_limits_round_trip_through_to_dict_and_from_config():
    """Contract test (section 8 #2 support): Limits round-trips."""
    assert Limits.from_config(LIM.to_dict()) == LIM
