"""QualityReport assembly and its JSON / CSV writers."""

import csv
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from .buildlog import Step
from .findings import Finding
from .limits import Limits, Status
from .verdict import AlignmentQuality, Capabilities, DeviceQuality

_LEVEL_RANK: dict[str, int] = {"fail": 0, "warn": 1, "info": 2}


@dataclass(frozen=True)
class QualityReport:
    built_at: str                        # ISO, UTC
    window: tuple[str, str]
    window_variant: str
    devices: Mapping[str, DeviceQuality]
    alignment: AlignmentQuality
    capabilities: Capabilities
    limits: Limits                       # the thresholds this verdict was reached under
    steps: tuple[Step, ...]              # from BuildLog
    status: Literal["ok", "marginal", "unusable"]

    @property
    def findings(self) -> tuple[Finding, ...]:
        """All findings, device then alignment, worst level first."""
        out: list[Finding] = []
        for dq in self.devices.values():
            out.extend(dq.findings)
        out.extend(self.alignment.findings)
        return tuple(sorted(out, key=lambda f: _LEVEL_RANK.get(f.level, 9)))


def status_of(report_findings: tuple[Finding, ...]) -> Status:
    """`fail` anywhere -> unusable; `warn` anywhere -> marginal; else ok."""
    if any(f.level == "fail" for f in report_findings):
        return "unusable"
    if any(f.level == "warn" for f in report_findings):
        return "marginal"
    return "ok"


def to_jsonable(obj: object) -> object:
    """A nested plain dict/list/scalar copy of `obj`: dataclasses and NamedTuples become dicts, arrays become lists."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):      # a NamedTuple
        return {k: to_jsonable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Mapping):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write(report: QualityReport, out_dir: Path) -> None:
    """`quality.json` (full, nested) and `findings.csv` (flat, one row per finding).

    CSV because a flat findings table is what gets scanned across nights once this is a cohort;
    JSON because the nested report is what the viewer reads.
    """
    out_dir = Path(out_dir)
    (out_dir / "quality.json").write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    with (out_dir / "findings.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "level", "subject", "message", "value", "threshold"])
        for f in report.findings:
            writer.writerow([f.code, f.level, f.subject, f.message, f.value, f.threshold])
