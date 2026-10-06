"""Preprocessing quality layer: public API only, no logic.

Answers "is this night usable, and how do we know" beside the existing signal chain
(`ax6apple.{signals,calibration,align,spectral}`), without changing any of it (D7).
"""

from .buildlog import BuildLog, Step
from .completeness import GapStats, gap_runs, gap_stats
from .findings import Finding, breach
from .limits import Level, Limits, Status
from .preflight import check_amplitude, check_rate, check_timing, preflight
from .report import QualityReport, write
from .verdict import AlignmentQuality, Capabilities, DeviceQuality, alignment_quality, capabilities, device_quality

__all__ = [
    "AlignmentQuality", "BuildLog", "Capabilities", "DeviceQuality", "Finding", "GapStats", "Level", "Limits",
    "QualityReport", "Status", "Step", "alignment_quality", "breach", "capabilities", "check_amplitude",
    "check_rate", "check_timing", "device_quality", "gap_runs", "gap_stats", "preflight", "write",
]
