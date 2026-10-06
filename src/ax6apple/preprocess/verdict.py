"""Per-device and alignment quality assembly, and capability gating.

Surfaces calibration and rotation/lag confidence that already exists but is never judged, and
recovers the counts section 0.2 discards.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ax6apple.align import Lag, Rotation
from ax6apple.calibration import Calibration
from ax6apple.signals import Recording, Timing

from .completeness import GapStats, judge as judge_gaps
from .preflight import preflight as run_preflight
from .findings import Finding, breach
from .limits import Limits


@dataclass(frozen=True)
class DeviceQuality:
    dev: str
    fs_nominal: int
    fs_median: float             # the un-rounded rate
    duration_s: float
    timing: Timing                # as computed today
    gaps: GapStats
    calibration: Calibration      # as computed today, now judged
    noise_floor_mg: float
    noise_floor_windows: int
    findings: tuple[Finding, ...]


def _judge_calibration(cal: Calibration, subject: str, lim: Limits) -> tuple[Finding, ...]:
    out = []
    if not cal.converged:
        out.append(Finding("calibration.not_converged", "warn", subject,
                           f"{subject}: calibration fit did not converge within its iteration cap"))
    if cal.low_orientation_coverage:
        out.append(Finding("calibration.low_coverage", "warn", subject,
                           f"{subject}: calibration orientation coverage is low ({cal.coverage:.3g})"))
    if cal.coverage < lim.coverage_warn:
        out.append(breach("calibration.coverage", "warn", subject, cal.coverage, lim.coverage_warn, "orientation coverage"))
    rms_mg = cal.rms_g * 1000
    if rms_mg > lim.rms_warn_mg:
        out.append(breach("calibration.rms", "warn", subject, rms_mg, lim.rms_warn_mg, "calibration RMS", " mg"))
    return tuple(out)


def device_quality(dev: str, rec: Recording, xyz: np.ndarray, cal: Calibration, noise: dict, grid_hz: int, n_out: int,
                   max_gap_s: float, lim: Limits) -> DeviceQuality:
    """Assembles one device's quality from values the build already has in hand."""
    from .completeness import gap_stats

    d = np.diff(rec.t_s)
    fs_median = 1.0 / float(np.median(d[d > 0])) if (d > 0).any() else float(rec.fs)
    gaps = gap_stats(rec.t_s, grid_hz, n_out, max_gap_s)
    findings = run_preflight(rec, xyz, dev, lim) + judge_gaps(gaps, dev, lim) + _judge_calibration(cal, dev, lim)
    return DeviceQuality(dev=dev, fs_nominal=rec.fs, fs_median=fs_median, duration_s=float(rec.t_s[-1] - rec.t_s[0]),
                         timing=rec.timing, gaps=gaps, calibration=cal, noise_floor_mg=noise["value_mg"],
                         noise_floor_windows=noise["n_windows"], findings=findings)


@dataclass(frozen=True)
class AlignmentQuality:
    lag_s: float
    lag_r: float
    lag_at_boundary: bool
    rotation_residual_deg: float
    rotation_uncertainty_deg: float
    rotation_n_windows: int
    rotation_n_examined: int     # see F11; today only the kept count survives on `Rotation` itself
    findings: tuple[Finding, ...]


def alignment_quality(lag: Lag, rot: Rotation, n_examined: int, lim: Limits) -> AlignmentQuality:
    """Judges lag r, lag boundary, rotation residual and rotation uncertainty."""
    findings = []
    if lag.r < lim.lag_r_warn:
        findings.append(breach("lag.weak_r", "warn", "pair", lag.r, lim.lag_r_warn, "clock-lag cross-correlation peak r"))
    if lag.at_boundary:
        findings.append(Finding("lag.at_boundary", "warn", "pair",
                                "the clock-lag peak sits at the edge of the search range: the true lag may lie outside it"))
    if rot.residual_deg > lim.rotation_residual_warn_deg:
        findings.append(breach("rotation.residual", "warn", "pair", rot.residual_deg, lim.rotation_residual_warn_deg,
                               "rotation RMS residual", " deg"))
    if rot.uncertainty_deg > lim.rotation_uncertainty_warn_deg:
        findings.append(breach("rotation.uncertainty", "warn", "pair", rot.uncertainty_deg,
                               lim.rotation_uncertainty_warn_deg, "rotation bootstrap uncertainty", " deg"))
    return AlignmentQuality(lag_s=lag.lag_s, lag_r=lag.r, lag_at_boundary=lag.at_boundary,
                            rotation_residual_deg=rot.residual_deg, rotation_uncertainty_deg=rot.uncertainty_deg,
                            rotation_n_windows=rot.n_windows, rotation_n_examined=n_examined, findings=tuple(findings))


@dataclass(frozen=True)
class Capabilities:
    """What this night's quality permits. Each is independently gated (D4).

    Mechanical statements about *this pipeline's* ability to compute something on this night, never
    a verdict on either device's quality (R3): field names and messages must stay mechanical.
    """

    enmo_agreement: bool       # needs both devices' completeness + calibration
    axis_comparison: bool      # additionally needs the rotation to fit -> False on the pilot night
    spectral_comparison: bool  # needs completeness + noise floors, NOT the rotation
    per_epoch_lag: bool        # needs lag_r above its limit


def capabilities(devices: Mapping[str, DeviceQuality], al: AlignmentQuality, lim: Limits) -> Capabilities:
    def device_ok(dq: DeviceQuality) -> bool:
        return not any(f.level == "fail" for f in dq.findings)

    both_ok = all(device_ok(dq) for dq in devices.values())
    rotation_fits = not any(f.code.startswith("rotation.") for f in al.findings)
    lag_ok = not any(f.code == "lag.weak_r" for f in al.findings)
    return Capabilities(enmo_agreement=both_ok, axis_comparison=both_ok and rotation_fits,
                        spectral_comparison=both_ok, per_epoch_lag=lag_ok)
