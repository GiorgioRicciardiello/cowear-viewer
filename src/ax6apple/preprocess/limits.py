"""Data-quality thresholds, config-backed.

Every number here is a threshold *on data quality*, read from `config.yaml`'s
`ax6apple.preprocess.limits`. Numerical-method constants that are part of an algorithm's own
identity (e.g. `calibration.PARAM_TOL`) stay in Python, not here.
"""

from dataclasses import dataclass, fields
from typing import Literal, Mapping

from ax6apple.config import AX6APPLE

Level = Literal["info", "warn", "fail"]
Status = Literal["ok", "marginal", "unusable"]


@dataclass(frozen=True)
class Limits:
    """Data-quality thresholds, from config `ax6apple.preprocess.limits`."""

    completeness_warn: float
    completeness_fail: float
    gap_run_warn: int
    rate_error_warn: float
    jitter_warn: float
    reordered_frac_warn: float
    duplicate_frac_warn: float
    mean_mag_g: tuple[float, float]
    axis_sd_g: tuple[float, float]
    zero_run_frac_warn: float
    coverage_warn: float
    rms_warn_mg: float
    lag_r_warn: float
    rotation_residual_warn_deg: float
    rotation_uncertainty_warn_deg: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, object] | None = None) -> "Limits":
        """`cfg` is the `ax6apple.preprocess.limits` mapping itself; `None` reads it from `config.yaml`."""
        if cfg is None:
            cfg = AX6APPLE["preprocess"]["limits"]
        kwargs = {}
        for f in fields(cls):
            v = cfg[f.name]
            kwargs[f.name] = tuple(v) if isinstance(v, (list, tuple)) else v
        return cls(**kwargs)

    def to_dict(self) -> dict[str, object]:
        """The inverse of `from_config`, for round-tripping into a report."""
        return {f.name: list(v) if isinstance(v := getattr(self, f.name), tuple) else v for f in fields(self)}
