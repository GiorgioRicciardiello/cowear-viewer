"""One judged observation, and the single place a threshold breach is worded."""

from dataclasses import dataclass

from .limits import Level


@dataclass(frozen=True)
class Finding:
    """One judged observation. `code` is stable and greppable; `message` is for humans."""

    code: str                      # "completeness.low", "rotation.residual", ...
    level: Level
    subject: str                   # "ax6" | "apple" | "pair"
    message: str
    value: float | None = None
    threshold: float | None = None


def breach(code: str, level: Level, subject: str, value: float, threshold: float, what: str, unit: str = "") -> Finding:
    """The single place a threshold breach is worded, so every message reads the same."""
    cmp = ">" if value > threshold else "<"
    message = f"{subject}: {what} is {value:.4g}{unit}, {cmp} the {level} limit of {threshold:.4g}{unit}"
    return Finding(code, level, subject, message, float(value), float(threshold))
