"""The build step recorder: the one mutable type (D5)."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal


@dataclass(frozen=True)
class Step:
    name: str
    status: Literal["ok", "failed", "skipped"]
    seconds: float
    detail: str | None = None


class BuildLog:
    """Records each build step's outcome and duration. The one mutable type (D5)."""

    def __init__(self) -> None:
        self._rows: list[Step] = []

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Times the block. On exception: records 'failed' with the exception text, then re-raises."""
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self._rows.append(Step(name, "failed", time.perf_counter() - start, str(exc)))
            raise
        else:
            self._rows.append(Step(name, "ok", time.perf_counter() - start))

    def skip(self, name: str, why: str) -> None:
        self._rows.append(Step(name, "skipped", 0.0, why))

    def rows(self) -> tuple[Step, ...]:
        return tuple(self._rows)
