"""Reading log files into events, with a count of what could not be understood."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .events import Event
from .parsers import MAX_LINE, PARSERS


@dataclass
class ReadStats:
    """How many lines were read, kept and skipped, per log format."""

    lines: dict[str, int] = field(default_factory=dict)
    events: dict[str, int] = field(default_factory=dict)

    def add(self, fmt: str, lines: int, events: int) -> None:
        self.lines[fmt] = self.lines.get(fmt, 0) + lines
        self.events[fmt] = self.events.get(fmt, 0) + events

    @property
    def skipped(self) -> int:
        return sum(self.lines.values()) - sum(self.events.values())


def _lines(path: str) -> Iterator[str]:
    """Lines of a file or of standard input ("-"). Invalid bytes never stop the read."""
    if path == "-":
        for line in sys.stdin:
            yield line
        return
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            yield line


def read_events(
    fmt: str, path: str, stats: ReadStats, *, tz: str = "UTC", year: int | None = None
) -> list[Event]:
    """Parse one file. `fmt` is auth, journald, web or windows."""
    parser = PARSERS[fmt]
    events: list[Event] = []
    count = 0
    for line in _lines(path):
        if not line.strip():
            continue
        count += 1
        if len(line) > MAX_LINE * 8:
            continue  # a line this long is not a log line
        event = parser(line, tz=tz, year=year)
        if event is not None:
            events.append(event)
    stats.add(fmt, count, len(events))
    return events


def read_all(
    inputs: Iterable[tuple[str, str]], *, tz: str = "UTC", year: int | None = None
) -> tuple[list[Event], ReadStats]:
    """Parse several (format, path) inputs and return the events sorted by time."""
    stats = ReadStats()
    events: list[Event] = []
    for fmt, path in inputs:
        if path != "-" and not Path(path).is_file():
            raise FileNotFoundError(path)
        events.extend(read_events(fmt, path, stats, tz=tz, year=year))
    events.sort(key=lambda e: e.ts)
    return events, stats
