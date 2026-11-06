"""From events to numbers: one feature vector per (entity, time window).

An *entity* is a source address (`ip:203.0.113.5`) or, for events with no address such as sudo
or a local Windows logon, an account on a host (`user:alice@web01`). A *window* is a fixed slice
of time, five minutes by default. Everything one entity did in one window becomes one row.

The features are generic on purpose: how many events of each kind, how fast, how varied, how
many errors, how odd the request paths look, whether the address or account was ever seen
before. There are no attack signatures here; `sec-mini-siem` has those. The models learn what
the normal rows look like and flag the rows that do not.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .events import (
    FAILED_LOGIN_KINDS,
    KINDS,
    SUCCESS_LOGIN_KINDS,
    Event,
    token,
)
from .net import is_private

DEFAULT_WINDOW = 300  # seconds
MAX_TOKENS = 64  # events per window the sequence model reads
MAX_TRACKED = 200_000  # entries kept per novelty set, so memory stays bounded
_SPECIAL = frozenset("'\"<>;()|`\\{}$")
_WEB_ERRORS = frozenset(
    {"web_denied", "web_notfound", "web_client_error", "web_server_error", "web_write_failed"}
)
_WEB_WRITES = frozenset({"web_write_ok", "web_write_failed"})
_WEB_KINDS = frozenset(k for k in KINDS if k.startswith("web_"))
_MIN_HOURS = 10  # logins needed before "unusual hour" means anything for an account
_RARE_HOUR = 0.02


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A column of the feature matrix. `log` columns hold log1p of a count."""

    name: str
    log: bool = False
    help: str = ""


_EXTRA: tuple[FeatureSpec, ...] = (
    FeatureSpec("events", True, "events in the window"),
    FeatureSpec("sources", False, "log families seen (auth, web, windows)"),
    FeatureSpec("distinct_users", True, "different accounts named"),
    FeatureSpec("distinct_paths", True, "different request paths"),
    FeatureSpec("distinct_hosts", True, "different hosts touched"),
    FeatureSpec("fail_ratio", False, "failed share of all login attempts"),
    FeatureSpec("web_error_ratio", False, "share of web requests that failed"),
    FeatureSpec("notfound_ratio", False, "share of web requests answered 404"),
    FeatureSpec("write_ratio", False, "share of web requests that are POST/PUT/DELETE"),
    FeatureSpec("gap_median", True, "median seconds between events"),
    FeatureSpec("gap_min", True, "shortest gap in seconds between events"),
    FeatureSpec("burst", True, "most events in any 10 seconds"),
    FeatureSpec("path_length", False, "mean request path length"),
    FeatureSpec("path_entropy", False, "mean character entropy of request paths (bits)"),
    FeatureSpec("path_special", False, "share of path characters used in injection and traversal"),
    FeatureSpec("agent_missing", False, "share of web requests without a User-Agent"),
    FeatureSpec("agent_distinct", True, "different User-Agents"),
    FeatureSpec("hour_sin", False, "time of day, sine"),
    FeatureSpec("hour_cos", False, "time of day, cosine"),
    FeatureSpec("private_source", False, "1 when the source address is private"),
    FeatureSpec("novel_source", False, "1 when this address was never seen before"),
    FeatureSpec("novel_user", False, "share of events naming an account not seen on that host"),
    FeatureSpec(
        "novel_pair", False, "share of successful logins from an address new to the account"
    ),
    FeatureSpec("odd_hour_login", False, "share of logins at an hour the account never uses"),
)
FEATURES: tuple[FeatureSpec, ...] = (
    *(FeatureSpec(f"n_{kind}", True, f"events of kind {kind}") for kind in KINDS),
    *_EXTRA,
)
FEATURE_NAMES = tuple(f.name for f in FEATURES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


_WEB_COLUMNS = frozenset({"distinct_paths", "web_error_ratio", "notfound_ratio", "write_ratio",
                          "path_length", "path_entropy", "path_special", "agent_missing",
                          "agent_distinct"})  # fmt: skip
_LOGIN_COLUMNS = frozenset({"fail_ratio", "novel_pair", "odd_hour_login"})


def applies(spec: FeatureSpec, window: Window) -> bool:
    """Whether a column says anything about this window (a web column needs web events)."""
    if spec.name in _WEB_COLUMNS:
        return any(e.kind in _WEB_KINDS for e in window.events)
    if spec.name in _LOGIN_COLUMNS:
        return any(e.kind in FAILED_LOGIN_KINDS | SUCCESS_LOGIN_KINDS for e in window.events)
    return True


@dataclass
class Window:
    """Everything one entity did in one slice of time."""

    entity: str
    start: float
    events: list[Event]


def make_windows(events: list[Event], size: int = DEFAULT_WINDOW) -> list[Window]:
    """Group events into (entity, window); sorted by time, then entity."""
    groups: dict[tuple[float, str], list[Event]] = defaultdict(list)
    for event in events:
        groups[(math.floor(event.ts / size) * size, event.entity)].append(event)
    return [
        Window(entity=entity, start=start, events=sorted(group, key=lambda e: e.ts))
        for (start, entity), group in sorted(groups.items())
    ]


class Tracker:
    """What has been seen so far: addresses, accounts per host, (account, address) pairs and
    the hours each account logs in. It is what turns "first time" into a number, and it is
    saved with the model so detection continues from where training stopped."""

    def __init__(self) -> None:
        self.sources: set[str] = set()
        self.users: set[tuple[str, str]] = set()
        self.pairs: set[tuple[str, str]] = set()
        self.hours: dict[str, list[int]] = {}

    def copy(self) -> Tracker:
        clone = Tracker()
        clone.sources = set(self.sources)
        clone.users = set(self.users)
        clone.pairs = set(self.pairs)
        clone.hours = {user: list(counts) for user, counts in self.hours.items()}
        return clone

    @staticmethod
    def _add(target: set[Any], item: tuple[str, str] | str) -> None:
        if len(target) < MAX_TRACKED:
            target.add(item)

    def update(self, window: Window) -> None:
        if window.entity.startswith("ip:"):
            self._add(self.sources, window.entity[3:])
        for event in window.events:
            if event.user and event.source != "web":
                self._add(self.users, (event.host, event.user))
            if event.kind in SUCCESS_LOGIN_KINDS and event.user:
                if event.ip:
                    self._add(self.pairs, (event.user, event.ip))
                if event.user in self.hours or len(self.hours) < MAX_TRACKED:
                    counts = self.hours.setdefault(event.user, [0] * 24)
                    counts[int(event.ts // 3600) % 24] += 1

    def to_json(self) -> dict[str, object]:
        return {
            "sources": sorted(self.sources),
            "users": sorted([h, u] for h, u in self.users),
            "pairs": sorted([u, i] for u, i in self.pairs),
            "hours": self.hours,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> Tracker:
        tracker = cls()
        tracker.sources = {str(s) for s in data["sources"]}  # type: ignore[attr-defined]
        tracker.users = {(str(h), str(u)) for h, u in data["users"]}  # type: ignore[attr-defined]
        tracker.pairs = {(str(u), str(i)) for u, i in data["pairs"]}  # type: ignore[attr-defined]
        tracker.hours = {
            str(u): [int(c) for c in counts]
            for u, counts in data["hours"].items()  # type: ignore[attr-defined]
        }
        return tracker


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def _row(window: Window, tracker: Tracker, size: int) -> list[float]:
    events = window.events
    n = len(events)
    kinds = Counter(e.kind for e in events)
    row = [math.log1p(kinds.get(kind, 0)) for kind in KINDS]

    web = [e for e in events if e.kind in _WEB_KINDS]
    web_n = len(web)
    failed = sum(kinds[k] for k in FAILED_LOGIN_KINDS)
    succeeded = sum(kinds[k] for k in SUCCESS_LOGIN_KINDS)
    paths = [e.path for e in web if e.path]
    path_chars = sum(len(p) for p in paths)
    agents = {e.agent for e in web if e.agent and e.agent != "-"}

    stamps = [e.ts for e in events]
    gaps = np.diff(stamps) if n > 1 else np.array([float(size)])
    buckets = Counter(int(t // 10) for t in stamps)

    hour = (window.start % 86400) / 3600
    logins = [e for e in events if e.kind in SUCCESS_LOGIN_KINDS and e.user]
    pair_new = sum(1 for e in logins if e.ip and (e.user, e.ip) not in tracker.pairs)
    pair_total = sum(1 for e in logins if e.ip)
    odd = 0
    for e in logins:
        counts = tracker.hours.get(e.user or "")
        if counts and sum(counts) >= _MIN_HOURS:
            odd += counts[int(e.ts // 3600) % 24] / sum(counts) < _RARE_HOUR
    named = [e for e in events if e.user and e.source != "web"]
    ip = window.entity[3:] if window.entity.startswith("ip:") else None

    row += [
        math.log1p(n),
        float(len({e.source for e in events})),
        math.log1p(len({e.user for e in events if e.user})),
        math.log1p(len(set(paths))),
        math.log1p(len({e.host for e in events if e.host})),
        failed / (failed + succeeded) if failed + succeeded else 0.0,
        sum(kinds[k] for k in _WEB_ERRORS) / web_n if web_n else 0.0,
        kinds["web_notfound"] / web_n if web_n else 0.0,
        sum(kinds[k] for k in _WEB_WRITES) / web_n if web_n else 0.0,
        math.log1p(float(np.median(gaps))),
        math.log1p(float(np.min(gaps))),
        math.log1p(max(buckets.values())),
        path_chars / len(paths) if paths else 0.0,
        sum(_entropy(p) for p in paths) / len(paths) if paths else 0.0,
        (sum(c in _SPECIAL for p in paths for c in p) + 2 * sum(p.count("..") for p in paths))
        / path_chars
        if path_chars
        else 0.0,
        sum(1 for e in web if not e.agent or e.agent == "-") / web_n if web_n else 0.0,
        math.log1p(len(agents)),
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        1.0 if ip and is_private(ip) else 0.0,
        1.0 if ip and ip not in tracker.sources else 0.0,
        sum(1 for e in named if (e.host, e.user or "") not in tracker.users) / len(named)
        if named
        else 0.0,
        pair_new / pair_total if pair_total else 0.0,
        odd / len(logins) if logins else 0.0,
    ]
    return row


def featurize(
    events: list[Event], size: int, tracker: Tracker
) -> tuple[list[Window], NDArray[np.float64]]:
    """Feature matrix for `events`. The tracker is updated as time advances, so a window sees
    only what happened before it. Pass a copy when the original must stay untouched."""
    windows = make_windows(events, size)
    matrix = np.zeros((len(windows), len(FEATURES)))
    for i, window in enumerate(windows):
        matrix[i] = _row(window, tracker, size)
        tracker.update(window)
    return windows, matrix


def tokens(window: Window) -> list[int]:
    """The window's events as (kind, gap) tokens in time order, for the sequence model."""
    out: list[int] = []
    previous: float | None = None
    for event in window.events[:MAX_TOKENS]:
        out.append(token(event.kind, None if previous is None else event.ts - previous))
        previous = event.ts
    return out
