from __future__ import annotations

import numpy as np

from loganomaly.events import Event
from loganomaly.features import (
    FEATURE_NAMES,
    FEATURES,
    Tracker,
    Window,
    applies,
    featurize,
    make_windows,
    tokens,
)


def ssh(
    ts: float, kind: str = "ssh_failed_password", ip: str = "203.0.113.9", user: str = "root"
) -> Event:
    return Event(ts=ts, source="auth", kind=kind, host="web01", ip=ip, user=user, raw="x")


def test_feature_names_unique_and_match_matrix() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES) == len(FEATURES)
    windows, matrix = featurize([ssh(10), ssh(20)], 300, Tracker())
    assert matrix.shape == (len(windows), len(FEATURES))
    assert np.isfinite(matrix).all()


def test_windows_group_by_entity_and_time() -> None:
    events = [ssh(10), ssh(20), ssh(400), ssh(15, ip="198.51.100.1")]
    windows = make_windows(events, 300)
    assert len(windows) == 3
    assert [len(w.events) for w in windows if w.entity == "ip:203.0.113.9"] == [2, 1]
    assert all(e.ts >= w.start and e.ts < w.start + 300 for w in windows for e in w.events)


def test_empty_input() -> None:
    windows, matrix = featurize([], 300, Tracker())
    assert windows == []
    assert matrix.shape == (0, len(FEATURES))


def test_novelty_features_fire_once() -> None:
    tracker = Tracker()
    events = [
        ssh(10, "ssh_accepted_key", user="alice"),
        ssh(1000, "ssh_accepted_key", user="alice"),
    ]
    _, matrix = featurize(events, 300, tracker)
    col = FEATURE_NAMES.index("novel_source")
    assert matrix[0, col] == 1.0
    assert matrix[1, col] == 0.0
    assert ("web01", "alice") in tracker.users


def test_a_window_does_not_see_itself_in_the_tracker() -> None:
    tracker = Tracker()
    _, first = featurize([ssh(10, "ssh_accepted_key", user="alice")], 300, tracker)
    assert first[0, FEATURE_NAMES.index("novel_pair")] == 1.0


def test_tracker_roundtrip_and_copy() -> None:
    tracker = Tracker()
    featurize(
        [
            ssh(10, "ssh_accepted_key", user="alice"),
            ssh(5000, "ssh_accepted_key", user="bob", ip="192.0.2.8"),
        ],
        300,
        tracker,
    )
    clone = Tracker.from_json(tracker.to_json())
    assert clone.to_json() == tracker.to_json()
    copy = tracker.copy()
    copy.sources.add("1.1.1.1")
    assert "1.1.1.1" not in tracker.sources


def test_tracker_is_bounded() -> None:
    from loganomaly import features

    tracker = Tracker()
    for i in range(features.MAX_TRACKED + 50):
        tracker.update(Window(f"ip:10.{i // 65536}.{i // 256 % 256}.{i % 256}", 0, []))
    assert len(tracker.sources) == features.MAX_TRACKED


def test_applies_filters_by_source() -> None:
    from loganomaly.features import _LOGIN_COLUMNS, _WEB_COLUMNS

    web = Window("ip:1.2.3.4", 0, [Event(1, "web", "web_ok", "h", "1.2.3.4", path="/")])
    login = Window("ip:1.2.3.4", 0, [ssh(1)])
    by_name = {f.name: f for f in FEATURES}
    for name in _WEB_COLUMNS:
        assert applies(by_name[name], web)
        assert not applies(by_name[name], login)
    for name in _LOGIN_COLUMNS:
        assert applies(by_name[name], login)
        assert not applies(by_name[name], web)
    assert applies(FEATURES[0], web)


def test_tokens_encode_gaps() -> None:
    window = Window("ip:1.2.3.4", 0, [ssh(0), ssh(1), ssh(10), ssh(100)])
    seq = tokens(window)
    assert len(seq) == 4
    assert len(set(seq)) == 3  # the first event and a slow one share a token by design
    assert seq[0] == seq[3]
