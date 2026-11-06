from __future__ import annotations

import io

import numpy as np
import pytest

from loganomaly.events import (
    GAP_NAMES,
    KIND_INDEX,
    KINDS,
    N_GAP,
    VOCAB,
    Event,
    describe_token,
    token,
)
from loganomaly.net import clean_ip, is_private
from loganomaly.sources import ReadStats, read_all, read_events


def test_vocabulary_is_consistent() -> None:
    assert len(set(KINDS)) == len(KINDS) == 27
    assert len(KINDS) * N_GAP + 1 == VOCAB
    seen = {token(k, g) for k in KINDS for g in (0.5, 10.0, 100.0, None)}
    assert len(seen) == len(KINDS) * N_GAP
    for kind in KINDS:
        assert token(kind, None) == KIND_INDEX[kind] * N_GAP + N_GAP - 1
        assert token(kind, 0.5) < token(kind, 10.0) < token(kind, 100.0)
        assert kind in describe_token(token(kind, 0.5))
    assert len(GAP_NAMES) == N_GAP
    assert describe_token(VOCAB - 1) == "the start of the window"


def test_event_entity() -> None:
    web = Event(ts=1.0, source="web", kind="web_ok", host="h", ip="1.2.3.4")
    login = Event(
        ts=1.0, source="auth", kind="ssh_failed_password", host="h", ip="1.2.3.4", user="bob"
    )
    assert web.entity == "ip:1.2.3.4"
    assert login.entity == "ip:1.2.3.4"
    local = Event(ts=1.0, source="auth", kind="sudo_command", host="h", user="bob")
    assert local.entity == "user:bob@h"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("203.0.113.5", "203.0.113.5"),
        ("-", None),
        ("", None),
        (None, None),
        ("not-an-ip", None),
        ("999.1.1.1", None),
    ],
)
def test_clean_ip(raw: str | None, expected: str | None) -> None:
    assert clean_ip(raw) == expected


def test_is_private() -> None:
    assert is_private("10.1.2.3")
    assert is_private("192.168.0.1")
    assert not is_private("8.8.8.8")


def test_read_events_counts_and_skips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "auth.log"
    path.write_text(
        "Oct  5 03:04:05 h sshd[1]: Failed password for root from 203.0.113.9 port 22 ssh2\n"
        "junk line\n"
        "\n"
    )
    stats = ReadStats()
    events = read_events("auth", str(path), stats, year=2026)
    assert len(events) == 1
    assert stats.skipped >= 1


def test_read_all_missing_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        read_all([("auth", str(tmp_path / "nope.log"))])


def test_read_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    line = '198.51.100.250 - - [05/Oct/2026:00:00:00 +0000] "GET / HTTP/1.1" 200 7139 "-" "x"\n'
    monkeypatch.setattr("sys.stdin", io.StringIO(line))
    events, _ = read_all([("web", "-")])
    assert len(events) == 1


def test_scaler_handles_constant_and_sparse_columns() -> None:
    from loganomaly.models.base import Scaler

    x = np.zeros((100, 3))
    x[:, 1] = np.arange(100)
    x[:3, 2] = 5.0
    scaler = Scaler.fit(x)
    z = scaler.transform(x)
    assert np.isfinite(z).all()
    assert np.abs(scaler.transform(np.array([[1e9, 0.0, 0.0]]))).max() <= 1e4
