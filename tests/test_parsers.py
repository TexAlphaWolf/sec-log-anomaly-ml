from __future__ import annotations

from datetime import UTC, datetime

import pytest

from loganomaly.parsers import (
    parse_auth_line,
    parse_journald_line,
    parse_web_line,
    parse_windows_line,
)

NOW = datetime(2026, 10, 20, tzinfo=UTC)


def auth(msg: str, prog: str = "sshd") -> str:
    return f"Oct  5 03:04:05 web01 {prog}[123]: {msg}"


@pytest.mark.parametrize(
    ("msg", "kind", "ip", "user"),
    [
        (
            "Failed password for root from 203.0.113.9 port 22 ssh2",
            "ssh_failed_password",
            "203.0.113.9",
            "root",
        ),
        (
            "Failed password for invalid user bob from 203.0.113.9 port 22 ssh2",
            "ssh_failed_password",
            "203.0.113.9",
            "bob",
        ),
        (
            "Invalid user admin from 203.0.113.9 port 4000",
            "ssh_invalid_user",
            "203.0.113.9",
            "admin",
        ),
        (
            "Accepted publickey for deploy from 192.0.2.50 port 3 ssh2: ED25519 SHA256:x",
            "ssh_accepted_key",
            "192.0.2.50",
            "deploy",
        ),
        (
            "Accepted password for deploy from 192.0.2.50 port 3 ssh2",
            "ssh_accepted_password",
            "192.0.2.50",
            "deploy",
        ),
    ],
)
def test_ssh_lines(msg: str, kind: str, ip: str, user: str) -> None:
    event = parse_auth_line(auth(msg), year=2026)
    assert event is not None
    assert (event.kind, event.ip, event.user, event.source) == (kind, ip, user, "auth")
    assert event.host == "web01"


def test_sudo_and_noise() -> None:
    line = auth("deploy : TTY=pts/0 ; PWD=/ ; USER=root ; COMMAND=/bin/ls", prog="sudo")
    event = parse_auth_line(line, year=2026)
    assert event is not None
    assert event.user == "deploy"
    assert parse_auth_line(auth("pam_unix(cron:session): session opened", prog="CRON")) is None
    assert parse_auth_line("garbage") is None
    assert parse_auth_line("") is None


def test_syslog_year_and_zone() -> None:
    line = auth("Failed password for root from 203.0.113.9 port 22 ssh2")
    utc = parse_auth_line(line, year=2026, tz="UTC")
    rome = parse_auth_line(line, year=2026, tz="Europe/Rome")
    assert utc is not None
    assert rome is not None
    assert utc.ts - rome.ts == 7200
    assert datetime.fromtimestamp(utc.ts, UTC) == datetime(2026, 10, 5, 3, 4, 5, tzinfo=UTC)


def test_syslog_without_year_never_lands_in_the_future() -> None:
    line = "Dec 31 23:59:59 h sshd[1]: Failed password for root from 203.0.113.9 port 22 ssh2"
    event = parse_auth_line(line, now=NOW)
    assert event is not None
    assert event.ts <= NOW.timestamp()


def test_iso_timestamp_auth_line() -> None:
    line = "2026-10-05T03:04:05+00:00 web01 sshd[9]: Invalid user x from 203.0.113.9 port 1"
    event = parse_auth_line(line)
    assert event is not None
    assert event.kind == "ssh_invalid_user"


def test_journald() -> None:
    line = (
        '{"__REALTIME_TIMESTAMP": "1791161412786693", "_HOSTNAME": "db01", '
        '"SYSLOG_IDENTIFIER": "sshd", "MESSAGE": "Invalid user postgres from 185.237.129.215 port 43839"}'
    )
    event = parse_journald_line(line)
    assert event is not None
    assert (event.kind, event.host, event.ip) == ("ssh_invalid_user", "db01", "185.237.129.215")
    assert event.ts == pytest.approx(1791161412.786693)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "not json",
        "[]",
        '{"MESSAGE": "x"}',
        '{"__REALTIME_TIMESTAMP": "abc", "MESSAGE": "x"}',
        '{"__REALTIME_TIMESTAMP": "1", "_HOSTNAME": "h", "SYSLOG_IDENTIFIER": "sshd", "MESSAGE": null}',
    ],
)
def test_journald_rejects_junk(line: str) -> None:
    assert parse_journald_line(line) is None


def web(request: str, status: int = 200, agent: str = "curl/8", ip: str = "203.0.113.7") -> str:
    return f'{ip} - - [05/Oct/2026:00:00:01 +0000] "{request}" {status} 512 "-" "{agent}"'


def test_web_line() -> None:
    event = parse_web_line(web("GET /index.html?a=1 HTTP/1.1"))
    assert event is not None
    assert (event.source, event.kind, event.method, event.ip) == (
        "web",
        "web_ok",
        "GET",
        "203.0.113.7",
    )
    assert event.path is not None
    assert event.path.startswith("/index.html")
    assert event.agent == "curl/8"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (200, "web_ok"),
        (301, "web_redirect"),
        (403, "web_denied"),
        (404, "web_notfound"),
        (418, "web_client_error"),
        (500, "web_server_error"),
    ],
)
def test_web_status_kinds(status: int, kind: str) -> None:
    event = parse_web_line(web("GET / HTTP/1.1", status=status))
    assert event is not None
    assert event.kind == kind


def test_web_rejects_junk_and_survives_hostile_input() -> None:
    assert parse_web_line("nonsense") is None
    assert parse_web_line("") is None
    evil = web("GET /" + "A" * 50_000 + " HTTP/1.1")
    event = parse_web_line(evil)
    assert event is None or len(event.raw) <= 8192


def windows(event_id: int, **data: str) -> str:
    import json

    record = {
        "TimeCreated": "2026-10-05T00:00:08.0000000Z",
        "EventID": event_id,
        "Computer": "WIN-DESK01",
        "Channel": "Security",
        "Data": data,
    }
    return json.dumps(record)


def test_windows_logon_and_failure() -> None:
    ok = parse_windows_line(
        windows(4624, TargetUserName="alice", LogonType="10", IpAddress="203.0.113.5")
    )
    bad = parse_windows_line(
        windows(4625, TargetUserName="alice", LogonType="3", IpAddress="203.0.113.5")
    )
    assert ok is not None
    assert bad is not None
    assert ok.source == bad.source == "windows"
    assert ok.kind != bad.kind
    assert ok.user == "alice"
    assert ok.ip == "203.0.113.5"


def test_windows_unknown_event_id_and_junk_skipped() -> None:
    assert parse_windows_line(windows(9999)) is None
    assert parse_windows_line("{}") is None
    assert parse_windows_line("nope") is None
