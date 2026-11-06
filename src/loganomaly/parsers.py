"""Parsers for the three log families: auth (sshd, sudo), web access logs and Windows events.

Every parser takes one line and returns an `Event`, or None when the line is not one it
understands. Nothing here raises on bad input: logs are written by strangers.

Accepted formats are the ones `sec-mini-siem` reads, so the same files work in both tools:

* auth     - `/var/log/auth.log` or `/var/log/secure`, classic syslog or ISO timestamps
* journald - `journalctl -o json`, same sshd/sudo messages as auth.log
* web      - nginx / Apache "combined" access log
* windows  - one JSON object per line, from the PowerShell snippet in docs/log-sources.md or
             from Winlogbeat
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, tzinfo
from urllib.parse import unquote_plus
from zoneinfo import ZoneInfo

from .events import Event
from .net import clean_ip

MAX_LINE = 8192  # longer lines are cut: a log line is never worth more memory than this
_MAX_RAW = 240
_MAX_FIELD = 256

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


def _zone(name: str) -> tzinfo:
    return UTC if name.upper() == "UTC" else ZoneInfo(name)


# --------------------------------------------------------------------------- auth / journald

_HEAD = r"(?P<host>\S+)\s+(?P<prog>[^\s\[:]+)(?:\[(?P<pid>\d+)\])?:\s+(?P<msg>.*)$"
_SYSLOG = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<hms>\d{2}:\d{2}:\d{2})\s+" + _HEAD
)
_ISO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\s+" + _HEAD
)
_FAILED = re.compile(
    r"^Failed (?:password|publickey|keyboard-interactive/pam) for (?:invalid user )?(?P<user>\S+)"
    r" from (?P<ip>\S+) port \d+"
)
_INVALID = re.compile(r"^Invalid user (?P<user>\S*) from (?P<ip>\S+)")
_ACCEPTED = re.compile(r"^Accepted (?P<method>\S+) for (?P<user>\S+) from (?P<ip>\S+) port \d+")
_PREAUTH = re.compile(
    r"^(?:Connection closed by (?:authenticating user \S+ |invalid user \S+ )?"
    r"(?P<ip1>[0-9a-fA-F:.]+) port \d+ \[preauth\]"
    r"|Did not receive identification string from (?P<ip2>\S+)"
    r"|Bad protocol version identification .* from (?P<ip3>\S+)"
    r"|maximum authentication attempts exceeded for (?:invalid user )?\S+ from (?P<ip4>\S+))"
)
_SUDO_FAIL = re.compile(r"^(?P<user>\S+)\s+:\s+\d+ incorrect password attempts?\s*;")
_SUDO_CMD = re.compile(
    r"^(?P<user>\S+)\s+:\s+(?:TTY=\S+\s+;\s+)?PWD=.*?\s+;\s+USER=\S+\s+;\s+COMMAND="
)


def classify_auth_message(prog: str, msg: str) -> tuple[str, str | None, str | None] | None:
    """(kind, user, ip text) for an sshd or sudo message we understand."""
    if prog == "sshd":
        if m := _FAILED.match(msg):
            return "ssh_failed_password", m["user"], m["ip"]
        if m := _INVALID.match(msg):
            return "ssh_invalid_user", m["user"] or None, m["ip"]
        if m := _ACCEPTED.match(msg):
            kind = "ssh_accepted_key" if m["method"] == "publickey" else "ssh_accepted_password"
            return kind, m["user"], m["ip"]
        if m := _PREAUTH.match(msg):
            return "ssh_preauth_closed", None, m["ip1"] or m["ip2"] or m["ip3"] or m["ip4"]
    elif prog == "sudo":
        if m := _SUDO_FAIL.match(msg):
            return "sudo_failure", m["user"], None
        if m := _SUDO_CMD.match(msg):
            return "sudo_command", m["user"], None
    return None


def _syslog_ts(mon: str, day: str, hms: str, tz: str, year: int, now: datetime) -> float | None:
    month = _MONTHS.get(mon)
    if month is None:
        return None
    hour, minute, second = (int(p) for p in hms.split(":"))
    try:
        moment = datetime(year, month, int(day), hour, minute, second, tzinfo=_zone(tz))
    except ValueError:
        return None
    # A December line read in January belongs to the previous year.
    if moment > now + timedelta(days=1):
        moment = moment.replace(year=year - 1)
    return moment.timestamp()


def _iso_ts(value: str, tz: str) -> float | None:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_zone(tz))
    return moment.timestamp()


def _auth_event(
    ts: float, host: str, classified: tuple[str, str | None, str | None], raw: str
) -> Event | None:
    kind, user, raw_ip = classified
    ip = clean_ip(raw_ip)
    if raw_ip is not None and ip is None:
        return None  # an address field that is not an address: do not trust the line
    return Event(
        ts=ts,
        source="auth",
        kind=kind,
        host=host[:_MAX_FIELD],
        ip=ip,
        user=user[:_MAX_FIELD] if user else None,
        raw=raw[:_MAX_RAW],
    )


def parse_auth_line(
    line: str, *, tz: str = "UTC", year: int | None = None, now: datetime | None = None
) -> Event | None:
    """One line of auth.log. Syslog timestamps carry no year or zone: pass both when it matters."""
    now = now or datetime.now(UTC)
    line = line[:MAX_LINE].strip()
    if m := _SYSLOG.match(line):
        ts = _syslog_ts(m["mon"], m["day"], m["hms"], tz, year or now.year, now)
    elif m := _ISO.match(line):
        ts = _iso_ts(m["ts"], tz)
    else:
        return None
    if ts is None:
        return None
    classified = classify_auth_message(m["prog"], m["msg"])
    return _auth_event(ts, m["host"], classified, line) if classified else None


def parse_journald_line(
    line: str, *, tz: str = "UTC", year: int | None = None, now: datetime | None = None
) -> Event | None:
    """One line of `journalctl -o json`; the message is the same text sshd writes to auth.log."""
    doc = _load_json(line)
    if doc is None:
        return None
    try:
        ts = int(str(doc["__REALTIME_TIMESTAMP"])) / 1_000_000
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    message = doc.get("MESSAGE")
    program = doc.get("SYSLOG_IDENTIFIER") or doc.get("_COMM")
    if not isinstance(message, str) or not isinstance(program, str):
        return None
    classified = classify_auth_message(program, message[:4096].strip())
    host = doc.get("_HOSTNAME")
    if classified is None:
        return None
    return _auth_event(ts, host if isinstance(host, str) else "", classified, line.strip())


def _load_json(line: str) -> dict[str, object] | None:
    try:
        doc = json.loads(line[: MAX_LINE * 8].strip().lstrip("﻿"))
    except (ValueError, RecursionError):  # not JSON, or nested so deeply that it is an attack
        return None
    return doc if isinstance(doc, dict) else None


# ------------------------------------------------------------------------------------- web

_COMBINED = re.compile(
    r"^(?P<ip>\S+) \S+ (?P<user>\S+) \[(?P<ts>[^\]]+)\] \"(?P<req>[^\"]*)\" "
    r"(?P<status>\d{3}) (?P<size>\S+)(?: \"(?P<ref>[^\"]*)\" \"(?P<ua>[^\"]*)\")?"
)
_WEB_TS = re.compile(
    r"^(?P<d>\d{2})/(?P<mon>[A-Z][a-z]{2})/(?P<y>\d{4}):(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2}) "
    r"(?P<off>[+-]\d{4})$"
)


def _web_kind(method: str, status: int) -> str:
    if method not in ("GET", "HEAD", "OPTIONS"):
        return "web_write_ok" if status < 400 else "web_write_failed"
    if status < 300:
        return "web_ok"
    if status < 400:
        return "web_redirect"
    if status in (401, 403):
        return "web_denied"
    if status == 404:
        return "web_notfound"
    return "web_client_error" if status < 500 else "web_server_error"


def parse_web_line(
    line: str, *, tz: str = "UTC", year: int | None = None, now: datetime | None = None
) -> Event | None:
    """One line of an nginx or Apache access log in the combined format."""
    line = line[:MAX_LINE].strip()
    m = _COMBINED.match(line)
    if not m:
        return None
    t = _WEB_TS.match(m["ts"])
    ip = clean_ip(m["ip"])
    if t is None or ip is None or t["mon"] not in _MONTHS:
        return None
    try:
        moment = datetime.strptime(
            f"{t['y']}-{_MONTHS[t['mon']]:02d}-{t['d']} {t['h']}:{t['mi']}:{t['s']} {t['off']}",
            "%Y-%m-%d %H:%M:%S %z",
        )
    except ValueError:
        return None
    parts = m["req"].split(" ")
    if len(parts) == 3:
        method, target = parts[0], parts[1]
    else:  # binary probes, TLS sent to a plain port, ...
        method, target = "", m["req"]
    status = int(m["status"])
    user = None if m["user"] == "-" else m["user"][:_MAX_FIELD]
    return Event(
        ts=moment.timestamp(),
        source="web",
        kind=_web_kind(method, status),
        host="",
        ip=ip,
        user=user,
        path=unquote_plus(target)[:512],
        method=method[:16],
        agent=(m["ua"] or None) and m["ua"][:_MAX_FIELD],
        raw=line[:_MAX_RAW],
    )


# --------------------------------------------------------------------------------- windows

_LOGON_TYPES = {
    2: "win_logon_interactive",
    7: "win_logon_interactive",  # unlock
    11: "win_logon_interactive",  # cached credentials
    3: "win_logon_network",
    10: "win_logon_remote",
    4: "win_logon_service",
    5: "win_logon_service",
}

# Event ID -> (kind or None for "decide from LogonType", field holding the account to blame)
_WINDOWS_EVENTS: dict[int, tuple[str | None, str]] = {
    4624: (None, "TargetUserName"),
    4625: ("win_logon_failed", "TargetUserName"),
    4672: ("win_privileged_logon", "SubjectUserName"),
    4698: ("win_scheduled_task_created", "SubjectUserName"),
    4720: ("win_user_created", "SubjectUserName"),
    4728: ("win_group_member_added", "SubjectUserName"),
    4732: ("win_group_member_added", "SubjectUserName"),
    4756: ("win_group_member_added", "SubjectUserName"),
    1102: ("win_audit_log_cleared", "SubjectUserName"),
    104: ("win_audit_log_cleared", "SubjectUserName"),
    7045: ("win_service_installed", "AccountName"),
}


def _clean(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value)
    return None if text in ("", "-") else text[:_MAX_FIELD]


def _windows_record(
    doc: dict[str, object],
) -> tuple[str, object, str, dict[str, object]] | None:
    """(timestamp text, event id, computer, data) from the PowerShell or Winlogbeat layout."""
    winlog = doc.get("winlog")
    if isinstance(winlog, dict):  # Winlogbeat
        data = winlog.get("event_data")
        return (
            str(doc.get("@timestamp", "")),
            winlog.get("event_id"),
            str(winlog.get("computer_name") or ""),
            data if isinstance(data, dict) else {},
        )
    if "EventID" in doc:  # the PowerShell snippet
        data = doc.get("Data")
        return (
            str(doc.get("TimeCreated", "")),
            doc.get("EventID"),
            str(doc.get("Computer") or ""),
            data if isinstance(data, dict) else {},
        )
    return None


def parse_windows_line(
    line: str, *, tz: str = "UTC", year: int | None = None, now: datetime | None = None
) -> Event | None:
    """One exported Windows event (JSON). Event IDs we do not use are skipped."""
    doc = _load_json(line)
    if doc is None or (record := _windows_record(doc)) is None:
        return None
    stamp, event_id, computer, data = record
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        spec = _WINDOWS_EVENTS.get(int(str(event_id)))
    except ValueError:
        return None
    if spec is None or moment.tzinfo is None:
        return None
    kind, user_field = spec
    if kind is None:  # 4624: the logon type tells interactive, network, remote and service apart
        logon_type = _clean(data.get("LogonType"))
        kind = (
            _LOGON_TYPES.get(int(logon_type), "win_logon_other")
            if (logon_type and logon_type.isdigit())
            else "win_logon_other"
        )
    ip_field = "IpAddress" if kind.startswith("win_logon") else ""
    return Event(
        ts=moment.timestamp(),
        source="windows",
        kind=kind,
        host=computer[:_MAX_FIELD],
        ip=clean_ip(_clean(data.get(ip_field))) if ip_field else None,
        user=_clean(data.get(user_field)),
        raw=line.strip()[:_MAX_RAW],
    )


Parser = Callable[..., Event | None]
PARSERS: dict[str, Parser] = {
    "auth": parse_auth_line,
    "journald": parse_journald_line,
    "web": parse_web_line,
    "windows": parse_windows_line,
}
