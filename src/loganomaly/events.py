"""The normalised event every parser produces.

Models never see raw log text. They see an `Event` with a `kind` taken from a small fixed
vocabulary, so the same detector works on sshd, nginx and Windows events alike.
"""

from __future__ import annotations

from dataclasses import dataclass

SOURCES = ("auth", "web", "windows")

# The vocabulary. The order is part of the model format: appending is safe, reordering is not.
AUTH_KINDS = (
    "ssh_failed_password",
    "ssh_invalid_user",
    "ssh_accepted_password",
    "ssh_accepted_key",
    "ssh_preauth_closed",
    "sudo_command",
    "sudo_failure",
)
WEB_KINDS = (
    "web_ok",
    "web_redirect",
    "web_denied",
    "web_notfound",
    "web_client_error",
    "web_server_error",
    "web_write_ok",
    "web_write_failed",
)
WINDOWS_KINDS = (
    "win_logon_interactive",
    "win_logon_network",
    "win_logon_remote",
    "win_logon_service",
    "win_logon_other",
    "win_logon_failed",
    "win_privileged_logon",
    "win_user_created",
    "win_group_member_added",
    "win_audit_log_cleared",
    "win_service_installed",
    "win_scheduled_task_created",
)
KINDS = AUTH_KINDS + WEB_KINDS + WINDOWS_KINDS
KIND_INDEX = {kind: i for i, kind in enumerate(KINDS)}

# Kinds that say "someone tried to log in and failed", and "someone logged in".
FAILED_LOGIN_KINDS = frozenset({"ssh_failed_password", "ssh_invalid_user", "win_logon_failed"})
SUCCESS_LOGIN_KINDS = frozenset(
    {
        "ssh_accepted_password",
        "ssh_accepted_key",
        "win_logon_interactive",
        "win_logon_network",
        "win_logon_remote",
    }
)


@dataclass(frozen=True, slots=True)
class Event:
    """One log line, reduced to what the models need."""

    ts: float  # seconds since the Unix epoch, UTC
    source: str  # one of SOURCES
    kind: str  # one of KINDS
    host: str
    ip: str | None = None
    user: str | None = None
    path: str | None = None  # web only: decoded request path and query
    method: str | None = None  # web only
    agent: str | None = None  # web only: User-Agent
    raw: str = ""  # the original line, shortened, for the report

    @property
    def entity(self) -> str:
        """Who the event is about: the source address, or the account when there is none."""
        if self.ip:
            return f"ip:{self.ip}"
        return f"user:{self.user or '-'}@{self.host or '-'}"


# The sequence model reads each event as (kind, gap since the previous event in the window).
GAP_EDGES = (2.0, 30.0)  # seconds: faster than 2 s, faster than 30 s, slower (or the first event)
GAP_NAMES = ("fast", "steady", "slow")
N_GAP = len(GAP_NAMES)
VOCAB = len(KINDS) * N_GAP + 1  # the last token marks "start of window"


def token(kind: str, gap: float | None) -> int:
    """Token for an event; `gap` is None for the first event of a window."""
    if gap is None or gap >= GAP_EDGES[1]:
        return KIND_INDEX[kind] * N_GAP + N_GAP - 1
    bucket = 0 if gap < GAP_EDGES[0] else 1
    return KIND_INDEX[kind] * N_GAP + bucket


def describe_token(value: int) -> str:
    """A token in words, for explanations."""
    if value >= len(KINDS) * N_GAP:
        return "the start of the window"
    return f"{KINDS[value // N_GAP]} ({GAP_NAMES[value % N_GAP]})"
