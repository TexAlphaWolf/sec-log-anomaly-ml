"""Small address helpers."""

from __future__ import annotations

import ipaddress


def clean_ip(value: str | None) -> str | None:
    """The normalised address, or None when `value` is not an IP address."""
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


def is_private(value: str) -> bool:
    """True for private, loopback and link-local addresses."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local
