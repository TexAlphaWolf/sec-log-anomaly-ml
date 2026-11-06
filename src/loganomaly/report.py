"""Turning an analysis into something a person can read: a table, JSON lines, a web page.

Log lines are written by whoever sends them, so the page treats every one of them as hostile
text: everything is escaped, and a Content-Security-Policy forbids scripts and remote loads.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from .explain import reasons
from .pipeline import Analysis
from .store import Bundle

SAMPLE_LINES = 3
MAX_QUIET_DOTS = 2500


@dataclass
class Finding:
    """One flagged window, with the reasons."""

    row: int
    time: float
    entity: str
    score: float
    ratio: float
    detectors: list[str]
    reasons: list[str]
    events: int
    sample: list[str]

    def severity(self) -> str:
        return "high" if self.ratio >= 3 else "medium" if self.ratio >= 1.5 else "low"

    def to_json(self) -> dict[str, object]:
        return {
            "time": _iso(self.time),
            "entity": self.entity,
            "score": round(self.score, 4),
            "times_threshold": round(self.ratio, 2),
            "severity": self.severity(),
            "flagged_by": self.detectors,
            "events": self.events,
            "reasons": self.reasons,
            "sample": self.sample,
        }


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def findings(
    bundle: Bundle, analysis: Analysis, detector: str = "vote", limit: int | None = None
) -> list[Finding]:
    """Flagged windows for `detector`, most unusual first."""
    if detector not in analysis.flags:
        raise KeyError(detector)
    rows = np.flatnonzero(analysis.flags[detector])
    rows = rows[np.argsort(-analysis.ratios[detector][rows], kind="stable")]
    reason_source = detector if detector in bundle.detectors else _best(analysis, rows)
    found = []
    for row in rows[:limit]:
        window = analysis.windows[int(row)]
        row_int = int(row)
        flagged = [name for name in bundle.detectors if analysis.flags[name][row_int]]
        source = reason_source if reason_source in flagged or not flagged else flagged[0]
        found.append(
            Finding(
                row=row_int,
                time=window.start,
                entity=window.entity,
                score=float(analysis.scores[detector][row_int]),
                ratio=float(analysis.ratios[detector][row_int]),
                detectors=flagged,
                reasons=reasons(bundle, analysis, row_int, source),
                events=len(window.events),
                sample=[e.raw for e in window.events[:SAMPLE_LINES]],
            )
        )
    return found


def _best(analysis: Analysis, rows: np.ndarray) -> str:
    """For a combined verdict, explain with the learned detector that flagged most windows."""
    learned = [n for n in ("autoencoder", "iforest", "sequence", "zscore") if n in analysis.flags]
    return max(learned, key=lambda n: int(analysis.flags[n][rows].sum()), default="zscore")


def text_table(items: list[Finding]) -> str:
    """A plain-text list for the terminal."""
    if not items:
        return "No anomalies found."
    lines = []
    for item in items:
        lines.append(
            f"{_iso(item.time)}  {item.severity():6s}  x{item.ratio:<5.1f} {item.entity}"
            f"  ({item.events} events)"
        )
        for reason in item.reasons:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ HTML

_CSS = """
:root{--bg:#fff;--fg:#1d2330;--mute:#667085;--line:#e3e7ee;--card:#f7f8fb;--high:#d92d20;
--medium:#dc6803;--low:#667085;--bar:#4f6bed}
@media (prefers-color-scheme:dark){:root{--bg:#10141c;--fg:#e6e9f0;--mute:#98a2b3;--line:#28303d;
--card:#171c27;--high:#f97066;--medium:#fdb022;--low:#98a2b3;--bar:#7c93ff}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,sans-serif}main{max-width:980px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:20px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:24px}.stat span{color:var(--mute);font-size:13px}
h2{font-size:16px;margin:24px 0 8px}svg{width:100%;height:auto;display:block;background:var(--card);
border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 10px;
border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mute);font-weight:600;
font-size:13px}.sev{font-weight:700}.sev.high{color:var(--high)}.sev.medium{color:var(--medium)}
.sev.low{color:var(--low)}code,pre{font:12px/1.4 ui-monospace,monospace}pre{margin:6px 0 0;
white-space:pre-wrap;word-break:break-all;color:var(--mute)}ul{margin:0;padding-left:18px}
.entity{font-family:ui-monospace,monospace;word-break:break-all}
@media (max-width:640px){.hide{display:none}}
"""


def _chart(analysis: Analysis, detector: str, flagged: set[int]) -> str:
    """Score over time as an SVG: every window a dot, anomalies in red, the threshold dashed."""
    width, height, pad = 960, 220, 34
    ratios = analysis.ratios[detector]
    if len(ratios) == 0:
        return ""
    times = np.array([w.start for w in analysis.windows])
    low, high = float(times.min()), float(times.max())
    top = max(2.0, float(np.percentile(ratios, 99.9)) * 1.1)

    def px(t: float) -> float:
        return pad + (t - low) / max(high - low, 1.0) * (width - 2 * pad)

    def py(r: float) -> float:
        return height - pad - min(r, top) / top * (height - 2 * pad)

    dots = []
    quiet = [int(i) for i in np.flatnonzero(~np.isin(np.arange(len(ratios)), list(flagged)))]
    if len(quiet) > MAX_QUIET_DOTS:  # a page of 100,000 dots helps nobody
        step = len(quiet) / MAX_QUIET_DOTS
        quiet = [quiet[int(k * step)] for k in range(MAX_QUIET_DOTS)]
    for i in sorted([*quiet, *flagged], key=lambda j: float(ratios[j])):  # anomalies on top
        red = int(i) in flagged
        colour = "var(--high)" if red else "var(--bar)"
        opacity = "0.95" if red else "0.35"
        radius = "3.4" if red else "1.8"
        dots.append(
            f'<circle cx="{px(times[i]):.1f}" cy="{py(float(ratios[i])):.1f}" r="{radius}"'
            f' fill="{colour}" opacity="{opacity}"/>'
        )
    y1 = py(1.0)
    first, last = _iso(low)[:10], _iso(high)[:10]
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Score over time">'
        f'<line x1="{pad}" x2="{width - pad}" y1="{y1:.1f}" y2="{y1:.1f}" stroke="var(--mute)"'
        f' stroke-dasharray="5 4"/>'
        f'<text x="{width - pad}" y="{y1 - 6:.1f}" text-anchor="end" fill="var(--mute)"'
        f' font-size="12">threshold</text>'
        f'<text x="{pad}" y="{height - 10}" fill="var(--mute)" font-size="12">{first}</text>'
        f'<text x="{width - pad}" y="{height - 10}" text-anchor="end" fill="var(--mute)"'
        f' font-size="12">{last}</text>' + "".join(dots) + "</svg>"
    )


def html_report(
    bundle: Bundle,
    analysis: Analysis,
    items: list[Finding],
    detector: str,
    read_lines: dict[str, int],
    skipped: int,
) -> str:
    """A self-contained page: summary, score timeline and the flagged windows with reasons."""
    esc = html.escape
    flagged = {item.row for item in items}
    stats = [
        (f"{sum(read_lines.values()):,}", "log lines read"),
        (f"{skipped:,}", "lines not understood"),
        (f"{len(analysis.windows):,}", "windows scored"),
        (f"{len(items):,}", "anomalies"),
        (sum(1 for i in items if i.severity() == "high"), "high severity"),
    ]
    cards = "".join(f'<div class="stat"><b>{esc(str(v))}</b><span>{esc(label)}</span></div>'
                    for v, label in stats)  # fmt: skip
    rows = []
    for item in items:
        reason_list = "".join(f"<li>{esc(r)}</li>" for r in item.reasons) or "<li>-</li>"
        sample = "\n".join(esc(line) for line in item.sample)
        rows.append(
            f"<tr><td>{esc(_iso(item.time))}</td>"
            f'<td class="sev {item.severity()}">{item.severity()}<br>'
            f'<span class="hide">x{item.ratio:.1f}</span></td>'
            f'<td class="entity">{esc(item.entity)}<br>{item.events} events</td>'
            f"<td><ul>{reason_list}</ul><pre>{sample}</pre></td></tr>"
        )
    body = (
        "".join(rows)
        if rows
        else '<tr><td colspan="4">No anomalies: every window is within the normal range.</td></tr>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src'
        " 'unsafe-inline'\">"
        "<title>Log anomaly report</title>"
        f"<style>{_CSS}</style></head><body><main>"
        "<h1>Log anomaly report</h1>"
        f'<p class="sub">Detector: {esc(detector)} &middot; '
        f"windows of {bundle.window // 60} minutes &middot; threshold at the "
        f"{bundle.quantile * 100:g}th percentile of normal activity</p>"
        f'<div class="stats">{cards}</div>'
        f"<h2>How unusual each window was (1.0 is the threshold)</h2>"
        f"{_chart(analysis, detector, flagged)}"
        f"<h2>Anomalies, most unusual first</h2>"
        f"<table><thead><tr><th>Window start (UTC)</th><th>Severity</th><th>Entity</th>"
        f"<th>Why, and the first events</th></tr></thead><tbody>{body}</tbody></table>"
        "</main></body></html>"
    )


def jsonl(items: list[Finding]) -> str:
    return "".join(json.dumps(item.to_json()) + "\n" for item in items)
