from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from loganomaly.cli import main
from loganomaly.events import Event
from loganomaly.explain import reasons
from loganomaly.pipeline import analyse
from loganomaly.report import MAX_QUIET_DOTS, findings, html_report, jsonl, text_table
from loganomaly.store import Bundle


@pytest.fixture(scope="module")
def scored(bundle: Bundle, events: list[Event]):  # type: ignore[no-untyped-def]
    cut = min(e.ts for e in events) + 3 * 86400
    return analyse(bundle, [e for e in events if e.ts >= cut])


def test_findings_sorted_with_reasons(bundle: Bundle, scored) -> None:  # type: ignore[no-untyped-def]
    items = findings(bundle, scored, "vote", 15)
    assert 0 < len(items) <= 15
    ratios = [i.ratio for i in items]
    assert ratios == sorted(ratios, reverse=True)
    assert all(i.reasons for i in items)
    assert all(i.severity() in {"low", "medium", "high"} for i in items)
    with pytest.raises(KeyError):
        findings(bundle, scored, "nope")


@pytest.mark.parametrize("name", ["zscore", "iforest", "autoencoder", "sequence"])
def test_every_detector_can_explain(bundle: Bundle, scored, name: str) -> None:  # type: ignore[no-untyped-def]
    items = findings(bundle, scored, name, 3)
    for item in items:
        assert reasons(bundle, scored, item.row, name)


def test_text_and_jsonl_outputs(bundle: Bundle, scored) -> None:  # type: ignore[no-untyped-def]
    items = findings(bundle, scored, "vote", 5)
    assert "No anomalies" in text_table([])
    assert items[0].entity in text_table(items)
    rows = [json.loads(line) for line in jsonl(items).splitlines()]
    assert len(rows) == len(items)
    assert {"time", "entity", "severity", "reasons", "flagged_by"} <= rows[0].keys()


def test_html_report_is_self_contained_and_escapes_log_text(bundle: Bundle, scored) -> None:  # type: ignore[no-untyped-def]
    items = findings(bundle, scored, "vote", 5)
    items[0].sample = ['<script>alert("x")</script>', '"><img src=x onerror=alert(1)>']
    items[0].reasons = ["<b>bold</b>"]
    page = html_report(bundle, scored, items, "vote", {"auth": 10}, 2)
    assert "Content-Security-Policy" in page
    assert "default-src 'none'" in page
    assert "<script" not in page
    assert "<img" not in page
    assert "&lt;script&gt;" in page
    assert not re.search(r"(src|href)=[\"']https?://", page)
    assert page.count("<circle") <= MAX_QUIET_DOTS + len(items) + 50


def test_html_report_with_no_findings(bundle: Bundle, scored) -> None:  # type: ignore[no-untyped-def]
    page = html_report(bundle, scored, [], "vote", {}, 0)
    assert "<html" in page


# ------------------------------------------------------------------------------------ CLI


def test_cli_full_workflow(
    dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model"
    report = tmp_path / "r.html"
    args = [
        "--auth",
        str(dataset_dir / "auth.log"),
        "--journald",
        str(dataset_dir / "journal.jsonl"),
        "--web",
        str(dataset_dir / "access.log"),
        "--windows",
        str(dataset_dir / "windows-security.jsonl"),
        "--year",
        "2026",
    ]
    assert main(["train", *args, "--out", str(model), "--models", "zscore,autoencoder"]) == 0
    assert (model / "model.json").is_file()
    capsys.readouterr()
    assert (
        main(
            [
                "detect",
                *args,
                "--model",
                str(model),
                "--detector",
                "autoencoder",
                "--top",
                "3",
                "--format",
                "jsonl",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert len([line for line in captured.out.splitlines() if line.strip()]) <= 3
    assert report.is_file()
    assert "Read events/lines" in captured.err


def test_cli_errors_are_friendly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="at least one log file"):
        main(["train", "--out", str(tmp_path / "m")])
    with pytest.raises(SystemExit, match="no such file"):
        main(["train", "--auth", str(tmp_path / "missing.log"), "--out", str(tmp_path / "m")])
    with pytest.raises(SystemExit, match="error"):
        main(["detect", "--auth", str(tmp_path / "x"), "--model", str(tmp_path / "none")])


def test_cli_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert "loganomaly 1.0.1" in capsys.readouterr().out


def test_cli_generate_and_evaluate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "d"
    assert main(["generate", str(out), "--days", "3", "--train-days", "2", "--seed", "2"]) == 0
    assert (out / "labels.jsonl").is_file()
    assert main(["evaluate", str(out)]) == 0
    assert "autoencoder" in capsys.readouterr().out


def test_cli_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--out", str(tmp_path / "demo"), "--days", "4", "--train-days", "2"]) == 0
    assert (tmp_path / "demo" / "report.html").is_file()
    assert (tmp_path / "demo" / "model" / "model.json").is_file()
    assert "Report:" in capsys.readouterr().out


def test_cli_benchmark_writes_markdown(tmp_path: Path) -> None:
    target = tmp_path / "results.md"
    assert (
        main(
            ["benchmark", "--seeds", "1", "--days", "4", "--train-days", "2", "--out", str(target)]
        )
        == 0
    )
    text = target.read_text()
    assert "autoencoder" in text.lower()
    assert "how-the-comparison-is-made" in text
    assert text.count("|") > 50
