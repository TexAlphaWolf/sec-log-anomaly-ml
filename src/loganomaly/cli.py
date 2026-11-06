"""The `loganomaly` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .evaluate import Evaluation, evaluate
from .pipeline import ALL_MODELS, TrainConfig, analyse, train
from .report import findings, html_report, jsonl, text_table
from .sources import read_all
from .store import ModelError, load, save
from .synth import DatasetSpec, generate, write

_SOURCES = (
    ("auth", "auth.log or /var/log/secure"),
    ("journald", "`journalctl -o json` output"),
    ("web", "nginx or Apache combined access log"),
    ("windows", "exported Windows events (JSON lines)"),
)


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    for name, help_text in _SOURCES:
        parser.add_argument(
            f"--{name}", action="append", default=[], metavar="FILE",
            help=f"{help_text}; repeat for several files, '-' reads standard input",
        )  # fmt: skip
    parser.add_argument("--year", type=int, help="year of auth.log lines, which carry none")
    parser.add_argument("--tz", default="UTC", help="time zone of auth.log lines (default UTC)")


def _inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    found = [(name, path) for name, _ in _SOURCES for path in getattr(args, name)]
    if not found:
        raise SystemExit("error: give at least one log file (--auth, --journald, --web, --windows)")
    return found


def _read(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    try:
        return read_all(_inputs(args), tz=args.tz, year=args.year)
    except FileNotFoundError as error:
        raise SystemExit(f"error: no such file: {error}") from error


def _print_stats(stats) -> None:  # type: ignore[no-untyped-def]
    parts = [f"{fmt}: {stats.events.get(fmt, 0):,}/{n:,}" for fmt, n in stats.lines.items()]
    print(f"Read events/lines  {'  '.join(parts)}  (skipped {stats.skipped:,})", file=sys.stderr)


# ---------------------------------------------------------------------------------- commands


def cmd_generate(args: argparse.Namespace) -> int:
    spec = DatasetSpec(days=args.days, train_days=args.train_days, seed=args.seed,
                       dirty=args.dirty)  # fmt: skip
    data = generate(spec)
    out = write(data, spec, args.out)
    print(f"Wrote {len(data.auth) + len(data.journal) + len(data.web) + len(data.windows):,} log "
          f"lines and {len(data.attacks)} labelled attacks to {out}")  # fmt: skip
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    events, stats = _read(args)
    _print_stats(stats)
    config = TrainConfig(window=args.window * 60, quantile=args.quantile, seed=args.seed,
                         models=tuple(args.models.split(",")))  # fmt: skip
    try:
        bundle = train(events, config)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    save(bundle, args.out)
    info = bundle.info
    print(f"Trained {', '.join(bundle.detectors)} on {info['windows']:,} windows "
          f"in {info['total_seconds']}s; model saved to {args.out}")  # fmt: skip
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    try:
        bundle = load(args.model)
    except ModelError as error:
        raise SystemExit(f"error: {error}") from error
    events, stats = _read(args)
    _print_stats(stats)
    analysis = analyse(bundle, events)
    detector = args.detector
    if detector not in analysis.flags:
        raise SystemExit(f"error: detector '{detector}' is not in this model "
                         f"(have: {', '.join(analysis.flags)})")  # fmt: skip
    items = findings(bundle, analysis, detector, args.top)
    if args.report:
        page = html_report(bundle, analysis, items, detector, stats.lines, stats.skipped)
        Path(args.report).write_text(page, encoding="utf-8")
        print(f"Report written to {args.report}", file=sys.stderr)
    sys.stdout.write(jsonl(items) if args.format == "jsonl" else text_table(items) + "\n")
    return 0


def _eval_table(ev: Evaluation) -> str:
    lines = [f"{'detector':12s} {'AUC':>6s} {'AP':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s} "
             f"{'alerts/day':>10s} {'campaigns':>10s}"]  # fmt: skip
    for r in ev.results:
        lines.append(f"{r.name:12s} {r.auc:6.3f} {r.ap:6.3f} {r.precision:6.2f} {r.recall:6.2f} "
                     f"{r.f1:6.2f} {r.false_alerts_per_day:10.1f} "
                     f"{r.campaigns_found:>5d}/{r.campaigns_total:<4d}")  # fmt: skip
    lines.append(
        f"\n{ev.test_windows:,} test windows, {ev.positive_windows} of them attacks, over "
        f"{ev.test_days:.0f} days. AUC and AP rank attacks above normal windows; precision,\n"
        "recall and F1 use each detector's own threshold; 'alerts/day' counts false alerts."
    )
    return "\n".join(lines)


def cmd_evaluate(args: argparse.Namespace) -> int:
    ev = evaluate(args.dataset, TrainConfig(quantile=args.quantile, seed=args.seed))
    print(_eval_table(ev))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate a dataset, train, evaluate and write a report: the whole tool in one go."""
    spec = DatasetSpec(days=args.days, train_days=args.train_days, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generating {spec.days} days of logs ({spec.train_days} for training)...")
    write(generate(spec), spec, out / "data")
    print("Training on the normal days and scoring the rest...\n")
    config = TrainConfig(seed=args.seed)
    print(_eval_table(evaluate(out / "data", config)))

    from .evaluate import load_events

    events, manifest = load_events(out / "data")
    bundle = train([e for e in events if e.ts < manifest["train_end"]], config)
    save(bundle, out / "model")
    analysis = analyse(bundle, [e for e in events if e.ts >= manifest["train_end"]])
    items = findings(bundle, analysis, "vote", 40)
    from .sources import ReadStats

    stats = ReadStats()
    page = html_report(bundle, analysis, items, "vote", {"demo": len(events)}, stats.skipped)
    (out / "report.html").write_text(page, encoding="utf-8")
    print(f"\nModel: {out / 'model'}\nReport: {out / 'report.html'} (open it in a browser)")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    text = run_benchmark(seeds=args.seeds, days=args.days, train_days=args.train_days)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Results written to {args.out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------------- the parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loganomaly",
        description="Unsupervised anomaly detection on auth, web and Windows logs.",
    )
    parser.add_argument("--version", action="version", version=f"loganomaly {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write a synthetic labelled dataset")
    p.add_argument("out", help="directory to write into")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--train-days", type=int, default=7)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--dirty", type=float, default=0.0, help="hide attacks in the training days (share of test)"
    )
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="learn what normal looks like from your logs")
    _add_inputs(p)
    p.add_argument("--out", required=True, help="model directory to write")
    p.add_argument("--window", type=int, default=5, help="window length in minutes (default 5)")
    p.add_argument(
        "--quantile", type=float, default=0.995, help="training share under the threshold"
    )
    p.add_argument("--models", default=",".join(ALL_MODELS), help="comma separated detectors")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("detect", help="score logs with a trained model")
    _add_inputs(p)
    p.add_argument("--model", required=True, help="model directory from `train`")
    p.add_argument("--detector", default="vote",
                   help="zscore, iforest, autoencoder, sequence or vote (default)")  # fmt: skip
    p.add_argument("--top", type=int, default=20, help="how many windows to show (default 20)")
    p.add_argument("--format", choices=("text", "jsonl"), default="text")
    p.add_argument("--report", metavar="FILE.html", help="also write a web page")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("evaluate", help="score the detectors on a generated dataset")
    p.add_argument("dataset", help="directory made by `generate`")
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("demo", help="generate, train, evaluate and report in one go")
    p.add_argument("--out", default="loganomaly-demo")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--train-days", type=int, default=7)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("benchmark", help="average the evaluation over several datasets")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--train-days", type=int, default=7)
    p.add_argument("--out", help="write the markdown results here")
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
