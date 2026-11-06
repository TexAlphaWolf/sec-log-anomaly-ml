# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.0.0] - 2026-11-07

### Added
- Parsers for Linux `auth.log`, `journalctl -o json`, nginx/Apache access logs and exported
  Windows Security events, all reduced to one event vocabulary of 27 kinds.
- 46 features per (entity, five-minute window): counts, rates, ratios, path entropy, time of day
  and "first time seen" novelty for sources, accounts and account/address pairs.
- Four detectors, written in NumPy: robust z-score baseline, Isolation Forest, autoencoder and a
  next-event sequence model (DeepLog-style); a vote that needs two of the three learned ones.
- Reasons for every alert in plain words, and a self-contained HTML report.
- Synthetic, labelled log generator with twelve attack types, and an evaluation that reports
  AUC, average precision, precision, recall, false alerts per day and campaigns found.
- `loganomaly demo`, `benchmark`, `generate`, `train`, `detect`, `evaluate`.
- Model files without pickle (JSON + `.npz`) with a SHA-256 integrity check.
- Docker image, CI on Python 3.11-3.14, model quality gate, CodeQL, dependency audit.
- `tools/export-windows-events.ps1` to export your own Windows Security log.
