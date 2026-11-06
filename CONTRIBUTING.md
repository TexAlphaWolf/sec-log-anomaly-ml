# Contributing

Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest --cov                    # fails under 90% coverage
python tools/quality_gate.py    # the detectors must not get clearly worse
```

CI runs the same on Python 3.11 to 3.14, builds the Docker image and runs the demo inside it.

## Guidelines

- **Runtime stays NumPy-only.** scikit-learn and SciPy are test-time references, not
  dependencies of the tool. If a model needs something new, write it in NumPy and test it
  against the reference.
- **New log format:** add a parser in `parsers.py` that returns an `Event` with a `kind` from the
  fixed vocabulary in `events.py`, with tests on real (anonymised) lines. Adding a kind at the end
  of the vocabulary is safe; reordering breaks saved models.
- **New feature:** add it to `features.py`, say in `docs/features.md` what it measures, and check
  `python tools/quality_gate.py`. Old models stop loading on purpose when the feature list
  changes.
- **New attack for the synthetic data:** add it to `synth.py` with its label, and a row in
  `docs/results.md` through `loganomaly benchmark`.
- **Logs are hostile.** Never render log data as HTML or run it through anything but the parsers.
