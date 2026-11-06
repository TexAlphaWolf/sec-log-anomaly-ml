# Security policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private reporting: open
the **Security** tab of this repository and choose **Report a vulnerability**. Include what you
found, how to reproduce it and the version you tested. This is a personal project, so I answer on
a best-effort basis, usually within a week.

## What is in scope

loganomaly reads log files that an attacker can influence and writes model files that you load
later, so the interesting reports are about:

- **Parsers**: a line that crashes them, hangs them (regular expression backtracking), or makes
  an attack invisible.
- **The HTML report**: any way for log content to run as script or load an outside resource.
- **Model files**: any way a crafted `model.json` or `weights.npz` could run code or read files.
- **Evasion that is cheap and general** (not "a patient attacker stays below the threshold": that
  is a documented limit, see [docs/limitations.md](docs/limitations.md)).

## Measures already in place

- **No pickle, anywhere.** Models are JSON plus a NumPy `.npz` loaded with `allow_pickle=False`,
  so loading a model cannot execute code.
- `weights.npz` is checked against the SHA-256 recorded in `model.json`, and the schema, feature
  list and event vocabulary must match, or loading stops with a clear error.
- The checksum catches damage and accidental mix-ups. **It is not a signature**: someone who can
  write to the model directory can replace both files. Keep models where only you can write.
- Log lines are capped in length; fields are shortened; unparseable lines are counted, not
  trusted.
- The HTML report escapes all log text, has a Content-Security-Policy of
  `default-src 'none'; style-src 'unsafe-inline'`, no script and no external resources (a test
  fails the build if that changes).
- The container runs as an unprivileged user and needs no network.
- CI runs CodeQL and a dependency audit; Dependabot keeps dependencies and actions current.
