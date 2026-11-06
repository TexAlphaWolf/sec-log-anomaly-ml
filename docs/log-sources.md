# Log sources

| Flag | Source | How to get it |
| --- | --- | --- |
| `--auth` | `/var/log/auth.log` (Debian, Ubuntu) or `/var/log/secure` (RHEL) | Copy the file |
| `--journald` | systemd journal | `journalctl -u ssh -o json --since "-14 days" > journal.jsonl` |
| `--web` | nginx or Apache access log, *combined* format | Copy `access.log` |
| `--windows` | Windows Security events as JSON lines | `tools\export-windows-events.ps1` (elevated PowerShell) |

Repeat a flag for several files; `-` reads standard input. Give the same flags to `train` and
`detect`.

## Time

`auth.log` lines carry no year and no time zone. Pass `--year 2026` and `--tz Europe/Rome` (the
zone the server logs in). Without a year, the current one is used, and a date that would be in the
future is moved back a year, so a log read in January that ends in December still works. Journal,
web and Windows timestamps carry their own zone.

## What is understood

- **SSH:** failed password, invalid user, accepted password, accepted key, connection closed
  before authentication.
- **sudo:** commands and failures.
- **Web:** every request, as one of ok / redirect / denied (401, 403) / not found / other client
  error / server error / write ok / write failed, with path, method and User-Agent.
- **Windows:** 4624 (logon, split by logon type into interactive, network, remote, service,
  other), 4625 (failed logon), 4672 (privileged logon), 4720 (user created), 4728/4732/4756
  (added to a group), 4698 (scheduled task), 1102 and 104 (audit log cleared), 7045 (service
  installed). Winlogbeat output is understood as well as the PowerShell export.

Everything else is skipped and counted: `train` and `detect` print how many lines were read,
understood and skipped.

## Using your own data

```bash
# 1. Learn normal from a quiet period (two weeks or more is better than two days)
loganomaly train --auth auth.log --web access.log --year 2026 --out model

# 2. Score a later period; the report shows why each window was flagged
loganomaly detect --auth auth.log.1 --web access.log.1 --year 2026 --model model --report report.html
```

Train on logs you believe are clean. If they are not, the model learns the attack as normal
(see "dirty training" in [results](results.md)). Logs contain addresses and account names: keep
reports and models private.
