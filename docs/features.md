# Features

Each window (one entity, five minutes) becomes 46 numbers. Counts are stored as `log1p(count)`,
so 3 failed logins and 3000 are far apart without one outweighing everything else.

| Group | Columns | What they capture |
| --- | --- | --- |
| Kind counts (27) | `n_<kind>` | How many events of each kind: failed passwords, invalid users, accepted keys, sudo, 404s, 5xx, writes, Windows logons, a cleared audit log, ... |
| Volume | `events`, `sources`, `distinct_users`, `distinct_paths`, `distinct_hosts` | How much, across how many log families, accounts, paths and machines |
| Login behaviour | `fail_ratio` | Failed share of login attempts |
| Web behaviour | `web_error_ratio`, `notfound_ratio`, `write_ratio`, `agent_missing`, `agent_distinct` | Scanners probe for pages that do not exist and rotate or omit User-Agents |
| Timing | `gap_median`, `gap_min`, `burst` | Humans are slow and irregular; tools are fast and regular |
| Path shape | `path_length`, `path_entropy`, `path_special` | Injection and traversal strings are long, random-looking and full of `' " < > % ../` |
| Time of day | `hour_sin`, `hour_cos` | Cyclic hour, so 23:00 and 01:00 are neighbours |
| Source | `private_source` | Internal vs. external |
| Novelty | `novel_source`, `novel_user`, `novel_pair`, `odd_hour_login` | First time this address, this account on this host, this account from this address, or this account at this hour |

## Applicability

A feature that is 0 because it does not apply is not the same as 0 because nothing unusual
happened. `path_entropy` of a window with no web requests means nothing. The reasons shown in
reports therefore skip web columns for windows without web events and login columns for windows
without logins.

## Novelty and training

The tracker is trained on the same days as the models and saved with them, so detection starts
where training stopped. During the first day (or 10% of the span, whichever is shorter) *every*
address is new, so those windows are left out of training: otherwise "new" would look normal.
