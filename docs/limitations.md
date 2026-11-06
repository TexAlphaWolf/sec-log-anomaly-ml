# Limitations

Read this before trusting an alert, or a number in this repository.

- **The benchmark is synthetic.** The data generator is mine and so are the attacks. Good numbers
  show that the detectors do what they were designed to do on attacks I could imagine, not that
  they find what a real attacker does on your network. The only fair test is your own logs, and
  the tool is built to make that easy ([log sources](log-sources.md)).
- **Unsupervised means "unusual", not "malicious".** A new admin, a deploy at 3 a.m. or a
  link on the front page of a news site will be flagged. The reasons are there so a person can
  decide in seconds.
- **It learns what you show it.** Attacks in the training period become normal. Training on a
  period with a known incident is a mistake the benchmark quantifies.
- **A patient attacker wins.** Low-and-slow activity (one attempt an hour) never fills a window.
  The `ssh_slow` campaigns exist to measure this, and they are the weakest result.
- **Seasonality is limited.** Hour of day is a feature; day of week, month-end jobs and holidays
  are not. A two-week training period will not have seen them.
- **Windows are per entity.** An attacker spreading one attack over many addresses (a botnet
  doing one attempt each) looks like many quiet entities. Aggregating by target is future work.
- **No enrichment.** No GeoIP, reputation or threat-intelligence lookups; the tool works offline.
- **Models are not signed.** The checksum detects damage, not tampering by someone who can write
  to the model directory.
- **Small models.** The networks are deliberately small so that they train in seconds on CPU and
  the code stays readable. A large deployment would want streaming input and a bigger model.
