# Models

Four detectors, all NumPy. They see the same scaled feature matrix (the sequence model also sees
the windows' events as tokens).

| Detector | Idea | Good at | Blind to |
| --- | --- | --- | --- |
| `zscore` | Mean of the log of the 3 largest robust z-scores. One line of arithmetic. | Any single feature far out of range | Combinations of ordinary values |
| `iforest` | Isolation Forest: 200 random trees on subsamples of 256; points that are isolated in few splits are odd. | Odd combinations in dense data | Sparse counts (see below) |
| `autoencoder` | 46 → 32 → 8 → 32 → 46, tanh, Adam, early stopping on a 10% holdout. Score is the rebuild error. | Windows that do not look like any normal window, with the reason per column | Attacks that look like heavy normal traffic |
| `sequence` | A small network predicts the next event from the last 4 (embedding 8, hidden 48), inspired by DeepLog. Each event is a token of (kind, speed: fast / steady / slow). Score is the mean surprise. | Wrong *order*: success right after a burst of failures, a login followed by account creation | Volume: a thousand perfectly normal requests are not surprising |
| `vote` | Flagged when 2 of the 3 learned detectors flag it | Fewer false alerts | Attacks only one model sees |

The baseline is in the table on purpose: if a one-line z-score does as well as a neural network
on a given data set, you should use the one-liner. [Results](results.md) shows where that is true.

## How the comparison is made

- **Data.** Labelled synthetic logs ([synth.py](../src/loganomaly/synth.py)): normal web traffic
  (visitors, crawlers, a health check, messy query strings), SSH users, deploys and backups,
  a Windows desktop, plus twelve kinds of attack at a known rate. Every attack is a *campaign*
  with a start, an end and the entities involved.
- **Split.** Train on the first days (attack-free unless `dirty` is set), test on the rest.
- **Labels.** A test window is positive when its entity is in a campaign that overlaps the
  window. Campaigns that are deliberately low and slow (`ssh_slow`) are *excluded from the window
  metrics* and reported only as "campaigns found", since flagging them window by window is not
  the question; the question is whether anyone notices.
- **Metrics.**
  - *AUC* and *average precision* rank windows; they do not need a threshold.
  - *Precision*, *recall* and *F1* use each detector's own threshold (99.5th percentile of
    training scores), which is what a user would get.
  - *False alerts per day* counts flagged windows that are not attacks.
  - *Campaigns found* counts attacks with at least one flagged window.
- **Seeds.** Every number in [results](results.md) is a mean ± standard deviation over several
  independently generated data sets.
- **Dirty training.** A realistic problem: the training days already contain some attacks. The
  benchmark repeats the comparison with 30% of the test attacks hidden in the training days.

Synthetic data is the only way to know the ground truth, and it is also the main caveat: it shows
that the models behave as designed, not how they would score on your network. See
[limitations](limitations.md).

## Why the Isolation Forest is weak here

Isolation Forest picks a random feature and a random cut between the minimum and maximum. In this
data most columns are zero in nearly every window, so most random cuts separate nothing, and a
value far outside the training range is no easier to isolate than one just outside it (the
forest cannot extrapolate beyond the range it saw). That is a property of the algorithm on sparse
counts, not of this implementation: it ranks the same points as scikit-learn's
(`tests/test_models.py::test_isolation_forest_matches_sklearn`).

## Reasons

A score says "unusual", not why. For a flagged window the report shows the features that fall
outside the range 99% of training windows stay inside (0.5th to 99.5th percentile), sorted by how
far out they are, and only those that apply to the window. When no single value is out of range
it says so and lists the features that are unusual *together*. For the sequence model the reason
is a transition: "ssh_accepted (slow) after ssh_failed_password (fast) has probability 0.004;
expected ssh_failed_password (fast) (0.91)".

## Thresholds and the cost of a false alert

The default quantile (0.995) accepts about one false alert per 200 normal windows. At
one-minute windows on a busy server that is a lot; at five minutes on a small one it is a few per
day. Use `--quantile 0.999` for fewer alerts at the price of recall; the benchmark prints both.
