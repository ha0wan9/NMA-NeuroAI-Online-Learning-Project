# Network-scale evaluation

Primary raw source: [`architecture-sweep-v2.json`](../../../playground/step1-static-mnist/runs/architecture-sweep-v2.json).

Deterministic derivatives: [`tables.md`](artifacts/tables.md),
[`summary.json`](artifacts/summary.json), and
[`summary.csv`](artifacts/summary.csv).

The pre-fix v1 run is invalid, preserved, and excluded from every value below.
All results are v2 means ± sample SD over paired seeds `7`, `42`, and `123`.

## H1.E1 — Shared 256×2 control

**E:** Learned parameters are 269,322. BP/PC final accuracy is
`98.267/98.193%` static, `18.541/18.329%` SplitMNIST, and
`91.544/91.326%` Permuted MNIST. Initialization, split, and task-stream checks
match the frozen protocol.

**I:** This is a valid architecture control. Its small BP–PC gaps do not prove
equivalence, but it is fit for paired architecture deltas.

Verdict: kept as baseline reference.

## H1.E2 — Width 512×2

Learned parameters rise to 669,706 (`2.487×` baseline; +400,384).

### Static MNIST

**E:** Final architecture-minus-baseline differences are
`+0.223 ± 0.040` pp for BP and `+0.253 ± 0.047` pp for PC. Paired values are
`+0.260/+0.180/+0.230` pp for BP and `+0.270/+0.200/+0.290` pp for PC.
Epoch-1 gains have mixed seed signs.

**I:** Width gives a small, consistent final improvement but misses the
registered `+0.5` pp scale-benefit threshold.

### SplitMNIST

**E:** BP final gain is `+0.377 ± 0.248` pp, below threshold. PC final gain is
`+0.528 ± 0.172` pp with paired values `+0.353/+0.535/+0.696` pp, satisfying
the registered mean gate. PC adaptation improves `+2.575` pp and prequential
accuracy `+7.914` pp. PC forgetting worsens `+1.467` pp to `97.154%`.

**I:** Wider PC has a narrow acquisition/plasticity benefit on SplitMNIST, but
it does not solve catastrophic forgetting and should not be generalized to
continual learning overall.

### Permuted MNIST

**E:** BP final gain is `+0.990 ± 1.260` pp with paired values
`+2.330/−0.172/+0.812`; it passes the mean/plasticity gate but has one negative
seed. PC final change is `−0.069 ± 1.377` pp with paired values
`+1.376/−1.366/−0.216`. Wide PC trails wide BP in every seed by
`−0.296/−1.946/−1.588` pp, and PC forgetting worsens `+1.266` pp.

**I:** Width is a BP-specific Permuted-MNIST candidate here. PC gains
prequential/adaptation performance but does not convert it into retained final
accuracy under the frozen configuration.

### Cost

**E:** Peak allocated memory increases approximately `7.1 MiB` for BP and
`15.0 MiB` for PC. Timed-processing ratios versus the same-method baseline are
roughly `1.00–1.10×` on this GPU, while parameter storage increases `2.49×`.

Verdict: partial/narrow support; keep as setting-specific follow-up, not a
default or general scale benefit.

## H2.E1 — Depth 256×4

Learned parameters rise to 400,906 (`1.489×` baseline; +131,584).

**E:** Static final change is `−0.160` pp for BP and `−0.660` pp for PC. Deep
PC epoch-1 gain falls `−2.814` pp, violating the plasticity guardrail.

**E:** SplitMNIST final change is `−6.367` pp for BP and `−6.579` pp for PC;
adaptation falls `−27.373/−26.417` pp. Lower forgetting is caused by far weaker
acquisition and is not a stability benefit.

**E:** Permuted-MNIST final change is `−7.922` pp for BP and `−39.941` pp for
PC. Deep-PC paired losses are `−40.686/−36.574/−42.562` pp. Forgetting worsens
`+10.242` pp for BP and `+45.733` pp for PC. Deep PC finishes at `51.385%`
versus deep BP `83.622%`.

**I:** Additional plain-MLP depth is consistently harmful under the fixed
learning rate and training budget. The severe deep-PC Permuted-MNIST result is
consistent across seeds and passed implementation diagnostics, so it is a
substantive negative result rather than a corrupted run. However, `T=5` is
exactly the propagation minimum for four PC layers; the result cannot establish
that depth is intrinsically incompatible with sufficiently relaxed PC.

**E:** Peak memory rises about `2.0 MiB` for BP and `7.9 MiB` for PC. Timed
processing rises approximately `1.08–1.21×` for BP and `1.29–1.36×` for PC.

Verdict: rejected under the frozen protocol; no promotion.

## Independent result-skeptic review

The reviewer found no mandatory rerun anomaly. It accepted the baseline,
classified width as partial scenario-specific support, rejected depth under
the frozen protocol, and rejected every global scale-law or default-architecture
claim. Required paired-sign, weak-learning, minimum-relaxation, cost, and
three-seed caveats are incorporated above.
