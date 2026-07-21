# Evaluation

Raw source: [`relaxation-sweep-v1.json`](../../../playground/step1-static-mnist/runs/relaxation-sweep-v1.json)

Deterministic derivatives: [`tables.md`](artifacts/tables.md),
[`summary.json`](artifacts/summary.json), and
[`summary.csv`](artifacts/summary.csv).

All values are mean ± sample SD across paired seeds `7`, `42`, and `123`.
Percentage differences are percentage points. The run used the CyberEngine
ROCm GPU and passed initialization, task-stream, finite-value, and artifact
completeness checks.

## H1.E1 — Static MNIST

**E:** BP final test accuracy was `98.267 ± 0.154%`. PC obtained
`84.457 ± 0.175%` at `T=1`, `98.193 ± 0.120%` at `T=5`,
`98.183 ± 0.117%` at `T=10`, and `98.137 ± 0.199%` at `T=20`.
The paired PC−BP final differences were:

- `T=1`: `−13.680`, `−14.060`, `−13.690` pp;
- `T=5`: `−0.020`, `−0.060`, `−0.140` pp;
- `T=10`: `+0.070`, `−0.060`, `−0.260` pp;
- `T=20`: `−0.180`, `−0.090`, `−0.120` pp.

**E:** Epoch-1 validation gain was `84.592%` for BP and
`84.614/84.614/84.675%` for PC at `T=5/10/20`. Their mean post-update
validation-curve values differed from BP by at most `0.028` pp.

**I:** Once error propagation is effective, static learning saturates: more
than five tested steps produced no material accuracy or learning-speed gain,
while synchronized timed training-loop cost rose monotonically. `T=5` is an
efficiency candidate on this finite grid, not an optimum, because `T=3/4`
were not tested and the `T≥5` accuracy differences are tiny.

Verdict: result kept as a saturation/threshold observation; no superiority,
optimality, or default-setting promotion.

## H1.E2 — SplitMNIST

**E:** BP final/prequential/adaptation/forgetting values were
`18.541/60.144/94.561/96.188%`. PC results were:

| T | Final average | Prequential | Adaptation gain | Forgetting |
|---:|---:|---:|---:|---:|
| 1 | 11.441 ± 1.465 | 12.877 ± 1.501 | 56.239 ± 0.957 | 57.160 ± 2.361 |
| 5 | 18.329 ± 0.115 | 57.512 ± 1.264 | 93.949 ± 1.060 | 95.687 ± 0.314 |
| 10 | 16.500 ± 3.896 | 56.487 ± 0.758 | 91.160 ± 4.162 | 94.487 ± 1.644 |
| 20 | 16.860 ± 0.619 | 54.547 ± 2.756 | 89.829 ± 3.559 | 92.373 ± 3.595 |

**E:** The `T=10` paired final differences were `+0.151`, `+0.444`, and
`−6.717` pp, exposing a seed-123 collapse. `T=20` was below BP in all three
seeds. BWT is exactly the negative of forgetting here and is not independent
corroboration.

**I:** From `T=5→10→20`, adaptation and prequential performance decline while
measured forgetting improves. Lower forgetting is therefore confounded by
weaker task acquisition: there is less learned performance available to lose.
All settings still catastrophically forget, and no PC depth is a winner.

Verdict: trade-off observation kept; selection remains inconclusive.

## H1.E3 — Permuted MNIST

**E:** BP final average accuracy was `91.544 ± 0.427%`. PC obtained
`51.008 ± 1.099%` at `T=1`, `91.326 ± 0.499%` at `T=5`,
`91.806 ± 0.566%` at `T=10`, and `91.037 ± 1.147%` at `T=20`.
Paired PC−BP final differences were:

- `T=5`: `+0.658`, `−0.752`, `−0.560` pp;
- `T=10`: `+0.916`, `−0.380`, `+0.250` pp;
- `T=20`: `+0.288`, `−1.792`, `−0.018` pp.

**E:** `T=10` beat `T=5` and `T=20` within each of the three seeds, and its
prequential delta versus BP was positive but small in every seed. Its final
delta versus BP still had mixed signs; adaptation and forgetting differences
also had mixed signs.

**I:** `T=10` is the best tested PC point for mean Permuted-MNIST final
accuracy, but the effect is small and seed-dependent. It is a follow-up
candidate, not evidence of superiority or a global optimum.

Verdict: inconclusive for promotion; repeat with more seeds before selection.

## Runtime and boundary condition

**E:** PC/BP synchronized timed training-loop ratios for `T=1/5/10/20` were:

- static: `1.018/1.775/2.574/4.232×`;
- SplitMNIST: `1.006/1.743/2.646/4.137×`;
- Permuted MNIST: `1.033/1.686/2.413/3.730×`.

These timers include loading/transfers and continual predict-before-update
work. One diagnostic batch is excluded symmetrically per static run and per
continual task. The fixed BP→T1→T5→T10→T20 order, one accelerator, and one
container make runtime secondary and hardware-specific.

`T=1` is below the upstream two-PC-layer propagation minimum (`T≥3`). It has
no prior latent update informing the final parameter gradient and only one
objective value, so objective decrease is not assessable. It is a degenerate
boundary condition rather than an ordinary low-depth operating point.

## Result-skeptic review

The independent review passed artifact validity and rejected monotonic
accuracy, global optimum, superiority, and project-default claims. It required
paired-seed disclosure, joint stability/plasticity interpretation, explicit
`T=1` treatment, and hardware/order runtime limitations; all are incorporated
above.
