# Continual MNIST PC–BP Reference Summary

Protocol: `continual-mnist-pc-bp-v1`

Raw artifact: [`continual-reference-v1.json`](continual-reference-v1.json)

This is a matched three-seed run on the CyberEngine ROCm backend. Values below
are mean ± sample standard deviation across paired seeds `7`, `42`, and `123`.
Accuracy-like values are shown as percentage points. `PC − BP` is computed
within each seed before aggregation.

## Protocol

- Model: `784 → 256 → 256 → 10`, ReLU, 269,322 learned parameters.
- Shared objective and parameter optimizer: one-hot half squared error and
  Adam with learning rate `0.001`.
- PC: 20 relaxation steps, latent-state SGD learning rate `0.01`, parameter
  update at the final relaxation step.
- Stream: one pass per task, batch size 500, no replay buffer, no task ID
  supplied to either model.
- Evaluation: predict-before-update prequential accuracy and a full task test
  matrix after every task boundary.
- SplitMNIST: five disjoint class-pair tasks with a shared 10-way output.
- Permuted MNIST: identity plus four fixed pixel permutations; each task
  revisits the base images under a different domain transformation.

## Permuted MNIST

| Metric | BP | PC | PC − BP |
|---|---:|---:|---:|
| Final average accuracy | 91.385 ± 0.537 | 91.841 ± 0.456 | +0.457 ± 0.337 |
| Prequential accuracy | 86.957 ± 0.179 | 87.085 ± 0.326 | +0.128 ± 0.149 |
| Backward transfer | −4.902 ± 0.800 | −4.347 ± 0.668 | +0.555 ± 0.319 |
| Average forgetting | 4.902 ± 0.800 | 4.347 ± 0.668 | −0.555 ± 0.319 |
| Mean adaptation gain | 84.409 ± 0.380 | 84.830 ± 0.511 | +0.421 ± 0.280 |
| Stream processing time | 6.107 ± 0.069 s | 22.686 ± 0.673 s | 3.715 ± 0.140× |

The paired final-accuracy differences were `+0.822`, `+0.390`, and `+0.158`
percentage points for seeds `7`, `42`, and `123`.

**Established fact:** PC has numerically higher mean final, prequential,
adaptation, and backward-transfer values in this matched artifact, and lower
mean forgetting. PC requires approximately 3.72 times the stream-processing
time.

**Interpretation:** The direction is consistent across the three paired seeds
for final accuracy and forgetting, but the magnitude is small. Three seeds are
not enough to claim robust superiority or statistical significance.

## SplitMNIST

| Metric | BP | PC | PC − BP |
|---|---:|---:|---:|
| Final average accuracy | 18.548 ± 0.272 | 17.173 ± 0.391 | −1.375 ± 0.552 |
| Prequential accuracy | 60.139 ± 0.570 | 54.583 ± 2.854 | −5.556 ± 2.663 |
| Backward transfer | −96.183 ± 0.303 | −92.448 ± 3.428 | +3.735 ± 3.153 |
| Average forgetting | 96.183 ± 0.303 | 92.448 ± 3.428 | −3.735 ± 3.153 |
| Mean adaptation gain | 94.564 ± 1.075 | 90.201 ± 3.002 | −4.363 ± 2.256 |
| Stream processing time | 1.104 ± 0.093 s | 4.448 ± 0.356 s | 4.058 ± 0.579× |

The paired final-accuracy differences were `−1.957`, `−0.857`, and `−1.311`
percentage points for seeds `7`, `42`, and `123`.

**Established fact:** Both methods show severe class-incremental forgetting.
PC has lower final and prequential accuracy and lower adaptation gain. Its BWT
is less negative and its forgetting value is numerically smaller.

**Interpretation:** The smaller PC forgetting value is not independent evidence
of better stability: PC also learns each arriving task less strongly, leaving
less peak performance available to lose. The joint result is a
stability–plasticity trade-off with weaker plasticity, not a clear PC win.

## Resource differences

Both methods have 269,322 learned parameters. At batch size 500, PC reaches a
peak of 256,000 temporary latent-state elements. Peak accelerator memory was:

| Scenario | BP | PC | PC − BP |
|---|---:|---:|---:|
| Permuted MNIST | 121,751,040 B | 127,454,720 B | +5,703,680 B |
| SplitMNIST | 121,543,168 B | 127,454,720 B | +5,911,552 B |

## Limitations and claim status

- **Established fact:** All 30 recorded first-batch PC relaxation objectives
  decreased, and all paired initialization checksums matched.
- **Limitation:** This is one shared hyperparameter configuration, not a
  matched hyperparameter search.
- **Limitation:** Batch size 500 is mini-batch continual learning, not strict
  sample-at-a-time online learning.
- **Limitation:** Repeated test evaluation is used only for the fixed accuracy
  matrix and did not affect training or tuning.
- **Limitation:** SplitMNIST with a shared 10-way head and no replay is a severe
  class-incremental setting; conclusions do not transfer automatically to
  task-incremental evaluation.
- **Limitation:** Permuted MNIST revisits the same base images under new domains,
  whereas SplitMNIST exposes each base image in one class-pair task. Scores
  should not be averaged across the two benchmarks.
- **Open question:** Whether the small Permuted-MNIST differences persist with
  more seeds, a predeclared paired confidence interval, or a matched tuning
  budget.

No biological-superiority or general continual-learning-superiority claim is
supported by this artifact alone.
