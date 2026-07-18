# PC Relaxation Plasticity Design

## Round 1 — 2026-07-17T21:53:49Z

### Constraints

- Budget: one full sweep, no more than six GPU-hours.
- Parallelism: one CyberEngine process to avoid GPU contention.
- Editable surface: Step 1 harness and this study directory.
- Protected protocol: data, streams, model, objective, optimizer, seeds, metrics.
- Baseline/control: one freshly rerun BP control shared across PC levels.
- Data version: torchvision MNIST with split seed `20260717`.

### Shortlist

| ID | Hypothesis | Intervention | Control | Diff | Metrics | Seeds | Cost | Decision gate | Dependency | Parallelism |
|---|---|---|---|---|---|---|---|---|---|---|
| H1.E1 | More relaxation changes static learning speed/final accuracy. | PC `T=1/5/10/20`. | Fresh BP. | `T` only. | Epoch-1 gain, curve mean, final test accuracy, runtime. | 7,42,123 | 10 epochs each | Direction/magnitude; monotonic only if all adjacent means agree. | Smoke pass | serial GPU |
| H1.E2 | More relaxation changes SplitMNIST plasticity/stability. | PC `T=1/5/10/20`. | Fresh BP. | `T` only. | Adaptation, prequential, final accuracy, forgetting/BWT, runtime. | 7,42,123 | one pass/task | Joint plasticity/stability interpretation. | Smoke pass | serial GPU |
| H1.E3 | More relaxation changes Permuted-MNIST plasticity/stability. | PC `T=1/5/10/20`. | Fresh BP. | `T` only. | Same as H1.E2. | 7,42,123 | one pass/task | Same gate, reported separately. | Smoke pass | serial GPU |

### Schedule

1. Run dependency-free tests and a remote full-path smoke sweep.
2. Launch the single combined full sweep and preserve its raw JSON.
3. Generate deterministic tables, evaluate checksums/dynamics, and synthesize.

### Rationale

The combined runner realizes one-factor conditions and a single fresh BP
control while avoiding four redundant BP reruns. Every PC condition still
uses the same per-seed initialization and stream as its control.

### Managed Review

- Reviewer role: design critic and implementation-intent reviewer
- Verdict: pass with warnings
- Blocking issues addressed: deterministic parser added; initialization and
  stream checksums plus finite metrics enforced; diagnostic batches excluded
  symmetrically from synchronized timed-sample accounting.
- Remaining warnings: `T=1` is below the two-layer propagation minimum; fixed
  ascending execution order weakly confounds secondary runtime comparisons;
  three seeds do not support broad superiority or optimality claims.
