# PC Relaxation-Step Experiment Report

## Answer

Increasing PC relaxation from one to five steps changes performance from a
propagation-deficient boundary condition to BP-like static and continual
learning. Beyond five steps, the effect is not monotonic and depends on the
setting: static accuracy is already saturated; SplitMNIST shows less measured
forgetting but progressively weaker adaptation; Permuted MNIST has its highest
tested PC final mean at `T=10`, followed by a decline at `T=20`. Runtime is the
only consistently monotonic outcome, rising to roughly `3.7–4.2×` BP at
`T=20`. No tested depth is globally optimal or supported as a project default.

## Experimental protocol

- Intervention: PC relaxation steps `T∈{1,5,10,20}` only.
- Control: one freshly rerun BP condition reused across PC depths.
- Repetitions: paired seeds `7`, `42`, and `123`.
- Shared model: `784→256→256→10`, ReLU, 269,322 learned parameters.
- Shared objective/optimizer: one-hot half squared error; Adam `lr=0.001`.
- PC latent optimizer: SGD `lr=0.01`.
- Static: full MNIST, ten epochs; plasticity proxies are epoch-1 validation
  gain/accuracy and the validation-curve mean.
- Continual: one pass per task, no replay and no model task ID; SplitMNIST is
  class-incremental and Permuted MNIST is domain-incremental. Plasticity
  proxies are adaptation gain and predict-before-update prequential accuracy.
- Environment: CyberEngine ROCm container recorded in the raw artifact.

Full deterministic tables are in [`artifacts/tables.md`](artifacts/tables.md),
and the evidence-level analysis is in [`04-evaluation.md`](04-evaluation.md).

## Findings

### 1. Five steps cross the useful propagation threshold

**Established fact:** `T=1` loses `13.81` pp static final accuracy,
`7.10` pp SplitMNIST final accuracy, and `40.54` pp Permuted-MNIST final
accuracy versus BP on the paired means. The upstream implementation warns that
two PC layers need at least three iterations for error propagation.

**Interpretation:** The large `T=1→5` jump should be read as crossing from a
degenerate propagation-deficient condition into effective relaxation, not as
evidence for smooth improvement with every additional step.

### 2. Static performance saturates at five tested steps

**Established fact:** PC final static accuracy at `T=5/10/20` is
`98.193/98.183/98.137%`, versus BP `98.267%`. Early validation gain and the
whole validation curve are effectively tied across these conditions.

**Interpretation:** Extra relaxation beyond five tested steps buys no material
static accuracy or plasticity benefit. `T=5` is the cheapest adequate tested
setting, but untested `T=3/4` prevent an optimum claim.

### 3. SplitMNIST exposes a stability–plasticity trade-off

**Established fact:** From `T=5→10→20`, mean adaptation falls
`93.949→91.160→89.829%`, prequential accuracy falls
`57.512→56.487→54.547%`, and forgetting falls
`95.687→94.487→92.373%`. Final accuracy is non-monotonic and below BP on the
mean at every depth.

**Interpretation:** Deeper relaxation appears more stable only alongside
weaker acquisition. This does not solve catastrophic forgetting and should not
be described as a continual-learning win.

### 4. Permuted MNIST favors an intermediate candidate, not an optimum

**Established fact:** PC final accuracy is `91.326/91.806/91.037%` at
`T=5/10/20`, while BP is `91.544%`. `T=10` beats the other two effective PC
depths within all three seeds, but its paired difference versus BP is
`+0.916/−0.380/+0.250` pp.

**Interpretation:** An intermediate depth may help domain-incremental
retention, but the small mixed-sign BP comparison and three-seed sample make
`T=10` only a replication candidate.

### 5. Cost scales more reliably than accuracy

**Established fact:** At `T=20`, synchronized timed processing is
`4.23×` BP in static, `4.14×` in SplitMNIST, and `3.73×` in Permuted MNIST.
Learned parameter count is unchanged; PC still requires temporary latent
state.

**Interpretation:** Complexity strongly penalizes large `T` when accuracy has
already saturated or become non-monotonic.

## Decision

- Do not use `T=1` except as a failure/boundary check.
- Keep `T=5` as the compute-efficient finite-grid candidate for static and
  SplitMNIST follow-up, without calling it optimal.
- Keep `T=10` as a Permuted-MNIST replication candidate only.
- Do not promote any relaxation depth to the project default from this study.
- If another sweep is approved, test `T=3/4` around the propagation threshold
  and repeat the `T=5/10` candidates with more paired seeds.

## Limitations

- `n=3`; means and sample SDs are descriptive, not a significance claim.
- The sweep changes iteration count, not a matched wall-clock or energy budget.
- Static/Split/Permuted scores are separate and must not be pooled.
- Plasticity metrics are operational learning measures, not direct biological
  evidence about synapses or brains.
- Low forgetting can be caused by weak learning; adaptation and prequential
  accuracy must accompany it.
- BWT and forgetting are algebraic sign inverses in these SplitMNIST results.
- Runtime is hardware-, container-, and execution-order-specific.
- No matched hyperparameter search was performed separately for each `T`.

## Reproduction

```bash
bash playground/step1-static-mnist/remote-run-relaxation-sweep.sh \
  --relaxation-steps 1 5 10 20 \
  --seeds 7 42 123 \
  --output playground/step1-static-mnist/runs/relaxation-sweep-v1-reproduction.json

python3 playground/step1-static-mnist/relaxation_analysis.py \
  --input playground/step1-static-mnist/runs/relaxation-sweep-v1-reproduction.json \
  --output-dir agents/research/pc-relaxation-plasticity/artifacts/reproduction
```

## Research Graph

Graph nodes `R-static`, `R-split`, and `R-permuted` encode the three findings.
They support decision `D-no-default`: retain finite-grid candidates for follow-up
without promotion. See [`research_graph.mmd`](research_graph.mmd) and
[`research_graph.json`](research_graph.json).
