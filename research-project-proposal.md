# Research Project Proposal

## Learning Without Letting Go: Predictive Coding Beyond Backpropagation in Continual Learning

**Status:** complete living proposal; wording and experimental mappings are
refined as evidence accumulates.

**Last synchronized with project results:** 2026-07-21.

## Motivation

Backpropagation (BP) is highly effective for training artificial neural
networks, but its standard implementation does not directly describe a known
biological learning mechanism. Important objections include weight transport,
global error propagation, synchronized layer-wise credit assignment, and the
relationship between neural activity and weight updates [1]. These objections
do not imply that the brain cannot implement BP-like computations; they motivate
testing learning rules with more local information pathways.

Predictive Coding (PC) offers a local-error formulation in which neural activity
is inferred before plasticity updates are applied [2]. This ordering and its
local prediction-error signals make PC a useful candidate for studying whether
biologically motivated learning dynamics produce different plasticity,
stability, robustness, or computational signatures from BP.

**Working hypothesis:** PC may be more robust than BP in online or continual
learning because its inference-before-plasticity dynamics could reduce harmful
interference. This is a falsifiable project hypothesis, not an established
fact. The project therefore compares PC and BP under matched static and
single-pass continual-learning protocols and reports plasticity jointly with
forgetting and resource cost.

## Three Research Questions

### Q1 — Static learning

What are the differences between Predictive Coding and Backpropagation learning
techniques when they use the same ANN architecture trained on the static MNIST
digits dataset?

Primary evidence: learning curves, early validation gain, final train/test
accuracy, loss, PC inference dynamics, runtime, parameter count, latent-state
memory, and paired-seed differences. Covariance or representation metrics may
be added when they answer a specific mechanism question.

### Q2 — Continual-learning behavior

Under a fixed single-pass continual-learning protocol, how do Predictive Coding
and Backpropagation differ in online accuracy, adaptation speed, forgetting,
backward transfer, and loss of plasticity?

Primary evidence: predict-before-update prequential accuracy, task-boundary
accuracy matrices, adaptation gain, average forgetting, backward transfer,
final average accuracy, representation drift, and label-alignment gain.

### Q3 — Inference budget

How does the Predictive Coding inference budget affect its
stability–plasticity behavior and computational cost in continual learning?

Primary intervention: the number of PC relaxation steps. Primary evidence:
static learning, continual adaptation and prequential accuracy, forgetting,
backward transfer, runtime, memory, and objective descent.

## Current Evidence Mapping

| Question | Implemented evidence | Current answer | Status |
|---|---|---|---|
| Q1 | Matched static MNIST BP–PC runs and `T=1/5/10/20` learning curves across seeds `7/42/123` | PC reaches BP-like static accuracy once relaxation is sufficient; `T=1` is propagation-deficient and extra relaxation adds cost without monotonic accuracy benefit. | Answered for the frozen MLP protocol |
| Q2 | SplitMNIST, Permuted MNIST, SplitCIFAR-10, accuracy matrices, prequential metrics, forgetting/BWT, RDM drift and label alignment | PC is BP-like on Permuted MNIST. In class-incremental settings, lower forgetting and lower drift accompany weaker acquisition; neither method solves catastrophic forgetting. | Descriptively answered; causal mechanism unresolved |
| Q3 | Relaxation sweep `T=1/5/10/20`, matched BP control, runtime and memory accounting | `T=5` is the cheapest effective tested default for the two-layer MLP. Higher `T` does not monotonically improve stability or plasticity, while runtime rises consistently. | Answered on the tested finite grid |

## Planned Refinements Without Replacing Q1–Q3

1. **Q1 refinement — comparison validity:** separate conventional one-update
   BP, archived repeated-update RBP, PC-last, and PC-all under validation-only
   selection and explicit sample/update/wall-clock budgets.
2. **Q2 refinement — causal stability test:** match new-task acquisition by
   intervening on early-layer plasticity, then test whether lower representation
   drift still predicts lower forgetting.
3. **Q3 refinement — depth × relaxation:** test whether the required relaxation
   margin grows with model depth using per-layer convergence diagnostics.

These are follow-up experiments under the existing questions, not replacement
research questions.

## References

1. Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton,
   G. (2020). *Backpropagation and the brain*. Nature Reviews Neuroscience,
   21, 335–346.
2. Song, Y., et al. (2024). *Inferring neural activity before plasticity as a
   foundation for learning beyond backpropagation*. Nature Neuroscience, 27,
   348–358.

## Claim Boundary

The proposal motivates a comparison. It does not assume that PC is already
more biologically plausible in every dimension, more robust than BP, or a
solution to catastrophic forgetting. Those conclusions require matched
experimental evidence and an explicit biological-plausibility audit.
