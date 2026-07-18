# PC Network Scale Design

## Round 1 — 2026-07-17T22:55:56Z

### Constraints

- Budget: one combined sweep, no more than six CyberEngine GPU-hours.
- Parallelism: serial conditions on one GPU.
- Frozen: data, streams, metrics, seeds, optimizer, objective, batch size,
  epochs, replay, task information, and PC `T=5`.
- Control: freshly rerun `256×2` architecture.

### Shortlist

| ID | Hypothesis | Intervention | Control | Metrics | Seeds | Decision gate | Parallelism |
|---|---|---|---|---|---|---|---|
| H1.E1 | Establish shared control. | `256×2`. | none | all planned metrics | 7,42,123 | valid matched reference | serial GPU |
| H1.E2 | Greater width may improve capacity-dependent performance. | `512×2`. | H1.E1 | accuracy, plasticity, forgetting, cost | 7,42,123 | ≥0.5 pp final gain without >1 pp plasticity loss | serial GPU |
| H2.E1 | Greater depth may change propagation and stability. | `256×4`. | H1.E1 | same | 7,42,123 | same | serial GPU |

### Schedule

1. Dependency-free checks and three-architecture GPU smoke.
2. Independent design/implementation review.
3. One full combined sweep, deterministic parser, result-skeptic review, report.

### Rationale

The design isolates width and depth against the same fresh baseline rather
than conflating them in a single larger model. Fixed `T=5` isolates architecture
and satisfies the maximum PC-layer propagation minimum.

### Managed Review

- Reviewer: design critic and implementation-intent reviewer
- Verdict: pass with warnings
- Blocking issues addressed: frozen `T=5` enforced; expected parameters,
  diagnostic descent, split/task checksums, and finite values asserted;
  architecture-minus-baseline deltas exposed in JSON, CSV, and Markdown.
- Remaining warnings: deep PC uses the minimum sufficient relaxation depth;
  architecture-specific optimization is out of scope; runtime order is fixed;
  three seeds do not support general superiority.
