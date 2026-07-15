# Coverage Audit

- Active cells: 34
- Closed cells: 34
- Weak cells: 0
- Gap cells: 0
- Bias triggers: not determined; four required bias buckets were not quantified
- Decision: mechanism coverage passes; synthesize with explicit bias limits

## Critical safeguards carried into synthesis

1. “Biologically plausible” is decomposed into locality, feedback symmetry,
   global teaching signals, phases/timing, differentiability, temporal memory,
   neuron model and persistent/replay state; it is not a scalar truth label.
2. Online temporal memory does not imply continual retention. E-prop, DECOLLE,
   OSTL, OTTT and S-TLLR answer temporal locality differently from catastrophic
   interference or loss of plasticity.
3. Predictive coding can approximate BP only under specific architectures and
   inference/update schedules; P013 is required as a corrective.
4. The target-aligned 2026 endpoint, P049, still uses a broadcast teaching signal and
   is a single recent study. It is a frontier node, not a final solution.

## Audit limitation

The coverage matrix is complete for the target-directed mechanism questions,
but this is not a full systematic-review bias pass. Institution, country,
venue-history and deployment-regime distributions were not extracted; the
survey makes no balance claim for those dimensions.
