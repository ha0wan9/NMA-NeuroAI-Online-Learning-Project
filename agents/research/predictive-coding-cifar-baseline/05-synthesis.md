# Predictive Coding CIFAR-10 baseline report

## Answer

The paper-backed static result is reproducible on this ROCm system: every BP
and PC method/seed cell falls within the preregistered ±1 percentage point
Figure 4i best-test target, with matched initialization/data and valid PC
relaxation. The same matched system does not yield a continual-learning win on
replay-free SplitCIFAR-10. PC forgets less, but it also has lower prequential,
adaptation, and final accuracy.

For research Question 2, the representation result explains what kind of
trade-off is occurring: PC changes old-task RDM geometry less in every measured
layer, with almost frozen early convolutional geometry, while also developing
less label-aligned structure. This is evidence for a shift toward stability
through reduced plasticity, not evidence that drift causes forgetting or that
the resulting trade-off is favorable.

## Protocol and provenance

- Static H1: archived seeds `1482555873`, `698841058`, `2283198659`; RBP and
  PC; 80 passes; source-winning per-method/per-seed learning rates; best-test
  accuracy compared with archived targets.
- Continual H2: paired seeds `7`, `42`, `123`; five class-pair tasks; shared
  ten-way head; one pass/task; no replay and no model task ID.
- Environment: CyberEngine Radeon ROCm container; exact Python, Torch,
  torchvision, image, device, source revision, and checksums are recorded in
  the immutable JSON artifacts.
- Representations: 20 fixed anchors/class, four layers, six task-boundary
  checkpoints, and a diagnostics-off same-revision behavioral control.
- Tracking: TensorBoard mirrors and self-contained Plotly artifacts are under
  `playground/step2-cifar10/runs/tensorboard/`.

Detailed evidence and claim labels are in [`04-evaluation.md`](04-evaluation.md).

## Confirmed findings

### Static reproduction passed

**Established fact:** H1.E2 reproduced BP best accuracy at
`68.9660 ± 0.3376%` and PC at `71.2619 ± 0.8729%`; all six cells passed the
registered tolerance. Mean absolute target error was `0.337` pp for BP and
`0.514` pp for PC.

**Interpretation:** This validates the implemented archived Figure 4i
configuration as a baseline on the current hardware/software stack. It does
not validate leakage-free selection, Figure 4j, or field-wide superiority.

### Continual learning exposes a trade-off

**Established fact:** PC's mean SplitCIFAR-10 forgetting is `81.421%` versus
BP's `85.954%`, but its acquisition, prequential, and final-accuracy metrics
are all lower. Both methods end near `17–18%` final average accuracy.

**Interpretation:** Stability cannot be evaluated alone. In this system, PC's
reduced forgetting accompanies weaker new-task learning and does not solve
catastrophic forgetting.

### Representations identify reduced plasticity, not a causal mechanism

**Established fact:** At the final boundary, mean PC old-task RDM drift is
lower than BP at `conv1`, `conv2`, `fc1`, and logits by `0.009335`, `0.024882`,
`0.066082`, and `0.081923`. All three seed-level layer means have the same
direction. Diagnostics-on behavior exactly matches the same-revision control.

**Established fact:** PC also has smaller initial-to-final label-alignment
gains at all four layers and lower adaptation, prequential accuracy, and final
accuracy. The simple drift–forgetting correlations are negative or inconsistent
and task-confounded.

**Interpretation:** Representation drift should be reported jointly with
plasticity and behavior. In this experiment, lower drift is partly a signature
of under-learning rather than an independently beneficial retention mechanism.

### Cost remains inside the approved envelope

**Established fact:** Static synchronized update time was `5.7466` hours;
PC took `1.228×` BP update time and about `72.9 MB` more peak allocated
accelerator memory. H2.E3 and its instrumentation control add approximately
`0.1498` stream-plus-evaluation accelerator-hours, keeping the documented
study total near `5.98` hours and below the ten-hour ceiling.

## Decision

- Adopt H1.E2 as the project's paper-protocol static CIFAR-10 baseline.
- Adopt H2.E2 as the matched replay-free SplitCIFAR-10 behavioral baseline.
- Adopt H2.E3 as an observational representation baseline and answer RQ2 with
  joint RDM-drift, label-alignment, adaptation, prequential, forgetting, and
  final-accuracy measurements.
- Do not claim general PC superiority: static best-test reproduction and
  continual efficacy are separate claims.

## Next steps

1. Preserve the current content-hash provenance manifest with the immutable
   artifacts. Before broad reuse, create a reviewed commit and separately
   correct the signed field named `absolute_target_difference`; do not rewrite
   the completed raw artifact.
2. Label the static control as the archived repeated-update RBP/paper-protocol
   BP. If the project needs a conventional one-update BP/SGD comparator,
   register and run it separately rather than relabeling this artifact.
3. Use the full TensorBoard/Plotly timelines to select concrete seed/task/layer
   case studies, but keep aggregate inference in original RDM space rather
   than from MDS/PCA display coordinates.
4. If within-task `T0:E0…E5` checkpoints remain required, preregister a causal
   five-chunk diagnostic extension. Do not silently replace the frozen
   one-pass task-boundary protocol with five repeated training epochs.
5. Treat a transformer/ViT experiment as a separate study. The
   current evidence and remaining budget support a small pilot, not a matched
   architecture-wide claim.

## Limitations

- Three seeds support the registered tolerance/descriptive gates, not a broad
  significance claim.
- Static learning rates and best epochs come from test-informed archived
  source selection; this is intentional reproduction behavior, not clean
  model development.
- Test evaluation preserves archived shuffled/drop-last and off-by-one subset
  behavior.
- The static comparator is the archived repeated-update RBP control, not a
  conventional one-update BP/SGD baseline; H2 also uses source-derived
  method-specific learning rates.
- The local PC library is newer than the archived nested copy, and the study
  tree is dirty/untracked; results are exact-as-practical rather than bitwise.
- Update time and peak allocated memory are hardware/runtime measures, not
  energy or biological-efficiency measurements.
- H2.E3 has only three independent seeds, 20 fixed anchors/class, four probed
  layers, and task-boundary sampling. Its twelve old-task rows are nested
  observations, not twelve independent replications.
- RDM correlations are task-confounded and do not establish mediation,
  monotonicity, or a causal drift-to-forgetting mechanism.
- H2.E3 neutrality is established against its same-revision control; small
  differences from the older H2.E2 artifact preclude a bitwise identity claim.

## Reproduction

```bash
bash playground/step2-cifar10/remote-run.sh \
  --output playground/step2-cifar10/runs/paper-reproduction-v1-reproduction.json

bash playground/step2-cifar10/remote-run-split.sh \
  --output playground/step2-cifar10/runs/split-cifar10-v1-reproduction.json

python playground/step2-cifar10/representation_analysis.py \
  --representations playground/step2-cifar10/runs/split-cifar10-representations-v1.json \
  --control playground/step2-cifar10/runs/split-cifar10-instrumentation-control-v1.json \
  --output agents/research/predictive-coding-cifar-baseline/artifacts/h2e3-representation-summary.json
```

## Research Graph

Graph results `R-H1-reproduced`, `R-H2-tradeoff`, and
`R-H2-representation-tradeoff` support the baseline decisions. Decision
`D-rq2-joint-metrics` records that RDM drift must be interpreted beside
plasticity and behavior. See
[`research_graph.mmd`](research_graph.mmd) and
[`research_graph.json`](research_graph.json).
