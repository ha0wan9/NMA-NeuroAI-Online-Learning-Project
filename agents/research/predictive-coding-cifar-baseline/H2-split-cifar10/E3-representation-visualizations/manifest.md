# H2.E3 representation visualizations

- Display ID: `H2.E3`
- Slug: `H2E3-representation-visualizations`
- Parent: `predictive-coding-cifar-baseline-H2E2-split-full-001`
- Code revision: `6ddd656` plus the approved dirty visualization diff, frozen
  by `../../artifacts/h2e3-full-launch-provenance.json`
- Data: unchanged torchvision CIFAR-10 split and stream seed `20260717`
- Probe seed: `42`; full seeds: `7, 42, 123`
- Intervention: observational TensorBoard diagnostics only
- Anchors: 20 test images/class for the full run, class sorted, selected with
  isolated seed `20292132`; anchor checksum is recorded in the JSON artifact
- Layers: outputs of `conv1`, `conv2`, `fc1`, and `logits`
- Figures: class-sorted correlation RDM, initialization-aligned classical MDS,
  interactive 3D class-centroid trajectories over aligned MDS geometry,
  interactive four-layer RDM timeline with slider and play/pause controls,
  penultimate RDM drift versus accuracy loss, test/validation accuracy matrices,
  and paired stability-plasticity comparison
- Scalars: label alignment, within/between dissimilarity, class separation,
  and per-old-task RDM drift at every task boundary
- Invariants: evaluation mode, no gradients, no parameter/optimizer mutation,
  isolated anchor RNG, unchanged stream/evaluation/primary metrics
- Status: completed and kept as an observational representation result
- Recorded cost: `0.0746` stream-plus-evaluation accelerator-hours for the
  diagnostics-on artifact; `0.0752` hours for the same-revision control
- Claim ceiling: no causal drift-to-forgetting, monotonic mediation,
  superiority, biological-mechanism, or general continual-learning claim

Probe command:

```bash
bash playground/step2-cifar10/remote-run-split.sh \
  --seeds 42 --max-train-samples-per-task 40 \
  --max-eval-samples-per-task 40 --representation-diagnostics \
  --anchor-samples-per-class 2 --device cpu \
  --output playground/step2-cifar10/runs/split-representation-smoke-v1.json
```

Full command:

```bash
bash playground/step2-cifar10/remote-run-split.sh \
  --representation-diagnostics --anchor-samples-per-class 20 \
  --output playground/step2-cifar10/runs/split-cifar10-representations-v1.json
```

Expected artifacts:

- immutable JSON under `playground/step2-cifar10/runs/`
- TensorBoard events under
  `playground/step2-cifar10/runs/tensorboard/<artifact-stem>/`
- self-contained Plotly HTML under each method run's
  `artifacts/representation_trajectory_3d.html` and `artifacts/rdm_timeline.html`

Completed artifacts:

- full representations: `playground/step2-cifar10/runs/split-cifar10-representations-v1.json`
- same-revision diagnostics-off control:
  `playground/step2-cifar10/runs/split-cifar10-instrumentation-control-v1.json`
- deterministic summary:
  `agents/research/predictive-coding-cifar-baseline/artifacts/h2e3-representation-summary.json`

Analysis command:

```bash
python playground/step2-cifar10/representation_analysis.py \
  --representations playground/step2-cifar10/runs/split-cifar10-representations-v1.json \
  --control playground/step2-cifar10/runs/split-cifar10-instrumentation-control-v1.json \
  --output agents/research/predictive-coding-cifar-baseline/artifacts/h2e3-representation-summary.json
```
