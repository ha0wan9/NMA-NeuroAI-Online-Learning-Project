# Step 2: CIFAR-10 paper reproduction and SplitCIFAR-10

This harness has two deliberately separate protocols:

1. `paper_experiment.py` reproduces Song et al. (2024), Fig. 4i as closely as
   practical from official source revision
   `625d52678b3d4cab2dbc403b1d6ee0cfc004aea0`.
2. `split_experiment.py` freezes the same architecture and learning-rule
   semantics before extending them to a causal, replay-free SplitCIFAR-10
   stream. This extension is not a result from the paper.

## Paper protocol

The archived experiment uses two stride-two convolutional layers
(`3→64→128`), a `8192→512→10` classifier, ReLU, half squared error, batch size
200, 80 passes, PC `T=16`, latent SGD `lr=0.5`, Adam with weight decay `0.01`,
and parameter updates at every inference step. The BP control is the archived
RBP condition: it also uses the trainer's 16 parameter-update steps per batch.

The paper selected the minimum test error across epochs and six learning
rates. To fit the registered ten-hour budget without repeating the completed
grid, this reproduction pre-registers the per-seed winning learning rates from
`source_data/fig4-i.csv`. It preserves the source protocol's test-informed
selection and data-loader quirks, and labels both as limitations.

Dependency-free checks:

```bash
python3 -m unittest playground/step2-cifar10/test_protocol.py
python3 playground/step2-cifar10/paper_experiment.py --print-config
python3 playground/step2-cifar10/split_experiment.py --print-config
```

GPU smoke tests:

```bash
bash playground/step2-cifar10/remote-run.sh
bash playground/step2-cifar10/remote-run-split.sh
```

Full registered runs:

```bash
bash playground/step2-cifar10/remote-run.sh \
  --output playground/step2-cifar10/runs/paper-reproduction-v1.json

bash playground/step2-cifar10/remote-run-split.sh \
  --output playground/step2-cifar10/runs/split-cifar10-v1.json
```

The wrappers reuse the pinned CyberEngine ROCm image from Step 1. CIFAR-10 is
downloaded into the persistent remote workspace; the local checkout remains
the canonical source and completed JSON artifacts are synchronized back.

## TensorBoard tracking

All future remote runs write live TensorBoard events by default under
`playground/step2-cifar10/runs/tensorboard/<artifact-stem>/`. Each method/seed
has a separate run. Static dashboards contain test accuracy/loss, samples,
cumulative update time, and PC inference dynamics. Split dashboards contain
per-task test/validation curves, prequential accuracy, continual-learning
summaries, and PC inference dynamics.

Backfill event files from an immutable JSON artifact:

```bash
bash playground/step2-cifar10/remote-tensorboard-export.sh \
  playground/step2-cifar10/runs/split-cifar10-v1.json
```

Serve the persistent remote log directory through an SSH tunnel, then open
`http://127.0.0.1:6006`:

```bash
bash playground/step2-cifar10/tensorboard-serve.sh
```

The static run launched before TensorBoard instrumentation remains untouched.
After `paper-reproduction-v1.json` completes, use the same exporter to backfill
its full epoch history. TensorBoard is pinned to `2.20.0` in the persistent
remote `.python-packages` directory; the ROCm image itself remains immutable.

Representation-enabled Split runs add `--representation-diagnostics`. They use
fixed class-balanced anchors and log learned-layer RDM heatmaps, aligned MDS
geometry, label alignment, class separation, old-task RDM drift, drift versus
accuracy loss, accuracy matrices, and a paired stability-plasticity comparison:

```bash
bash playground/step2-cifar10/remote-run-split.sh \
  --representation-diagnostics --anchor-samples-per-class 20 \
  --output playground/step2-cifar10/runs/split-cifar10-representations-v1.json
```

The feature hooks execute under `eval()` and `no_grad()` after task-boundary
evaluation. Anchor selection uses a separate generator, so diagnostics cannot
change training order, parameters, optimizer state, or primary metrics.
