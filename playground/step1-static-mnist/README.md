# Step 1: Matched Static-MNIST PC–BP Protocol

Protocol identifier: `step1-static-mnist-v1`

This harness implements the workflow chart's first scientific comparison:
verify that Predictive Coding (PC) and Backpropagation (BP) both learn
stationary MNIST before either method enters a continual-learning protocol.
It reuses the experiment organization of `../NMA-Microlearning.ipynb` and calls
the PC implementation pinned in `../predictive-coding/` without modifying or
copying that upstream library.

For a high-level visual explanation of the baseline workflow, fairness
controls, evidence artifact, and manual commands, open the
[standalone HTML guide](guide.html). The guide introduces the original static
feasibility harness; this README is authoritative for the subsequently expanded
and evaluated protocol.

## Frozen comparison

| Dimension | Value |
|---|---|
| Dataset | MNIST, deterministic 80/20 train/validation split, untouched test set |
| Preprocessing | `ToTensor()` followed by flattening; no normalization |
| Architecture | `784 → 256 → 256 → 10`, ReLU, biases enabled |
| Initialization | Identical Linear-layer tensors for each paired BP–PC seed |
| Supervised objective | Half squared error against one-hot labels |
| Parameter optimizer | Adam, learning rate `0.001`, both methods |
| Batch size | `500` |
| Epochs | `10` |
| Paired seeds | `7`, `42`, `123` |
| PC latent inference | SGD, learning rate `0.01`, `20` relaxation steps |
| Replay/future access | None |
| Evaluation | Shared validation/test function; test evaluated only after training |

The primary x-axis is `samples_seen`, not epoch alone. Each run records loss,
accuracy (including final training-set accuracy for overfit checks), runtime,
learned parameter count, temporary latent-state size,
environment versions, initial-parameter checksum, and accelerator peak memory
when ROCm/CUDA is used. The first PC batch also records loss, energy, and total
objective across relaxation steps.

## Setup

Initialize the pinned PC submodule:

```bash
git submodule update --init --recursive
```

Use Python 3.10 or newer with PyTorch, torchvision, NumPy, pandas, matplotlib,
seaborn, and tqdm. The latter four packages are imported by the pinned PC
library. Install PyTorch and torchvision using the command appropriate for the
machine from the official PyTorch installation selector, then install the
remaining packages in an isolated environment.

For a reproducible local CPU environment on Python 3.11, the integration branch
also retains the locked `uv` configuration:

```bash
uv sync --locked --extra cpu
uv run --extra cpu python -m unittest \
  playground/step1-static-mnist/test_protocol.py
uv run --extra cpu python \
  playground/step1-static-mnist/experiment.py --print-config
```

The NMA repository licenses software under BSD-3-Clause and educational
content under CC BY 4.0. The pinned PredictiveCoding upstream did not include a
license file when added here, so this harness imports the submodule and does
not redistribute its source.

## Checks and runs

Dependency-free protocol checks:

```bash
python3 -m unittest playground/step1-static-mnist/test_protocol.py
python3 playground/step1-static-mnist/experiment.py --print-config
```

One-seed CPU-friendly smoke run:

```bash
python3 playground/step1-static-mnist/experiment.py \
  --seeds 42 \
  --epochs 1 \
  --batch-size 64 \
  --pc-steps 4 \
  --max-train-samples 256 \
  --max-valid-samples 256 \
  --max-test-samples 256 \
  --output playground/step1-static-mnist/runs/smoke-seed42.json
```

Frozen reference run:

```bash
python3 playground/step1-static-mnist/experiment.py \
  --output playground/step1-static-mnist/runs/reference-v1.json
```

The runner refuses to overwrite an existing artifact. Preserve failed and
negative runs rather than reusing their path.

## CyberEngine GPU backend

CyberEngine is used as a remote execution instance while this local checkout
remains the source of truth. The wrapper uses SSH and `rsync` to incrementally
copy the working tree into an isolated remote workspace, runs the experiment
inside a rootless Podman ROCm container, and synchronizes run artifacts back to
the local `runs/` directory.

The backend is an AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`), so the
correct PyTorch backend is ROCm. PyTorch still exposes ROCm devices through its
`torch.cuda` API. The runtime derives from AMD's official ROCm 7.2.1,
Python 3.12, PyTorch 2.9.1 image:

```text
docker.io/rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

This selection follows AMD's
[Ryzen ROCm support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
and [PyTorch installation guidance](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installryz/native_linux/install-pytorch.html).

`Containerfile.cyberengine` adds only `seaborn==0.13.2`, which is imported by
the pinned PC library but absent from the base image. Set up and verify the
remote runtime once with:

```bash
bash playground/step1-static-mnist/setup-cyberengine.sh
```

This creates the rootless local image
`localhost/nma-neuroai-step1-rocm:7.2.1` on CyberEngine and runs an actual GPU
matrix multiplication as its acceptance check.

Prerequisites on the local machine are an SSH alias named `cyberengine` and
`rsync`. The remote needs rootless Podman plus accessible `/dev/kfd` and
`/dev/dri` devices. No Docker daemon or remote Git commit is required. The
wrapper adds `--device cuda` unless a device was supplied explicitly, so a
broken ROCm connection fails instead of silently running the experiment on CPU.

Run the default one-seed smoke test from the repository root:

```bash
bash playground/step1-static-mnist/remote-run.sh
```

Pass normal experiment arguments to override the smoke configuration:

```bash
bash playground/step1-static-mnist/remote-run.sh \
  --seeds 7 42 123 \
  --output playground/step1-static-mnist/runs/reference-v1.json
```

Optional environment overrides are `CYBERENGINE_HOST`,
`CYBERENGINE_WORKSPACE`, and `CYBERENGINE_IMAGE`. Synchronization excludes Git
metadata, local datasets, caches, and the run directory; remote run artifacts
are copied back after both successful and failed runs. The remote workspace is
disposable and must not be treated as the canonical repository.

## Gate interpretation

The static gate passes only when both methods learn above chance, paired
initialization checksums match, and PC's first-batch total objective decreases
over relaxation. A failed gate triggers debugging of inference dynamics,
weight updates, initialization, or model complexity. It does not justify
changing the continual-learning protocol or making comparative performance
claims.

Passing this gate establishes implementation feasibility, not superiority or
biological plausibility. Any PC–BP conclusion still requires review across all
paired seeds and disclosure of PC's inference-time cost.

## Continual-learning extension

`continual_experiment.py` runs two separate matched benchmarks. Their absolute
scores are not pooled because they expose different amounts and types of data:

| Scenario | Regime | Five tasks | Model information |
|---|---|---|---|
| SplitMNIST | Class-incremental | `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, `(8,9)` | Shared 10-way head; no task ID |
| Permuted MNIST | Domain-incremental | Identity plus four fixed pixel permutations | Shared 10-way head; no task ID |

Both protocols use a deterministic 80/20 split of the MNIST training set, one
pass per task, no replay buffer, no future samples, matched BP–PC initialization,
and an explicitly realized sample order whose checksum is recorded. Task
boundaries are available to the experiment driver for evaluation, but are not
passed into either model. Permuted MNIST revisits the base images under each
new transformation; SplitMNIST exposes each base training image in only its
class-pair task.

Before every update, the current batch is predicted to obtain causal
prequential accuracy. After each task boundary, every task test set is evaluated
to produce an accuracy matrix `R`, where `R[k][j]` is accuracy on task `j` after
training through task `k-1`. The recorded summary metrics are:

- final average accuracy: mean of the final row of `R`;
- backward transfer: mean `R[T][j] - R[j+1][j]` over non-final tasks;
- average forgetting: mean historical post-learning maximum minus final
  accuracy over non-final tasks;
- adaptation gain: `R[j+1][j] - R[j][j]` for each task;
- overall and per-task prequential accuracy;
- stream-processing and evaluation time, learned parameters, final and peak PC
  latent-state size, and peak accelerator memory.

Run the default two-scenario, one-seed smoke test:

```bash
bash playground/step1-static-mnist/remote-run-continual.sh
```

Run the frozen full-data, paired three-seed protocol:

```bash
bash playground/step1-static-mnist/remote-run-continual.sh \
  --scenarios split permuted \
  --seeds 7 42 123 \
  --output playground/step1-static-mnist/runs/continual-reference-v1.json
```

The two scenarios are reported separately. A favorable result on one does not
establish general continual-learning superiority, and any mean difference must
be accompanied by its individual paired seeds.

## Relaxation-step ablation

`relaxation_sweep.py` varies only PC latent relaxation depth over
`T=1/5/10/20`, reruns a single shared BP control, and evaluates all three
settings with paired seeds `7/42/123`. Static plasticity is operationalized by
epoch-1 validation gain/accuracy and the validation-curve mean. Continual
plasticity uses adaptation gain and predict-before-update prequential accuracy;
forgetting/BWT remains a separate stability measure.

Run the full CyberEngine sweep:

```bash
bash playground/step1-static-mnist/remote-run-relaxation-sweep.sh
```

Regenerate deterministic JSON/CSV/Markdown tables:

```bash
python3 playground/step1-static-mnist/relaxation_analysis.py \
  --input playground/step1-static-mnist/runs/relaxation-sweep-v1.json \
  --output-dir agents/research/pc-relaxation-plasticity/artifacts
```

For this two-PC-layer model, the upstream trainer warns that fewer than three
iterations cannot propagate error through every layer. `T=1` is retained as a
requested boundary/failure condition, and its one-point objective trace cannot
establish objective decrease. The complete claim-labelled report is
[`agents/research/pc-relaxation-plasticity/05-synthesis.md`](../../agents/research/pc-relaxation-plasticity/05-synthesis.md).

## Architecture-scale ablation

`architecture_sweep.py` compares the original `256×2` MLP with a `512×2`
width intervention and a `256×4` depth intervention. PC relaxation is frozen
at `T=5`, which is sufficient for all planned networks and exactly the minimum
for four PC layers. All methods and architectures use paired seeds and shared
static/continual data checksums.

```bash
bash playground/step1-static-mnist/remote-run-architecture-sweep.sh

python3 playground/step1-static-mnist/architecture_analysis.py \
  --input playground/step1-static-mnist/runs/architecture-sweep-v2.json \
  --output-dir agents/research/pc-network-scale/artifacts
```

The complete report is
[`agents/research/pc-network-scale/05-synthesis.md`](../../agents/research/pc-network-scale/05-synthesis.md).
