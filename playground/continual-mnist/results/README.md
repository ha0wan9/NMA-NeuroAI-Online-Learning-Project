# Continual MNIST Results — Audited Experiment Artifacts

**Purpose:** This directory preserves all outputs from the predictive-coding + CLASSP continual learning sweep on permuted MNIST (July 21, 2026). Every result — including negative, failed, weak, tuning, and superseded outcomes — is retained as auditable evidence.

---

## Experiment Matrix

| Axis | Values |
|------|--------|
| **Methods** | BP (backpropagation), PC (predictive coding, T=20), iPC (PC with update_p_at='all') |
| **Optimizers** | Adam, CLASSP(p∈{1,2,3}, threshold∈{0,1e-6,1e-5}) |
| **Task counts** | 20 tasks (24 runs), 50 tasks (6 runs) |
| **Learning rates** | 1e-3 (default), 3e-4 (tuned) |
| **Seeds** | 42 (single seed for all experiments) |

### Canonical baselines
| Name | File | Description |
|------|------|-------------|
| BP Adam baseline | `bp_adam_lr3e-4.json` | Best vanilla BP (Adam, lr=3e-4) |
| PC vanilla baseline | `pc_vanilla_lr3e-4.json` | Best vanilla PC (Adam, lr=3e-4, T=20, update_p_at='last') |
| BP Adam (lr=1e-3) | `bp_adam.json` | Higher-LR comparison (underperforms) |
| PC vanilla (lr=1e-3) | `pc_vanilla.json` | Higher-LR comparison (underperforms) |

### CLASSP sweep (20 tasks)
All CLASSP experiments use the default lr=1e-3. Files follow the pattern:
- `bp_classp_p{N}.0_th{value}.json` — BP + CLASSP
- `pc_classp_p{N}.0_th{value}.json` — PC + CLASSP

### iPC (20 tasks)
- `pc_ipc.json` — PC with update_p_at='all', lr=1e-3
- `pc_ipc_lr3e-4.json` — PC with update_p_at='all', lr=3e-4
Both are **negative results** (15-17% accuracy; iPC fails on small MLPs).

### 50-task scaling benchmarks
- `bp_adam_50tasks.json` — BP Adam baseline
- `pc_vanilla_50tasks.json` — PC vanilla, lr=1e-3
- `pc_vanilla_lr3e-4_50tasks.json` — PC vanilla, lr=3e-4
- `pc_classp_p2_th1e-5_50tasks.json` — Best PC+CLASSP config
- `bp_classp_p2_th1e-5_50tasks.json` — Matching BP+CLASSP config
- `pc_ipc_50tasks.json` — iPC scaling test (**negative result**)

### Notable negative and superseded results
| File | Issue |
|------|-------|
| `pc_ipc.json`, `pc_ipc_lr3e-4.json`, `pc_ipc_50tasks.json` | iPC fails on MLP (15.6%, 11.8% acc) — architecture-dependent |
| `pc_classp_p2.0_th1e-06.json` | Threshold too aggressive (20.1% acc) |
| `pc_classp_p1.0_th1e-06.json` | Threshold degrades PC (46.5% acc) |
| `bp_adam.json` | Superseded by lr=3e-4 variant |
| `pc_vanilla.json` | Superseded by lr=3e-4 variant |

---

## Hardware and Environment

| Detail | Value |
|--------|-------|
| **GPU** | NVIDIA GeForce RTX 4090 (25.28 GB) |
| **CUDA** | 12.6 |
| **PyTorch** | 2.10.0+cu126 |
| **Python** | 3.11 (project .venv) |
| **PC library** | Bogacz-Group/PredictiveCoding (git submodule, pinned) |
| **Architecture** | MLP 784 → 256 → 256 → 10, ReLU, biases enabled |
| **Batch size** | 500 |
| **PC configuration** | T=20 inference steps, SGD x-optimizer (lr=0.01), update_x_at='all' |
| **Loss** | PC: half squared error (one-hot targets); BP: CrossEntropyLoss |

---

## Metrics Computed

- **Final average accuracy:** mean of test accuracy on all tasks after the final task.
- **Forgetting:** peak accuracy on task *j* (measured immediately after training task *j*) minus final accuracy on task *j*. Averaged over tasks 1..N-1 (task 0 excluded from average to avoid initial-untrained contamination).
- **Backward transfer (BWT):** final accuracy on task *j* minus initial (pre-training) accuracy on task *j*.
- **Task 0 curve:** test accuracy on task 0 measured after every task — the primary forgetting trajectory.

---

## Provenance

- **Git revision:** committed on `integration/combined-experiments` (see `git log`).
- **Dataset checksum:** Not recorded. MNIST is downloaded via `torchvision.datasets.MNIST` with default settings.
- **Stream checksum:** Not recorded. Sample order within each task is determined by `DataLoader(shuffle=True)` with `torch.manual_seed(seed + task_id)` but no explicit sample-order hash was saved.
- **Dependency versions:** Not pinned beyond PyTorch 2.10.0. Other packages (numpy, torchvision, tqdm) were latest compatible at time of run. PC library is a pinned git submodule.
- **Seeds:** All experiments use `seed=42` for task-permutation generation and model initialization. Task data loaders use `seed=42 + task_id` for each task's shuffle.

---

## Known Missing Provenance

1. **Dataset checksum:** MNIST training/test split integrity was not verified against a known hash.
2. **Stream checksum:** The exact order in which samples were presented within each task epoch was not recorded or checksummed.
3. **Dependency lockfile:** `uv.lock` exists at the repo root but was not verified against the packages actually installed during these runs.
4. **GPU driver version:** Not recorded (nvidia-smi showed 560.35.03 at time of audit).
5. **PC library commit:** The pinned PredictiveCoding submodule commit was not explicitly recorded in the result files.

---

## Limitations

- **Single seed (42).** All conclusions are based on one random initialization and one task-permutation sequence. The reported numerical differences have not been tested for statistical significance. Multi-seed replication (e.g., seeds 7, 42, 123 as used elsewhere in this project) is required before drawing general conclusions.
- **Shallow architecture.** Results apply to a 2-hidden-layer MLP (256 units). Behavior on deeper networks, convolutional architectures, or transformers is unknown and likely to differ.
- **Single epoch per task.** Each task is seen exactly once. Multi-epoch training or replay-buffer protocols may produce different relative rankings.
- **No compositional baselines.** EWC, SI, GEM, and experience replay were not run. CLASSP is compared only to vanilla optimizers, not to other continual learning methods.
- **Descriptive differences do not establish causality, synergy, biological superiority, or generalization.** The observation that PC+CLASSP > BP+CLASSP is a descriptive fact for this seed and configuration. It does not demonstrate that PC "causes" better CLASSP utilization, that the mechanisms are synergistic, or that the result generalizes beyond permuted MNIST.
- **Mechanistic claims are hypotheses.** Explanations involving "cleaner gradients," "better signal-to-noise ratio," or "complementary mechanisms" are interpretations consistent with the observed numbers. They have not been directly tested through gradient analysis, ablation, or causal intervention.

---

## Minimal Reproduction Commands

```bash
# From repo root, with .venv activated:

# Canonical BP baseline
python playground/continual-mnist/experiment.py \
  --method bp --n-tasks 20 --lr 0.0003

# Canonical PC baseline
python playground/continual-mnist/experiment.py \
  --method pc --n-tasks 20 --T 20 --lr 0.0003

# Best PC+CLASSP config (20 tasks)
python playground/continual-mnist/experiment.py \
  --method pc --n-tasks 20 --T 20 --classp \
  --classp-p 3 --classp-threshold 1e-5

# Full sweep (generates all files in this directory)
python playground/continual-mnist/sweep.py
```

**Note:** Reproducing identical numerical results requires identical MNIST download, identical PyTorch/torchvision versions, identical CUDA determinism settings, and identical sample ordering within each task's DataLoader shuffle.

---

## File Manifest

30 curated JSON result files + this README + .gitignore. All other outputs (checkpoints, TensorBoard logs, caches, future experiment runs) are excluded by the local `.gitignore`.
