# CUDA smoke-gate report

- **Protocol:** `matched-lr-paper-v1`
- **Gate date:** 2026-07-22
- **Status:** passed; awaiting the human design/commit checkpoint
**Scope:** two tasks, 1,000 training and 1,000 test examples per task, seed 42

## Outcome

All four CUDA cells completed through the production training/evaluation path.
The directory validator accepted all four result/receipt pairs and confirmed a
single source identity, finite scientific fields, expected sample/update
counts, GPU use, paired permutations, paired training order, paired actual
initial tensors, persistent optimizer steps, canonical matrix dimensions, and
recomputed metrics.

The smoke accuracies below follow only four optimizer updates per cell. They
are engineering observations and are not evidence for or against the full
study hypothesis.

| Condition | Final avg. acc. | Online predict-before-update | Mean adaptation | Mean BWT | Mean forgetting | Runtime (s) | Peak CUDA bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bp-adam` | 0.1340 | 0.1050 | 0.0440 | 0.0100 | 0.0000 | 1.786 | 79,266,304 |
| `bp-classp` | 0.1365 | 0.1100 | 0.0530 | -0.0030 | 0.0030 | 1.775 | 78,188,544 |
| `pc-adam` | 0.1360 | 0.1055 | 0.0460 | 0.0100 | 0.0000 | 2.151 | 81,314,816 |
| `pc-classp` | 0.1165 | 0.1010 | 0.0320 | 0.0000 | 0.0000 | 1.929 | 80,237,056 |

Every cell processed exactly 2,000 samples and four parameter-optimizer
updates. Optimizer state steps were `[2, 4]` after the two task boundaries.
The paired training-order checksum was
`72a5b913247188812b8d2472f989c76061f6f570af44e31b95b2f641f21c2123`;
the paired initialization checksum was
`25ff0adb760be358d5f17386bda7e9ad96394f4800cd434aa954ed55be59e522`.

## Diagnostics neutrality

`pc-classp` was rerun on CUDA with diagnostics disabled. The validator compared
source, environment, data, initialization, optimizer state, counts, every
online batch accuracy, the full test matrix, all metrics, and the final linear
parameter checksum. All fields were identical. The diagnostics-off final
average accuracy was also `0.1165`.

## Source and environment

- repository base commit:
  `5b357cb6f40a9da6f041262dee20b99818c59f6c`;
- smoke source was intentionally tracked-dirty because this is the pre-commit
  checkpoint; full mode will reject this state;
- predictive-coding submodule:
  `5bf803c3636a39928488941f05539c70ab4df0b1`, clean;
- runner SHA-256:
  `fdc6866c90de667623c4ee2a24a63fc6adde84a603f088af3a0f13530be56c9b`;
- corrected optimizer SHA-256:
  `048350f80db2c753dacc506fd24a2f51f64b6ce651859e5f3e8ae84077756a90`;
- protocol SHA-256:
  `65828849f0ac2276ec0fd37e1b8b270808d02e5eefe0c19055aa813dfce4242b`;
- validated four-cell source identity:
  `bae0770592651d220c393ee801b4e290a7251aa87730ea47ff79aa332f34293d`;
- GPU: NVIDIA GeForce RTX 4090, driver 560.35.03, CUDA 12.6;
- Python 3.11.2, PyTorch 2.10.0+cu126, torchvision 0.25.0+cu126,
  NumPy 2.4.6;
- deterministic PyTorch algorithms enabled with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Staged artifacts

These outputs remain in ignored run staging until the human checkpoint decides
whether they should be curated. They are not staged in Git and no raw JSON was
rewritten.

| Artifact | SHA-256 |
|---|---|
| `smoke-gate-review/...bp-adam__seed-42.json` | `c523d87e7a04170ea3e0c2fc335c616d1e9e6a68308880a376f5b3aa693066b5` |
| `smoke-gate-review/...bp-classp__seed-42.json` | `3ef6ac7ee38282646a8c8722007264d26983559b2437e606a5104615fa50c913` |
| `smoke-gate-review/...pc-adam__seed-42.json` | `efb229208ee13a75f4ebb77bde243e5baff66873cc05cb2f70307982eecf311f` |
| `smoke-gate-review/...pc-classp__seed-42.json` | `cc046a952bfb35f81186271e3fa0991ab70861657c1b9efcf66531ac62876c0e` |
| `neutrality-off-review/...pc-classp__seed-42.json` | `99768c9ca9168d04f956dc56ae1a8b69a02e40a12e02d17013384e54403f8f5a` |

Each result has a separate receipt recording and validating its full-file hash
and byte size.

## Reproduction and validation

The commands below are the original invocations. Reproduction must use a new,
empty output directory because the runner refuses to overwrite these artifacts.

```bash
for condition in bp-adam bp-classp pc-adam pc-classp; do
  .venv/bin/python playground/continual-mnist/run_matched_lr_validation.py \
    --mode smoke --condition "$condition" --seed 42 --device cuda \
    --diagnostics on \
    --output-dir playground/continual-mnist/run-staging/smoke-gate-review
done

.venv/bin/python playground/continual-mnist/validate_matched_lr_results.py \
  --mode smoke --require-complete \
  --output-dir playground/continual-mnist/run-staging/smoke-gate-review
```

Validation establishes internal schema consistency, provenance and artifact
integrity, deterministic pairing, and reproduction metadata. It does not prove
the paper interpretation or empirical hypothesis true; those remain human
review and full-study questions.
