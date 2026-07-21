# Combined Experiment Integration Verification

Date: 2026-07-21

Branch: `integration/combined-experiments`

## Status

**Resolved workflow:** The `haoran-experiments` studies are integrated with
the static-MNIST harness branch without deleting raw, negative, invalid, or
superseded outputs. The evolved Haoran runner and its matching tests own the
three add/add conflicts; the locked `uv` environment, planning material, and
standalone static-MNIST guide remain available.

**Established fact:** The imported preferred artifacts parse as JSON, every
artifact named by the imported run ledgers exists, paired continual-MNIST
initialization checksums match, and the recorded continual metrics are finite.
The stored CIFAR parent-artifact and derivative hashes also match their files.

**Established fact:** Regenerating the relaxation and architecture summaries
from their preferred raw artifacts reproduces the tracked JSON and Markdown
byte-for-byte. CSV content is identical after normalizing the generated CRLF
line endings to the tracked LF convention.

**Open question:** The imported numerical results have not yet been reproduced
from the clean combined revision. Until that run succeeds, describe them as
traceable imported results rather than independently reproduced results.

## Preferred evidence and exclusions

- Continual MNIST: `playground/step1-static-mnist/runs/continual-reference-v1.json`.
- Relaxation: `playground/step1-static-mnist/runs/relaxation-sweep-v1.json`.
- Architecture: `playground/step1-static-mnist/runs/architecture-sweep-v2.json`.
- Static CIFAR-10: `playground/step2-cifar10/runs/paper-reproduction-v1.json`.
- SplitCIFAR-10: `playground/step2-cifar10/runs/split-cifar10-v1.json`.

Artifacts whose names or run index mark them as smoke, pre-fix, invalid, or
superseded remain preserved but do not supply the final report's conclusions.
In particular, the architecture `v1-pre-init-order-fix` run is excluded from
comparative claims.

## Source-provenance correction

Earlier outputs record the runtime and image identity but do not embed the
repository revision copied to CyberEngine. Future Step 1 and Step 2 launchers
record these additive environment fields:

- `repository_commit` and `repository_dirty`;
- `pc_submodule_commit` and `pc_submodule_dirty`.

Older JSON remains valid when these fields are absent. This metadata change
does not alter the frozen datasets, streams, models, training, or metrics.

## Verification commands

```bash
uv sync --locked --extra cpu
uv run --extra cpu python -m unittest discover \
  -s playground/step1-static-mnist -p 'test_*.py'
uv run --extra cpu python -m unittest discover \
  -s playground/step2-cifar10 -p 'test_*.py'
python3 scripts/validate_repository.py
git diff --check --cached
```

Deterministic summary checks use the two analysis programs with temporary
output directories, followed by byte comparisons for JSON/Markdown and
line-ending-normalized comparisons for CSV.

## Planned reproduction gate

The imported CyberEngine launcher targets a contributor-specific SSH alias that
is not available on the integration workstation. The workstation instead uses
an NVIDIA RTX 4090 with driver 560.35.03 and the locked PyTorch 2.10 CUDA 12.6
environment. `scripts/run-local-gpu.sh` requires CUDA, records source identity,
and does not silently fall back to CPU.

After the local-GPU adaptation is committed and clean, reproduce
`continual-mnist-pc-bp-v1` with both scenarios, BP and PC, seeds `7/42/123`,
and PC `T=20`. This deliberately reproduces the archived reference; it does
not replace the later, separate interpretation that `T=5` is the cheapest
effective tested setting. Exact runtime and memory agreement with the archived
ROCm execution is not expected and remains outside the reproduction gate.

Acceptance requires matching protocol fields, stream and initialization
checksums, model/state counts, accuracy matrices, and derived non-timing
metrics. Runtime and peak memory remain descriptive. Preserve and report any
mismatch without changing the protocol or overwriting the source artifact.
