# Predictive Coding CIFAR Baseline

> Master record for the static CIFAR-10 reproduction and matched
> SplitCIFAR-10 extension. Protocol fields below are frozen unless an explicit
> protocol-change record is added before an affected run.

## Identity

- Study ID: `predictive-coding-cifar-baseline`
- Owner: project team
- Created UTC: `2026-07-18T00:00Z`
- Study root: `agents/research/predictive-coding-cifar-baseline/`
- Branch: `haoran-experiments` (existing user-owned experiment branch)
- Adapter: `agents/research/adapter.yaml`, with study-specific overrides below
- Status: `synthesized`; baseline and registered representation extension complete
- Parent study: `pc-relaxation-plasticity`

## Question

Can the paper's prospective-configuration Predictive Coding convolutional
network be reproduced on static CIFAR-10 within a ten-GPU-hour budget, and how
does the same matched BP-PC system behave in a causal, replay-free,
class-incremental SplitCIFAR-10 stream?

## Success Criteria

- Primary static metric: best CIFAR-10 top-1 test accuracy under the archived
  per-epoch test-informed selection protocol (`maximize`). Final accuracy is
  retained as a secondary endpoint.
- Static reproduction gate: each method/seed best accuracy must be within
  `1.0` percentage point of the authors' archived `fig4-i.csv` value, with no
  unresolved protocol deviation; aggregate direction is reported separately.
- Continual primary metric: final average accuracy over five SplitCIFAR-10
  tasks (`maximize`).
- Feasibility gate: both BP and PC learn above chance, matched initialization
  and stream checksums agree, values remain finite, and the PC diagnostic
  objective decreases during relaxation.
- Secondary guardrails: learning-curve accuracy, prequential accuracy,
  adaptation, forgetting/BWT, runtime, parameters, latent-state memory, and
  peak accelerator memory.
- Required repetitions: paper seeds `1482555873`, `698841058`, and
  `2283198659` for H1; paired project seeds `7`, `42`, and `123` for H2 when
  calibration projects completion inside the budget. Otherwise preserve a
  preregistered partial result and do not make promotion claims.

## Scope

- In scope:
  - exact-as-practical reproduction of Song et al. (2024), Fig. 4i-j, using
    the authors' public configuration and source data;
  - a matched project-controlled BP-PC static CIFAR-10 comparison;
  - five-task SplitCIFAR-10, shared ten-way head, one pass per task, no replay,
    and no model task ID.
- Out of scope:
  - transformers, spiking/Difference PC, replay, EWC, architecture search,
    broad hyperparameter search, CIFAR-100, and claims of field-wide SOTA or
    biological superiority.

## Protocol

- Baseline/control: paper-matched Backpropagation plus a project-controlled BP
  run initialized identically to PC within each paired seed.
- Data version source: torchvision CIFAR-10 backed by the canonical Toronto
  archive; deterministic train/validation indices and task streams recorded by
  checksum; test data is evaluation-only.
- Eval command/parser: study-specific commands under
  `playground/step2-cifar10/`, frozen before the first full run.
- Editable surface: `playground/step2-cifar10/` and this study root.
- Protected files: post-freeze data splits, task construction, evaluation and
  metrics; seed set; comparator; preprocessing; architecture; objective;
  optimizer; relaxation configuration; replay budget; and evaluation cadence.
- Run naming: `predictive-coding-cifar-baseline-<HnEn>-<experiment-name>`.
- Budget: at most ten CyberEngine Radeon 8060S GPU-hours: `0.3` calibration,
  `4.7` static, `3.0` continual, and `2.0` reserve/repetition. Transformer work
  from the earlier planning discussion is explicitly excluded by the user's
  approved execution scope in this study.
- Stop policy: preserve every attempt; stop on non-finite values, checksum or
  initialization mismatch, unavailable GPU, failed PC objective descent, or
  the ten-hour ceiling. Retry once only for external infrastructure failure.

## Adapter

Use `agents/research/adapter.yaml`. Study overrides change the data source and
artifact root to CIFAR-10 and `playground/step2-cifar10/runs/`; the CyberEngine
ROCm image and local-canonical/remote-disposable rule remain unchanged.

## H/E Tracks

| Track | Directory | Hypothesis / route | Status | Decision gate |
|---|---|---|---|---|
| `H1` | `H1-static-reproduction/` | Paper PC can be reproduced on static CIFAR-10. | completed; promotable within claim ceiling | Registered source-data tolerance and feasibility checks. |
| `H2` | `H2-split-cifar10/` | PC and BP exhibit distinguishable acquisition-retention behavior on SplitCIFAR-10. | completed; behavioral and representation baselines synthesized | Report paired plasticity, stability, final accuracy, and cost without a superiority presumption. |

## Managed Review Policy

- Design critic: required before a full GPU launch.
- Result skeptic: required before any reproduction or BP-PC comparison claim.
- Methodology auditor: required before synthesis or reuse as a project baseline.
- Optional reviewers: none; the lead owns source reconciliation and claims.

## Protocol Changes

| UTC | Change | Reason | Approved by | Effective from run |
|---|---|---|---|---|
| 2026-07-18T20:20Z | Register the paper's actual three seeds, per-seed source-data winning learning rates, 80 archived passes, and ±1 pp reproduction tolerance for H1; register fixed cross-seed source learning rates and project seeds for H2. | Official code/source-data inspection resolved fields that were unknown at frame time; avoids rerunning the completed 36-cell tuning grid or tuning on SplitCIFAR test data. | User-approved paper reproduction scope; source-derived preparation detail | First H1/H2 smoke |
| 2026-07-18T18:55Z | Increase H1.E1 smoke from 16 to 40 train examples/class while retaining the paper batch size. | The first smoke yielded zero batches because 160 total examples were smaller than `batch_size=200` with `drop_last=True`; 400 examples yield two representative full batches. | Lead correction after invalid smoke; no hypothesis or full-run protocol change | `predictive-coding-cifar-baseline-H1E1-smoke-002` |
| 2026-07-18T19:03Z | Reallocate unused continual-learning reserve and run all three registered static seeds despite a 6.2-hour throughput projection. | H2.E2 completed in about 0.08 GPU-hours; the combined projection remains below the user-approved 10 GPU-hour cap, and retaining all paper seeds materially strengthens the reproduction. | Lead budget allocation within approved cap; no metric-facing protocol change | `predictive-coding-cifar-baseline-H1E2-paper-full-001` |
| 2026-07-18T19:18Z | Add TensorBoard as an additive tracking mirror for future runs and deterministic backfills from immutable JSON. | User requested TensorBoard tracking. The active static process is not restarted or modified; its event history will be exported after the JSON artifact completes. | User | Future runs; H2.E2 backfill completed |
| 2026-07-18T19:44Z | Register H2.E3 fixed-anchor representation diagnostics and TensorBoard figures. | User requested implementation of the W1D3-inspired RDM, geometry, drift, accuracy-matrix, and stability-plasticity visualizations. Diagnostics are observational and do not change H2.E2 metrics or claims. | User | `predictive-coding-cifar-baseline-H2E3-representation-smoke-001` |

## Pointers

- `01-survey.md`
- `02-design.md`
- `03-monitor.md`
- `04-evaluation.md`
- `05-synthesis.md`
- `runs.jsonl`
- `research_graph.mmd`
- `research_graph.json`
- `artifacts/`
- `audits/`
