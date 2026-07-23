# PC×CLASSP matched-LR optimizer-state-policy study

- **Protocol ID:** `matched-lr-state-policy-v1`
- **Status:** registered single-GPU implementation protocol; source approval
  pending
- **Registered:** 2026-07-23
- **Parent protocol:** `matched-lr-paper-v1` (completed and immutable)
- **Execution device:** one stable local NVIDIA GeForce RTX 4090
- **Claim type:** working hypothesis until all 48 full cells pass complete
  validation and the human conclusion checkpoint is approved

## Question and claim ceiling

This fixed-learning-rate 2×2×2 factorial asks whether resetting parameter
optimizer state at task boundaries changes CLASSP differently from Adam, and
whether that optimizer-specific reset effect differs between Predictive Coding
(PC) and Backpropagation (BP):

```text
{BP, PC} × {Adam, CLASSP} × {persistent, task-reset}
```

For metric `Y`, method `m`, optimizer `o`, and held-out seed `r`:

```text
δ(m,o;r) = Y(m,o,task-reset;r) − Y(m,o,persistent;r)
R_PC(r)  = δ(PC,CLASSP;r) − δ(PC,Adam;r)
R_BP(r)  = δ(BP,CLASSP;r) − δ(BP,Adam;r)
Ω(r)     = R_PC(r) − R_BP(r)
```

The confirmatory final-accuracy gates are hierarchical:

1. CLASSP-specific reset rescue under PC requires positive `R_PC` in at least
   four of five held-out seeds.
2. PC-specific rescue is evaluated only if gate 1 passes, and requires
   positive `Ω` in at least four of five held-out seeds.

Every seed and outcome is reported regardless of direction. Seed 42 is an
openly inspected bridge and is excluded from confirmatory summaries. A passing
gate supports only the corresponding configuration-specific contrast. This
protocol cannot establish statistical significance, mechanistic synergy,
biological superiority, or validity beyond this Permuted-MNIST MLP.

## Frozen scientific conditions

| ID | Method | Optimizer | Parameter-optimizer state |
|---|---|---|---|
| `bp-adam-persistent` | BP | Adam | cumulative |
| `bp-adam-task-reset` | BP | Adam | reset before tasks 1–19 |
| `bp-classp-persistent` | BP | paper-equation CLASSP | cumulative |
| `bp-classp-task-reset` | BP | paper-equation CLASSP | reset before tasks 1–19 |
| `pc-adam-persistent` | PC | Adam | cumulative |
| `pc-adam-task-reset` | PC | Adam | reset before tasks 1–19 |
| `pc-classp-persistent` | PC | paper-equation CLASSP | cumulative |
| `pc-classp-task-reset` | PC | paper-equation CLASSP | reset before tasks 1–19 |

At every boundary the learned network parameters persist. A task-reset cell
clears only the state mapping of the existing parameter optimizer; it does not
recreate the optimizer or its parameter groups. The PC trainer retains that
same parameter-optimizer object for the entire cell. PC latent activities and
their SGD optimizer reset at every batch in both state policies.

Full task-reset cells must record exactly 19 resets at task indices `1..19`.
Persistent optimizer step counts are cumulative across tasks. Task-reset step
counts restart within each task. Boundary records include parameter checksums
immediately before and after the policy action, state before and after reset,
and the PC trainer identity audit when applicable.

## Frozen common configuration

The following values are inherited unchanged from `matched-lr-paper-v1`:

- parameter learning rate `3e-4`;
- MLP `784→256→256→10`, ReLU, biases enabled;
- 20-task domain-incremental Permuted MNIST, identity task first, then 19
  permutations generated once with permutation seed `0`;
- torchvision default 60,000-example train and 10,000-example test splits;
- one pass per task, no replay, no task-ID input, batch size `500`;
- explicit per-task sample order determined only by experimental seed and task
  index and shared by all eight conditions;
- the same realized initial linear tensors for BP and PC within seed;
- one-hot half-squared error summed over batch and classes;
- one parameter update per training batch;
- CLASSP `p=2`, strict squared-gradient threshold `1e-5`, epsilon `1e-5`,
  always-on decay;
- PC `T=20`, latent SGD learning rate `0.01`, latent reset every batch, and
  parameter update only at the final inference step;
- deterministic PyTorch algorithms and CUDA-only full cells.

No tuning, seed substitution, outcome-dependent stopping, or post-freeze
protocol change is permitted. A flaw found after freeze preserves the v1
artifacts and creates a new protocol version.

## Seeds and Williams order

Held-out seeds are `7`, `123`, `2026`, `31415`, and `271828`. Bridge seed `42`
is reported separately. Six seeds by eight conditions produce exactly 48 full
cells.

Canonical condition indices are:

```text
1 BP–Adam–persistent
2 BP–Adam–task-reset
3 BP–CLASSP–persistent
4 BP–CLASSP–task-reset
5 PC–Adam–persistent
6 PC–Adam–task-reset
7 PC–CLASSP–persistent
8 PC–CLASSP–task-reset
```

The Williams base row is `[1,2,8,3,7,4,6,5]`. Subsequent rows add one modulo
eight. Rows are assigned in order to seeds
`[42,7,123,2026,31415,271828]`. The two smoke passes both use seed 42's row.

Every cell runs sequentially on the one manifest-frozen RTX 4090. The RTX
3060, WD SSD, SSH alias, Git bundles, artifact transfer, cross-GPU calibration,
seed-to-host assignment, gaming computer, and distributed execution are
outside v1.

## Learning-first stages and checkpoints

### 1. Implementation gate

Refactor and verify the protocol, runner, validator, local orchestrator, and
tests. Present the source diff and pre-commit packet. Stop for human source
approval. Do not run preflight or a study cell from uncommitted sources.

### 2. Local commissioning

After an approved source commit, run RTX 4090 preflight and two complete,
sequential eight-condition smoke passes. Both passes must validate, match
exactly on scientific/state artifacts and final network parameters, and show
the expected persistent/reset state trajectories.

### 3. Scientific-intuition gate

Run the eight full seed-42 cells sequentially. The JSON and Markdown bridge
reports may expose metric values, optimizer trajectories, reset transients,
parameter movement, BP/PC first updates, continual metrics, runtime, and
memory. Record observations separately from interpretations, limitations,
counterclaims, and a discriminating check. Stop for human review.

The bridge may authorize continuation or expose a flaw. It may not tune v1,
replace a seed, or substitute for confirmation. Held-out authorization records
the exact SHA-256 of the reviewed Markdown bridge report.

### 4. Held-out confirmation

After exact protocol confirmation and bridge-report review, run the remaining
40 cells as five sequential eight-condition seed blocks. Partial held-out
stdout, status, and validation expose structural progress, paths, hashes,
failures, counts, and runtimes only. They do not expose metric values or effect
directions.

Confirmatory analysis is unreachable until all 48 full result/receipt pairs
validate. Complete closeout produces validation JSON, Markdown synthesis,
intended and actual run order, the operational log, and an artifact SHA-256
manifest. Stop before committing results.

## Frozen local manifest

`preflight` creates one self-hashed, immutable manifest containing:

- protocol and schema versions;
- ordered conditions and bridge/held-out seeds;
- the Williams condition order for every seed;
- repository commit/tree/clean status, predictive-coding submodule identity,
  source-file hashes, parent-runner identity, and `uv.lock` hash;
- locked Python, NumPy, PyTorch, torchvision, CUDA, and cuDNN identities;
- MNIST raw-file hashes;
- hostname, GPU model, UUID, driver, CUDA runtime, compute capability, and
  idle-memory baseline for the local RTX 4090;
- bridge visibility/exclusion and complete-only held-out analysis policy;
- its own canonical SHA-256.

Preflight requires committed study sources, a clean tracked repository, a
clean predictive-coding submodule, `uv lock --check`,
`uv sync --locked --extra cuda --check`, locally available MNIST with matching
hashes, CUDA, an RTX 4090, at least 20 GiB free disk, at least 16 GiB available
RAM, and the idle gate.

The one-condition runner refuses a source, dependency, dataset, host/GPU, or
manifest mismatch.

## Observation-only intuition diagnostics

Diagnostics are enabled for every study cell and record:

- per-task network-parameter displacement L2 and relative L2;
- parameter-optimizer step range, state tensor/elements/bytes, and parameters
  with state;
- aggregate Adam first- and second-moment L2 norms;
- CLASSP accumulator sum, L2 norm, and nonzero fraction;
- the first-batch gradient L2 and network-parameter update magnitude after
  every boundary, including the initial task transition;
- optimizer state before and after every reset or retain action;
- network-parameter preservation checksums and PC parameter-optimizer identity.

These reads may add synchronization, allocation, runtime, and peak-memory
overhead. Synthetic neutrality checks require identical stream identity,
initialization, counts, online observations, accuracy matrix, continual
metrics, and final network parameters with diagnostics on versus off. Runtime
and peak memory are not neutrality targets.

## Local execution and artifacts

The ignored authoritative layout is:

```text
playground/continual-mnist/run-staging/matched-lr-state-policy-v1/
├── manifest.json
├── operational-log.jsonl
├── smoke/
│   ├── pass-1/
│   └── pass-2/
├── full/
│   ├── seed-42/
│   └── heldout/seed-{7,123,2026,31415,271828}/
└── reports/
```

Each condition directory contains immutable `attempt-NNN` directories. Result,
receipt, failure, idle sample, log, manifest, review, report, and final-summary
writes are atomic/exclusive or same-content-only. A valid sealed pair is never
rerun or overwritten. Incomplete and failed attempts remain in place and do
not count as scientific cells.

Before every cell, two idle samples must pass 10 seconds apart:

- GPU utilization at most 5%;
- temperature at most 40 °C;
- used memory within 512 MiB of the preflight baseline;
- no foreign graphics or compute process;
- at least 16 GiB available RAM;
- at least 20 GiB free disk.

The idle wait stops after 10 minutes. A runner or invariant failure stops
before the next cell and writes a hash-addressed human-review halt. `resume`
requires the exact structural review SHA-256, skips valid sealed pairs,
preserves incomplete attempts, and allocates a new attempt directory only for
the incomplete cell. Held-out resume additionally requires the same bridge
report SHA-256 and exact protocol confirmation.

No local result is copied elsewhere or curated into tracked results before the
final human review checkpoint.

## Public local orchestrator interface

`playground/continual-mnist/run_state_policy_protocol.py` exposes:

```text
preflight
smoke
bridge
heldout --confirm-protocol matched-lr-state-policy-v1 \
  --bridge-review-sha256 SHA256
resume --stage {smoke,bridge,heldout} \
  --reviewed-failure-sha256 SHA256
status
validate --stage {smoke,bridge,complete}
```

Held-out resume also supplies `--confirm-protocol` and
`--bridge-review-sha256`. `status` never displays held-out metric values or
directions.

Every full cell receives the explicit runner contract:

```text
--mode full
--condition CONDITION
--seed SEED
--device cuda
--diagnostics on
--run-manifest MANIFEST
--confirm-protocol matched-lr-state-policy-v1
--output-dir ATTEMPT
```

The completed `matched-lr-paper-v1` runner remains the read-only owner of the
stream, initialization, objective, and metric implementations.

## Metrics and complete-only analysis

The accuracy matrix and continual metrics retain the parent protocol's exact
definitions. Outcomes are online predict-before-update accuracy, adaptation,
BWT, raw forgetting, final accuracy, runtime, and peak memory. Raw forgetting
is lower-is-better; differences remain on the raw scale.

Complete validation requires exactly 48 valid result/receipt pairs, enforces
source/data/stream/initialization/count/state/host/order invariants, reports
every `δ`, `R_PC`, `R_BP`, and `Ω`, applies the hierarchical gates, and keeps
seed 42 separate from held-out summaries.

Validation establishes code-path behavior, provenance, artifact integrity,
and reproducibility metadata. It does not prove that premises are relevant,
source interpretations are faithful, or empirical conclusions are true.
