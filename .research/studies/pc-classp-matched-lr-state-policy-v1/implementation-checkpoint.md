# matched-lr-state-policy-v1 implementation checkpoint

- **Checkpoint:** single-GPU source implementation and synthetic verification
- **Date:** 2026-07-23
- **Status:** ready for human source review; not committed
- **Protocol:** `matched-lr-state-policy-v1`
- **Scientific results:** none produced or inspected

## Outcome

The uncommitted dual-host draft has been replaced with a learning-first local
protocol for one sequential RTX 4090. The registered protocol, one-cell
runner, observation-only diagnostics, validator/analysis, local orchestrator,
and focused tests are implemented.

The completed `matched-lr-paper-v1` runner, validator, tests, protocol, and
result tree have no working-tree diff. The conceptual
`matched-lr-state-policy-model.html` is preserved unchanged with SHA-256
`9c1e7c02b83826aa1cffc506a332980946ca71c822ebf506478b7fc972f1b3e5`.

The implementation stops at the required source checkpoint. It has not:

- committed or pushed any file;
- run the RTX 4090 preflight or created a study manifest;
- loaded MNIST for a state-policy study cell;
- run either smoke pass, a full bridge cell, or a held-out cell;
- generated or inspected any scientific metric or effect direction;
- connected to another host, used SSH, created a bundle, transferred an
  artifact, or written to the WD SSD;
- changed dependencies, CI/CD, credentials, mounts, ownership, or
  permissions.

## Proposed source set

| Path | Role |
|---|---|
| `.research/studies/pc-classp-matched-lr-state-policy-v1/protocol.md` | Frozen single-GPU scientific, diagnostic, artifact, analysis, and recovery contract |
| `.research/studies/pc-classp-matched-lr-state-policy-v1/implementation-checkpoint.md` | This review and pre-commit delivery packet |
| `playground/continual-mnist/run_state_policy_study.py` | One-condition/one-seed runner, state intervention, and read-only intuition diagnostics |
| `playground/continual-mnist/validate_state_policy_results.py` | Pairing/invariant validation, repeated-smoke and bridge reports, and complete-only confirmation |
| `playground/continual-mnist/run_state_policy_protocol.py` | Local preflight, sequential stages, idle gates, resume/review, status, validation, and closeout |
| `playground/continual-mnist/test_state_policy_study.py` | Synthetic algorithm, persistent parity, neutrality, reset, validator, and contrast tests |
| `playground/continual-mnist/test_state_policy_protocol.py` | Manifest, idle, failure/review, visibility, and end-to-end lifecycle tests |
| `playground/continual-mnist/experiment-compute-strategy.md` | Explicitly deferred multi-host extension note |
| `playground/continual-mnist/matched-lr-state-policy-model.html` | Unchanged conceptual model |

The superseded `state_policy_controller.py` and
`test_state_policy_controller.py` do not exist in the proposed source set and
must not be committed.

## Scientific implementation audit

### Preserved factorial and intervention

- All eight BP/PC × Adam/CLASSP × persistent/task-reset condition IDs are
  explicit.
- The learning rate, architecture, 20-task stream, data order,
  initialization, objective, PC inference, CLASSP settings, and continual
  metrics are inherited unchanged from the frozen parent runner.
- Bridge seed 42 and held-out seeds `7`, `123`, `2026`, `31415`, and `271828`
  are unchanged.
- The Williams base row and all six seed rows are materialized into the frozen
  manifest.
- Both policies retain exactly one parameter-optimizer object.
- `task-reset` clears only `optimizer.state` immediately before tasks after
  task 0.
- Full task-reset cells require exactly 19 resets at indices `1..19`.
- Boundary records prove that learned linear-network parameters are unchanged
  by the state action.
- PC retains the supplied parameter optimizer in `PCTrainer`; PC latent state
  and latent SGD reset every batch in both policies.

### Read-only intuition diagnostics

Every study cell records:

- per-task network-parameter displacement L2 and relative L2;
- parameter-optimizer step range and state tensor/elements/bytes;
- Adam first- and second-moment aggregate L2 norms;
- CLASSP accumulator sum, L2 norm, and nonzero fraction;
- first-batch gradient and network-parameter update magnitude after each task
  transition;
- detailed state before and after every retain/reset action;
- parameter-preservation and PC optimizer-identity checks.

PC transient latents are excluded from network-parameter displacement. The
diagnostic path may add observation overhead but does not mutate optimizer
state or parameters. Synthetic diagnostics-on/off comparisons cover all eight
conditions and require identical streams, initialization, counts, online
observations, accuracy matrices, metrics, boundary audits, and final network
parameters after removing the diagnostic-only boundary fields.

### Analysis gate

- Seed 42 is openly inspectable and excluded from confirmation.
- Smoke and bridge reports may show learning/state values.
- Held-out runner stdout and `status` contain structural fields, paths,
  hashes, counts, failures, and runtimes only.
- Held-out execution requires exact protocol confirmation and the SHA-256 of
  the reviewed bridge Markdown report.
- Confirmatory analysis is reachable only after all 48 full result/receipt
  pairs validate.
- Every held-out `δ`, `R_PC`, `R_BP`, and `Ω` is reported.
- The hierarchical final-accuracy gates are unchanged.
- Raw forgetting remains lower-is-better and all differences remain on the
  raw scale.

## Local orchestration audit

The public interface is:

```text
preflight
smoke
bridge
heldout
resume --stage {smoke,bridge,heldout}
status
validate --stage {smoke,bridge,complete}
```

- `preflight` requires clean committed study sources, a clean predictive-
  coding submodule, `uv lock --check`, a synchronized locked CUDA environment,
  local MNIST hashes, CUDA, the RTX 4090 identity, minimum RAM/disk, and two
  idle samples before freezing a self-hashed manifest.
- The manifest contains repository/submodule/source/lock identities,
  dependencies, MNIST hashes, GPU identity, all Williams rows, stage/analysis
  policy, and its own canonical hash.
- Smoke is exactly two sequential eight-condition passes using seed 42's row.
- Bridge is exactly eight sequential full seed-42 cells and produces adjacent
  JSON/Markdown intuition reports.
- Held-out is five sequential eight-condition seed blocks.
- Every cell receives two acceptable idle samples 10 seconds apart, with a
  10-minute limit, utilization/temperature/baseline-memory thresholds, no
  foreign graphics/compute process, and minimum RAM/disk.
- Results, receipts, failures, logs, idle records, reports, and attempts are
  non-overwriting. Valid pairs are skipped.
- A runner or validation failure stops before another cell and writes a
  hash-addressed halt.
- `resume` requires the exact structural review SHA-256, preserves all prior
  attempts, and allocates a new attempt directory only where no valid pair
  exists.
- Complete closeout writes validation JSON, Markdown synthesis, intended and
  actual order, operational log, and artifact hashes.

## Verification record

Commands use the existing locked environment and synthetic fixtures; they do
not run a real state-policy study cell:

```bash
.venv/bin/python playground/continual-mnist/test_state_policy_protocol.py
.venv/bin/python playground/continual-mnist/test_state_policy_study.py
.venv/bin/python playground/continual-mnist/test_classp_optimizer.py
.venv/bin/python playground/continual-mnist/test_matched_lr_validation.py
uv lock --check
uv sync --locked --extra cuda --check
python3 -m py_compile \
  playground/continual-mnist/run_state_policy_study.py \
  playground/continual-mnist/validate_state_policy_results.py \
  playground/continual-mnist/run_state_policy_protocol.py \
  playground/continual-mnist/test_state_policy_study.py \
  playground/continual-mnist/test_state_policy_protocol.py
python3 scripts/validate_repository.py
```

Observed results:

- new local-protocol suite: 10/10 passed, including a synthetic
  preflight → repeated smoke → bridge → reviewed held-out → 48-cell complete
  validation lifecycle;
- new study/validator suite: 9/9 top-level tests passed, including all eight
  two-task conditions, diagnostics neutrality for all eight conditions, four
  20-task reset fixtures, frozen persistent parity, single-GPU/order
  rejection, and hierarchical contrast checks;
- frozen CLASSP suite: 7/7 passed;
- frozen matched-LR suite: 14/14 passed;
- locked dependency checks: `uv lock --check` passed with 49 packages
  resolved, and `uv sync --locked --extra cuda --check` passed with 42
  installed packages and no changes required;
- Python compilation: passed;
- frozen `matched-lr-paper-v1` source/result diff check: passed;
- trailing-whitespace scan of the proposed Markdown/Python source: passed;
- repository validation: passed (12 required files, 91 Markdown files,
  37 Python files, and 5 provenance artifacts checked).

The first repository-validation pass found a pre-existing 4 KiB macOS
AppleDouble sidecar named `._implementation-checkpoint.md`. It was not project
Markdown and was already covered by the repository's `._*` ignore rule, but
the validator attempted to decode it. The sidecar was moved intact to ignored
`.harness/quarantine/state-policy-v1/` and remains recoverable. No research
artifact was deleted.

The clean-source host-description probe is expected to refuse execution until
these files are committed:

```text
HOST DESCRIPTION FAILED: RuntimeError: study execution requires committed sources
```

It must be rerun after an approved source commit before local commissioning.
No claim is made yet about real MNIST, CUDA execution, RTX 4090 idle state,
runtime, memory, or empirical outcomes.

## Pre-commit delivery packet

### User-facing docs

- Registers the single-GPU protocol and this implementation checkpoint.
- Relabels the compute strategy as a deferred multi-host extension.
- Preserves the conceptual HTML model unchanged.
- Does not change existing shared documentation or completed results.

### Agent-facing docs

- No canonical `AGENTS.md` or `agents/*.md` behavior changed.
- Runtime behavior lives only in the versioned study protocol and local
  orchestrator.

### Research and evaluation impact

- Registers a new evaluation protocol; it does not revise
  `matched-lr-paper-v1`.
- Adds the fixed-LR parameter-optimizer state-policy factor and
  observation-only intuition diagnostics.
- Preserves the frozen estimands, seeds, hierarchy, and complete-only
  confirmatory gate.
- Produces no scientific result at this checkpoint.

### Behavior or trigger changes

- Adds a local, sequential orchestration CLI for preflight, two smoke passes,
  bridge review, held-out authorization, resume, structural status, and
  validation.
- Removes distributed execution, SSH, bundle/transfer, cross-GPU calibration,
  storage-mount orchestration, and hardware assignment from v1.
- Does not alter CI/CD, dependencies, lock data, or an existing public
  interface.

### Validation

- `git diff --check`: no tracked diff; the proposed untracked source set
  separately passed the trailing-whitespace scan.
- `python3 scripts/validate_repository.py`: passed.
- Markdown links and cited source identifiers in touched files: passed by the
  repository validator; no new external evidence claim is made.
- Experiment reproduction command, configuration, seeds, ordering, and stage
  gates: frozen in `protocol.md`; execution remains blocked until source
  approval and commit.

### Commit scope

Only the nine paths in **Proposed source set** are intended for the proposed
source commit.

### Excluded local state

`USER.md`, `.venv/`, caches, `.harness/` (including the recoverable AppleDouble
sidecar), ignored run staging, data, credentials, and any future raw run
artifacts are excluded.

### Commit and push status

- Commit: not created; explicit approval required.
- Push: not performed and not authorized.

## Human decision requested

Review the proposed source set and this packet. If approved, the next bounded
checkpoint is:

1. create the source commit without pushing;
2. run local RTX 4090 `preflight`;
3. run both eight-condition smoke passes;
4. inspect the smoke/optimizer-state report and state audits;
5. run the eight full seed-42 bridge cells;
6. review observations, interpretations, limitations, counterclaim, bridge
   report SHA-256, and bridge-derived held-out ETA;
7. stop again before the 40 held-out cells.

If the bridge exposes an implementation or protocol flaw, preserve every
artifact and create a new protocol version. Do not silently modify v1.

Validation establishes code-path behavior, provenance checks, artifact
integrity, and reproducibility metadata. Scientific semantics and every human
checkpoint remain review decisions.

## Memory writeback check

- Durable lesson: none beyond the candidate protocol and runtime contract
  already owned by these versioned study artifacts.
- Evidence: focused synthetic and frozen regression checks listed above.
- Claim status: implementation behavior verified locally; scientific and real
  RTX 4090 behavior remain unverified.
- Memory owner: study protocol and implementation checkpoint.
- Writeback decision: skip canonical `AGENTS.md`/`agents/*.md` writeback.
- Reason: the protocol is still at source review and no reusable
  repository-wide rule or validated scientific result has been established.
