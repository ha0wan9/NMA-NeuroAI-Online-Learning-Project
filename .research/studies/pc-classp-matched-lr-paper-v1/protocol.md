# PC×CLASSP matched-LR paper-equation study

- **Protocol ID:** `matched-lr-paper-v1`
- **Status:** algorithm and CUDA smoke gates passed; human design checkpoint pending
- **Registered:** 2026-07-22
- **Normative method source:** Ludwig (2024), arXiv:2405.09637, equation (1)
  and Algorithm 1
- **Claim type:** working hypothesis until the complete held-out study passes
  validation

## Scientific question and claim ceiling

This 2×2 factorial asks whether paper-equation CLASSP changes performance more
under Predictive Coding (PC) than under Backpropagation (BP) when parameter
learning rate, objective, architecture, initialization, task stream, sample
order, data access, and evaluation are matched.

For each held-out seed:

```text
ΔPC = PC+CLASSP − PC+Adam
ΔBP = BP+CLASSP − BP+Adam
ΔΔ  = ΔPC − ΔBP
```

A positive final-accuracy interaction in at least four of five held-out seeds
supports only a configuration-specific differential benefit. It does not
establish mechanistic synergy, cleaner PC gradients, biological superiority,
or generalization beyond this Permuted-MNIST MLP.

## Corrective reset and legacy boundary

All July 21 seed-42 JSON results under
`playground/continual-mnist/results/` are retained as `legacy-v0`
observations. They are not evidence for cross-task CLASSP protection or a
PC×CLASSP interaction because:

1. Adam and CLASSP were recreated at every task boundary, so neither retained
   optimizer history across the continual stream.
2. The old CLASSP implementation erased accumulator history on coordinates
   that passed the threshold before adding the current contribution, so active
   coordinates did not accumulate history across steps.
3. BP used cross-entropy while PC used one-hot half-squared error.
4. The exploratory sweep used a single seed, unequal baseline/sweep learning
   rates, noncanonical continual metrics, and incomplete provenance.

The stopped `matched-lr-validation-v1` draft is preserved and described in
[`superseded-design-audit.md`](superseded-design-audit.md). No legacy JSON is
deleted or rewritten.

## CLASSP algorithm contract

For coordinate `i` at optimizer step `t`, let `g[i]` be its current gradient
and `A[i]` its persistent accumulator. The implementation performs:

```text
mask[i] = g[i]^2 > threshold
A[i]    = A[i] + |g[i]|^p       if mask[i], otherwise unchanged
w[i]    = w[i] - lr*g[i]/(epsilon + A[i])^(1/p)
                                      if mask[i] and apply_decay
w[i]    = w[i] - lr*g[i]        if mask[i] and not apply_decay
w[i]    = w[i]                   otherwise
```

The accumulator and optimizer object exist once for the entire task stream.
`apply_decay=False` still accumulates history and applies the accepted
unscaled SGD update; it is available for the schedule described in the
paper's section 4.1. This study uses always-on decay.

Equation (1) prints a sum ending at `t-1`, while Algorithm 1 adds the current
gradient before calculating the scaling factor. This implementation follows
Algorithm 1's ordering. With `p=2`, zero threshold, epsilon represented as an
initial accumulator, and no post-root epsilon, it matches PyTorch AdaGrad.

### Reference-code divergences

The official repository was audited at commit
`ea3fe3c67279e27db8edea73d5f3ad522bb15dc1`; `CLASSP.py` has Git blob
`2bdf432b3f8bfdc4fc249cc6afadb9f150313032`.

- Official code: `(grad ** 2).any() > threshold` converts the whole tensor to a
  Boolean before comparing it, so it does not implement coordinate-wise
  thresholding.
- Official code: `grad_sum.add_(grad.abs()) ** power` discards the powered
  result, so the stored accumulator is an L1 sum for every `power`.
- Paper-equation implementation: applies the threshold and `abs(g)^p`
  accumulator elementwise, preserves rejected-coordinate state, and rejects
  nonfinite gradients.
- The official MNIST→Fashion-MNIST experiment changes threshold and scaling
  between datasets. This study does not reproduce that schedule, architecture,
  task pair, or loss. It is explicitly a **paper-equation variant** on a fixed
  Permuted-MNIST protocol.

The local normative PDF has SHA-256
`0e528058503ba9424969661bccf13df9150f709bd0a073c5cc4bdc98b80e8e56`.

## Frozen factorial

| Condition | Method | Parameter optimizer | LR | PC `T` |
|---|---|---|---:|---:|
| `bp-adam` | BP | persistent Adam | `3e-4` | — |
| `bp-classp` | BP | persistent paper-equation CLASSP | `3e-4` | — |
| `pc-adam` | PC | persistent Adam | `3e-4` | `20` |
| `pc-classp` | PC | persistent paper-equation CLASSP | `3e-4` | `20` |

Common full-study configuration:

- 20-task domain-incremental Permuted MNIST; identity task first and 19 fixed
  pixel permutations generated once with permutation seed `0`;
- torchvision default 60,000-example training and 10,000-example test splits;
- one pass per task, no replay, no task ID input, batch size `500`;
- MLP `784→256→256→10`, ReLU, biases enabled;
- the same realized linear tensors in BP and PC for each paired seed;
- explicit per-task sample orders derived only from experimental seed and task
  index, shared by all four conditions;
- one-hot half-squared error summed over batch and classes for BP and PC;
- PC parameter update at the last of `T=20` inference steps, latent SGD
  learning rate `0.01`, reset for each batch;
- CLASSP `p=2`, strict squared-gradient threshold `1e-5`, epsilon `1e-5`, and
  always-on decay;
- deterministic PyTorch algorithms and sequential cell execution on CUDA.

Epsilon was not fixed in the stopped draft. `1e-5` is frozen here because it
is the paper reference implementation's default and the computer-vision
experiment's declared value. It is not tuned on this task.

## Seeds and inclusion

- Primary held-out seeds: `7`, `123`, `2026`, `31415`, `271828`.
- Legacy-bridge seed: `42`, reported separately and excluded from the primary
  interaction summary because it informed configuration selection.
- Four conditions across six seeds produce exactly 24 full cells.
- No tuning or seed substitution is permitted after this registration.

## Canonical observations and metrics

Training records prediction accuracy on **every training batch before that
batch's update**, plus correct/sample counts and per-task/overall aggregates.

The test matrix `R` has shape `(N+1)×N`. Row 0 is before any training; row
`i+1` is after acquiring task `i`; column `j` evaluates task `j`.

- adaptation for task `j`: `R[j+1,j] - R[j,j]`;
- BWT for tasks `0..N-2`: `R[N,j] - R[j+1,j]`;
- forgetting for tasks `0..N-2`:
  `max(R[j+1:N+1,j]) - R[N,j]`;
- final average accuracy: mean of the final row.

The analysis reports all held-out seed values, mean, sample standard deviation,
range, and positive/zero/negative sign counts for `ΔPC`, `ΔBP`, and `ΔΔ`
on final accuracy and secondary metrics. Forgetting differences remain on the
raw scale, where lower is better.

## Execution modes and artifacts

`smoke` mode uses the same path with two tasks and the first 1,000 examples
from each test set, while presenting a deterministic 1,000-example shuffled
training subset per task. It is frozen to seed `42`.

`full` mode has no implicit default. It requires CUDA, the exact confirmation
`--confirm-protocol matched-lr-paper-v1`, a permitted seed, committed study
sources, clean tracked repository/submodule revisions, and diagnostics enabled.

Every invocation runs one cell and requires explicit `--mode`, `--condition`,
`--seed`, `--device`, and `--output-dir`. Results and receipts are written
atomically with exclusive creation. Existing result, receipt, or failure paths
cause refusal rather than skip or overwrite. Failures receive a separate
atomic JSON artifact with traceback and available provenance.

Each result records resolved configuration, source revisions and source-file
hashes, tracked-dirty state, environment and GPU, raw dataset hashes, task
permutations, sample orders, initialization, sample/update counts, optimizer
state across boundaries, all online batch accuracies, the full matrix, metrics,
timestamps, runtime, memory, a canonical-payload hash, and an external receipt
containing the complete JSON hash.

Staging belongs under ignored
`playground/continual-mnist/run-staging/`. Only reviewed, validated final
artifacts may later be force-added to a tracked results location.
`session_logs/` remains excluded.

## Gates and stopping rules

1. **Algorithm gate:** hand-check thresholding, persistent state, `p=1` and
   `p=2`, and `apply_decay`; compare the AdaGrad special case.
2. **Smoke gate:** run all four conditions on CUDA in smoke mode, validate the
   complete set, confirm streams/initialization/source identity, finite values,
   and counts, then compare a separate diagnostics-on/off pair.
3. **Human checkpoint:** present design, paper/reference-code audit, tests,
   smoke results, compute estimate, changed files, and pre-commit packet. Do
   not commit or start full cells before explicit approval.
4. **Full study:** after the approved clean commit, run 24 cells sequentially.
   Validate each cell and paired invariants immediately. Preserve and stop on
   the first failed invariant or failure artifact.
5. **Synthesis checkpoint:** accept the study only if all 24 cells and receipts
   validate. Generate the interaction summary only from those raw JSON files.
   Results/synthesis require a second review before commit.

Expected full-study wall time is approximately 4–6 hours on the local RTX
4090, including the expanded `(N+1)×N` evaluation.

## Commands

```bash
# Algorithm and synthetic runner gates
.venv/bin/python playground/continual-mnist/test_classp_optimizer.py
.venv/bin/python playground/continual-mnist/test_matched_lr_validation.py

# One CUDA smoke cell (repeat for all four condition IDs)
.venv/bin/python playground/continual-mnist/run_matched_lr_validation.py \
  --mode smoke --condition bp-adam --seed 42 --device cuda \
  --output-dir playground/continual-mnist/run-staging/smoke-gate

# Validate the four-cell smoke set
.venv/bin/python playground/continual-mnist/validate_matched_lr_results.py \
  --mode smoke --require-complete \
  --output-dir playground/continual-mnist/run-staging/smoke-gate

# Full example after human approval and a clean source commit
.venv/bin/python playground/continual-mnist/run_matched_lr_validation.py \
  --mode full --condition bp-adam --seed 7 --device cuda \
  --confirm-protocol matched-lr-paper-v1 \
  --output-dir playground/continual-mnist/run-staging/full
```

Any change to the task stream, split, seed set, metrics, objective, optimizer
persistence, hyperparameters, or comparator creates a new protocol version and
requires all affected conditions to be rerun.
