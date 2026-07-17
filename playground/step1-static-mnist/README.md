# Step 1: Static MNIST Feasibility

This harness checks that Predictive Coding (PC) and Backpropagation (BP) can
train through one shared, deterministic static-MNIST path. It is a software and
method feasibility check only. It is not an online or continual-learning
experiment, does not test forgetting, and does not authorize a claim that one
method outperforms the other.

For a high-level-first visual explanation of the workflow, experiment design,
fairness controls, evidence artifact, and manual commands, open the
[standalone HTML guide](guide.html).

## Provenance and dependency boundary

The team-owned runner adapts experiment-orchestration ideas developed in the
project contributor's `PredictiveCoding` repository at commit `dba6e9a`. The
PC implementation itself remains in the Bogacz Group Git submodule pinned at
commit `5bf803c3636a39928488941f05539c70ab4df0b1`.

The upstream repository did not contain a license when reviewed. Its source is
not copied into this harness. Do not redistribute or modify the upstream code
without clarification from its maintainers.

## Setup and checks

From the repository root, using Python 3.11:

```bash
git submodule update --init --recursive
uv sync --locked --extra cpu
uv run --extra cpu python -m unittest playground/step1-static-mnist/test_protocol.py
uv run --extra cpu python playground/step1-static-mnist/experiment.py --print-config
```

`--print-config` does not load MNIST or train a model.

## Smoke run

```bash
uv run --extra cpu python playground/step1-static-mnist/experiment.py \
  --config smoke --device cpu --methods pc bp --seeds 0
```

The first run downloads MNIST to the ignored repository-local `data/`
directory. The smoke configuration uses a stratified 1,000-example training
subset, a separate stratified 500-example test subset, one epoch, and a
`784 -> 64 -> 64 -> 10` network. PC uses five latent-inference iterations per
parameter update. BP and PC receive matching initial trainable parameters,
sample identities, realized order, loss, optimizer family, and evaluation set.

The command creates `results/smoke-seed-0.json`. Run artifacts are
non-overwriting: choose a new `--output-dir` to repeat a run. The JSON records
configuration, stream hashes, accuracy, resource counters, environment and Git
provenance, warnings, and failures. Runtime is descriptive rather than a
compute-matched comparison because PC performs additional inference work.

## Interpretation boundary

A successful record establishes only that both methods execute under this
small static protocol. Accuracy differences from one seed are not comparative
evidence. Continual-learning streams, metrics, and matched PC/BP evaluation
remain governed by the protocol and baseline gates in the project contract.
