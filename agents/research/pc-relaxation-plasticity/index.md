# PC Relaxation Plasticity

## Identity

- Study ID: `pc-relaxation-plasticity`
- Owner: project team
- Created UTC: `2026-07-17T21:53:49Z`
- Study root: `agents/research/pc-relaxation-plasticity/`
- Branch: `codex/step1-pc-bp-protocol` (existing user-requested experiment branch)
- Adapter: `agents/research/adapter.yaml`
- Status: `synthesized`
- Parent study: none

## Question

How does changing Predictive Coding latent relaxation from 1 to 5, 10, or 20
steps affect plasticity, accuracy, forgetting, runtime, and temporary state in
matched static MNIST, SplitMNIST, and Permuted MNIST evaluations?

## Success Criteria

- Primary metric: final accuracy within each scenario (`maximize`).
- Decision gate: characterize the direction and magnitude of adjacent-step and
  BP-control differences; call a monotonic effect only if all adjacent
  three-seed means move in the same direction.
- Secondary guardrails: static first-epoch validation gain, continual
  adaptation gain and prequential accuracy, forgetting/BWT, runtime, memory,
  matching initialization checksums, and decreasing first-batch PC objective.
  The objective-decrease check is not assessable at `T=1`, which contains only
  one recorded objective value and is below the two-layer propagation minimum.
- Required repetitions: paired seeds `7`, `42`, and `123` at every level.

## Scope

- In scope: PC steps `1/5/10/20`, one fresh BP control, static MNIST,
  SplitMNIST, Permuted MNIST, raw artifacts, analysis, and report.
- Out of scope: tuning state learning rate, parameter learning rate, batch
  size, architecture, task construction, replay, objective, or seed selection.

## Protocol

- Baseline/control: fresh BP run inside `relaxation-sweep-v1.json`.
- Data version source: torchvision MNIST; deterministic split seed `20260717`;
  continual task stream checksums recorded in the raw artifact.
- Eval command/parser: adapter `metric_parser`.
- Editable surface: adapter `editable_surface`.
- Protected files: adapter `protected_files`; user approved the full rerun on
  `2026-07-17` and no accuracy/data metric change was ultimately required.
- Run naming: `pc-relaxation-plasticity-<HnEn>-<experiment-name>`.
- Budget: one four-level sweep, maximum six CyberEngine GPU-hours.
- Stop policy: adapter `stop_policy`.

## H/E Tracks

| Track | Directory | Hypothesis / route | Status | Decision gate |
|---|---|---|---|---|
| `H1` | `H1-relaxation-depth/` | More relaxation changes the accuracy–plasticity–cost trade-off. | evaluated | Paired multi-seed characterization in all three settings. |

## Managed Review Policy

- Design critic required when: before the full GPU sweep.
- Result skeptic required when: any superiority or optimal-step claim is made.
- Methodology auditor required when: promoting a setting as a project default.
- Optional reviewers: implementation-intent reviewer before launch.

## Protocol Changes

| UTC | Change | Reason | Approved by | Effective from run |
|---|---|---|---|---|
| 2026-07-17T21:53:49Z | Add discarded warm-up, explicit GPU synchronization, and symmetric exclusion of one diagnostic batch per static run / continual task from timed-sample accounting. Accuracy and learned updates remain unchanged. | Make runtime comparison across relaxation depths interpretable without T-dependent diagnostic logging overhead. | User | `pc-relaxation-sweep-v1` |

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
