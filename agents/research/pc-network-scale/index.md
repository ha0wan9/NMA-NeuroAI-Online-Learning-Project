# PC Network Scale

## Identity

- Study ID: `pc-network-scale`
- Owner: project team
- Created UTC: `2026-07-17T22:55:56Z`
- Study root: `agents/research/pc-network-scale/`
- Branch: `codex/step1-pc-bp-protocol`
- Adapter: `agents/research/adapter.yaml`, with launch/parser overrides below
- Status: `synthesized`
- Parent study: `pc-relaxation-plasticity`

## Question

How do greater width and greater depth change BP and PC accuracy, operational
plasticity, forgetting, runtime, parameter count, and GPU memory in matched
static MNIST, SplitMNIST, and Permuted MNIST evaluations?

## Success Criteria

- Primary metrics: final accuracy within each scenario (`maximize`).
- Decision gate: report paired architecture-minus-baseline effects separately
  for BP and PC. Call a scale benefit only if the mean final accuracy improves
  by at least `0.5` pp without a plasticity regression larger than `1.0` pp;
  cost and all paired seed signs must be disclosed.
- Secondary guardrails: static epoch-1 gain/curve mean, continual adaptation
  and prequential accuracy, forgetting/BWT, parameters, memory, runtime,
  matching BP–PC initialization, shared data/task checksums, and decreasing PC
  diagnostic objectives.
- Required repetitions: paired seeds `7`, `42`, and `123`.

## Scope

- Baseline: `784→256→256→10` (`256×2`).
- Width intervention: `784→512→512→10` (`512×2`).
- Depth intervention: `784→256→256→256→256→10` (`256×4`).
- PC relaxation: fixed at `T=5`, sufficient for four PC layers.
- Out of scope: width–depth factorial combinations, CNNs, normalization,
  optimizer tuning, replay, data/stream changes, or project-default promotion.

## Protocol

- Control: fresh `256×2` BP and PC runs in the same artifact.
- Data: torchvision MNIST, split seed `20260717`, permutation seed `0`, and
  recorded split/task checksums.
- Launch: `bash playground/step1-static-mnist/remote-run-architecture-sweep.sh`.
- Parser: `python3 playground/step1-static-mnist/architecture_analysis.py --input playground/step1-static-mnist/runs/architecture-sweep-v2.json --output-dir agents/research/pc-network-scale/artifacts`.
- Editable surface and protected protocol: project adapter. The user explicitly
  requested the architecture intervention on `2026-07-18`; data, metrics,
  optimizers, seeds, and evaluation cadence remain frozen.
- Budget: one combined sweep, at most six CyberEngine GPU-hours.
- Stop policy: preserve failures; stop on non-finite values, checksum/stream
  mismatch, unavailable GPU, or the six-hour ceiling.

## H/E Tracks

| Track | Directory | Hypothesis / route | Status | Decision gate |
|---|---|---|---|---|
| `H1` | `H1-width/` | More width may improve capacity-dependent accuracy/plasticity. | partial support | Compare `512×2` with `256×2` per method. |
| `H2` | `H2-depth/` | More depth may change PC credit propagation and stability. | rejected under frozen protocol | Compare `256×4` with `256×2` per method. |

## Managed Review Policy

- Design and implementation-intent review: required before full launch.
- Result-skeptic review: required before any scale-benefit claim.
- Methodology audit: required before adopting a larger default.

## Protocol Changes

| UTC | Change | Reason | Approved by | Effective from run |
|---|---|---|---|---|
| 2026-07-17T22:55:56Z | Generalize hidden width/layer count while preserving original defaults; add static split checksums. | User requested larger-network study. | User | `pc-network-scale-v1` |
| 2026-07-17T23:20:00Z | Restore BP-first module construction and rerun every condition as v2; preserve v1 as invalid. | Interleaved BP/PC construction changed the seeded baseline initialization relative to the frozen protocol. | Protocol audit | `pc-network-scale-v2` |

## Pointers

- `01-survey.md`
- `02-design.md`
- `03-monitor.md`
- `04-evaluation.md`
- `05-synthesis.md`
- `runs.jsonl`
- `research_graph.mmd`
- `research_graph.json`
