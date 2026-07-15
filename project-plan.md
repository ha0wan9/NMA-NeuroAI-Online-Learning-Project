# Online/Continual Learning Project Plan

**Planning date:** July 15, 2026

**Team size:** 8 people

**Project length:** Two weeks

**Current phase:** Literature Review (Day 3)

## 1. Project objective

Test whether one or two biologically plausible learning rules can learn under a
causal online or continual stream while remaining plastic, limiting forgetting,
and coping with a selected harder condition such as noise or non-stationarity.
Compare them with a reproducible Backpropagation/SGD baseline and, only when it
answers a research question, a Hebbian baseline.

The team will assess both task performance and biological plausibility. A
spiking architecture by itself is not a learning rule, and cross-paper results
will not be presented as matched head-to-head evidence.

## 2. Source decisions and current status

This plan translates the decisions in
[`meeting-notes/2026-07-14-meeting-1.md`](meeting-notes/2026-07-14-meeting-1.md)
and the gates in
[`agents/research-project-contract.md`](agents/research-project-contract.md)
into team ownership and daily work.

### Confirmed

- The theme is Online/Continual Learning.
- The project will implement one or two novel learning rules and compare them
  with Backpropagation; Hebbian learning is a possible secondary reference.
- Predictive Coding, SNN-based local plasticity, and BTSP are candidate routes,
  not final selections.
- The primary test concerns robustness under a harder online setting.
- Learning speed, accuracy, and signal-to-noise behavior are candidate
  signatures; forgetting and resource cost are also required when applicable.
- The proposal is due on Day 4, July 16, and must include motivation, exactly
  three research questions, and the workflow.

### Open or provisional

- **Open question:** Which one or two rules will be implemented?
- **Open question:** Which online/continual scenario, dataset, stream shift, and
  noise model will be frozen?
- **Open question:** Is a Hebbian baseline necessary for one of the three
  research questions?
- **Open question:** What are the exact mid-project and final-presentation dates?
- **Current gate risk:** The SOTA survey remains `audit-needs-roundN`; five weak
  evidence cells require stronger independent coverage before literature
  synthesis and candidate selection are defensible.

## 3. Team structure

Replace `Member 1` through `Member 8` with names at the next team check-in. Each
artifact has one directly responsible individual (DRI), even when a pair works
on it together.

| Pair | Member and primary role | Initial responsibility | Later-phase responsibility |
|---|---|---|---|
| Coordination and evidence | **Member 1 — Project lead and proposal editor** | Own scope, decisions, proposal integration, and daily unblock | Integrate reports and presentation; enforce milestone gates |
| Coordination and evidence | **Member 2 — Evidence and plausibility lead** | Close literature weak cells; own claim labels and evidence matrix | Audit biological-plausibility claims and limitations |
| Methods | **Member 3 — Predictive Coding lead** | Assess update rule, online evidence, code, and feasibility | Implement Candidate A if selected; otherwise review the chosen rule |
| Methods | **Member 4 — SNN/local-rule lead** | Assess STDP, three-factor, BTSP, or other local-rule evidence | Implement Candidate B if selected; otherwise support Candidate A and ablations |
| Baseline and protocol | **Member 5 — Data and baseline lead** | Draft dataset/stream options and Backpropagation baseline design | Implement deterministic stream and baseline; record smoke-run command |
| Baseline and protocol | **Member 6 — Evaluation and protocol lead** | Map each research question to comparator, metric, and falsifier | Implement evaluator; own leakage checks and protocol freeze |
| Reproducibility and results | **Member 7 — Testing and reproducibility lead** | Audit candidate code availability, environments, and compute risk | Own focused checks, configs, seeds, run provenance, and reproduction |
| Reproducibility and results | **Member 8 — Analysis and communication lead** | Draft result-table/figure shells and presentation narrative | Analyze matched runs, uncertainty, negative results, and visuals |

### Pair operating rules

- Each pair has a 10-minute daily sync; the full team has a 15-minute daily
  checkpoint led by Member 1.
- The DRI writes the artifact; the paired member reviews it before handoff.
- Members 3 and 4 do not begin comparative candidate implementation before the
  baseline gate passes. Isolated equation or API feasibility checks are allowed
  but cannot support performance claims or change the baseline protocol.
- Members 5 and 6 jointly own fairness across comparators; Members 2 and 7
  independently review claim traceability and reproducibility.
- Every blocker or scope change is recorded with its owner, affected artifact,
  decision, and reruns required.

## 4. Work plan and gates

Dates after July 16 are relative because the meeting note does not specify the
mid-project or final-presentation dates. Member 1 must replace them with course
dates as soon as the team confirms them.

| Window | Goal and exit condition | Primary owners | Deliverables |
|---|---|---|---|
| **Day 3 — Jul 15** | Close the Literature Review gate | Members 2–4; review by 1, 5, 6 | Definitions, demand survey, evidence matrix, five weak-cell follow-ups, ranked shortlist of 2–4 rules, three draft research questions |
| **Day 4 — Jul 16** | Submit an answerable and feasible proposal | Member 1; all members approve their sections | Motivation, exactly 3 research questions and labelled hypotheses, 1–2 chosen rules, baselines, workflow, dataset/stream options, metrics, compute cap, risks, draft protocol ID |
| **Days 5–6** | Freeze design, then pass the Baseline gate | Members 5–7 | Deterministic data/stream pipeline, Backpropagation/SGD baseline, evaluator, leakage checks, smoke run and reference run with command/config/output |
| **Days 7–9** | Pass the Implementation gate | Members 3, 4, and 7 | 1–2 rule implementations, equation/update-order checks, plausibility/approximation notes, matched-path smoke runs |
| **Mid-project checkpoint — date TBD** | Demonstrate an end-to-end baseline and honest implementation status | Members 1 and 8; evidence from all | Short methods/protocol summary, baseline result, candidate smoke status, risks and decisions needed |
| **Days 10–11** | Freeze protocol and run matched evaluation | Member 6; runs by 3–5 and provenance by 7 | Versioned protocol, same realized streams/seeds/cadence, matched runs, raw records, failures and deviations |
| **Days 12–13** | Analyze and synthesize without cherry-picking | Members 2, 7, and 8 | Metrics with uncertainty, robustness/forgetting/resource results, plausibility audit, negative results, answers to all 3 questions |
| **Day 14 / final — date TBD** | Deliver reproducible conclusions | Members 1 and 8; all members review | Final report/presentation, result tables/figures, limitations, reproduction instructions, follow-up questions |

## 5. Immediate 24-hour checklist

### Literature pair — Members 1 and 2

- [ ] Assign the five weak evidence cells and record independent sources.
- [ ] Complete the evidence matrix fields required by the research contract.
- [ ] Label every important statement as established fact, interpretation,
  working hypothesis, speculation, or open question.
- [ ] Produce a ranked shortlist with evidence gaps and feasibility risks shown.

### Methods pair — Members 3 and 4

- [ ] Write a one-page feasibility card for each serious candidate: exact update
  rule, required signals/state, online evidence, available code, expected effort,
  biological caveats, and simplest two-week test.
- [ ] Recommend one primary rule and one optional second rule.
- [ ] Define a fallback rule in case the primary implementation fails by Day 7.

### Baseline/protocol pair — Members 5 and 6

- [ ] Propose at most two dataset-and-stream designs, including why each exposes
  plasticity, forgetting, noise, or non-stationarity.
- [ ] Draft a comparison table covering information access, replay/memory,
  parameter/compute budget, tuning budget, seeds, cadence, and metrics.
- [ ] Map each draft research question to a manipulation, comparator, metric,
  expected artifact, and falsifying outcome.

### Reproducibility/results pair — Members 7 and 8

- [ ] Verify candidate code and environment claims; separate “repository exists”
  from “minimal run reproduced.”
- [ ] Draft empty result tables and figure specifications before experiments.
- [ ] Create the risk register and flag decisions needed for the proposal.

### Full-team proposal checkpoint — July 16

- [ ] Select exactly three answerable research questions.
- [ ] Select one required candidate rule; add a second only if it fits the time
  and compute budget without risking the first matched comparison.
- [ ] Approve the baseline, harder scenario, metrics, roles, and fallback scope.
- [ ] Give the design a draft protocol ID; list every item still requiring freeze.

## 6. Required experiment contract

Before comparative runs, Member 6 records and the full team approves:

- online regime: single-pass, bounded replay, task-incremental,
  domain-incremental, or class-incremental;
- realized sample order, boundaries, class exposure, and task-ID access;
- splits, preprocessing, augmentation, revisit policy, and leakage checks;
- replay capacity, persistent state, parameter count, compute/time budget, and
  update frequency;
- seeds, evaluation cadence, metrics, stopping rules, search budget, and result
  inclusion rules.

All comparators receive the same permitted information, stream, evaluation
cadence, and tuning budget. Any post-freeze change creates a new protocol
version and requires affected comparators to be rerun.

## 7. Metrics and evidence outputs

The proposal should select only metrics that answer a research question, from:

- online or prequential performance and learning speed;
- final and average accuracy;
- forgetting or backward transfer;
- robustness under the chosen noise or distribution shift;
- signal-to-noise behavior when operationally defined;
- compute time, update cost, parameter count, persistent state, and replay use;
- a dimension-by-dimension biological-plausibility audit covering locality,
  weight transport, global errors, update timing, differentiability, temporal
  credit, neuron/communication model, and memory requirements.

Every displayed comparison links to its configuration, seed, raw run record,
and protocol version. Cross-paper impressions remain literature context rather
than project results.

## 8. Risk register and fallback rules

| Risk | Trigger | Mitigation / fallback | Owner |
|---|---|---|---|
| Literature gate remains open | Five weak cells are not independently supported by proposal review | Narrow claims and candidate scope; record gaps; do not claim synthesis is complete | Member 2 |
| Candidate is underspecified or code cannot run | No verified equation/API feasibility by end of Day 4 | Choose the best-supported feasible candidate or reduce to one rule | Members 3–4 |
| Baseline slips | No end-to-end smoke run by end of Day 6 | Stop candidate integration and swarm on baseline/evaluator | Member 5 |
| Two candidates exceed capacity | Candidate A is not through smoke tests by Day 8 | Drop Candidate B; preserve one fair, reproducible comparison | Member 1 |
| Unfair comparison | Comparator has different future information, replay, tuning, or cadence | Correct before freeze; after freeze, version protocol and rerun affected methods | Member 6 |
| Compute budget is exceeded | Reference-run estimate does not fit remaining time | Reduce model/stream size equally for all comparators before freeze | Member 7 |
| Results are noisy or negative | High seed variance or no improvement | Report uncertainty and negative result; do not change metrics or select favorable seeds | Member 8 |
| Integration bottleneck | A DRI misses a daily handoff | Pair reviewer takes over the artifact; Member 1 rebalances work that day | Member 1 |

## 9. Definition of done

The project is complete when:

- the proposal's three research questions are each answered with matched
  evidence or explicitly marked unresolved;
- at least the Backpropagation baseline and one biologically plausible rule run
  end-to-end under the same frozen protocol;
- conclusions are traceable to primary sources or matched runs;
- raw records, seeds, configurations, failures, negative results, and deviations
  are retained;
- biological claims name both supported dimensions and approximations;
- another contributor can reproduce the smoke run using the documented command;
- the final materials state limitations and distinguish facts,
  interpretations, hypotheses, speculation, and open questions.

## 10. Decision log template

Use this table in the proposal or a linked project record. Do not silently
change a gate, candidate, or protocol assumption.

| Date | Decision or trigger | Evidence | Affected artifacts/runs | Owner | Rerun required? |
|---|---|---|---|---|---|
| 2026-07-15 | Eight-person plan created; assignments await names | Meeting 1 and project contract | Planning only | Member 1 | No |
