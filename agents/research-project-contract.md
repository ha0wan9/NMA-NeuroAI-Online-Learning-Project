---
artifact_name: research-project-contract
origin: project-owned
project_scope: this repo only
owner: agent-facing
review_policy: team review when milestone gates, claim discipline, or evaluation invariants change
last_reviewed: 2026-07-15
---

# Research Project Contract

This contract governs the two-week Online/Continual Learning group project. It converts the July 14 brainstorming decisions into research gates without imposing a generic software phase lock.

## Scope and Scientific Objective

The project investigates whether one or two biologically plausible learning rules can remain plastic under streaming, noisy, or non-stationary conditions while limiting catastrophic forgetting. Candidate families include Predictive Coding and SNN-based plasticity; BTSP, local three-factor rules, eligibility-trace methods, and other alternatives remain candidates until the literature gate supports them. Backpropagation is the principal reference baseline, with Hebbian learning included when it clarifies the comparison.

A candidate architecture is not automatically a biologically plausible learning rule. Claims apply to a specified model, update rule, information pathway, and experimental protocol.

## Milestone Gates

### 1. Brainstorm — complete (July 14)

Entry: group formation and broad idea collection.

Deliverables: Online/Continual Learning chosen as the group subject; Backpropagation comparison and the weight-transport motivation identified; Predictive Coding, SNN-based rules, BTSP, and Hebbian learning recorded as candidate directions or references.

Exit gate: the meeting record documents the subject, candidate families, target difficulties, comparison intent, and Day 3/Day 4 schedule. This gate is satisfied by `meeting-notes/2026-07-14-meeting-1.md`.

### 2. Literature Review — Day 3 (July 15)

Entry: Brainstorm exit gate satisfied.

Required deliverables:

- operational definitions of online learning, continual learning, and the biological-plausibility dimensions used to screen methods
- a demand survey describing the unresolved need: plasticity, stability, locality, streaming constraints, and which Backpropagation limitations matter here
- an evidence matrix for each candidate: primary source, rule, biological claim, online/continual setting, data access, replay or memory budget, comparator, metric, result, code availability, and limitation
- explicit coverage of Predictive Coding and SNN learning rules, plus evidence-led consideration of other local or biologically motivated rules
- a ranked shortlist of two to four candidate rules and drafts of three research questions

Exit gate:

- every important claim has a primary or authoritative source and a claim-status label
- each shortlisted rule has implementation evidence, a plausible two-week test, and at least one relevant online or continual benchmark or protocol
- performance claims distinguish matched evidence from cross-paper impressions
- known evidence gaps and feasibility risks are visible rather than silently resolved

### 3. Proposal — Day 4 (July 16)

Entry: Literature Review exit gate satisfied.

Required deliverables:

- motivation grounded in the demand survey
- exactly three answerable research questions with claim-labelled hypotheses
- one or two candidate learning rules, Backpropagation baseline, and any justified secondary baseline
- workflow, roles, implementation scope, datasets, stream construction, metrics, compute limits, schedule, and risks
- a draft protocol identifier and a list of decisions that must freeze before comparison

Exit gate: the group can map every research question to a comparator, manipulation, metric, expected artifact, and falsifying outcome; the implementation fits the remaining project time.

### 4. Baseline

Entry: Proposal exit gate satisfied.

Required deliverables:

- a reproducible Backpropagation/SGD online-learning baseline; add a Hebbian baseline only when it answers a proposal question
- deterministic data preparation and stream generation, explicit preprocessing, seeds, evaluation cadence, and checkpoint policy
- a minimal smoke run and one reference run with configuration, environment assumptions, command, and outputs recorded

Exit gate: the baseline runs end-to-end, the evaluator produces the planned metrics, leakage checks pass, and another contributor can reproduce the smoke run from the documented command.

**No novel-rule implementation begins before this exit gate.** Feasibility prototypes may test an isolated equation or API, but they cannot produce comparative claims and must not mutate the baseline protocol.

### 5. Implementation

Entry: Baseline exit gate satisfied.

Required deliverables:

- one or two shortlisted learning rules implemented against the baseline data and evaluation interfaces
- checks for local update equations, state reset/carry behavior, tensor or spike timing, and online update order
- an implementation note mapping code operations to the claimed biological-plausibility dimensions and naming approximations
- a smoke run using the same observation and logging path as the baseline

Exit gate: focused checks and smoke runs pass, known approximations are documented, and no candidate receives privileged future information, memory, replay, or tuning access.

### 6. Evaluation

Entry: Implementation exit gate satisfied and the protocol is frozen before comparative runs.

Protocol freeze must version and lock:

- online/continual scenario, sample order, task or distribution boundaries, class exposure, and task-ID availability
- train/validation/test partitions, preprocessing, augmentation, revisit policy, and leakage checks
- replay contents and capacity, persistent state, parameter count, compute/time budget, and update frequency
- seed set, evaluation cadence, metrics, stopping rules, hyperparameter-search budget, and result inclusion rules

Required deliverables:

- matched runs for all comparators using the frozen protocol and seed set
- learning speed and online/prequential performance; final or average accuracy; forgetting or backward transfer when applicable; robustness to the chosen noise/non-stationarity; compute and memory cost
- uncertainty across seeds, failures, negative results, and deviations
- a biological-plausibility audit alongside task performance rather than a single unsupported plausibility score

Exit gate: every displayed comparison is reproducible, uses matched data access and budgets or discloses the mismatch, and remains traceable to configuration and run artifacts. Any post-freeze protocol change creates a new protocol version and requires all affected comparators to be rerun.

### 7. Synthesis

Entry: Evaluation exit gate satisfied.

Required deliverables: mid-project/final materials, method and protocol summary, result tables or figures, answers to the three research questions, negative results, biological-plausibility limitations, and follow-up questions.

Exit gate: every conclusion is traceable to a primary source or matched run; facts are separated from interpretation; hypotheses are reported as supported, weakened, unresolved, or falsified; limitations and reproducibility instructions are included.

## Biological-Plausibility Claim Discipline

Evaluate each learning rule explicitly on:

- locality of information available at a synapse or neuron
- need for symmetric feedback weights or weight transport
- global error signals, layer locking, and update timing
- differentiability and surrogate-gradient assumptions
- temporal credit mechanism and permitted eligibility or neuromodulatory signals
- neuron and communication model, including whether spikes are essential or merely an implementation choice
- persistent state, replay, and memory requirements

Use these labels in research artifacts:

- **Established fact:** directly supported by cited evidence or a verified run.
- **Interpretation:** a reasoned reading of evidence; alternatives remain possible.
- **Working hypothesis:** a falsifiable expectation to test in this project.
- **Speculation:** a plausible idea without adequate evidence here.
- **Open question:** unresolved and not treated as a conclusion.

Do not call a method comparable to or better than Backpropagation unless the comparison is matched on data stream, architecture or disclosed capacity, information access, replay/memory budget, tuning budget, metric, and uncertainty. Cross-paper differences are demand-survey evidence, not a head-to-head result. Distinguish an algorithmic plausibility argument from biological evidence about real neural mechanisms.

## Online and Continual Evaluation Invariants

- Define whether the protocol is single-pass online, online with bounded replay, task-incremental, domain-incremental, or class-incremental; do not mix their conclusions.
- Updates are causal: no future samples, labels, task boundaries, or test information unless the protocol explicitly grants them to every comparator.
- Record stream construction and seeds; use the same realized stream for matched comparisons.
- Report revisits, replay, persistent per-example state, and external memory as part of the algorithm budget.
- Use the same evaluation cadence and metric implementation for all rules.
- Give comparators comparable tuning information and search budgets; never tune on the test stream unless that is an explicit, shared online-evaluation rule.
- Separate plasticity, stability/forgetting, robustness, accuracy, and resource cost. One metric does not establish overall superiority.
- Keep raw run records and negative outcomes; do not select only favorable seeds or stream orders.

## Controlled Feedback Loops

Milestones may send work backward when evidence or experiments expose a real problem:

- Literature evidence may refine proposal wording or remove an unsupported candidate.
- Proposal feasibility checks may narrow a research question before approval.
- A failed baseline returns work to Baseline; it does not authorize weakening comparison criteria.
- Implementation findings may trigger a documented proposal or protocol revision before freeze.
- Evaluation anomalies may return to Baseline or Implementation for diagnosis while preserving the original run record.
- A post-freeze change invalidates affected comparisons until every comparator is rerun under a new protocol version.

Each loop records the trigger, changed assumption, affected artifacts, decision owner, and reruns required. Feedback is allowed; silently bypassing an exit gate is not.
