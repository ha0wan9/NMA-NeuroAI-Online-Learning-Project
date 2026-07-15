# Biologically Plausible Online Learning: Reading Roadmap Survey

> Provenance note: most mechanism claims were checked against publisher or
> proceedings abstracts; a smaller subset was checked in full-text HTML. The
> The short summaries and project-fit judgments are curator interpretations, not
> verbatim claims by the authors. No cross-paper accuracy ranking is attempted.
>
> Target adaptation: this is **Stage 1 — Domain Introduction** for a
> reader with a general scientific background. Bare `P###` references use the
> canonical shared
> [`paper_index.md`](paper_index.md); source-ID provenance is in
> [`paper-id-crosswalk.md`](paper-id-crosswalk.md). Continue to
> [Stage 2 — SOTA Results Explorer](index.md) after completing the core spine.

## §1 Research question and scope

The survey asks for the shortest defensible path from basic learning rules to
current research on biologically plausible online learning for a reader with
general scientific literacy. The endpoint is not “the brain uses algorithm X.” It is a
comparison of which constraints each route addresses: synaptic locality,
feedback symmetry, global teaching signals, forward/backward phases,
non-differentiable spikes, temporal credit, bounded memory and continual
retention (P041, P002, P001).

The associated two-week project makes the route target-directed: papers that
do not change the conceptual prerequisites, experimental protocol or feasible
shortlist are omitted even when they are historically influential.

## §2 Domain background

Backpropagation is the computational reference for multilayer credit assignment
(P031), while Oja's rule is a small example of an adaptive neuron extracting a
principal component from a stochastic input stream (P032). Sequential exposure
creates a second problem—interference with earlier learning (P034)—and modern
continual-learning work identifies a distinct failure mode: a network can
gradually lose the ability to learn new data (P007).

For reading purposes, the roadmap treats synaptic plasticity, spatial credit,
temporal credit and continual stability as separate axes, anchored by different
paper clusters (C036). A method may address more than one axis; the chart does
not infer unreported properties from a “biologically plausible” label.

## §3 Two-axis taxonomy and timeline

### Primary axis: mechanism route

1. **Local plasticity:** STDP, eligibility traces and BTSP (P036, P024, P037, P038).
2. **Spatial credit:** feedback alignment, dendritic segregation, equilibrium
   dynamics and predictive coding (P025, P039, P028, P011, P040, P026, P041, P002, P013).
3. **Temporal credit:** e-prop, DECOLLE, OSTL, OTTT and S-TLLR (P018, P019, P042, P022, P020, P043).
4. **Online/continual stability:** protocol definitions, predictive-coding
   systems and plasticity maintenance (P044, P001, P016, P045, P046, P007, P014).
5. **Frontier integration:** phaseless/dendritic real-time rules,
   astrocyte-gated timescales and generalized latent equilibrium (P047, P048, P049, P050).

### Secondary axis: unresolved constraint

Each route is read again along four questions: Where is the teaching
signal generated? What state must persist through time? Does the rule require a
separate inference or backward phase? What prevents new learning from erasing
old learning? These are curator comparison axes, not a claim that every paper
was experimentally tested on all four (C036).

### Timeline

- **1980–1989:** cognitive-code framing, streaming PCA, BP and
  catastrophic interference establish the problem vocabulary (P031, P032, P033, P034, P035).
- **1998–2018:** measured spike timing, BTSP, and circuit and energy
  alternatives motivate candidate mechanisms framed in terms of biological
  plausibility (P036, P024, P037, P025, P039, P028, P011, P040, P026).
- **2019–2023:** reviews clarify the objections; causal SNN and predictive-
  coding systems begin to address online streams directly (P041, P002, P013, P018, P019, P042, P022,
  P044, P001, P016, P045, P046).
- **2024–2026:** BTSP algorithms, loss of plasticity, prospective
  configuration, phaseless alignment, neural-similarity controls and
  multi-timescale rules form the target-aligned frontier (P038, P020, P043,
  P007, P014, P047, P048, P049, P050).

## §4 Datasets and benchmarks for evaluation

Before comparing algorithms, fix whether the stream is task-, domain- or
class-incremental and whether task identity is available at test time (P044).
For the project, the minimum protocol should report current-task learning,
retention/forgetting, learning speed and compute or memory under matched online
budgets (P046, P007).

Event datasets used by SNN papers test temporal processing and hardware-facing
constraints, but they do not automatically test long-horizon continual
retention (P019, P022, P020). Conversely, Split-MNIST/FashionMNIST-style
continual protocols are convenient but can conceal temporal-credit difficulty
(P016, P046). The two-week project should therefore pair one simple stream
benchmark with one non-stationary or sequential condition.

## §5 Comparison of method and architecture routes

| Route | What the representative source establishes | What is not inferred here |
|---|---|---|
| BP baseline (P031) | introduces backpropagation for multilayer neuron-like units | a biological implementation or continual-learning property |
| Plasticity anchors (P032, P036, P024, P037, P038) | streaming PCA, spike-timing dependence, eligibility traces and BTSP/one-shot memory in their reported settings | deep spatial credit or system-level retention from those anchors alone |
| Spatial-credit routes (P025, P039, P028, P011, P040, P026, P041, P002, P013, P047, P048) | random feedback, segregated dendrites, two-phase energy dynamics, predictive-coding approximations and explicit scaling/equivalence critiques | online or continual behavior unless the individual paper tests it |
| Temporal/SNN routes (P018, P019, P042, P022, P020, P043) | online e-prop, DECOLLE without stored gradient history, separation of spatial and temporal gradient components, constant-memory OTTT and sequence-length-independent S-TLLR complexity | long-horizon retention from temporal locality alone |
| Online/continual systems (P044, P001, P016, P045, P046, P007, P014) | protocol taxonomy, task-free/streaming systems, multi-metric evaluation and loss of plasticity | a shared biological mechanism across all systems |
| Integration frontier (P049, P050) | eligibility-plus-gating with a broadcast signal, and local present-time approximations to BPTT | independent replication, consensus or complete biological sufficiency |

Unknown implementation details must be treated as unknown rather than inferred
from a “biologically plausible” label. P026 and P013 are the required checks
against over-generalizing from elegant mechanisms.

## §6 Paper capsules

### Stage A — concepts and failure modes

- **P031** — Defines BP, the reference algorithm whose spatial and
  temporal dependencies the rest of the roadmap modifies. Why it matters: no
  alternative can be evaluated without stating what it keeps or discards.
- **P032** — Shows a simplified neuron tending toward principal-component
  analysis for a stochastic input stream. Why it matters: a small beginner
  example of streaming self-organization.
- **P033** — Historical paper on how a brain might build a cognitive code.
  Why it matters: period context and optional concept reading only.
- **P034** — Establishes catastrophic interference under sequential
  learning. Why it matters: motivates retention metrics.
- **P035** — Historical Nature commentary on the neural-network resurgence.
  Why it matters: period context; it does not replace P041/P002 as the modern
  objection map.

### Stage B — biological plasticity

- **P036** — Experimental timing dependence of synaptic strengthening
  and weakening. Why it matters: turns Hebbian coincidence into a measurable
  temporal rule.
- **P024** — Reviews eligibility traces and third-factor modulation.
  Why it matters: conceptual bridge from local events to delayed credit.
- **P037** — Establishes BTSP as a seconds-scale, non-classical
  plasticity rule. Why it matters: optional biological branch toward one-shot
  adaptation.
- **P038** — Models BTSP as binary-synapse content-addressable memory.
  Why it matters: makes the BTSP branch algorithmically testable.

### Stage C — spatial credit assignment

- **P025** — Shows fixed random feedback can support learning. Why it
  matters: removes exact symmetric feedback, not every global dependency.
- **P039** — Uses segregated dendritic compartments for sensory and
  feedback signals. Why it matters: a concrete neuron/circuit interpretation.
- **P028** — Uses differences between free and nudged equilibria. Why it
  matters: canonical energy-based alternative; phase cost remains visible.
- **P011** — Constructs a predictive-coding network approximating BP.
  Why it matters: essential precursor for the project's PC branch.
- **P040** — Combines dendrites and cortical microcircuits. Why it matters:
  demonstrates how hidden-layer errors may become locally usable.
- **P026** — Tests biologically motivated routes on harder tasks. Why it
  matters: prevents “plausible on MNIST” from becoming a scalability claim.
- **P041** — Compares error-backpropagation theories in the brain. Why
  it matters: a compact taxonomy before choosing a rule family.
- **P002** — Reassesses BP's relationship to brain mechanisms. Why it
  matters: replaces the binary plausible/implausible framing with explicit
  assumptions.
- **P013** — Identifies conditions and differences in PC/BP equivalence.
  Why it matters: required critical reading for the PC branch.

### Stage D — temporal and spiking credit

- **P018** — E-prop factors learning into eligibility traces and learning
  signals. Why it matters: a direct project candidate for causal online SNN
  learning without BPTT history.
- **P019** — DECOLLE uses fixed local readouts and instantaneous local
  losses. Why it matters: scalable online learning with a clear locality trade.
- **P042** — OSTL separates spatial and temporal gradient components. Why
  it matters: makes the two credit problems explicit.
- **P022** — OTTT derives forward-in-time SNN updates with constant
  temporal memory. Why it matters: useful performance-oriented comparator.
- **P020** — S-TLLR, by Marco P. E. Apolinario (TMLR, 2025), uses an
  STDP-inspired temporal local rule. Why it matters: recent option for
  low-memory online experiments.
- **P043** — Compares e-prop and BPTT by neural similarity at matched task
  accuracy. Why it matters: warns that architecture and initialization may
  dominate the apparent brain-model difference.

### Stage E — online and continual systems

- **P044** — Defines incremental-learning scenarios. Why it matters:
  protocol choices can reverse apparent method rankings.
- **P001** — Reviews neuronal and non-neuronal mechanisms relevant to lifelong
  learning. Why it matters: broadens the mechanism search beyond a single
  synaptic update rule.
- **P016** — Applies predictive coding to task-free stream learning. Why it
  matters: direct PC-to-continual bridge.
- **P045** — Combines predictive coding, spiking neurons and data streams.
  Why it matters: a pre-2024 bridge across the project's candidate
  families.
- **P046** — Evaluates online continual-learning methods using accuracy,
  forgetting, stability and representation quality. Why it matters: source for
  project metrics and comparator design.
- **P007** — Demonstrates loss of plasticity over continuing training. Why
  it matters: adds new-task learnability to the evaluation, not only forgetting.

### Stage F — 2024–2026 convergence frontier

- **P014** — Infers neural activity before weight updates. Why it matters:
  a direct recent PC-style route for batch-one/changing environments.
- **P047** — Learns feedback projections continuously without separate
  phases. Why it matters: jointly attacks symmetry and phase locking.
- **P048** — Proposes dendritic localized learning across multiple network
  types. Why it matters: recent scalable spatial-credit candidate; direct
  streaming/continual evidence remains limited.
- **P049** — Adds an astrocyte-mediated gate to eligibility traces and a
  broadcast teaching signal. Why it matters: the roadmap's target-aligned
  endpoint, but still one recent study with a global signal.
- **P050** — Uses generalized latent equilibrium for local present-time
  spatial and temporal credit. Why it matters: conceptual endpoint and future
  work, not the recommended two-week implementation.

## §7 Downstream application relevance

Potential application settings include adaptive neural interfaces,
neuromorphic event streams, and agents exposed to distribution shift. For the
mechanisms represented by P020, P045, and P049, an important project constraint
is not merely low-power inference but bounded-state learning that continues
after deployment. The associated project should therefore evaluate learning
during the stream rather than train only offline and deploy a frozen spiking
model.

## §8 Cross-cutting analysis

Three distinctions prevent category errors. First, `online` may mean one
forward-time update rather than long-term continual retention (P018, P019, P042, P022, P020).
Second, `local` may still include a globally broadcast error or reward signal
(P018, P049). Third, matching task accuracy does not by itself establish neural
similarity or brain mechanism identity (P043).

## §9 Research-team clusters

The map intentionally crosses several partially separated communities:
predictive coding and theoretical neuroscience (P011, P041, P013, P014),
spiking/neuromorphic learning (P018, P019, P042, P022, P020, P043), biological synaptic plasticity
(P036, P024, P037, P038), continual learning (P044, P046, P007), and cortical/dendritic
credit assignment (P039, P040, P047, P048, P049, P050). Route and publication-year counts
were checked; institution and country distributions were not quantified in
this version, so no geographic-balance claim is made.

## §10 Open challenges

1. Fixing feedback symmetry does not guarantee useful deep performance
   (P026).
2. Predictive-coding equivalence to BP is conditional rather than universal
   (P013).
3. A causal eligibility trace can still rely on a global learning signal and
   can still forget across tasks (P018, P049).
4. Biological resemblance should include neural-data fit, not only task
   accuracy or a local-looking equation (P043).
5. Loss of plasticity and catastrophic forgetting need separate metrics
   (P007).

## §11 Frontiers and future directions

P047 explores continuously plastic feedback projections without separated
phases; P048 is a dendritic spatial-credit route; P049 adds eligibility traces,
a broadcast signal and an astrocyte-mediated gate; P050 proposes local
present-time approximations to temporal credit (P047, P048, P049, P050).
Recency is not consensus: these nodes should be tested with matched evaluations
against simple tuned baselines (P026, P046). A near-term project question is whether slow plasticity
gates add retention beyond what is obtained by eligibility traces plus ordinary
regularization under the same memory and compute budget (P018, P046, P049).

## §12 Direct answer and project shortlist

For a reader with a general scientific background, the main spine is:

**P031 → P032 → P034 → P036 → P024 → P026 → P041 → P002 → P018 → P044 →
P016 → P045 → P046 → P007 → P049.**

For the two-week project, this roadmap selects two implementation branches:

- **Eligibility/SNN:** P024 → P018 → P020 → P043 → P049. Start from e-prop or
  a small S-TLLR example; use P049 as the continual-learning extension and
  critique target.
- **Predictive coding:** P011 → P013 → P016 → P045 → P014. Start from a small
  predictive-coding network; use P016/P014 to define the online and changing-
  environment hypothesis.

BTSP (P037 → P038) is a high-value third branch if the project prioritizes
one-shot associative memory rather than deep supervised classification. This is
a target-directed recommendation, not a field-wide ranking (C037).

## §13 Recommended reading list

### Entry tier

P031, P032, P034, P041 and P044 establish the vocabulary in roughly one day of
selective reading.

### Deep tier

Choose one mechanism branch: PC (P011, P013, P016, P045, P014), SNN temporal
credit (P024, P018, P019, P042, P022, P020, P043), or BTSP (P037, P038).

### Critical tier

P026, P013, P043, P046 and P007 constrain scalability, equivalence, neural
interpretation, protocol fairness and plasticity claims.

### Overview tier

P041, P002 and P001 provide complementary review-level maps.

## §14 Reproducibility status

Not audited in this version. Some publisher pages point to public code, but
repository health, dependency resolution, benchmark parity and independent
reproduction were not checked systematically. No R1–R4 tier is assigned.
Before choosing the two-week implementation, perform a separate repository and
minimal-run audit on the final two candidates.

## §15 Status

Synthesis complete. The independent claims-adversary gate passed after three
correction rounds. The interactive artifact was verified in a real browser at
736px and 320px widths; node selection, prerequisite display and the P049
archive link worked with zero console errors. Institution/geography balance and
repository-level reproducibility remain explicitly unassessed.

**Phase**: synthesize  **Survey**: biologically-plausible-online-learning-roadmap  **Status**: synthesized; v1.1-target-adapted  **Next**: Stage 2 at [`index.md`](index.md)
