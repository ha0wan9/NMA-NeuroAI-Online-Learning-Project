# Biologically Plausible Learning Rules for Online and Continual Learning

> Master record for this BFS literature survey. Scope and sub-questions are
> fixed at frame time; later phases add evidence without silently redefining
> the question.

## Identity

- Survey ID: `bioplausible-online-learning`
- Created UTC: `2026-07-15T10:49Z`
- Owner: NMA NeuroAI Online/Continual Learning Group
- Status: `audit-needs-roundN`
- Survey root: `.research/surveys/bioplausible-online-learning/`
- Parent / superseded: none

## Research Question

Among learning rules claimed to be biologically plausible, which have the
strongest theoretical and empirical evidence for learning effectively under
causal online or continual-learning conditions, and under which matched
conditions do they approach or exceed Backpropagation in plasticity, stability,
accuracy, learning speed, robustness, or resource cost? The survey must cover
Predictive Coding and local learning rules used with spiking neural networks,
seek additional biologically plausible alternatives, expose hidden
implausibilities and unfair comparisons, and produce a defensible shortlist of
one or two rules that can be tested within the remaining two-week project.

The survey maps evidence; it does not assume that any candidate already beats
Backpropagation. A spiking architecture alone is not counted as a learning
rule, and a bio-inspired mechanism is not counted as biologically plausible
without an explicit information and update pathway.

## Two-stage research path

1. **Stage 1 — Domain Introduction:** Researchers with a general scientific
   background first use the
   [interactive reading roadmap](biologically-plausible-online-learning-roadmap.html)
   and [roadmap index](reading-roadmap-index.md) to build the problem vocabulary,
   mechanism taxonomy, and comparison cautions needed for this project.
2. **Stage 2 — SOTA Results Explorer:** This master survey then explores current
   results, methods, protocols, and implementation feasibility against its fixed
   research question and ten SOTA sub-questions. Begin with the
   [interactive evidence atlas](sota-results-explorer.html), which re-clusters
   the retained papers through six categorical research lenses without treating
   visual distance as an evidence claim.

Stage 1 supplies context and hands readers off to Stage 2; its separate audit
does not change the SOTA coverage gate.

### Shared evidence registry

The two stages use one canonical 50-paper registry,
[`P001`–`P050`](paper_index.md). The original SOTA IDs `P001`–`P030` remain
stable, and 20 roadmap-only papers were appended as `P031`–`P050`. The roadmap
proposal's `SQ1`–`SQ8` namespace and this SOTA proposal's `SQ1`–`SQ10` namespace
remain separate; matching numbers are not equivalent. Claim provenance is in
the shared [`claims.jsonl`](claims.jsonl), and source-ID reconciliation is in
[`paper-id-crosswalk.md`](paper-id-crosswalk.md).

Under the currently recorded evidence, the imported roadmap closes none of the
five weak SOTA cells and does not authorize synthesis. The survey therefore
retains status `audit-needs-roundN`.

## Scope

| Dimension | In scope | Out of scope |
|---|---|---|
| Learning signal / rule | Synapse- or neuron-level updates; local errors; predictive-coding updates; spike-local plasticity; three-factor or eligibility-based rules; non-spiking local credit assignment; locally implementable stability mechanisms | Architecture-only claims with no specified learning rule; ordinary Backpropagation variants with no biological-plausibility argument |
| Learning setting | Causal single-pass online learning; bounded-replay online learning; task-, domain-, and class-incremental continual learning; streaming noise or distribution shift | Pure batch/offline performance unless the paper establishes a foundational mechanism or a directly transferable learning-rule result |
| Biological plausibility | Locality, weight transport, feedback symmetry, global error requirements, update timing, differentiability, temporal credit, neuromodulation, neuron/communication model, persistent state and replay | Vague “brain-inspired” labels without an operational mechanism; biological analogy used as proof |
| Method routes | Predictive Coding; local spiking rules including STDP-like and three-factor routes; feedback alignment, equilibrium/target-propagation and related non-spiking local credit routes; metaplasticity, homeostasis, consolidation and complementary-memory mechanisms | Hardware-only neuromorphic work that does not evaluate a learning rule; evolutionary or population search unless tied directly to the framed online-learning question |
| Model / task scale | Small-to-medium simulations and benchmarks suitable for a two-week project; larger studies when they materially test scalability or comparison claims | Frontier-scale training that cannot inform a feasible project choice; biological wet-lab results with no identifiable learning-rule implication |
| Evidence type | Primary theory and experiment papers; authoritative surveys and critical reviews; benchmark/dataset papers; peer-reviewed work and clearly labelled preprints; code/repository evidence for reproducibility | Blogs, marketing pages, or uncited summaries as scientific evidence; unverifiable performance claims |
| Time range | Foundational work through 2026-07-15; older seminal papers retained when still structurally relevant | Superseded versions when a canonical peer-reviewed version exists, except when version history itself matters |
| Language | English full text or sufficiently detailed English metadata | Sources that cannot be assessed reliably from accessible language or metadata |
| Deployment target | Research simulation and feasible CPU/GPU experiments; neuromorphic relevance when it changes the rule, information budget, or energy/latency claim | Public production deployment; device benchmarking with no connection to learning-rule performance |
| Comparison claim | Matched or transparently near-matched comparison on stream, model capacity, information access, replay/memory, tuning, metric and uncertainty | Cross-paper leaderboard comparisons presented as head-to-head evidence |

## Sub-Questions

The first five questions partition mechanism families; SQ6-SQ10 are explicitly
cross-cutting comparison, protocol, feasibility, and criticism questions.

1. **SQ1 — Plausibility criteria:** Which operational criteria can distinguish a biologically plausible learning rule from a merely bio-inspired architecture or training metaphor, and how are locality, weight transport, temporal credit, differentiability and global coordination assessed?
2. **SQ2 — Predictive Coding:** Under online or continual conditions, how are Predictive Coding learning updates implemented, what information do they require, and what evidence supports or contradicts competitive performance relative to Backpropagation?
3. **SQ3 — Spike-local learning:** Which explicitly specified local, STDP-like, three-factor, eligibility-based or neuromodulated rules used in spiking neural networks learn effectively from streams, and how do they compare with surrogate-gradient or Backpropagation baselines?
4. **SQ4 — Other local credit assignment:** Which non-spiking alternatives—such as feedback-alignment, equilibrium-propagation, target-propagation, dendritic or related local-error routes—retain biological-plausibility claims while supporting online or continual updates?
5. **SQ5 — Plasticity–stability mechanisms:** Which biologically motivated metaplasticity, homeostatic, consolidation, structural-plasticity or complementary-memory mechanisms reduce catastrophic forgetting, and which are learning-rule alternatives versus add-ons that still depend on Backpropagation?
6. **SQ6 — Competitive performance:** In matched or near-matched evidence across method routes, which rules approach or exceed Backpropagation, on which axes (online accuracy, learning speed, forgetting, robustness, memory, compute or energy), and with what uncertainty?
7. **SQ7 — Protocol dependence:** How do single-pass versus replay, task/domain/class incrementality, task-ID access, stream order, evaluation cadence and tuning budget change conclusions about plasticity and superiority?
8. **SQ8 — Benchmarks and metrics:** Which datasets, stream constructions, metrics and baselines support a fair, leakage-resistant comparison that can be executed within the project timeline?
9. **SQ9 — Reproducibility and feasibility:** Which candidate rules have usable code, sufficiently specified update equations, manageable compute, and integration complexity compatible with implementing one or two methods after a verified baseline?
10. **SQ10 — Failure modes and negative evidence:** What empirical failures, scalability limits, hidden non-local operations, offline phases, unrealistic signals or biological critiques weaken the strongest claims in this literature?

## Active Evidence Dimensions

Only checked cells participate in the audit closure gate. Dataset evidence is
reserved for protocol/benchmark questions rather than required separately for
every method family.

| Sub-question | theory | experiment | survey | critical-review | dataset |
|---|---:|---:|---:|---:|---:|
| SQ1 | ✓ | — | ✓ | ✓ | — |
| SQ2 | ✓ | ✓ | — | ✓ | — |
| SQ3 | ✓ | ✓ | — | ✓ | — |
| SQ4 | ✓ | ✓ | — | ✓ | — |
| SQ5 | ✓ | ✓ | ✓ | ✓ | — |
| SQ6 | — | ✓ | ✓ | ✓ | — |
| SQ7 | ✓ | ✓ | ✓ | ✓ | ✓ |
| SQ8 | — | ✓ | ✓ | ✓ | ✓ |
| SQ9 | — | ✓ | ✓ | ✓ | — |
| SQ10 | ✓ | ✓ | ✓ | ✓ | — |

## Star Rating Rubric

Canonical rubric: `deep-survey-bfs/references/paper-rating-rubric.md`. Round 1
must score relevance, authority, recency and evidence strength independently:

```text
weighted_score = relevance * 1.5 + authority + recency + evidence_strength
★★★: weighted_score >= 10 and relevance >= 2
★★:  weighted_score >= 6
★:   weighted_score >= 3
drop: weighted_score < 3
```

Project-specific relevance calibration:

- `3`: directly answers an SQ with an explicit learning rule and a relevant
  online/continual or biological-plausibility analysis.
- `2`: answers an SQ precisely but uses an adjacent setting, such as an offline
  mechanism paper with a clear online-learning implication.
- `1`: background only; useful vocabulary or context without answering an SQ.
- `0`: wrong topic or architecture-only evidence with no learning-rule claim.

Evidence strength is evaluated relative to paper type: experiments require
data, controls and reproducibility; surveys require synthesis breadth; theory
papers require a complete and influential framework. Star rating measures
evidence value for this survey, not whether a method beats Backpropagation.

## Bias Audit Thresholds

Any single bucket exceeding its threshold among ★★★ papers triggers targeted
re-search before synthesis. Counts use each paper once even when it supports
multiple SQs.

| Bucket | Threshold | Notes |
|---|---:|---|
| Institution | 60% | Trigger searches outside the dominant lab or collaboration |
| Country | 60% | Record explicit country metadata; do not infer when unknown |
| Year | 60% | Preserve foundational work while adding independent recent evidence |
| Method route | 60% | Routes: predictive-coding, spike-local, other-local-credit, stability-memory, protocol/critique |
| Deployment regime | 60% | Offline-only, single-pass, bounded-replay, continual, neuromorphic |
| Venue type (preprint) | 60% | Confirm canonical venue through OpenReview, DBLP or journal metadata |

## Round 1 Source Plan

Round 1 will retain approximately `max(30, 3 × 10) = 30` unique papers after
central deduplication. Candidate retrieval may be larger.

| Source | Keywords / queries | Candidate cap |
|---|---|---:|
| arXiv | `(ti:"predictive coding" OR abs:"predictive coding") AND (abs:"online learning" OR abs:"continual learning") AND (cat:cs.LG OR cat:cs.NE)` | 10 |
| arXiv | `(ti:"spiking neural network" OR ti:"spiking neural networks") AND (abs:"online learning" OR abs:"continual learning") AND (cat:cs.NE OR cat:cs.LG)` | 10 |
| arXiv | `(abs:"biologically plausible" OR abs:"local learning rule") AND (abs:"online learning" OR abs:"continual learning") AND (cat:cs.LG OR cat:cs.NE)` | 10 |
| arXiv | `(ti:"feedback alignment" OR ti:"equilibrium propagation" OR ti:"target propagation") AND (cat:cs.LG OR cat:cs.NE)` | 8 |
| arXiv | `(abs:"three-factor learning rule" OR abs:"eligibility trace" OR abs:metaplasticity) AND (cat:cs.NE OR cat:q-bio.NC)` | 8 |
| OpenReview | ICLR, NeurIPS, ICML and TMLR records; exact phrases above plus `continual learning`, `local learning`, `biologically plausible`, `spiking`, `predictive coding` | 20 |
| DBLP | Exact-title and author queries for canonical venue/year confirmation; forward searches for accepted successors | 15 confirmations |
| Semantic Scholar | Topic search per SQ, then backward/forward citation BFS from one canonical seed per method route; maximum two hops before gap audit | As needed |
| PubMed / journal sites | Primary neuroscience mechanisms and authoritative critical reviews for STDP, three-factor rules, eligibility traces, metaplasticity and predictive coding | 10 |

Search ordering is breadth-first: first obtain at least one plausible candidate
for each SQ and active evidence route, then deepen only the gaps identified by
the audit. `all:` arXiv searches and citation-count-only ranking are prohibited.

## Round 1 Inclusion and Exclusion Decisions

Include a paper when it supports at least one SQ and identifies a learning
rule, an evaluation protocol, an operational biological claim, or a critical
failure mode. Foundational offline work may enter for theory, but it cannot by
itself establish online/continual performance.

Exclude or demote:

- SNN papers that use ordinary Backpropagation or surrogate gradients without
  evaluating a distinct local rule relevant to an SQ;
- continual-learning papers that use a biological metaphor but retain an
  otherwise standard global Backpropagation update, unless they inform SQ5 or
  provide a necessary baseline;
- hardware papers without learning-rule evidence;
- duplicate preprint/venue versions, keeping the canonical accepted version;
- performance claims lacking enough protocol detail to establish data access,
  replay, comparator or metric.

## Pointers

### Stage 1 — Domain Introduction

- Entry points: [roadmap index](reading-roadmap-index.md) and
  [interactive HTML](biologically-plausible-online-learning-roadmap.html)
- Reading artifacts: [beginner-to-frontier route](reading-roadmap.md) and
  [roadmap survey](reading-roadmap-survey.md)
- Coverage and search: [coverage matrix](reading-roadmap-coverage-matrix.md) and
  [search log](reading-roadmap-search-log.md)
- Audits: [coverage](audits/reading-roadmap-coverage-audit.md),
  [bias](audits/reading-roadmap-bias-audit.md), and
  [claims-adversary](audits/reading-roadmap-claims-adversary.md)

### Shared registry artifacts

- [Paper index](paper_index.md) — stable canonical paper rows (`P001`–`P050`)
- [Claims ledger](claims.jsonl) — claim-level provenance across both stages
- [Paper-ID crosswalk](paper-id-crosswalk.md) — roadmap source IDs reconciled
  to the canonical registry

### Stage 2 — SOTA Results Explorer

- [Interactive evidence atlas](sota-results-explorer.html) — one-card-per-paper
  clustering by method, primary question, learning regime, research function,
  limiting caveat, and evidence form
- [Round 1 search log](round1-search-log.md) — queries, central score ledger,
  exclusions, reserve candidates, and predicted gaps
- [Coverage matrix](coverage_matrix.md) — active SOTA sub-question ×
  evidence-dimension coverage
- [Round 1 audit](audits/audit-r1.md) — formal coverage, concentration, and bias
  audit recording the five weak cells
- `survey.md` — synthesized prose, created only after the SOTA audit passes

## Changelog

| UTC | Change | Reason |
|---|---|---|
| 2026-07-15T10:49Z | Created fixed frame with 10 SQs and active evidence cells | Day 3 demand survey and literature-review gate |
| 2026-07-15T12:23Z | Added 30 centrally deduplicated and scored Round 1 papers; advanced to `round1-done` | Broad BFS retrieval completed across all method routes; formal gap audit is next |
| 2026-07-15T12:31Z | Audited 35 active evidence cells and 28 ★★★ papers; set `audit-needs-roundN` | Five cells are covered but source-concentrated; no global 60% bias trigger fired |
| 2026-07-15T17:30Z | Integrated the target-adapted Stage 1 roadmap and two-stage navigation | Share the `P001`–`P050` evidence registry while preserving the five weak SOTA cells and `audit-needs-roundN` gate |
