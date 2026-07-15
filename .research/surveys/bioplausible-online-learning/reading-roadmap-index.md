# Biologically Plausible Online Learning Roadmap

> **Stage 1 — Domain Introduction.** This target-directed reading roadmap is for
> a reader with a general scientific background. It establishes the problem
> vocabulary, mechanisms
> and evaluation cautions needed before
> [Stage 2 — SOTA Results Explorer](index.md).

## Identity

- Survey ID: `biologically-plausible-online-learning-roadmap`
- Version: `v1.1`
- Created UTC: `2026-07-15T12:24Z`
- Last updated UTC: `2026-07-15T13:46Z`
- Owner: repository owner + Codex research assistant
- Status: `synthesized; v1.1-target-adapted` (Stage 1 only; the Stage 2 SOTA
  audit status is unchanged)
- Evidence cutoff: `2026-07-15`
- Claims-adversary gate: `pass`
- Survey root: `.research/surveys/bioplausible-online-learning/`
- Parent: `neuroai-field-map`
- Source provenance: adapted from
  `papers/literature-maps/biologically-plausible-online-learning-roadmap/` in
  `cyber-neuroai-lab`; there is no content or runtime dependency on that source
  path

## Research Question

Which papers form the shortest defensible reading path for a reader with a
general scientific background to understand biologically plausible online
learning, including why backpropagation is disputed as a biological learning mechanism,
how local plasticity and temporal credit work, how stability is maintained in
non-stationary streams, and which Predictive Coding, spiking, three-factor,
BTSP, and related learning rules are most relevant to the associated two-week
Neural AI Online Learning Project?

## Scope

| Dimension | In scope | Out of scope |
|---|---|---|
| Reader | General scientific literacy; calculus/probability helpful but not assumed | A route that begins with specialist computational-neuroscience knowledge |
| Learning setting | Causal, streaming, single-pass or bounded-memory online learning; continual/non-stationary settings | Offline i.i.d. training presented as evidence of online performance |
| Biological claim | Locality, feedback/weight transport, global signals, timing, temporal credit, neuron model, persistent state and replay | A single unsupported “plausibility score” |
| Method families | Backprop baseline; Hebbian/STDP; three-factor/eligibility/BTSP; Predictive Coding and energy-based relaxation; dendritic/feedback routes; SNN online learning; continual-learning mechanisms | Unrelated NeuroAI branches, pure hardware surveys, delayed-reward RL except where it clarifies temporal credit |
| Evidence | Primary papers plus a small number of field-defining reviews/critical papers; authoritative landing/archive links | Citation-count ranking or claims inferred only from blogs |
| Time | Foundational antecedents through 2026-07-15 | Work first released after the cutoff |
| Delivery | A compact dependency graph with required/optional papers and project relevance | Exhaustive bibliography or full implementation tutorial |

## Synthesis Snapshot

- **Selected corpus:** 35 papers across six dependency stages; estimated-size
  Top-K allocations are `5 / 4 / 9 / 6 / 6 / 5`.
- **Default newcomer route:** 15 papers from P031 (backpropagation) through
  P049 (the target-aligned frontier node for online and continual learning).
- **Project branches:** Predictive Coding/prospective configuration and
  eligibility/SNN are the two implementation branches selected for the
  associated two-week project; BTSP is the optional one-shot-memory branch
  (C037, curator recommendation rather than a field-wide ranking).
- **Interactive delivery:** the standalone HTML shows each node's domain,
  reading priority, prerequisite chain, short review, target relevance, caveat,
  and archive link.

## Sub-Questions

1. **SQ1 — Problem foundations:** What minimum sequence explains neurons as adaptive units, supervised credit assignment, and the online-learning setting?
2. **SQ2 — Biological objections:** Which specific assumptions of backpropagation are biologically disputed, and which objections have been weakened or refined?
3. **SQ3 — Local plasticity:** How do Hebbian learning, STDP, three-factor rules, eligibility traces, and BTSP turn local events into lasting updates?
4. **SQ4 — Online stability:** Which mechanisms and evaluation concepts separate fast plasticity from catastrophic interference under streaming or non-stationary data?
5. **SQ5 — Spatial credit alternatives:** How do feedback alignment, dendritic segregation, equilibrium/contrastive methods, target-based methods, and Predictive Coding approximate or replace backpropagation?
6. **SQ6 — Temporal and spiking routes:** Which SNN learning rules avoid or approximate BPTT while operating causally online, and what biological assumptions remain?
7. **SQ7 — Project decision (v1.1 clarified):** Which one or two rule families best fit the associated two-week project, and which implementation, code-health, scale, and matched-protocol checks remain before final selection?
8. **SQ8 — Limits and frontier:** What do current approaches still fail to establish, and what 2024–2026 work materially changes the roadmap?

## Active Evidence Dimensions

| Sub-question | theory | experiment | survey | critical-review | dataset |
|---|---:|---:|---:|---:|---:|
| SQ1 | ✓ | ✓ | ✓ |  |  |
| SQ2 | ✓ | ✓ | ✓ | ✓ |  |
| SQ3 | ✓ | ✓ | ✓ | ✓ |  |
| SQ4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| SQ5 | ✓ | ✓ | ✓ | ✓ |  |
| SQ6 | ✓ | ✓ | ✓ | ✓ | ✓ |
| SQ7 | ✓ | ✓ | ✓ | ✓ | ✓ |
| SQ8 | ✓ | ✓ | ✓ | ✓ |  |

## Star Rating Rubric

Use the Deep Survey BFS four-dimensional rubric: exact roadmap relevance,
evidence quality, lineage/project decision value, and source/metadata
confidence. A paper is required reading only when it is both structurally
necessary for the dependency path and at least ★★★; optional papers may be ★★
when they expose a distinct branch or limitation.

## Bias Audit Thresholds

| Bucket | Threshold | Notes |
|---|---:|---|
| Institution | 60% | Trigger a search outside the dominant lab |
| Country | 60% | Report historical concentration; do not silently rebalance roots |
| Year | 60% | Apply to the post-2015 frontier subset separately |
| Method route | 60% | No single PC/SNN/local-credit route may stand in for the field |
| Deployment regime | 60% | Distinguish offline approximation from causal online evidence |
| Venue type (preprint) | 60% | Frontier preprints require an explicit status warning |

## Audit Outcome and Known Limits

- All 34 active mechanism-coverage cells have representative sources from at
  least two author teams; the evidence ledger remains predominantly
  abstract/metadata-level and is therefore recorded at medium confidence.
- The final claims-adversary gate passed after overclaims about rankings,
  reproducibility and historical mechanism details were removed.
- Method-route and publication-year distributions were quantified. Institution,
  country, venue-history and deployment-regime distributions were not; no
  balance conclusion is claimed for those dimensions.
- Repository health, runnable environments, benchmark parity and independent
  reproduction were not audited. Candidate implementation requires a separate
  code/reproduction pass.
- The search is target-directed through the evidence cutoff, not an exhaustive
  systematic review of every 2026 paper.

## Round 1 Source Plan

| Source | Keywords / queries | Cap |
|---|---|---:|
| Existing map | Hebbian, catastrophic forgetting, predictive coding, plausible backprop, SNN, neurogenesis, dynamics | all relevant seeds |
| Associated project | Predictive Coding, SNN, BTSP, Hebbian, online/continual constraints | all project-owned sources |
| arXiv | title/abstract searches for biologically plausible + online/continual/local/streaming and 2024–2026 successors | 45 candidates |
| OpenReview | ICLR/NeurIPS/TMLR papers on local credit, SNN, predictive coding and continual learning | 20 candidates |
| Publisher/proceedings | DOI, PMLR, NeurIPS, Nature, Cell, eLife, Frontiers and journal pages | all selected papers |
| Citation BFS | backward/forward search from Whittington–Bogacz, Lillicrap, Bellec/e-prop, Bittner/BTSP and continual-learning seeds | as needed |

## Shared evidence namespace

- [`paper_index.md`](paper_index.md) is the canonical shared 50-paper registry
  used by both stages; a bare `P###` in these adapted artifacts always means
  that canonical ID.
- [`claims.jsonl`](claims.jsonl) is the shared remapped claims ledger. It is not
  a Domain Introduction-only copy.
- [`paper-id-crosswalk.md`](paper-id-crosswalk.md) records roadmap v1.1 source
  IDs and their canonical equivalents for provenance. Legacy roadmap IDs are
  not a second active citation namespace.
- This adaptation changes only Stage 1 navigation and identifiers. It does not
  change Stage 2 coverage, audit status or interpretations in
  [`index.md`](index.md).

## Pointers

- [`index.md`](index.md) — continue to Stage 2 · SOTA Results Explorer
- [`paper_index.md`](paper_index.md) — shared canonical 50-paper registry
- [`claims.jsonl`](claims.jsonl) — shared claim provenance ledger
- [`paper-id-crosswalk.md`](paper-id-crosswalk.md) — roadmap v1.1 source-ID provenance
- [`reading-roadmap-coverage-matrix.md`](reading-roadmap-coverage-matrix.md) — Stage 1 sub-question × evidence coverage
- [`reading-roadmap.md`](reading-roadmap.md) — human-readable staged route
- [`reading-roadmap-survey.md`](reading-roadmap-survey.md) — full Domain Introduction synthesis
- [`biologically-plausible-online-learning-roadmap.html`](biologically-plausible-online-learning-roadmap.html) — standalone interactive roadmap
- [`reading-roadmap-search-log.md`](reading-roadmap-search-log.md) — evidence cutoff, source strategy and known search limits
- [`audits/reading-roadmap-bias-audit.md`](audits/reading-roadmap-bias-audit.md) — bias-threshold audit
- [`audits/reading-roadmap-claims-adversary.md`](audits/reading-roadmap-claims-adversary.md) — final claims-adversary gate
- [`audits/reading-roadmap-coverage-audit.md`](audits/reading-roadmap-coverage-audit.md) — mechanism-coverage audit

## Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-07-15 | Framed eight-sub-question target-directed survey | User requested a beginner-to-frontier interactive reading roadmap tied to the online-learning project |
| 2026-07-15 | Merged and centrally deduplicated 35 Round-1 papers | Reached all mechanism branches while keeping a 15-paper executable core |
| 2026-07-15 | Closed all 34 active mechanism-coverage cells | No gap-driven Round N required before synthesis |
| 2026-07-15 | Narrowed evidence claims and documented unmeasured bias dimensions | Claims-adversary review found several metadata-only and comparative overclaims |
| 2026-07-15 | Final claims-adversary gate passed; interactive HTML verified at desktop and 320px widths | Synthesis and visualization ready for use |
| 2026-07-15 | Published v1.1 index supplement: synthesis snapshot, explicit audit limits, and clarified SQ7 | Aligned the master record with the final survey and reproducibility audit |
| 2026-07-15 | Adapted the Domain Introduction to the shared 50-paper registry and local Stage 1/Stage 2 navigation | Preserved roadmap v1.1 evidence while making canonical IDs and stage ownership explicit |

*Survey index v1.1. Delta from v1: added the synthesis snapshot and audit
outcome, clarified SQ7 without changing its scope, and linked the parent field
map back to this child survey. The target adaptation remaps legacy source IDs to
the shared canonical registry without changing the 35-paper corpus, claims or
survey section anchors.*
