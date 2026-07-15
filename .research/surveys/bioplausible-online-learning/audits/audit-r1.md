# Formal Coverage and Bias Audit — R1

- Survey: `bioplausible-online-learning`
- Audited UTC: `2026-07-15T12:31Z`
- Evidence input: `P001`–`P030`, with 28 confirmed ★★★ papers
- Decision: `audit-needs-roundN`
- Next phase: targeted `roundn`

## Gate Summary

| Measure | Result |
|---|---:|
| Sub-questions | 10 |
| Active cells | 35 |
| Closed cells | 30 |
| Weak cells | 5 |
| Gap cells | 0 |
| Global 60% bias triggers | 0 |

All active cells contain candidate evidence, but five do not meet the
distinct-source closure rule. Because the user has not explicitly accepted
those limitations, zero empty cells is insufficient to pass the audit.

## Weak-Cell Findings

| Cell | Current evidence | Why weak | Closure evidence needed |
|---|---|---|---|
| SQ2 / critical-review | P013 | single source | Independent peer-reviewed critique of PC exactness, settling cost, compute or biological assumptions |
| SQ4 / critical-review | P002,P026 | shared DeepMind lineage | Independent critical or negative comparison of FA, EP, DTP, dendritic or local-error routes |
| SQ7 / dataset | P010 | single benchmark | Later distinct-lab benchmark with explicit stream, boundaries, task-ID and replay assumptions |
| SQ8 / survey | P009 | single survey | Independent survey comparing benchmarks, metrics and protocol assumptions |
| SQ8 / dataset | P010 | single benchmark | Second benchmark with stream construction, metrics, baselines and leakage controls |

No active cell is empty. `coverage_check.py` reports 35 cells with at least one
★★★ paper; the weak labels above are the concentration overlay required by the
coverage protocol.

## Automated Bias Audit

The audit used confirmed ★★★ rows only and counted each paper once even when
it supports multiple SQs.

| Bucket | Dominant value | Count | Share | Threshold | Result |
|---|---|---:|---:|---:|---|
| Institution | University of Oxford | 3/28 | 10.7% | 60% | clean |
| Country | USA | 8/28 | 28.6% | 60% | clean |
| Year | 2022 | 7/28 | 25.0% | 60% | clean |
| Venue | Nature Communications | 6/28 | 21.4% | 60% | clean |
| Method route | spike-local | 7/28 | 25.0% | 60% | clean |
| Venue type | preprint | 0/28 | 0.0% | 60% | clean |

Method-route distribution: spike-local 7; other-local-credit 6;
predictive-coding 6; stability-memory 5; protocol-critique 4.

`P026` has a confirmed institution but `Country=unknown`; it remains explicit
and is not guessed into a country bucket.

## Deployment-Regime Audit

This manual audit classifies the primary reported evaluation, not the method's
name or potential execution mode. In particular, a temporally causal gradient
computed inside multi-epoch stationary training is `offline-only`, not
`single-pass`.

| Primary regime | Paper IDs | Count / 24 | Share |
|---|---|---:|---:|
| offline-only | P011,P012,P013,P015,P017,P020,P022,P025,P026,P029,P030 | 11 | 45.8% |
| continual | P003,P004,P005,P006,P007,P008,P009,P010,P014,P016 | 10 | 41.7% |
| single-pass | P018,P027 | 2 | 8.3% |
| neuromorphic | P021 | 1 | 4.2% |
| bounded-replay | none | 0 | 0.0% |

Four review-only papers (`P001`, `P002`, `P023`, `P024`) are excluded from the
deployment denominator because they synthesize multiple regimes. No regime
exceeds 60%. The absence of a bounded-replay primary paper is a limitation to
carry into synthesis, but it does not by itself trigger the configured
single-bucket dominance rule.

## Venue-Type Audit

Among the 28 ★★★ rows, 17 are journal papers/reviews, 10 are main-conference
papers and one is a workshop paper. No unresolved preprint-only row is present,
so the preprint concentration is 0%.

## Round N Task List

1. `RN1-PC-CRIT` — harden SQ2/critical-review with an independent PC critique;
   begin with the already identified Zahid et al. 2023 reserve candidate.
2. `RN1-LOCAL-CRIT` — harden SQ4/critical-review outside the DeepMind lineage.
3. `RN1-PROTOCOL-DATA` — add an independent chronological or task-free
   benchmark for SQ7/dataset; inspect CLEAR and OpenLORIS first.
4. `RN1-BENCH-SURVEY` — add an independent recent benchmark/metric survey for
   SQ8/survey.
5. `RN1-BENCH-DATA` — use the same benchmark search batch to close SQ8/dataset,
   requiring explicit stream construction, metrics, baselines and leakage
   controls.

Tasks 3 and 5 may be closed by the same high-quality benchmark paper, but each
cell retains its own acceptance test. Round N must score and append any new
paper centrally without renumbering `P001`–`P030`.

## Decision

- Coverage gate: **not passed** — five weak cells remain.
- Global bias gate: **passed** — no configured dominance trigger.
- Synthesis authorization: **withheld** until targeted Round N closes the weak
  cells or the user explicitly accepts each recorded limitation.

**Phase**: audit  **Survey**: bioplausible-online-learning  **Status**: audit-needs-roundN  **Next**: roundn
