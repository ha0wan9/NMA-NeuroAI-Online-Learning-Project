# Beginner-to-Frontier Reading Route

This is the executable companion to
[`reading-roadmap-survey.md`](reading-roadmap-survey.md). Follow only the
**core spine** first; take one branch after the mechanism taxonomy is clear.

## Core spine — 15 stops

1. P031 — what backpropagation computes.
2. P032 — how a simple adaptive rule extracts a principal component from a stream.
3. P034 — why sequential learning destroys prior solutions.
4. P036 — how spike timing changes a biological synapse.
5. P024 — how eligibility traces bridge local events and delayed teaching.
6. P026 — scalability reality check.
7. P041 — taxonomy of brain-compatible credit-assignment proposals.
8. P002 — modern correction to the binary “BP is impossible” story.
9. P018 — e-prop as causal online temporal credit.
10. P044 — choose Task-IL, Domain-IL or Class-IL before comparing methods.
11. P016 — predictive coding under task-free continual streams.
12. P045 — predictive coding + spikes + continual streams.
13. P046 — fair online-continual evaluation.
14. P007 — measure loss of plasticity as well as forgetting.
15. P049 — target-aligned 2026 endpoint, read with the broadcast-signal and
    replication caveats visible.

## Branch A — predictive coding and prospective configuration

P011 → P013 → P016 → P045 → P014. This is the selected branch if the
project compares a small predictive-coding network against BP under batch-one,
noise or distribution shift.

## Branch B — spiking temporal credit

P024 → P018 → P019/P042/P022 → P020 → P043 → P049. This is the selected
branch if the project prioritizes causal temporal updates, bounded temporal
memory or event data.

## Branch C — BTSP and one-shot memory

P037 → P038. This is a narrower, comparatively self-contained branch if the
project focuses on fast one-shot storage rather than general deep
classification.

## Top-K allocation by estimated route size

| Stage | Estimated literature size | Included Top-K | Selection logic |
|---|---|---:|---|
| Foundations | very large | 5 | only prerequisites that change the endpoint |
| Synaptic plasticity | large | 4 | STDP → eligibility → BTSP bridge |
| Spatial credit assignment | very large | 9 | largest prerequisite branch and main source of plausibility objections |
| Temporal / spiking credit | large | 6 | one canonical rule plus representative scalable alternatives and a critical control |
| Online / continual systems | large | 6 | protocol, PC stream systems, fair evaluation and plasticity loss |
| 2024–2026 convergence | emerging | 5 | recent joint-constraint papers, all marked frontier |

The 35-node map is therefore a dependency-selected Top-K, not a citation-count
leaderboard. The 15-stop spine is the default route for a newcomer.
