# Round 1 Search Log — Biologically Plausible Online Learning

- Executed UTC: `2026-07-15T12:23Z`
- Phase result: `round1-done`
- Input frame: 10 sub-questions, 5 method routes, 35 active evidence cells
- Retained cap: `max(30, 3 × 10) = 30` unique canonical works
- Next phase: formal coverage and bias audit

This is a retrieval and scoring record, not a synthesis. A ★★★ score means
that a paper is high-value evidence for this survey; it does **not** mean its
method is biologically plausible overall or that it beats Backpropagation.

## Breadth-First Method

Searches began with title- and abstract-anchored arXiv queries, then resolved
canonical status and metadata through OpenReview or conference proceedings,
DBLP, and journal/publisher pages. Citation counts were not used to inflate
scores. Four read-only clusters searched in parallel: Predictive Coding,
spike-local rules, other local-credit routes, and stability/protocol evidence.
All clusters returned before central deduplication, scoring and ID assignment.

The lead merged duplicate preprint/venue versions, applied the shared
four-dimension rubric once, retained one canonical row per work, and assigned
`P001`–`P030`. Online gradient computation was not treated as evidence of
continual learning, and an SNN architecture was not treated as a learning rule.

## Central Score Ledger

Score order is `relevance / authority / recency / evidence`; weighted score is
`1.5 × relevance + authority + recency + evidence`.

| ID | Canonical primary source | R/A/N/E | Weighted | Central inclusion decision and limiting caveat |
|---|---|---:|---:|---|
| P001 | [Biological underpinnings for lifelong learning machines](https://www.nature.com/articles/s42256-022-00452-0) | 3/3/2/3 | 12.5 | Canonical lifelong-learning biology map; a perspective, not a matched algorithm comparison. |
| P002 | [Backpropagation and the brain](https://www.nature.com/articles/s41583-020-0277-3) | 3/3/1/3 | 11.5 | Canonical plausibility criteria and alternatives review; no new continual experiment. |
| P003 | [Continual Learning Through Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html) | 3/3/1/2 | 10.5 | Direct online importance mechanism; remains an add-on to gradient/BP training. |
| P004 | [Continual Reinforcement Learning with Complex Synapses](https://proceedings.mlr.press/v80/kaplanis18a.html) | 3/3/1/2 | 10.5 | Direct multiscale-synapse evidence; simple tasks and gradient-based base learner. |
| P005 | [Synaptic metaplasticity in binarized neural networks](https://www.nature.com/articles/s41467-021-22768-y) | 3/3/2/3 | 12.5 | Strong coded metaplasticity result; does not replace global credit assignment. |
| P006 | [Bayesian continual learning and forgetting](https://www.nature.com/articles/s41467-025-64601-w) | 3/3/3/3 | 13.5 | Leading task-boundary-free metaplastic update with code; biological inspiration is not physiological validation. |
| P007 | [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) | 3/3/3/3 | 13.5 | Large, long-horizon demand evidence; its continual-BP remedy is not a plausible rule alternative. |
| P008 | [Re-evaluating Continual Learning Scenarios](https://arxiv.org/abs/1810.12488) | 3/2/1/3 | 10.5 | Strong protocol and baseline correction; workshop status and no candidate rule. |
| P009 | [A continual learning survey: Defying forgetting](https://doi.org/10.1109/TPAMI.2021.3057446) | 3/3/2/3 | 12.5 | Broad survey plus empirical comparison; conclusions are task-incremental. |
| P010 | [CORe50](https://proceedings.mlr.press/v78/lomonaco17a.html) | 3/2/1/3 | 10.5 | Actual stream-oriented dataset/protocol evidence; not a learning-rule result. |
| P011 | [Whittington & Bogacz predictive-coding approximation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/) | 3/3/1/2 | 10.5 | Foundational local PC rule; restrictive settling/parameter assumptions and no CL. |
| P012 | [A Theoretical Framework for Inference Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f242c4cba2467637256722cb679642bd-Abstract-Conference.html) | 3/3/2/2 | 11.5 | Precise batch-one/implicit-SGD theory; online optimization is not non-stationary CL. |
| P013 | [Predictive coding and backpropagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0266102) | 3/2/2/3 | 11.5 | Direct critical test of exactness and cost; scope is conventional offline training. |
| P014 | [Prospective configuration](https://www.nature.com/articles/s41593-023-01514-1) | 3/3/2/3 | 12.5 | Direct online/continual claims with matched models; several protocols remain minibatch or offline. |
| P015 | [Stable, Fast PC Networks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/554414e570a85eb3118e988c5d77986f-Abstract-Conference.html) | 3/3/2/3 | 12.5 | Strong feasibility advance; “incremental” means during inference, not continual learning. |
| P016 | [Lifelong Neural Predictive Coding](https://proceedings.neurips.cc/paper_files/paper/2022/hash/26f5a4e26c13d1e0a47f46790c999361-Abstract-Conference.html) | 3/3/2/3 | 12.5 | Direct class-incremental evidence; extra controller/context/memory confounds the rule. |
| P017 | [Linear-memory online SNN learning](https://doi.org/10.1038/s41467-026-68453-w) | 3/3/3/3 | 13.5 | Leading temporal-online scale evidence; deep learning signals are spatially backpropagated. |
| P018 | [e-prop](https://doi.org/10.1038/s41467-020-17236-y) | 3/3/1/2 | 10.5 | Canonical causal eligibility-trace rule; learning signals and CL evidence remain limited. |
| P019 | [DECOLLE](https://doi.org/10.3389/fnins.2020.00424) | 3/2/1/2 | 9.5 | Practical layer-local updates; uses label losses, surrogate derivatives and offline epochs. |
| P020 | [S-TLLR](https://openreview.net/forum?id=CNaiJRcX84) | 3/3/2/3 | 12.5 | Near-BPTT multi-task evidence and resource accounting; full locality depends on DFA. |
| P021 | [ETLP](https://doi.org/10.1088/2634-4386/ad6733) | 3/2/2/2 | 10.5 | Explicit three-factor local rule and hardware demo; global supervised targets reach layers. |
| P022 | [Online Training Through Time](https://openreview.net/forum?id=Siv3nHYHheI) | 3/3/2/3 | 12.5 | Strong scale and temporal-memory result; retains spatial BP/global objective. |
| P023 | [Brain-Inspired Learning on Neuromorphic Substrates](https://doi.org/10.1109/JPROC.2020.3045625) | 3/3/1/3 | 11.5 | Canonical SNN locality framework; synthesis rather than matched superiority evidence. |
| P024 | [Eligibility traces and three-factor plasticity](https://doi.org/10.3389/fncir.2018.00053) | 3/3/1/3 | 11.5 | Physiological support for the motif; not for algorithm-specific deep error signals. |
| P025 | [Feedback Alignment](https://www.nature.com/articles/ncomms13276) | 3/3/1/2 | 10.5 | Removes exact weight transport; retains explicit output error and feedback signaling. |
| P026 | [Scalability of biologically motivated algorithms](https://proceedings.neurips.cc/paper_files/paper/2018/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html) | 3/3/1/3 | 11.5 | Important negative FA/TP/DTP evidence; offline and tied to 2018 models/tuning. |
| P027 | [Meta-learned plasticity with random feedback](https://www.nature.com/articles/s41467-023-37562-1) | 3/3/2/2 | 11.5 | Batch-one inner updates; global outer-loop meta-optimization and IID streams. |
| P028 | [Equilibrium Propagation](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full) | 3/3/1/1 | 9.5 | Foundational local rule; two phases, settling and symmetry weaken online plausibility. |
| P029 | [Scaling Difference Target Propagation](https://proceedings.mlr.press/v162/ernoult22a.html) | 3/3/2/3 | 12.5 | Strong scaling evidence; learned feedback explicitly approximates BP targets. |
| P030 | [Training with Local Error Signals](https://proceedings.mlr.press/v97/nokland19a.html) | 3/3/1/3 | 11.5 | Strong layer-local scaling evidence; labels and auxiliary gradient computations remain. |

## Retained Breadth

- By method route: predictive-coding 6; spike-local 8; other-local-credit 7;
  stability-memory 5; protocol-critique 4.
- By year: 2016: 1; 2017: 4; 2018: 4; 2019: 1; 2020: 3;
  2021: 2; 2022: 7; 2023: 1; 2024: 5; 2025: 1; 2026: 1.
- By canonical status: 19 journal articles/reviews, 10 main-conference
  papers, and 1 workshop paper; no unresolved preprint-only row was retained.
- By rating: 28 ★★★ and 2 ★★. This concentration reflects a capped,
  high-relevance evidence set, not 28 demonstrations of superiority.
- By sub-question: SQ1 20; SQ2 6; SQ3 8; SQ4 7; SQ5 7; SQ6 26; SQ7 21;
  SQ8 4; SQ9 24; SQ10 28.
- Zero-★★★ sub-questions: none. This candidate-level count does not close
  the active sub-question × evidence-dimension cells; that is the audit's job.

## Informational Coverage Preview

`coverage_check.py` found at least one ★★★ paper in all 35 active cells and
reported no empty cell. It nevertheless flagged five concentrated cells as
weak rather than diverse:

- SQ2 / critical-review: single source (`P013`)
- SQ4 / critical-review: one lab lineage (`P002`, `P026`)
- SQ7 / dataset: single source (`P010`)
- SQ8 / survey: single source (`P009`)
- SQ8 / dataset: single source (`P010`)

This preview is informational. It does not change `coverage_matrix.md` or pass
the formal audit; the audit must also count institution, country, year, route,
deployment regime and preprint concentration among ★★★ papers.

## Deduplication and Scored Reserves

Preprint and accepted versions were merged for Synaptic Intelligence,
Complex Synapses, metaplasticity, MESU, loss of plasticity, DECOLLE, e-prop,
S-TLLR, ETLP, OTTT, iPC, Lifelong Neural Predictive Coding, DTP and the
meta-learned random-feedback rule. Earlier or workshop versions do not receive
separate IDs.

The cap displaced useful but redundant or narrower evidence rather than
invalidating it. High-priority reserves for a targeted round are:

- Predictive Coding: Song et al. (2020) exact BP implementation; Zahid et al.
  (2023) critical evaluation; μPC (2025) depth scaling; BayesPCN (2022)
  sequential associative memory.
- Spike-local: OSTL, OSTTP, FPTT and SuperSpike. They are useful for temporal
  credit or scalability but retain spatial BP/global loss, have small evidence,
  or overlap retained routes.
- Other local credit: Deep Feedback Control, dendritic microcircuits,
  Continual EP and Forward-Forward. These complete mechanism routes but lack
  decisive continual evidence in the retained cap.
- Stability/protocol: Benna–Fusi complex-synapse theory, Farquhar–Gal robust
  evaluation, the 2025 continuously-changing-environments perspective and
  Avalanche. They become targets if the audit finds a theory, protocol or
  implementation-feasibility cell weak.

## Predicted Gaps Before Formal Audit

1. **No general matched superiority result.** There is no clean, multi-dataset,
   single-pass non-stationary comparison in which a standard biologically
   plausible rule and BP receive the same architecture, stream, replay,
   persistent state, tuning budget, metric and uncertainty treatment.
2. **Temporal online is often mislabeled as continual.** BrainTrace, OTTT,
   S-TLLR, iPC and Continual EP can update forward in time without demonstrating
   task-, domain- or class-incremental learning.
3. **Locality is usually partial.** The strongest methods commonly retain a
   global error/target, label broadcast, trained feedback, symmetric dynamics,
   surrogate derivatives, iterative settling, an offline outer loop or BP in
   another axis.
4. **Stability mechanisms are usually wrappers.** Synaptic Intelligence,
   complex synapses, metaplasticity, MESU and continual BP address stability or
   plasticity while leaving global credit assignment largely conventional.
5. **Benchmark evidence is thin.** SQ8 has only four retained papers and only
   CORe50 is an explicit dataset paper. A two-week protocol will still need a
   leakage-resistant small benchmark, stream construction and matched budgets.
6. **Reproducibility is unassessed.** Repository availability was used only as
   score evidence. Every `Repro` field remains `pending` until the dedicated
   reproducibility procedure is loaded and applied.
7. **Institutional dependence remains plausible.** Predictive Coding evidence
   is clustered around Oxford/Sussex lineages; the audit must quantify this and
   search independent replications if the ★★★ threshold is exceeded.

## Round 1 Handoff

- `paper_index.md` now contains stable IDs `P001`–`P030`.
- `claims.jsonl` remains empty because full-text claim extraction has not begun.
- `coverage_matrix.md` remains `not-audited`; Round 1 may preview coverage but
  cannot close cells.
- Next action: run the formal coverage and bias audit, then issue narrowly
  targeted Round N searches only for documented gaps.

**Phase**: round1  **Survey**: bioplausible-online-learning  **Status**: round1-done  **Next**: audit
