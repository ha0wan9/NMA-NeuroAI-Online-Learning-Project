# Coverage Matrix — bioplausible-online-learning

- Generated: `2026-07-15T12:31Z`
- Phase: `audit`
- Status: `audit-needs-roundN`
- Evidence round: `R1`

## Cell Map

Cells list only ★★★ paper IDs. `closed` means adequate candidate coverage for
this gate, `weak` means covered but source-concentrated, and `—` is inactive.

| Sub-question | theory | experiment | survey | critical-review | dataset |
|---|---|---|---|---|---|
| SQ1 | closed: P011,P012,P013,P014,P018,P020,P021,P022,P023,P024,P025,P027,P029 | — | closed: P001,P002,P023,P024 | closed: P001,P002,P013,P023,P024,P026 | — |
| SQ2 | closed: P011,P012,P013,P014,P015 | closed: P011,P012,P013,P014,P015,P016 | — | **weak: P013** | — |
| SQ3 | closed: P017,P018,P020,P021,P022,P023,P024 | closed: P017,P018,P020,P021,P022 | — | closed: P023,P024 | — |
| SQ4 | closed: P025,P027,P029 | closed: P025,P026,P027,P029,P030 | — | **weak: P002,P026** | — |
| SQ5 | closed: P003,P004,P005,P006 | closed: P003,P004,P005,P006,P007,P009 | closed: P001,P009 | closed: P001,P009 | — |
| SQ6 | — | closed: P003,P004,P005,P006,P007,P008,P009,P011,P012,P013,P015,P017,P018,P020,P021,P022,P025,P026,P027,P029,P030 | closed: P001,P002,P009,P023 | closed: P001,P002,P008,P009,P013,P023,P026 | — |
| SQ7 | closed: P003,P004,P005,P006,P012,P014,P017,P018,P020,P021,P022,P024,P027 | closed: P003,P004,P005,P006,P007,P008,P009,P010,P012,P014,P016,P017,P018,P020,P021,P022,P026,P027 | closed: P001,P009,P024 | closed: P001,P008,P009,P024,P026 | **weak: P010** |
| SQ8 | — | closed: P007,P008,P009,P010 | **weak: P009** | closed: P008,P009 | **weak: P010** |
| SQ9 | — | closed: P003,P004,P005,P006,P007,P009,P010,P012,P013,P014,P015,P016,P017,P018,P020,P021,P022,P026,P027,P029,P030 | closed: P009,P023 | closed: P009,P013,P023,P026 | — |
| SQ10 | closed: P003,P004,P005,P006,P011,P013,P014,P015,P017,P018,P020,P021,P022,P023,P024,P025,P027,P029 | closed: P003,P004,P005,P006,P007,P008,P009,P011,P013,P014,P015,P016,P017,P018,P020,P021,P022,P025,P026,P027,P029,P030 | closed: P001,P002,P009,P023,P024 | closed: P001,P002,P008,P009,P013,P023,P024,P026 | — |

## Status Per Cell

| State | Count | Meaning |
|---|---:|---|
| closed | 30 | At least one ★★★ paper and no concentration rule triggered |
| weak | 5 | Evidence exists but is single-source or single-lab |
| gap | 0 | No active cell is empty after R1 |
| inactive | 15 | Dimension was excluded at frame time |

Weak cells:

- SQ2 / critical-review — only `P013`.
- SQ4 / critical-review — `P002` and `P026` share the DeepMind lineage.
- SQ7 / dataset — only `P010`.
- SQ8 / survey — only `P009`.
- SQ8 / dataset — only `P010`.

## Bias Audit

The confirmed ★★★ population is 28 papers. Automated counts use each paper
once. No bucket exceeds the fixed 60% trigger.

| Bucket | Dominant value | Count / denominator | Share | Decision |
|---|---|---:|---:|---|
| Institution | University of Oxford | 3 / 28 | 10.7% | clean |
| Country | USA | 8 / 28 | 28.6% | clean |
| Year | 2022 | 7 / 28 | 25.0% | clean |
| Venue | Nature Communications | 6 / 28 | 21.4% | clean |
| Method route | spike-local | 7 / 28 | 25.0% | clean |
| Deployment regime | offline-only | 11 / 24 method/experiment papers | 45.8% | clean; 4 review-only papers excluded |
| Venue type | preprint | 0 / 28 | 0.0% | clean |

Deployment regimes were conservatively classified by the primary reported
evaluation, not by algorithm names: temporal-online but multi-epoch studies
remain `offline-only`. The full mapping is in `audits/audit-r1.md`.

## Round N Task List

| Task | Sub-question / dimension | Current state | Concrete target | Search strategy |
|---|---|---|---|---|
| RN1-PC-CRIT | SQ2 / critical-review | weak: P013 only | A second peer-reviewed critical evaluation of PC exactness, settling cost or biological constraints from an independent lab, preferably 2023+ | Canonical-check Zahid et al. 2023 from the reserve set, then forward citation BFS from P013 in publisher records, OpenReview and DBLP |
| RN1-LOCAL-CRIT | SQ4 / critical-review | weak: P002,P026; one lab lineage | Independent negative or scaling assessment of FA, EP, DTP, dendritic or local-error learning outside DeepMind | Forward-citation BFS from P026; query limitation/scaling/reproduction language in NeurIPS, TMLR and peer-reviewed neuroscience journals |
| RN1-PROTOCOL-DATA | SQ7 / dataset | weak: P010 only | A distinct-lab, later benchmark with chronological or task-free streams and explicit task-ID, replay and boundary assumptions | Verify CLEAR and OpenLORIS primary records, then compare protocol fields against CORe50 before scoring |
| RN1-BENCH-SURVEY | SQ8 / survey | weak: P009 only | An independent recent survey that compares continual-learning benchmarks, metrics and evaluation assumptions | Citation BFS forward from P009 plus title/abstract queries for continual-learning evaluation, benchmark and metrics surveys |
| RN1-BENCH-DATA | SQ8 / dataset | weak: P010 only | A second benchmark paper with stream construction, metrics, baselines and leakage controls suitable for small-project adaptation | Share the CLEAR/OpenLORIS search batch with RN1-PROTOCOL-DATA; accept only a primary dataset/benchmark paper with canonical venue metadata |

## Audit Decision

- [ ] All active cells are `closed` or accepted-`weak`.
- [x] Bias audit is clean or accepted with a documented limitation.
- [ ] Each closed cell draws on at least two distinct labs when evidence exists.

Decision: `audit-needs-roundN`. The user has not accepted the five weak-cell
limitations, so the next phase is a targeted Round N, not synthesis.

**Phase**: audit  **Survey**: bioplausible-online-learning  **Status**: audit-needs-roundN  **Next**: roundn
