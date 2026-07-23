# predictive-coding-cifar-baseline Audit 2026-07-19T01:05Z

## Aggregate Verdict

`warnings`

- Trigger: baseline synthesis after H1.E2 completion
- Reference protocol revision: `index.md` through the 2026-07-18 H2.E3
  representation registration
- Auditor: managed `methodology-auditor`

## Checks

| Check | Verdict | Evidence | Action |
|---|---|---|---|
| Identity | aligned | Study ID, root, H/E IDs, ledger, reports, and graph agree. | none |
| Adapter | aligned | The adapter now uses study-specific data/metric metadata and explicitly includes the Step 2 CIFAR editable and protected surfaces. | none |
| Protocol | aligned | Archived H1 behavior is intentional/disclosed; H2 remains paired, causal, one-pass, and replay-free. | preserve claim labels |
| Baseline | warning | BP/PC are matched, but the static control is archived repeated-update RBP rather than conventional one-update BP/SGD; H2 learning rates are source-derived by method. | label RBP consistently; register conventional BP separately if required |
| Data integrity | warning | Checksums pass; H1 deliberately uses test-informed selection and shuffled/drop-last evaluation. | never describe H1 as leakage-free model selection |
| Decision validity | aligned | Six H1 cells pass ±1 pp; result-skeptic accepted the narrow claim; H2 remains descriptive. | none |
| Tracking | aligned | Ledger validator passes and preserves the invalid zero-batch smoke. | none |
| Budget | warning | Recorded update time stays below ten hours but is not end-to-end occupancy or energy. | carry forward accounting caveat |
| Scope | aligned | Reports reject general superiority and keep completed H2.E3 within an observational representation claim. | none |
| Reproducibility | warning | Commands, seeds, environment, revisions, checksums, and artifacts exist; the H2.E3 launch is content-hash frozen, but dirty/untracked state and the newer PC library prevent bitwise reproduction. | create a reviewed commit before broad reuse |
| Graph consistency | aligned | Mermaid/JSON nodes, decisions, relations, and completed H2.E3 state agree. | none |

## Findings

- H1.E2 may be reused only as an exact-as-practical reproduction of the
  archived, test-informed Figure 4i best-test protocol.
- H2.E2 may be reused as a descriptive matched replay-free SplitCIFAR-10
  stability–plasticity baseline.
- H2.E3 may be reused as an observational representation baseline: PC has
  lower old-task RDM drift and weaker label-aligned acquisition. It does not
  support a causal drift-to-forgetting or favorable-trade-off claim.
- Neither artifact supports general PC, continual-learning, conventional-BP,
  or biological-superiority claims.
- The artifact field `absolute_target_difference` stores a signed difference
  and should be renamed or corrected before the next schema revision.

## Managed Reviewer

- Role: `methodology-auditor`
- Verdict: `pass-with-warnings`
- Blocking findings adopted: none
- Findings rejected with artifact evidence: a compact secondary review rated
  the newer local PC library as protocol `drift`. The lead retains a
  reproducibility `warning` because the library difference was disclosed
  before evaluation, the registered target was explicitly
  exact-as-practical rather than bitwise reproduction, and all six external
  accuracy gates passed. The corrective pin/equivalence action is still
  adopted before broad reuse.

## Required Action

- Freeze exact code/artifact identity before broad baseline reuse.
- Reconcile adapter defaults with the study-specific CIFAR contract.
- Preserve RBP, test-informed-selection, update-time, and newer-library
  caveats in downstream reports.
- Interpret H2.E3 drift only beside label alignment, adaptation, prequential
  accuracy, forgetting, and final accuracy.

## H2.E3 closeout addendum — 2026-07-19T16:53Z

- Design review: pass; exact paired seeds, one-pass replay-free stream, anchor
  RNG isolation, evaluation-only hooks, and frozen source hashes verified.
- Result-skeptic verdict: kept as an observational representation result with
  no blocker.
- Instrumentation check: diagnostics-on and same-revision diagnostics-off
  behavior matches exactly across all six runs and registered fields.
- Claim ceiling: three independent seeds; task rows are nested; correlations
  are task-confounded; no causal mediation, monotonic drift–forgetting,
  superiority, biological-mechanism, or general continual-learning claim.
- Validation: deterministic reanalysis, ledger validator, repository
  validator, protocol tests, source compilation, provenance hashes, graph
  references, JSON parsing, and `git diff --check` all pass.
