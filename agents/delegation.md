---
artifact_name: delegation
instantiated_from: project-meta/templates/delegation.md
source_reference: project-meta/references/multi-agent-protocols.md
project_scope: this repo only
owner: agent-facing
review_policy: user review before first behavior-changing commit
last_reviewed: 2026-07-15
---

# Delegation

Use delegation only for concrete, bounded, independent work. Parallel literature discovery is read-only by default; parallel experiment edits require disjoint file ownership.

## Roles

- Lead: fixes the research question and inclusion criteria, prepares context packages, assigns ownership, deduplicates evidence, reconciles contradictions, integrates results, and owns final memory writeback.
- Explorer: performs a narrow read-only search or repository investigation and returns source pointers plus limitations.
- Worker: creates or edits only the explicitly assigned artifact and reports commands and outputs.
- Reviewer: checks evidence traceability, protocol consistency, scope, and reproducibility; returns `PASS`, `SUGGEST`, or `BLOCKER`.

## Context Package

Every delegated task must state:

```text
Role: <Explorer | Worker | Reviewer>
Goal: <one evidence question or one artifact>
Read first:
- <smallest required project path or source list>
Ownership: <read-only | exact writable paths>
Constraints:
- <research gate, claim label, protocol invariant, and time budget>
Output format: <evidence rows | patch summary | review verdict>
Review criteria:
- <observable pass/fail conditions>
Memory policy: <report only | suggest a durable update; Lead writes canonical memory>
```

## Literature Search Contract

- Divide searches by non-overlapping method family, benchmark/protocol question, or biological-plausibility dimension.
- Return title, authors, year, stable URL or DOI, source type, exact supported claim, studied setting, comparator, metric, and caveat.
- Prefer primary papers for algorithm and performance claims; reviews may map the field but do not replace primary evidence for important claims.
- Do not infer that a method beats Backpropagation from incomparable datasets, architectures, data access, replay budgets, or evaluation protocols.
- The Lead performs duplicate removal, variant disambiguation, claim reconciliation, and final inclusion.

## Experiment Contract

- Give each Worker disjoint files or directories and a frozen protocol identifier.
- Require the smallest runnable command, configuration, seeds, and artifact paths in the handoff.
- Stop on a reviewer `BLOCKER`; do not continue integration until the Lead resolves it.
- Never let an implementation worker change the baseline or evaluation protocol silently.
