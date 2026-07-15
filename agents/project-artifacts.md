---
artifact_name: project-artifact-manifest
instantiated_from: project-meta/templates/project-artifact-manifest.md
source_reference: project-meta/references/documentation-delivery.md
project_scope: this repo only
owner: agent-facing
review_policy: user review before artifact policy changes
last_reviewed: 2026-07-15
---

# Project Artifact Manifest

## Project Meta-Instantiated Artifacts

Artifacts:
- path: agents/delegation.md
  artifact_name: delegation
  instantiated_from: project-meta/templates/delegation.md
  source_reference: project-meta/references/multi-agent-protocols.md
  owner: agent-facing
  review_policy: user review before first behavior-changing commit
  last_reviewed: 2026-07-15
  refresh_trigger: delegation roles, ownership rules, evidence handoff, or review gates change
- path: agents/execution-rules.md
  artifact_name: execution-rules
  instantiated_from: project-meta/templates/execution-rules.md
  source_reference: project-meta/references/execution-policy.md
  owner: agent-facing
  review_policy: user review when stop or ask categories change
  last_reviewed: 2026-07-15
  refresh_trigger: approval tiers, budgets, runtime enforcement, or research protocol safeguards change
- path: agents/pre-commit-delivery.md
  artifact_name: pre-commit-delivery
  instantiated_from: project-meta/templates/pre-commit-delivery.md
  source_reference: project-meta/references/documentation-delivery.md
  owner: agent-facing
  review_policy: user review before commit
  last_reviewed: 2026-07-15
  refresh_trigger: delivery sections, required checks, or commit policy change
- path: agents/memory-writeback-check.md
  artifact_name: memory-writeback-check
  instantiated_from: project-meta/templates/memory-writeback-check.md
  source_reference: project-meta/references/repo-memory-crud.md
  owner: agent-facing
  review_policy: user review when memory policy changes
  last_reviewed: 2026-07-15
  refresh_trigger: durable-memory eligibility, ownership, or closeout evidence rules change
- path: agents/project-artifacts.md
  artifact_name: project-artifact-manifest
  instantiated_from: project-meta/templates/project-artifact-manifest.md
  source_reference: project-meta/references/documentation-delivery.md
  owner: agent-facing
  review_policy: user review before artifact policy changes
  last_reviewed: 2026-07-15
  refresh_trigger: a Project Meta-instantiated artifact or its provenance changes

`USER.md` is an ignored, local-only preference artifact and is intentionally excluded from the tracked artifact manifest.

## Project-Owned Artifact

Project-owned artifacts:
- path: agents/research-project-contract.md
  origin: project-owned
  owner: agent-facing
  review_policy: team review when milestone gates, claim discipline, or evaluation invariants change
  last_reviewed: 2026-07-15
  refresh_trigger: the approved proposal, claim discipline, or evaluation protocol changes

The project-owned research contract is not instantiated from a Project Meta template.
