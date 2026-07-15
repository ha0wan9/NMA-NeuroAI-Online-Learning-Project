---
artifact_name: execution-rules
instantiated_from: project-meta/templates/execution-rules.md
source_reference: project-meta/references/execution-policy.md
project_scope: this repo only
owner: agent-facing
review_policy: user review when stop or ask categories change
last_reviewed: 2026-07-15
---

# Execution Rules

Bounded-execution agents must read this file before non-trivial edits. These rules are advisory; runtime sandboxing, approvals, branch protection, and CI provide enforcement.

## Decision Tiers

- **MUST STOP:** halt and obtain explicit approval.
- **SHOULD ASK:** show a scoped plan and wait for confirmation.
- **MAY PROCEED WITH NOTE:** act inside the approved scope and disclose it in the Structured delivery.

## MUST STOP

Halt and ask before:

- destructive Git or filesystem operations, force-push, branch deletion, or history rewrite
- any push without an explicit user request
- unrequested network commands, external writes, package installation, or dependency changes
- secrets, credentials, permissions, private participant data, `.env`, or license-sensitive dataset changes
- CI/CD, build, packaging, deployment, or workflow changes
- changing a frozen dataset split, stream, replay budget, task boundary, evaluation cadence, seed set, metric, tuning budget, or comparator
- overwriting or deleting raw data, run outputs, negative results, or provenance records
- public interfaces, structural top-level renames, generated bulk rewrites, or work outside the approved scope
- editing under a read-only task such as status, validate, deliver, audit, review, or literature-only exploration
- editing when the exact target files cannot be named

## SHOULD ASK

Show a plan and wait before:

- commits, pull requests, merges, or branch creation
- changes spanning more than one research milestone or logical subsystem
- a new top-level directory, test framework, dataset, model family, or experiment runner
- changing canonical memory or a topical agent contract
- changing an approved research question, hypothesis, candidate-rule shortlist, or baseline before protocol freeze

A branch or pull request is optional; do not create one merely because work is complex.

## MAY PROCEED WITH NOTE

Within an approved scope, agents may:

- edit a single research or documentation artifact
- perform read-only source discovery already authorized by the research request
- add focused tests or checks beside an approved implementation
- run experiments against a frozen protocol without changing that protocol
- correct typos in files already being edited

## Soft Budgets

```yaml
change_budget:
  default_files_soft_limit: 5
  default_lines_soft_limit: 300
  semantic_scope_escalation:
    one_milestone: may proceed with note
    cross_milestone: should ask
    cross_repo: must stop
```

Semantic scope is primary; file count is only a warning signal.

## Worker Plan

Before a non-trivial edit, emit:

```text
Goal:
Files to inspect:
Files likely to change:
Out of scope:
Commands likely to run:
Approval needed: yes / no
```

## Worker Constraints

A Worker must not expand scope, refactor opportunistically, introduce dependencies, alter mirrors before canonical memory, or promote uncertainty into durable memory. It must not claim a check passed without command output. If validation cannot run, identify the missing check and mark the result unverified. Literature and experiment handoffs must include source or run provenance, limitations, and the smallest reproduction command when applicable.
