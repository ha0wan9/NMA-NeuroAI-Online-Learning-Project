---
artifact_name: memory-writeback-check
instantiated_from: project-meta/templates/memory-writeback-check.md
source_reference: project-meta/references/repo-memory-crud.md
project_scope: this repo only
owner: agent-facing
review_policy: user review when memory policy changes
last_reviewed: 2026-07-15
---

# Memory Writeback Check

Run this check at the close of meaningful work. Canonical memory is for durable project guidance, not a lab notebook or session log.

## Eligible Writeback

Write only validated, reusable material such as an approved protocol invariant, stable repository path, accepted validation command, or recurring workflow correction. A scientific result belongs first in a versioned research or experiment artifact with source/run provenance; only a stable navigation or workflow consequence belongs in agent memory.

Never write secrets, credentials, private data, local machine state, transient task logs, unsupported conclusions, speculative claims, or unresolved hypotheses into canonical memory. Explicit stable collaboration preferences belong only in ignored `USER.md`.

## Closeout Record

```text
Durable lesson:
- <candidate lesson or none>

Evidence or validation:
- <primary source, review decision, command output, or run artifact>

Claim status:
- <established fact | interpretation | resolved workflow | still a hypothesis>

Memory owner:
- <AGENTS.md | agents/*.md | USER.md | research artifact | none>

Writeback decision:
- <write now | suggest only | skip>

Reason:
- <why a future agent will or will not need it>
```

Replace stale guidance instead of appending contradictions. The Lead owns final writeback after delegated work.
