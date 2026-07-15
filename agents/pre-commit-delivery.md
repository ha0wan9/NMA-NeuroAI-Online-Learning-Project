---
artifact_name: pre-commit-delivery
instantiated_from: project-meta/templates/pre-commit-delivery.md
source_reference: project-meta/references/documentation-delivery.md
project_scope: this repo only
owner: agent-facing
review_policy: user review before commit
last_reviewed: 2026-07-15
---

# Pre-Commit Delivery

No agent commits automatically. Before any requested commit, show this review packet and wait for approval:

```text
User-facing docs:
- <paths and research-facing changes>

Agent-facing docs:
- <paths and future-agent behavior changes>

Research and evaluation impact:
- <milestone gate, claim status, baseline, protocol, or result affected>

Behavior or trigger changes:
- <what agents will do differently>

Validation:
- git diff --check: <result>
- python3 scripts/validate_repository.py: <result>
- Markdown links and cited source identifiers in touched files: <result>
- experiment reproduction command, configuration, and seeds: <result or not applicable>

Commit scope:
- <tracked files intended for commit>

Excluded local state:
- USER.md, .harness/, credentials, caches, raw private data, and other local-only files

Commit and push status:
- commit: not created / explicitly requested and approved
- push: not performed unless explicitly requested
```

If user-facing research claims or documentation changed, call them out for review. Failed or unavailable validation must be visible; never present an unverified result as passing.
