# Agent Instructions

This repository is a two-week collaborative NeuroAI research and evaluation project on biologically plausible learning rules for online and continual learning. `AGENTS.md` is the canonical Codex-primary project-memory loader.

## Bootstrap Order

1. Read this file first.
2. Read `USER.md` when it exists. It is local-only and ignored by Git.
3. Inspect `README.md` lightly; read it fully only when changing shared documentation or when global project semantics are needed.
4. Before a non-trivial edit, read `agents/execution-rules.md`.
5. Load only the topical files routed below that are needed for the task.

## Topic Routing

- Research milestones, claim labels, biological-plausibility criteria, baselines, and evaluation protocol: `agents/research-project-contract.md`.
- Multi-agent literature or experiment work: `agents/delegation.md`.
- CI/CD design, workflow maintenance, and repository validation: `agents/ci-cd.md`.
- Delivery before a commit: `agents/pre-commit-delivery.md`.
- Durable-memory decisions at task closeout: `agents/memory-writeback-check.md`.
- Project Meta artifact provenance and refresh triggers: `agents/project-artifacts.md`.
- Meeting-derived schedule and decisions: `meeting-notes/`, beginning with `meeting-notes/2026-07-14-meeting-1.md`.

## Global Guardrails

- Treat research and Markdown artifacts as primary until an approved experiment requires code or notebooks.
- Keep important claims traceable to primary or authoritative sources; do not fabricate citations, results, or completed work.
- Distinguish fact, interpretation, working hypothesis, speculation, and open question.
- Preserve negative results and limitations. Do not promote unresolved hypotheses into canonical memory.
- Keep secrets, credentials, private data, local machine state, and `USER.md` out of tracked project memory.
- Do not commit or push automatically. A push requires an explicit user request.
- Record minimal verification or reproduction commands for experiment artifacts.
- Preserve the existing repository structure; use lowercase kebab-case for new project files unless an established convention requires otherwise.

## Closeout

For meaningful work, run the available checks, prepare the Structured delivery, and apply `agents/memory-writeback-check.md`. Update only the smallest canonical file that owns a validated durable lesson.
