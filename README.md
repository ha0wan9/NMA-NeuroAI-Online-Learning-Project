# NMA NeuroAI Online Learning Project

## Shared documentation

- [Online learning](https://docs.google.com/document/d/1KPN6ZmiHvTHzIgHrToI5Lz1579pkk8hePS8xrIrLj_k/edit?tab=t.0) — team-maintained living documentation for shared project reference.

## Meeting Notes

- [July 14, 2026 — Meeting 1: Project Brainstorming and Group Formation](meeting-notes/2026-07-14-meeting-1.md)

## Research workflow

This two-week research project studies biologically plausible learning rules for online and continual learning. The current workflow moves from brainstorming to literature review, proposal, reproducible baselines, implementation of one or two candidate rules, fixed-protocol evaluation, and evidence-linked synthesis. Established results, interpretations, working hypotheses, and speculation should be labeled distinctly.

## Literature Review

Follow the literature review in order:

1. **Stage 1 — Domain Introduction:** Researchers with a general scientific
   background should begin
   with the [interactive reading roadmap](.research/surveys/bioplausible-online-learning/biologically-plausible-online-learning-roadmap.html)
   and its [roadmap index](.research/surveys/bioplausible-online-learning/reading-roadmap-index.md).
2. **Stage 2 — SOTA Results Explorer:** After the introduction, continue to the
   [interactive SOTA Results Explorer](.research/surveys/bioplausible-online-learning/sota-results-explorer.html)
   for a multi-dimensional view of results, methods, protocols, and feasibility
   evidence, with the [SOTA survey index](.research/surveys/bioplausible-online-learning/index.md)
   as its canonical frame.

Both stages use one shared 50-paper registry: the
[interactive paper registry](.research/surveys/bioplausible-online-learning/paper-registry.html)
([canonical Markdown](.research/surveys/bioplausible-online-learning/paper_index.md)),
[claim ledger](.research/surveys/bioplausible-online-learning/claims.jsonl), and
[paper-ID crosswalk](.research/surveys/bioplausible-online-learning/paper-id-crosswalk.md).
The introduction uses a 35-paper subset. The SOTA survey retains status
`audit-needs-roundN` with five weak cells; this navigation does not claim that
the Literature Review gate is complete or authorize synthesis.

## Validation

Run the dependency-free repository checks from the project root:

```bash
python3 scripts/validate_repository.py
```

To validate another checkout, run `python3 scripts/validate_repository.py --root PATH`.

The validator checks the agreed repository harness, relative Markdown links, Project Meta artifact provenance, Python syntax, and exclusion of tracked local preference files. See [CI/CD guidance](agents/ci-cd.md) for the exact contract.

## CI/CD

GitHub Actions runs validation for pull requests, pushes to `main`, and manual dispatches. A validated push to `main` also produces a seven-day workflow artifact containing a `git archive` snapshot of tracked project files. This is continuous delivery for research review and handoff; there is no public deployment target.

Changes to CI/CD permissions, triggers, delivery semantics, or deployment targets require explicit approval under the project contract.
