# NMA NeuroAI Online Learning Project

## Shared documentation

- [Online learning](https://docs.google.com/document/d/1KPN6ZmiHvTHzIgHrToI5Lz1579pkk8hePS8xrIrLj_k/edit?tab=t.0) — team-maintained living documentation for shared project reference.

## Meeting Notes

- [July 14, 2026 — Meeting 1: Project Brainstorming and Group Formation](meeting-notes/2026-07-14-meeting-1.md)

## Research workflow

This two-week research project studies biologically plausible learning rules for online and continual learning. The current workflow moves from brainstorming to literature review, proposal, reproducible baselines, implementation of one or two candidate rules, fixed-protocol evaluation, and evidence-linked synthesis. Established results, interpretations, working hypotheses, and speculation should be labeled distinctly.

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
