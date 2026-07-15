# CI/CD

This repository uses a dependency-free GitHub Actions workflow for validation and continuous delivery of reviewable research snapshots.

## Local validation

Run the same check used by CI from the repository root:

```bash
python3 scripts/validate_repository.py
```

An alternate repository root can be checked with `python3 scripts/validate_repository.py --root PATH`.

## Triggers and permissions

The workflow runs for every pull request, every push to `main`, and manual `workflow_dispatch` runs. Its top-level GitHub token permission is read-only `contents: read`. It does not use `pull_request_target`, install packages, or call external services.

## Validation checks

The validation job uses Python 3.12 and the standard library only. It checks:

- presence of the agreed core and agent-harness files;
- relative links in Markdown files;
- scalar provenance frontmatter and manifest agreement for Project Meta artifacts;
- compilation of repository Python files; and
- that local `USER.md` and `USER.template.md` preference files are not tracked.

Errors are aggregated so one run reports all detected issues.

## Continuous delivery

After validation succeeds on a push to `main`, the delivery job runs `git archive` against the checked-out commit. It uploads the resulting tracked-file `tar.gz` as `research-snapshot-<commit SHA>` with seven-day retention.

This artifact is a reviewable, immutable-at-a-commit research snapshot. It is not a public deployment, release, website publication, or package publication. Pull requests and manual workflow runs never execute the delivery job.

Any future change to workflow triggers, permissions, action pins, retention, packaging semantics, or addition of a public deployment target requires an explicit plan and user approval before editing or committing the change.
