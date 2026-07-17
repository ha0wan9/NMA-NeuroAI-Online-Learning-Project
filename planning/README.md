# Working-Session Planning

This directory contains short-lived execution plans for the remaining project
sessions. These plans translate the milestone gates in
[`../agents/research-project-contract.md`](../agents/research-project-contract.md)
and the role allocation in [`../project-plan.md`](../project-plan.md) into
time-boxed work. They do not freeze or override the scientific protocol.

## Current plan set

| Date | Available team time | Primary objective | Plan |
|---|---:|---|---|
| 2026-07-17 | 1 hour | Set up the baseline path and dispatch the continual-learning protocol study | [`2026-07-17.md`](2026-07-17.md) |
| 2026-07-20 | 3 hours | Freeze protocol v0.1 and pass the BP continual-baseline gate | [`2026-07-20.md`](2026-07-20.md) |
| 2026-07-21 | 3 hours | Integrate PC after the baseline gate and attempt a matched-path smoke run | [`2026-07-21.md`](2026-07-21.md) |

The continual-learning research assignment and its Monday deadline are in
[`continual-learning-research-dispatch.md`](continual-learning-research-dispatch.md).

Dates after July 21 remain draft-only until the exact mid-project and final
presentation dates are known. At the July 21 closeout, create the next dated
plan from the template below and align it with the actual gate status rather
than the originally expected calendar day.

## Daily planning template

```markdown
# YYYY-MM-DD Working-Session Plan

**Available time:**
**Gate at session start:**
**Session owner:**
**Primary outcome:**

## Entry check

- [ ] Previous handoff and run artifacts reviewed.
- [ ] Current blockers and protocol decisions named.

## Time boxes

| Elapsed time | Owner | Work | Required output |
|---|---|---|---|
| T+00–T+00 | | | |

## Exit criteria

- [ ]

## Handoff and decision record

- Gate status: passed / not passed / not assessed
- Evidence or run path:
- Blocker owner:
- Next decision and deadline:
```

## Operating rules

- Each output has one directly responsible individual (DRI), even when several
  people contribute.
- Use elapsed time (`T+00`) when the meeting start time is not yet fixed.
- A plan records intended work; only cited run artifacts can establish that a
  gate passed.
- Do not silently change a stream, metric, comparator, information allowance,
  replay budget, seed set, evaluation cadence, or compute budget in a daily
  plan. Record the decision in the protocol or project decision log first.
- Preserve failed runs, negative results, and deviations.
