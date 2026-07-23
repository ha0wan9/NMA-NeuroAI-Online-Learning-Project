# H2.E3 implementation-intent review

Date: 2026-07-18T19:52Z.

Initial verdict: **warn — launch may proceed**. No blocking issue.

The independent reviewer confirmed that H2.E3 preserves the frozen seeds,
architecture, learning rates, optimizer path, task stream, evaluation cadence,
and metrics. Representation hooks run in evaluation mode under `no_grad()`,
detach activations, remove hooks, restore model mode, and never receive the
trainer or optimizer. The CPU probe exactly reproduced both methods' previous
initial checksums and behavioral metrics.

The reviewer warned that the anchor indices used an isolated generator but the
anchor `DataLoader` itself did not. This was corrected before the full launch:
the anchor loader now owns a separate generator seeded with `20292133`.

Follow-up verdict: **pass — the full H2.E3 visualization run may launch**.
The reviewer verified that anchor selection and loader iteration now use two
independent fixed generators and that ordinary stream/evaluation loaders are
unchanged.

Remaining provenance warning: the study implementation is still untracked in
the user-owned dirty worktree, so the source state is described by revision
`6ddd656` plus the recorded approved diff rather than a commit. This does not
authorize a commit or push.
