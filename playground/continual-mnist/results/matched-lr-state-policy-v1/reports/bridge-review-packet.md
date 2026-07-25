# matched-lr-state-policy-v1 bridge review packet

- Review status: `human authorized held-out execution`
- Protocol: `matched-lr-state-policy-v1`
- Bridge seed: `42` (openly inspected; excluded from confirmation)
- Validated cells: `8/8`
- Manifest SHA-256:
  `9e88aac4a68a886a056a88a91826168faedd95593fa7db1df210ddd3d7459791`
- Reviewed bridge Markdown SHA-256:
  `39d743c44b67677c6c179baddb13dfd0994ffeb68a2fcda3358efc606ec18e26`
- Independent validation: `passed`
- Failures, incomplete attempts, or unpaired artifacts: none

## Why the SHA-256 matters

SHA-256 maps the exact bytes of a file to a 256-bit digest written as 64
hexadecimal characters. The same bytes produce the same digest; changing even
one byte almost certainly produces a different digest.

Here it is an integrity fingerprint for the exact bridge Markdown reviewed at
this checkpoint. The held-out launcher recomputes the report digest and accepts
authorization only when it equals the supplied
`--bridge-review-sha256`. This binds later execution to this reviewed report
and detects silent edits or substitution.

The digest is not encryption, a scientific score, or proof of authorship. By
itself it does not establish that the report's interpretation is correct.

Recheck command:

```bash
sha256sum \
  playground/continual-mnist/run-staging/matched-lr-state-policy-v1/reports/bridge-intuition-report.md
```

## Observations

These are recorded facts from the aggregate bridge reports and validator.

1. All eight seed-42 full cells passed provenance, pairing, frozen-order,
   source/data/host, stream, initialization, count, and state-policy checks.
2. All conditions recorded 19 task-boundary actions. Learned parameters were
   preserved at every boundary. Persistent state was unchanged by each retain
   action; task-reset state was empty immediately after every clear action.
   Persistent optimizers ended at step 2400; task-reset optimizers ended at
   step 120 after the final task.
3. Reset-minus-persistent final-average-accuracy contrasts were:

   | Method/optimizer | Reset effect |
   |---|---:|
   | BP–Adam | -25.326 percentage points |
   | BP–CLASSP | -10.052 percentage points |
   | PC–Adam | -27.898 percentage points |
   | PC–CLASSP | -2.187 percentage points |

   Thus the registered relative contrasts for this open seed were
   `R_BP = +15.274` points, `R_PC = +25.711` points, and
   `Ω = +10.437` points.
4. For online predict-before-update accuracy, Adam reset effects were negative
   under BP (-2.155 points) and PC (-2.640 points), whereas CLASSP reset
   effects were positive under BP (+17.956 points) and PC (+17.045 points).
   The optimizer-relative contrasts were similar under BP and PC:
   `R_BP = +20.111`, `R_PC = +19.685`, and `Ω = -0.426` points.
5. Reset worsened BWT and forgetting in every method/optimizer pair. The
   deterioration was smaller for CLASSP than Adam, especially under PC.
   CLASSP reset simultaneously improved adaptation relative to CLASSP
   persistence under both BP and PC.
6. Mean first updates after task transitions increased under reset by about
   2.00× for BP–Adam, 4.75× for BP–CLASSP, 1.97× for PC–Adam, and 4.17× for
   PC–CLASSP.
7. Total parameter movement under reset was 0.95× persistent for BP–Adam,
   1.96× for BP–CLASSP, 0.94× for PC–Adam, and 1.59× for PC–CLASSP.
8. Total cell runtime was 6,776.291 seconds (about 1 hour 53 minutes). The
   bridge-derived held-out estimate is 33,881.457 seconds (about 9 hours
   25 minutes); the required 20% reservation margin gives about 11 hours
   18 minutes.

## Interpretations

1. The final-accuracy contrasts point in the direction required by both
   registered hierarchical gates for seed 42. This is directional intuition,
   not confirmatory evidence, because the seed is openly inspected and
   excluded from the five-seed gate.
2. “Relative reset rescue” must not be mistaken for an absolute benefit. Reset
   reduced final accuracy in all four method/optimizer pairs. The positive
   `R_PC` occurs because PC–CLASSP was harmed much less than PC–Adam, not
   because PC–CLASSP reset exceeded PC–CLASSP persistence.
3. The almost-zero online `Ω`, combined with similar CLASSP online gains under
   BP and PC, does not suggest a PC-specific online-learning benefit at this
   seed.
4. The enlarged first updates and increased CLASSP parameter movement make an
   optimizer-transient explanation credible. Reset appears to restore
   plasticity while sacrificing retention; that tradeoff can explain improved
   online/adaptation metrics alongside worse final accuracy and forgetting.
5. No implementation or frozen-protocol flaw is visible in the bridge
   validation or state audits. The bridge therefore serves its intended role:
   it exposes a meaningful, nontrivial pattern that should be tested without
   changing v1.

## Limitations

- This is one openly inspected seed, with no estimate of across-seed
  uncertainty and no confirmatory status.
- The protocol tests one fixed learning rate, one Permuted-MNIST MLP, one
  stream definition, and one state-reset intervention.
- Diagnostics add synchronization, allocation, and timing overhead. Synthetic
  neutrality tests cover scientific outputs and final parameters, not runtime
  or peak memory.
- Positive relative contrasts do not establish statistical significance,
  mechanistic synergy, biological superiority, or generality.
- Reviewing seed 42 creates a real risk of narrative overfitting; v1 must not
  be tuned in response.

## Counterclaim

The apparent CLASSP-specific “rescue” may be caused mainly by optimizer
transient scale: resetting state produces much larger first updates and harms
Adam more severely. On this account, the result does not reflect a special
PC×CLASSP continual-learning mechanism.

## Discriminating check

Run the frozen five held-out seeds without inspecting partial effects. After
all 40 held-out cells validate, check:

1. whether `R_PC > 0` for at least four of five held-out seeds;
2. only if that passes, whether `Ω > 0` for at least four of five seeds;
3. whether first-update amplification and the adaptation/forgetting tradeoff
   recur across seeds; and
4. whether any apparent final-accuracy pattern is better explained by Adam
   reset damage than by an absolute CLASSP reset benefit.

All seeds and outcomes must be retained regardless of direction.

## Decision checkpoint

- AI recommendation: `authorize-heldout`
- Rationale: the bridge passed all structural and scientific-invariant checks,
  exposed no v1 flaw, and produced a pattern for which the preregistered
  held-out gates are discriminating.
- Human decision: `authorize-heldout`
- Authorization recorded: `2026-07-23`, from the user's explicit
  “authorized, proceed” instruction
- Allowed human decisions: `authorize-heldout` or `new-protocol-required`

Validation establishes provenance, artifact integrity, frozen invariants, and
reproducibility metadata. It does not prove that the scientific interpretation
is true.
