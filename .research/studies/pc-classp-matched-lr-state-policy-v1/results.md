# PC×CLASSP matched-LR optimizer-state-policy results

- **Protocol ID:** `matched-lr-state-policy-v1`
- **Study status:** complete validation passed; human conclusion approved
- **Conclusion approved:** 2026-07-24
- **Frozen source commit:** `fa7ba098ab5ce89636137554aa0ea55f9419b50b`
- **Manifest SHA-256:**
  `9e88aac4a68a886a056a88a91826168faedd95593fa7db1df210ddd3d7459791`
- **Validated full cells:** 48/48
- **Open bridge seed:** `42`, reported separately and excluded from
  confirmation
- **Held-out seeds:** `7`, `123`, `2026`, `31415`, and `271828`

This is the tracked scientific result for the frozen
[`protocol.md`](protocol.md). The
[curated raw snapshot](../../../playground/continual-mnist/results/matched-lr-state-policy-v1/)
is an exact copy of the ignored local execution tree and preserves individual
cells, receipts, diagnostics, order, logs, and validation artifacts.

## Question and intervention

The study tested whether clearing parameter-optimizer state at task boundaries
changes continual learning differently for Adam and CLASSP, and whether that
optimizer-specific reset effect differs between Predictive Coding (PC) and
Backpropagation (BP):

```text
{BP, PC} × {Adam, CLASSP} × {persistent, task-reset}
```

All conditions used the same 20-task domain-incremental Permuted-MNIST stream,
MLP `784→256→256→10`, fixed parameter learning rate `3e-4`, one pass per task,
batch size `500`, no replay, no task-ID input, and matched per-seed
initialization and sample order.

Network parameters always persisted. The task-reset intervention cleared only
the existing parameter optimizer's state before tasks 1–19. It did not
reinitialize the model, optimizer object, or parameter groups. PC latent state
reset every batch under both parameter-optimizer policies.

For each method `m`, optimizer `o`, metric `Y`, and seed `r`, the direct reset
effect was:

```text
δ(m,o;r) = Y(m,o,task-reset;r) − Y(m,o,persistent;r)
```

The registered relative contrasts were:

```text
R_PC(r) = δ(PC,CLASSP;r) − δ(PC,Adam;r)
R_BP(r) = δ(BP,CLASSP;r) − δ(BP,Adam;r)
Ω(r)    = R_PC(r) − R_BP(r)
```

## Execution and validation observations

- Both seed-42 smoke passes completed all eight conditions. Their scientific
  outputs, optimizer-state audits, and final parameter values matched exactly.
- The open seed-42 bridge completed 8/8 cells in 6,776.291 seconds
  (approximately 1 hour 53 minutes).
- The held-out stage completed 40/40 cells in 34,019.378 seconds
  (approximately 9 hours 27 minutes).
- No full cell failed, remained incomplete, or lacked its paired receipt.
- Complete validation passed for all 48 full cells, including source, dataset,
  stream, initialization, count, state-policy, hardware, order, pairing, and
  artifact-hash invariants.
- The intended and actual full-cell orders matched.

Validation establishes provenance, artifact integrity, frozen invariants, and
reproducibility metadata. It does not establish that the scientific
interpretation is true.

## Confirmatory final-accuracy observations

Values are held-out means of task-reset minus persistent state, in percentage
points. Final average accuracy is higher-is-better.

| Configuration | Mean reset effect | Seed directions |
|---|---:|---:|
| BP–Adam | -17.2615 | negative 5/5 |
| BP–CLASSP | -7.2963 | negative 5/5 |
| PC–Adam | -25.3206 | negative 5/5 |
| PC–CLASSP | +0.7706 | positive 5/5 |

The PC–CLASSP direct effect ranged from +0.0635 to +1.8890 percentage
points. It was the only positive final-accuracy reset effect among the four
method–optimizer configurations.

The registered relative contrasts were:

| Contrast | Held-out mean | Seed directions |
|---|---:|---:|
| `R_BP` | +9.9652 | positive 5/5 |
| `R_PC` | +26.0912 | positive 5/5 |
| `Ω` | +16.1260 | positive 5/5 |

Both hierarchical gates passed:

1. Gate 1 required positive `R_PC` in at least four of five held-out seeds and
   passed 5/5.
2. Gate 2 was therefore evaluated. It required positive `Ω` in at least four
   of five held-out seeds and passed 5/5.

These large relative contrasts are not direct PC–CLASSP gains. For example,
the mean `R_PC` combines the small positive PC–CLASSP reset effect with the
large negative PC–Adam reset effect.

Open bridge seed 42 had a -2.1870-point direct PC–CLASSP reset effect, unlike
the five slightly positive held-out effects. Its `R_PC` and `Ω` directions
were nevertheless positive. This difference reinforces why seed 42 was
excluded from confirmation.

## Plasticity and retention observations

Mean held-out reset effects, in percentage points:

| Metric | BP–Adam | BP–CLASSP | PC–Adam | PC–CLASSP |
|---|---:|---:|---:|---:|
| Online predict-before-update accuracy ↑ | -2.4899 | +17.8588 | -3.1251 | +17.2175 |
| Mean adaptation ↑ | +0.0688 | +13.9613 | +0.3696 | +11.7646 |
| Mean BWT ↑ | -17.8552 | -21.1331 | -26.1914 | -11.4026 |
| Mean forgetting ↓ | +17.8552 | +21.1046 | +26.1914 | +11.3665 |

Task reset increased CLASSP online accuracy and adaptation under both learning
methods. Its online `Ω` was approximately zero and had mixed signs across
seeds, so the immediate plasticity improvement was not PC-specific.

Reset worsened BWT and forgetting for all four method–optimizer combinations.
The intervention therefore exposed a plasticity–stability tradeoff rather than
a general improvement.

## Approved interpretation

Under this frozen protocol, clearing CLASSP state appears to restore effective
plasticity at task transitions under both BP and PC. The distinctive
PC–CLASSP result appears in the final balance: it incurred a smaller retention
penalty than the other PC reset condition, allowing its adaptation benefit to
produce a small positive final-average-accuracy effect.

The approved scoped conclusion is:

> Across five held-out seeds in the 20-task Permuted-MNIST protocol, resetting
> optimizer state increased immediate adaptation for CLASSP but increased
> forgetting for every tested method–optimizer combination. Predictive Coding
> with CLASSP experienced the smallest retention penalty and consequently
> obtained a small positive final-average-accuracy reset effect of +0.77
> percentage points. The positive interaction contrast was consistent across
> all five seeds, but it does not establish a special mechanism, statistical
> significance, biological superiority, or generalization beyond this
> protocol.

## Counterclaim and limitations

A credible alternative is that the interaction arises from optimizer
transition dynamics—especially Adam's severe reset sensitivity—rather than a
special PC×CLASSP continual-learning mechanism. Reset changes the first
updates after task boundaries, and the present factorial distinguishes
configurations rather than isolating that mechanism.

The study does not establish:

- statistical significance or a population-level effect;
- a causal PC×CLASSP mechanism or biological synergy;
- reset as a generally superior continual-learning strategy;
- generality beyond the fixed learning rate, MLP, Permuted-MNIST stream, and
  five held-out seeds.

The most discriminating follow-up is a new preregistered protocol that matches
first-update magnitude across reset and persistent conditions and separately
ablates Adam moments, Adam's step counter, and CLASSP accumulator state. It
must be a new protocol and must not modify v1.

## Reproduction and provenance

Structural status:

```bash
.venv/bin/python \
  playground/continual-mnist/run_state_policy_protocol.py status
```

Complete validation:

```bash
.venv/bin/python \
  playground/continual-mnist/run_state_policy_protocol.py validate \
  --stage complete
```

Curated raw-snapshot integrity:

```bash
cd playground/continual-mnist/results/matched-lr-state-policy-v1
sha256sum --check reports/artifact-manifest.sha256
```

Aggregate artifact identities:

| Artifact | SHA-256 |
|---|---|
| Smoke optimizer-state report | `94a9a5198450bc60d3636b44a67e97329702eb668cdc7a24d5510fb95603cbe9` |
| Reviewed bridge Markdown report | `39d743c44b67677c6c179baddb13dfd0994ffeb68a2fcda3358efc606ec18e26` |
| Complete synthesis Markdown | `3056bebd126b6f4825922e14b79d6cd98cc2aa532c555a799a15cfecf3831b30` |
| Complete validation JSON | `51422b4bc02cbd8f3b2bc3c695f4701c567d45d31d85a59a37c5088b7998669b` |
| Artifact SHA-256 manifest | `694c78828262f2f1cf96dbdf5ddd551919e44aee8ea3262a31432285d6b5c7a9` |

The interactive companion is
[`playground/continual-mnist/matched-lr-state-policy-results.html`](../../../playground/continual-mnist/matched-lr-state-policy-results.html).
