# matched-lr-state-policy-v1 bridge intuition report

- Status: `passed`
- Validated bridge cells: `8` of `8`
- Seed: `42` (open bridge; excluded from confirmation)
- Manifest SHA-256: `9e88aac4a68a886a056a88a91826168faedd95593fa7db1df210ddd3d7459791`
- Bridge runtime: `6776.291` seconds
- Held-out runtime estimate from bridge: `33881.457` seconds

## Observations

| Condition | Online | Final | Adaptation | BWT | Forgetting ↓ | Parameter movement ΣL2 | First update mean L2 | Final state bytes | Runtime (s) | Peak CUDA bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bp-adam-persistent` | 0.836718 | 0.7571 | 0.84094 | -0.194079 | 0.194079 | 73.2084 | 0.0520584 | 2154600 | 780.005 | 82660352 |
| `bp-adam-task-reset` | 0.815172 | 0.503845 | 0.84044 | -0.458242 | 0.458242 | 69.8691 | 0.10401 | 2154600 | 786.81 | 82660352 |
| `bp-classp-persistent` | 0.541262 | 0.613245 | 0.59219 | -0.101895 | 0.1019 | 12.8522 | 0.0273367 | 1077288 | 787.885 | 81582592 |
| `bp-classp-task-reset` | 0.720824 | 0.51273 | 0.72468 | -0.341411 | 0.341411 | 25.2028 | 0.129715 | 1077288 | 778.873 | 81582592 |
| `pc-adam-persistent` | 0.839134 | 0.789415 | 0.84146 | -0.158421 | 0.158421 | 66.3525 | 0.0494054 | 2154600 | 913.721 | 84708864 |
| `pc-adam-task-reset` | 0.812738 | 0.510435 | 0.836775 | -0.447963 | 0.447963 | 62.3916 | 0.097149 | 2154600 | 896.6 | 84708864 |
| `pc-classp-persistent` | 0.560243 | 0.633285 | 0.606285 | -0.0958579 | 0.0964474 | 12.1025 | 0.0262214 | 1077288 | 920.847 | 83631104 |
| `pc-classp-task-reset` | 0.730692 | 0.611415 | 0.717125 | -0.239137 | 0.239137 | 19.2149 | 0.10943 | 1077288 | 911.551 | 83631104 |

Optimizer trajectories, per-boundary reset transients, Adam moments, CLASSP accumulators, first-batch gradients/updates, and parameter preservation checks are retained in the adjacent JSON report.

## Interpretations

- Human review pending. Record interpretations here only after checking the observations against the implementation and protocol.

## Limitations

- This is one openly inspected seed and is excluded from confirmation.
- Read-only diagnostics add synchronization, allocation, and timing overhead; neutrality tests cover scientific outputs and final parameters, not runtime or peak memory.
- A bridge anomaly can trigger a new protocol version; it cannot justify silent tuning of v1.

## Counterclaim

- Apparent reset effects could reflect optimizer transient scale rather than a method-specific continual-learning mechanism.

## Discriminating check

- Compare boundary-first gradients, update magnitudes, state trajectories, and task displacements across the matched Adam/CLASSP and BP/PC cells before authorizing held-out execution.

Validation establishes artifact integrity, provenance, frozen configuration, and recorded invariants. It does not establish that a scientific interpretation is true.
