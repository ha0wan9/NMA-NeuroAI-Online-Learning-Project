# matched-lr-state-policy-v1 smoke and optimizer-state report

- Status: `passed`
- Validated cells: `16`
- Complete passes: `2`
- Exact scientific repetition: `True`
- Manifest SHA-256: `9e88aac4a68a886a056a88a91826168faedd95593fa7db1df210ddd3d7459791`

The values below are commissioning diagnostics from open bridge seed 42, not held-out evidence.

| Condition | Final steps | State bytes | Adam m L2 | Adam v L2 | CLASSP sum | CLASSP L2 | CLASSP nonzero | Task displacement L2 sum | First updates L2 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bp-adam-persistent` | 4..4 | 2154600 | 83.8137 | 39.4455 | — | — | — | 0.449014 | 0.120416 |
| `bp-adam-task-reset` | 2..2 | 2154600 | 44.6665 | 13.854 | — | — | — | 0.490266 | 0.135059 |
| `bp-classp-persistent` | 4..4 | 1077288 | — | — | 314839 | 41018.8 | 0.919401 | 0.35253 | 0.112222 |
| `bp-classp-task-reset` | 2..2 | 1077288 | — | — | 126576 | 15136.3 | 0.741347 | 0.394496 | 0.127923 |
| `pc-adam-persistent` | 4..4 | 2154600 | 68.3503 | 37.5955 | — | — | — | 0.450339 | 0.120881 |
| `pc-adam-task-reset` | 2..2 | 2154600 | 33.2751 | 13.1045 | — | — | — | 0.493017 | 0.136058 |
| `pc-classp-persistent` | 4..4 | 1077288 | — | — | 194861 | 40676.3 | 0.552372 | 0.251843 | 0.0772937 |
| `pc-classp-task-reset` | 2..2 | 1077288 | — | — | 76336.9 | 15626.3 | 0.392215 | 0.275392 | 0.0856745 |

Validation establishes exact repeated outputs and recorded optimizer invariants on this synthetic-size smoke configuration. It does not establish a scientific effect.
