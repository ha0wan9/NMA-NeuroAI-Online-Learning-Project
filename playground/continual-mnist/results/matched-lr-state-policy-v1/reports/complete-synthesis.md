# matched-lr-state-policy-v1 complete validation and synthesis

- Status: `passed`
- Validated cells: `48` of `48`
- Manifest SHA-256: `9e88aac4a68a886a056a88a91826168faedd95593fa7db1df210ddd3d7459791`
- Source identity: `12ac2a6a8fce44343ae842e885fe01ee5a1783194a5453878f660fa47684851d`
- Bridge seed excluded from confirmation: `42`

Validation establishes artifact integrity, pairing, provenance, frozen configuration, and recorded state-policy invariants. It does not by itself establish that the scientific interpretation is true.

## Hierarchical final-accuracy gates

- Gate 1: `True` (5/5 positive held-out seeds).
- Gate 2 evaluated: `True`; result: `True` (5/5 positive held-out seeds before hierarchical masking).
- Claim ceiling: No statistical-significance, mechanistic-synergy, biological-superiority, or general-validity claim follows from these gates.

## final_average_accuracy

higher is better.

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | -0.203385 | -0.05281 | -0.257055 | 0.00306 | 0.150575 | 0.260115 | 0.10954 |
| 123 | -0.157935 | -0.080035 | -0.25397 | 0.000635 | 0.0779 | 0.254605 | 0.176705 |
| 2026 | -0.18114 | -0.08726 | -0.272005 | 0.006045 | 0.09388 | 0.27805 | 0.18417 |
| 31415 | -0.165685 | -0.07772 | -0.259625 | 0.01889 | 0.087965 | 0.278515 | 0.19055 |
| 271828 | -0.15493 | -0.06699 | -0.223375 | 0.0099 | 0.08794 | 0.233275 | 0.145335 |

Bridge seed 42 (reported, excluded from summaries):

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.253255 | -0.100515 | -0.27898 | -0.02187 | 0.15274 | 0.25711 | 0.10437 |

## online_predict_before_update_accuracy

higher is better.

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | -0.02606 | 0.178434166667 | -0.0270008333333 | 0.173351666667 | 0.204494166667 | 0.2003525 | -0.00414166666667 |
| 123 | -0.0244475 | 0.18041 | -0.0313658333333 | 0.174386666667 | 0.2048575 | 0.2057525 | 0.000895 |
| 2026 | -0.0248416666667 | 0.17868 | -0.0293008333333 | 0.171685833333 | 0.203521666667 | 0.200986666667 | -0.002535 |
| 31415 | -0.022365 | 0.175774166667 | -0.0362033333333 | 0.17028 | 0.198139166667 | 0.206483333333 | 0.00834416666667 |
| 271828 | -0.0267816666667 | 0.179643333333 | -0.0323858333333 | 0.171170833333 | 0.206425 | 0.203556666667 | -0.00286833333333 |

Bridge seed 42 (reported, excluded from summaries):

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.021545 | 0.1795625 | -0.0263958333333 | 0.170449166667 | 0.2011075 | 0.196845 | -0.0042625 |

## mean_adaptation

higher is better.

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.002745 | 0.134275 | 0.00668 | 0.112105 | 0.13153 | 0.105425 | -0.026105 |
| 123 | -0.00196 | 0.13984 | -0.00303 | 0.1164 | 0.1418 | 0.11943 | -0.02237 |
| 2026 | -0.00157 | 0.14178 | 0.006635 | 0.123915 | 0.14335 | 0.11728 | -0.02607 |
| 31415 | -0.002315 | 0.142095 | 0.0034 | 0.122585 | 0.14441 | 0.119185 | -0.025225 |
| 271828 | 0.00654 | 0.140075 | 0.004795 | 0.113225 | 0.133535 | 0.10843 | -0.025105 |

Bridge seed 42 (reported, excluded from summaries):

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.0005 | 0.13249 | -0.004685 | 0.11084 | 0.13299 | 0.115525 | -0.017465 |

## mean_bwt

higher is better.

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | -0.210694736842 | -0.189021052632 | -0.267089473684 | -0.120405263158 | 0.0216736842105 | 0.146684210526 | 0.125010526316 |
| 123 | -0.162184210526 | -0.222563157895 | -0.262226315789 | -0.125626315789 | -0.0603789473684 | 0.1366 | 0.196978947368 |
| 2026 | -0.187189473684 | -0.227436842105 | -0.281184210526 | -0.116952631579 | -0.0402473684211 | 0.164231578947 | 0.204478947368 |
| 31415 | -0.171563157895 | -0.211868421053 | -0.268094736842 | -0.0993473684211 | -0.0403052631579 | 0.168747368421 | 0.209052631579 |
| 271828 | -0.161126315789 | -0.205763157895 | -0.230973684211 | -0.1078 | -0.0446368421053 | 0.123173684211 | 0.167810526316 |

Bridge seed 42 (reported, excluded from summaries):

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.264163157895 | -0.239515789474 | -0.289542105263 | -0.143278947368 | 0.0246473684211 | 0.146263157895 | 0.121615789474 |

## mean_forgetting

lower is better; all differences remain on the raw forgetting scale.

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.210694736842 | 0.189021052632 | 0.267089473684 | 0.119847368421 | -0.0216736842105 | -0.147242105263 | -0.125568421053 |
| 123 | 0.162184210526 | 0.222563157895 | 0.262226315789 | 0.125626315789 | 0.0603789473684 | -0.1366 | -0.196978947368 |
| 2026 | 0.187189473684 | 0.226636842105 | 0.281184210526 | 0.116321052632 | 0.0394473684211 | -0.164863157895 | -0.204310526316 |
| 31415 | 0.171563157895 | 0.211868421053 | 0.268094736842 | 0.0993473684211 | 0.0403052631579 | -0.168747368421 | -0.209052631579 |
| 271828 | 0.161126315789 | 0.205142105263 | 0.230973684211 | 0.107184210526 | 0.0440157894737 | -0.123789473684 | -0.167805263158 |

Bridge seed 42 (reported, excluded from summaries):

| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | R BP | R PC | Ω |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.264163157895 | 0.239510526316 | 0.289542105263 | 0.142689473684 | -0.0246526315789 | -0.146852631579 | -0.1222 |

## Runtime and peak memory

All cells used the one frozen local RTX 4090 identity.

| Seed | Condition | Runtime (s) | Peak CUDA bytes |
|---:|---|---:|---:|
| 7 | `bp-adam-persistent` | 784.679314 | 82660352 |
| 42 | `bp-adam-persistent` | 780.004916 | 82660352 |
| 123 | `bp-adam-persistent` | 809.647806 | 82660352 |
| 2026 | `bp-adam-persistent` | 801.355382 | 82660352 |
| 31415 | `bp-adam-persistent` | 789.681831 | 82660352 |
| 271828 | `bp-adam-persistent` | 789.078361 | 82660352 |
| 7 | `bp-adam-task-reset` | 787.753939 | 82660352 |
| 42 | `bp-adam-task-reset` | 786.810355 | 82660352 |
| 123 | `bp-adam-task-reset` | 785.620810 | 82660352 |
| 2026 | `bp-adam-task-reset` | 791.313281 | 82660352 |
| 31415 | `bp-adam-task-reset` | 787.805392 | 82660352 |
| 271828 | `bp-adam-task-reset` | 791.906754 | 82660352 |
| 7 | `bp-classp-persistent` | 794.544576 | 81582592 |
| 42 | `bp-classp-persistent` | 787.884797 | 81582592 |
| 123 | `bp-classp-persistent` | 798.627686 | 81582592 |
| 2026 | `bp-classp-persistent` | 791.582054 | 81582592 |
| 31415 | `bp-classp-persistent` | 796.374618 | 81582592 |
| 271828 | `bp-classp-persistent` | 782.980513 | 81582592 |
| 7 | `bp-classp-task-reset` | 793.423418 | 81582592 |
| 42 | `bp-classp-task-reset` | 778.872744 | 81582592 |
| 123 | `bp-classp-task-reset` | 785.440062 | 81582592 |
| 2026 | `bp-classp-task-reset` | 784.857140 | 81582592 |
| 31415 | `bp-classp-task-reset` | 783.647691 | 81582592 |
| 271828 | `bp-classp-task-reset` | 787.747298 | 81582592 |
| 7 | `pc-adam-persistent` | 917.960617 | 84708864 |
| 42 | `pc-adam-persistent` | 913.720531 | 84708864 |
| 123 | `pc-adam-persistent` | 909.111075 | 84708864 |
| 2026 | `pc-adam-persistent` | 911.710512 | 84708864 |
| 31415 | `pc-adam-persistent` | 910.319366 | 84708864 |
| 271828 | `pc-adam-persistent` | 909.333068 | 84708864 |
| 7 | `pc-adam-task-reset` | 914.379127 | 84708864 |
| 42 | `pc-adam-task-reset` | 896.599540 | 84708864 |
| 123 | `pc-adam-task-reset` | 904.156306 | 84708864 |
| 2026 | `pc-adam-task-reset` | 908.139404 | 84708864 |
| 31415 | `pc-adam-task-reset` | 899.063040 | 84708864 |
| 271828 | `pc-adam-task-reset` | 912.716455 | 84708864 |
| 7 | `pc-classp-persistent` | 901.838842 | 83631104 |
| 42 | `pc-classp-persistent` | 920.847087 | 83631104 |
| 123 | `pc-classp-persistent` | 912.123342 | 83631104 |
| 2026 | `pc-classp-persistent` | 909.681697 | 83631104 |
| 31415 | `pc-classp-persistent` | 905.071569 | 83631104 |
| 271828 | `pc-classp-persistent` | 902.953651 | 83631104 |
| 7 | `pc-classp-task-reset` | 913.375985 | 83631104 |
| 42 | `pc-classp-task-reset` | 911.551372 | 83631104 |
| 123 | `pc-classp-task-reset` | 915.345729 | 83631104 |
| 2026 | `pc-classp-task-reset` | 914.662427 | 83631104 |
| 31415 | `pc-classp-task-reset` | 912.566832 | 83631104 |
| 271828 | `pc-classp-task-reset` | 916.801335 | 83631104 |
