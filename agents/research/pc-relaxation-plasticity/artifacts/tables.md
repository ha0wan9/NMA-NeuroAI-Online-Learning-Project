# Relaxation sweep deterministic tables

Protocol: `pc-relaxation-sweep-v1`; seeds: `[7, 42, 123]`.
Values are mean ± sample SD. Accuracy-like metrics are percentage points.
Runtime is synchronized timed training-loop/stream-processing seconds per
1,000 timed samples; it includes loading/transfers and, in continual runs,
predict-before-update work. One
diagnostic batch is excluded symmetrically from each static run and each
continual task.

## Static MNIST

| Method | Final test | Epoch-1 validation | Epoch-1 gain | Curve mean | s/1k updates |
|---|---:|---:|---:|---:|---:|
| BP | 98.267 ± 0.154 | 94.742 ± 0.160 | 84.592 ± 0.434 | 97.294 ± 0.055 | 0.021 ± 0.000 |
| PC T=1 | 84.457 ± 0.175 | 71.983 ± 1.785 | 61.833 ± 1.214 | 80.160 ± 0.216 | 0.021 ± 0.000 |
| PC T=5 | 98.193 ± 0.120 | 94.764 ± 0.128 | 84.614 ± 0.464 | 97.295 ± 0.062 | 0.037 ± 0.001 |
| PC T=10 | 98.183 ± 0.117 | 94.764 ± 0.158 | 84.614 ± 0.435 | 97.321 ± 0.057 | 0.053 ± 0.000 |
| PC T=20 | 98.137 ± 0.199 | 94.825 ± 0.123 | 84.675 ± 0.469 | 97.285 ± 0.090 | 0.087 ± 0.005 |

## SplitMNIST

| Method | Final average | Prequential | Adaptation gain | Forgetting | BWT | s/1k updates |
|---|---:|---:|---:|---:|---:|---:|
| BP | 18.541 ± 0.274 | 60.144 ± 0.572 | 94.561 ± 1.079 | 96.188 ± 0.296 | -96.188 ± 0.296 | 0.022 ± 0.002 |
| PC T=1 | 11.441 ± 1.465 | 12.877 ± 1.501 | 56.239 ± 0.957 | 57.160 ± 2.361 | -57.160 ± 2.361 | 0.022 ± 0.000 |
| PC T=5 | 18.329 ± 0.115 | 57.512 ± 1.264 | 93.949 ± 1.060 | 95.687 ± 0.314 | -95.687 ± 0.314 | 0.039 ± 0.002 |
| PC T=10 | 16.500 ± 3.896 | 56.487 ± 0.758 | 91.160 ± 4.162 | 94.487 ± 1.644 | -94.487 ± 1.644 | 0.059 ± 0.001 |
| PC T=20 | 16.860 ± 0.619 | 54.547 ± 2.756 | 89.829 ± 3.559 | 92.373 ± 3.595 | -92.373 ± 3.595 | 0.092 ± 0.009 |

## Permuted MNIST

| Method | Final average | Prequential | Adaptation gain | Forgetting | BWT | s/1k updates |
|---|---:|---:|---:|---:|---:|---:|
| BP | 91.544 ± 0.427 | 86.955 ± 0.270 | 84.294 ± 0.309 | 4.699 ± 0.711 | -4.699 ± 0.711 | 0.025 ± 0.000 |
| PC T=1 | 51.008 ± 1.099 | 47.504 ± 1.619 | 56.722 ± 1.146 | 22.433 ± 2.386 | -22.433 ± 2.386 | 0.026 ± 0.000 |
| PC T=5 | 91.326 ± 0.499 | 87.010 ± 0.385 | 84.103 ± 0.516 | 4.964 ± 0.653 | -4.964 ± 0.653 | 0.042 ± 0.000 |
| PC T=10 | 91.806 ± 0.566 | 87.129 ± 0.264 | 83.569 ± 0.979 | 4.383 ± 0.881 | -4.383 ± 0.881 | 0.060 ± 0.002 |
| PC T=20 | 91.037 ± 1.147 | 87.046 ± 0.436 | 84.815 ± 0.204 | 5.317 ± 1.491 | -5.317 ± 1.491 | 0.092 ± 0.002 |
