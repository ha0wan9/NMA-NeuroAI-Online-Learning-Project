# `matched-lr-paper-v1` complete-study synthesis

- **Validation status:** passed; 24/24 full cells and receipts accepted.
- **Primary cohort:** held-out seeds `7`, `123`, `2026`, `31415`, and `271828`.
- **Legacy bridge:** seed `42`, reported separately and excluded from all primary summaries.
- **Source:** repository `482ce0b1d82cd211a74d0a0e46a6dba4668e5516`; predictive-coding submodule `5bf803c3636a39928488941f05539c70ab4df0b1`.
- **Run window:** `2026-07-22T22:23:41Z` to `2026-07-23T04:10:50Z` (20829 s; 5 h 47 min 9 s).
- **Validated source identity:** `761b8d8adcbf0a558995b38b692f3c3fd03901dccf1c58c4bc66014fe283c33e`.

All differences below use `ΔPC = PC+CLASSP − PC+Adam`, `ΔBP = BP+CLASSP − BP+Adam`, and `ΔΔ = ΔPC − ΔBP`. Values are raw accuracy/metric-scale differences; `0.01` equals one percentage point for accuracy metrics. Forgetting stays on its raw scale, where lower is better.

## Registered claim gate

**Established observation:** the registered gate failed: positive final-accuracy `ΔΔ` occurred in 1/5 held-out seeds, below the required 4/5.

**Interpretation:** this frozen Permuted-MNIST MLP configuration does not support the registered configuration-specific differential-benefit hypothesis. CLASSP reduced final accuracy relative to Adam in both PC and BP in all five held-out seeds, and final-accuracy `ΔΔ` was negative in four of five seeds. The result does not establish a general claim about CLASSP, mechanistic synergy, gradient quality, statistical significance, or biological superiority.

Secondary observations do not override the registered final-accuracy gate. Online-accuracy and adaptation `ΔΔ` were positive in 5/5 seeds. BWT `ΔΔ` was negative in 4/5. Forgetting `ΔPC` and `ΔBP` were both negative in 5/5 seeds (less forgetting than the corresponding Adam cell), while raw forgetting `ΔΔ` was positive in 5/5, meaning that reduction was smaller under PC than BP.

## Held-out seed-level interactions

### Final average accuracy

Direction: higher is better.

| Seed | ΔPC | ΔBP | ΔΔ |
|---:|---:|---:|---:|
| 7 | -0.173360 | -0.188140 | 0.014780 |
| 123 | -0.205310 | -0.161260 | -0.044050 |
| 2026 | -0.188595 | -0.158645 | -0.029950 |
| 31415 | -0.190500 | -0.146620 | -0.043880 |
| 271828 | -0.164855 | -0.110245 | -0.054610 |

### Online predict-before-update accuracy

Direction: higher is better.

| Seed | ΔPC | ΔBP | ΔΔ |
|---:|---:|---:|---:|
| 7 | -0.282746 | -0.297884 | 0.015138 |
| 123 | -0.292059 | -0.305603 | 0.013544 |
| 2026 | -0.282986 | -0.296429 | 0.013443 |
| 31415 | -0.279384 | -0.290002 | 0.010618 |
| 271828 | -0.279401 | -0.298344 | 0.018943 |

### Mean adaptation

Direction: higher is better.

| Seed | ΔPC | ΔBP | ΔΔ |
|---:|---:|---:|---:|
| 7 | -0.230510 | -0.242450 | 0.011940 |
| 123 | -0.244070 | -0.265215 | 0.021145 |
| 2026 | -0.253995 | -0.264520 | 0.010525 |
| 31415 | -0.238905 | -0.254830 | 0.015925 |
| 271828 | -0.235200 | -0.247825 | 0.012625 |

### Mean BWT

Direction: higher is better.

| Seed | ΔPC | ΔBP | ΔΔ |
|---:|---:|---:|---:|
| 7 | 0.046358 | 0.046058 | 0.000300 |
| 123 | 0.020195 | 0.082658 | -0.062463 |
| 2026 | 0.032526 | 0.078847 | -0.046321 |
| 31415 | 0.021421 | 0.081189 | -0.059768 |
| 271828 | 0.048274 | 0.125111 | -0.076837 |

### Mean forgetting

Direction: lower is better; differences are reported on the raw metric scale.

| Seed | ΔPC | ΔBP | ΔΔ |
|---:|---:|---:|---:|
| 7 | -0.045800 | -0.046058 | 0.000258 |
| 123 | -0.020195 | -0.082658 | 0.062463 |
| 2026 | -0.031895 | -0.078047 | 0.046153 |
| 31415 | -0.021421 | -0.081189 | 0.059768 |
| 271828 | -0.047658 | -0.124489 | 0.076832 |

## Held-out summaries

Sign counts are shown as positive/zero/negative.

| Metric | Contrast | Mean | Sample SD | Range | Signs +/0/− |
|---|---|---:|---:|---:|---:|
| Final average accuracy | ΔPC | -0.184524 | 0.015779 | [-0.205310, -0.164855] | 0/0/5 |
| Final average accuracy | ΔBP | -0.152982 | 0.028300 | [-0.188140, -0.110245] | 0/0/5 |
| Final average accuracy | ΔΔ | -0.031542 | 0.027336 | [-0.054610, 0.014780] | 1/0/4 |
| Online predict-before-update accuracy | ΔPC | -0.283315 | 0.005188 | [-0.292059, -0.279384] | 0/0/5 |
| Online predict-before-update accuracy | ΔBP | -0.297652 | 0.005563 | [-0.305603, -0.290002] | 0/0/5 |
| Online predict-before-update accuracy | ΔΔ | 0.014337 | 0.003046 | [0.010618, 0.018943] | 5/0/0 |
| Mean adaptation | ΔPC | -0.240536 | 0.009018 | [-0.253995, -0.230510] | 0/0/5 |
| Mean adaptation | ΔBP | -0.254968 | 0.010050 | [-0.265215, -0.242450] | 0/0/5 |
| Mean adaptation | ΔΔ | 0.014432 | 0.004244 | [0.010525, 0.021145] | 5/0/0 |
| Mean BWT | ΔPC | 0.033755 | 0.013296 | [0.020195, 0.048274] | 5/0/0 |
| Mean BWT | ΔBP | 0.082773 | 0.028100 | [0.046058, 0.125111] | 5/0/0 |
| Mean BWT | ΔΔ | -0.049018 | 0.029622 | [-0.076837, 0.000300] | 1/0/4 |
| Mean forgetting | ΔPC | -0.033394 | 0.013011 | [-0.047658, -0.020195] | 0/0/5 |
| Mean forgetting | ΔBP | -0.082488 | 0.027896 | [-0.124489, -0.046058] | 0/0/5 |
| Mean forgetting | ΔΔ | 0.049095 | 0.029392 | [0.000258, 0.076832] | 5/0/0 |

## Legacy bridge (seed 42; excluded)

| Metric | ΔPC | ΔBP | ΔΔ | Direction |
|---|---:|---:|---:|---|
| Final average accuracy | -0.156130 | -0.143855 | -0.012275 | higher is better |
| Online predict-before-update accuracy | -0.278891 | -0.295456 | 0.016565 | higher is better |
| Mean adaptation | -0.235175 | -0.248750 | 0.013575 | higher is better |
| Mean BWT | 0.062563 | 0.092184 | -0.029621 | higher is better |
| Mean forgetting | -0.061974 | -0.092179 | 0.030205 | lower is better; differences are reported on the raw metric scale |

The bridge shows a negative final-accuracy `ΔΔ` but is not confirmatory evidence because seed 42 informed configuration selection.

## Integrity and operations

- Exactly 24 fresh runner processes exited with code 0; each produced one result and one receipt.
- Every cell recorded 1,200,000 samples, 2,400 optimizer updates, a finite `21×20` accuracy matrix, matched initialization, persistent optimizer state, CUDA provenance, and the frozen source commits.
- The operational log contains 48 qualifying idle samples (two before each cell), 24 passed idle gates, 24 starts, 24 exits, 24 partial validations, and zero failed operational events.
- GPU: RTX 4090 UUID `GPU-53dcafb1-294a-ca41-713f-376e6cb56ff2`; driver `560.35.03`; CUDA runtime `12.6`; PyTorch `2.10.0+cu126`; torchvision `0.25.0+cu126`; NumPy `2.4.6`.
- Final post-study GPU state was idle: 0% utilization, 31°C, 4 MiB used, and no compute process.
- Complete validation recomputed metric integrity, verified receipts, source identity, CUDA use, and all per-seed permutation, sample-order, and initialization checksums.

Validation establishes internal consistency, provenance, artifact integrity, pairing, and reproducibility metadata. It does not prove the paper interpretation or scientific conclusion true.

## Curated artifacts

- [`complete-validation.json`](complete-validation.json): authoritative complete validator output and held-out interaction analysis.
- [`legacy-bridge-analysis.json`](legacy-bridge-analysis.json): seed-42 condition values and deltas, excluded from primary summaries.
- [`run-order.json`](run-order.json): scheduled/actual order, timestamps, runtimes, gates, and artifact names.
- [`operational-log.tsv`](operational-log.tsv): pre/post GPU telemetry, process lists, idle gates, validations, and ETA events.
- `matched-lr-paper-v1__full__*.json` and matching `*.receipt.json`: 24 immutable raw results and receipts.
- [`artifact-manifest.sha256`](artifact-manifest.sha256): SHA-256 manifest for all curated files except the manifest itself.

## Reproduction and checkpoint

One cell is reproduced from the clean source commit with:

```bash
.venv/bin/python playground/continual-mnist/run_matched_lr_validation.py \
  --mode full --condition CONDITION --seed SEED --device cuda \
  --diagnostics on --confirm-protocol matched-lr-paper-v1 \
  --output-dir NEW_IGNORED_EMPTY_DIRECTORY
```

The study is stopped at the second human checkpoint. No result was committed or pushed.
