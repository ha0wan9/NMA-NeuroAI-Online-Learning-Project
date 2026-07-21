# Predictive Coding Static and Continual-Learning Experiments

Branch delivery: `haoran-experiments`

This report consolidates the matched Predictive Coding (PC) and
Backpropagation (BP) experiments developed in this research session. It covers
the static-MNIST feasibility gate, continual MNIST benchmarks, relaxation-step
ablation, network-scale ablation, compute cost, negative results, protocol
corrections, and reproducibility artifacts.

## Executive conclusion

**Established fact:** PC can match BP closely on static MNIST when latent
relaxation is sufficient. One relaxation iteration is propagation-deficient;
five iterations reach the static saturation region for the tested two-layer
network. More iterations increase cost without a monotonic accuracy benefit.

**Established fact:** Continual-learning outcomes are setting-dependent.
Permuted MNIST permits BP-like PC performance, while class-incremental
SplitMNIST produces severe catastrophic forgetting for both methods. Lower
forgetting sometimes coincides with weaker task acquisition and is not, by
itself, evidence of better continual learning.

**Established fact:** Doubling network width provides narrow benefits but does
not improve every method/scenario. Doubling plain-MLP depth is harmful under
the frozen optimizer and `T=5` protocol, with a particularly large deep-PC
Permuted-MNIST failure.

**Decision:** Retain the `256×2` architecture and PC `T=5` as the
compute-efficient experimental defaults. Treat `512×2` as a targeted follow-up
for PC–SplitMNIST or BP–Permuted-MNIST. Do not promote the `256×4` architecture
under the current protocol.

No result here establishes biological superiority, general PC superiority,
or a state-of-the-art claim.

## Shared experimental foundation

| Dimension | Frozen value |
|---|---|
| Dataset | torchvision MNIST with deterministic 80/20 train/validation split |
| Split seed | `20260717` |
| Paired training seeds | `7`, `42`, `123` |
| Base architecture | `784→256→256→10`, ReLU, biases enabled |
| Base learned parameters | 269,322 |
| Objective | One-hot half squared error |
| Parameter optimizer | Adam, learning rate `0.001` |
| PC latent optimizer | SGD, learning rate `0.01` |
| Batch size | 500 |
| Replay/future access | None |
| Backend | CyberEngine ROCm GPU in a pinned rootless Podman image |

BP and PC Linear tensors are copied from the same initialization within every
paired condition. Static split indices and continual task streams are
checksum-audited. All full runs reject non-finite metrics before artifact
creation.

### Static setting

Static MNIST runs for ten epochs. Operational plasticity is represented by
epoch-1 validation gain/accuracy and the mean post-update validation curve.
Final train and test accuracy, synchronized timed processing, parameters,
latent state, and accelerator memory are also recorded.

### Continual settings

- SplitMNIST: five class-pair tasks, shared 10-way head, no model task ID.
- Permuted MNIST: identity plus four fixed permutations, no model task ID.
- One pass per task and no replay.
- Predict-before-update prequential accuracy.
- Full task test matrices after every task boundary.
- Final average accuracy, adaptation gain, forgetting, backward transfer,
  runtime, and memory.

SplitMNIST is class-incremental; Permuted MNIST is domain-incremental. Their
absolute scores are never pooled.

## Study 1 — Relaxation depth

Protocol: `pc-relaxation-sweep-v1`

Raw artifact:
[`relaxation-sweep-v1.json`](../../playground/step1-static-mnist/runs/relaxation-sweep-v1.json)

Study report:
[`pc-relaxation-plasticity/05-synthesis.md`](pc-relaxation-plasticity/05-synthesis.md)

PC relaxation was varied over `T∈{1,5,10,20}` while all other fields remained
fixed. A freshly rerun BP control was shared within each setting.

### Accuracy results

| Setting | BP | PC T=1 | PC T=5 | PC T=10 | PC T=20 |
|---|---:|---:|---:|---:|---:|
| Static final test | 98.267 | 84.457 | 98.193 | 98.183 | 98.137 |
| Split final average | 18.541 | 11.441 | 18.329 | 16.500 | 16.860 |
| Permuted final average | 91.544 | 51.008 | 91.326 | 91.806 | 91.037 |

Values are mean percentages across three paired seeds.

### Findings

1. **Propagation threshold.** The upstream two-PC-layer implementation needs
   at least three iterations to propagate error through all layers. `T=1` is a
   degenerate boundary condition, not an ordinary low-depth operating point.

2. **Static saturation.** PC `T=5/10/20` finishes within `0.13` percentage
   points of BP on the mean. Early validation gain and the whole validation
   curve are also effectively tied. Additional relaxation beyond five tested
   steps provides no material static benefit.

3. **SplitMNIST trade-off.** From `T=5→10→20`, adaptation falls
   `93.949→91.160→89.829%`, prequential accuracy falls
   `57.512→56.487→54.547%`, and measured forgetting falls
   `95.687→94.487→92.373%`. The lower forgetting is confounded by weaker
   acquisition rather than demonstrating a continual-learning win.

4. **Permuted-MNIST intermediate candidate.** `T=10` has the highest tested PC
   final mean and beats `T=5/20` within every seed. Its PC−BP paired differences
   still have mixed signs, so it is a replication candidate rather than an
   optimum or superiority result.

5. **Cost.** At `T=20`, synchronized timed processing is approximately
   `4.23×` BP on static MNIST, `4.14×` on SplitMNIST, and `3.73×` on Permuted
   MNIST. Runtime is the only consistently monotonic result.

### Relaxation decision

- Exclude `T=1` except as a failure/boundary check.
- Use `T=5` as the cheapest effective tested default for the two-layer model.
- Keep `T=10` only as a Permuted-MNIST follow-up candidate.
- A future threshold study should test `T=3/4` with more seeds.

## Study 2 — Network scale

Final protocol: `pc-network-scale-v2`

Raw artifact:
[`architecture-sweep-v2.json`](../../playground/step1-static-mnist/runs/architecture-sweep-v2.json)

Study report:
[`pc-network-scale/05-synthesis.md`](pc-network-scale/05-synthesis.md)

The experiment compared the original `256×2` network with a width intervention
(`512×2`) and a depth intervention (`256×4`). PC remained fixed at `T=5`, which
is exactly the propagation minimum for four PC layers.

### Accuracy results

| Architecture | Parameters | Static BP / PC | Split BP / PC | Permuted BP / PC |
|---|---:|---:|---:|---:|
| `256×2` | 269,322 | 98.267 / 98.193 | 18.541 / 18.329 | 91.544 / 91.326 |
| `512×2` | 669,706 | 98.490 / 98.447 | 18.917 / 18.857 | 92.534 / 91.257 |
| `256×4` | 400,906 | 98.107 / 97.533 | 12.173 / 11.750 | 83.622 / 51.385 |

Values are mean percentages across three paired seeds.

### Width findings

- Static final improvement is consistent but small: `+0.223` pp BP and
  `+0.253` pp PC, below the registered `+0.5` pp gate.
- Wide PC passes the SplitMNIST mean gate: final `+0.528` pp, adaptation
  `+2.575` pp, and prequential accuracy `+7.914` pp. All final seed changes are
  positive. Forgetting nevertheless worsens to `97.154%`.
- Wide BP improves mean Permuted-MNIST final accuracy by `+0.990` pp, but one
  paired seed is negative.
- Wide PC changes Permuted final accuracy by `−0.069` pp and trails wide BP in
  all three seeds.
- Width increases parameters `2.487×` and adds about `7.1 MiB` BP / `15.0 MiB`
  PC peak allocated memory on this runtime.

**Interpretation:** Width offers method–scenario-specific capacity benefits,
not a general scale law or default replacement.

### Depth findings

- Static final change: `−0.160` pp BP and `−0.660` pp PC.
- Split final change: `−6.367` pp BP and `−6.579` pp PC, with severe adaptation
  loss for both methods.
- Permuted final change: `−7.922` pp BP and `−39.941` pp PC.
- Deep-PC Permuted losses are large and consistent in all seeds; forgetting
  worsens by `45.733` pp.

**Interpretation:** Plain depth is optimization- and interference-prone under
the frozen learning rate and training budget. BP also degrades, so the result
is not solely a PC implementation failure. The additional PC collapse applies
to the minimally relaxed `T=5` configuration; it does not prove that deeper PC
is intrinsically impossible.

### Network-scale decision

- Retain `256×2` as the shared default.
- Use `512×2` only for targeted follow-up.
- Reject `256×4` under the frozen protocol.
- If depth remains important, pre-register a depth×relaxation (`T=5/7/10`)
  study and a separate depth-appropriate optimization/residualization study.

## Validity and negative-result preservation

The first architecture-scale full run interleaved BP and PC module creation,
changing RNG consumption and therefore the seeded `256×2` initialization.
Although BP and PC remained paired within that run, it did not reproduce the
frozen baseline mapping. The run was marked invalid, preserved as
[`architecture-sweep-v1-pre-init-order-fix.json`](../../playground/step1-static-mnist/runs/architecture-sweep-v1-pre-init-order-fix.json),
and fully rerun after restoring BP-first construction.

All three corrected v2 baseline seed checksums match the prior relaxation
study. No invalid v1 value contributes to the final network-scale conclusions.

Other superseded smoke and accounting artifacts remain indexed in
[`runs/README.md`](../../playground/step1-static-mnist/runs/README.md).

## Cross-study conclusions

1. PC needs sufficient inference iterations, but more iterations are not
   automatically better.
2. Static parity does not predict continual retention.
3. Stability and plasticity must be reported jointly; low forgetting can be a
   symptom of weak acquisition.
4. Width can improve acquisition selectively, but capacity alone does not
   solve interference.
5. Plain depth is a poor default under the current optimizer and relaxation
   protocol.
6. Compute and memory costs materially constrain PC settings that yield only
   small accuracy differences.

## Limitations

- Three paired seeds provide descriptive mean and sample SD, not broad
  statistical superiority.
- MNIST, SplitMNIST, and Permuted MNIST are controlled baselines rather than
  real-world data.
- Batch size 500 is mini-batch continual learning, not strict sample-at-a-time
  online learning.
- Hyperparameters were matched rather than separately optimized per method,
  relaxation depth, or architecture.
- `T=5` has no relaxation margin for the four-layer PC network.
- Timings include training-loop work and are CyberEngine/order-specific.
- Peak allocated memory depends on the runtime; learned parameter counts are
  architecture facts.
- Operational plasticity metrics are not direct biological evidence.

## Reproduction

Relaxation sweep:

```bash
bash playground/step1-static-mnist/remote-run-relaxation-sweep.sh \
  --relaxation-steps 1 5 10 20 \
  --seeds 7 42 123 \
  --output playground/step1-static-mnist/runs/relaxation-sweep-reproduction.json
```

Network-scale sweep:

```bash
bash playground/step1-static-mnist/remote-run-architecture-sweep.sh \
  --seeds 7 42 123 \
  --pc-steps 5 \
  --output playground/step1-static-mnist/runs/architecture-sweep-v2-reproduction.json
```

Deterministic parsers:

```bash
python3 playground/step1-static-mnist/relaxation_analysis.py \
  --input playground/step1-static-mnist/runs/relaxation-sweep-v1.json \
  --output-dir agents/research/pc-relaxation-plasticity/artifacts

python3 playground/step1-static-mnist/architecture_analysis.py \
  --input playground/step1-static-mnist/runs/architecture-sweep-v2.json \
  --output-dir agents/research/pc-network-scale/artifacts
```

## Artifact map

- Experiment harness and CyberEngine instructions:
  [`playground/step1-static-mnist/README.md`](../../playground/step1-static-mnist/README.md)
- Relaxation protocol, ledger, review, and synthesis:
  [`pc-relaxation-plasticity/`](pc-relaxation-plasticity/)
- Network-scale protocol, invalid-run record, review, and synthesis:
  [`pc-network-scale/`](pc-network-scale/)
- Raw and superseded run index:
  [`playground/step1-static-mnist/runs/README.md`](../../playground/step1-static-mnist/runs/README.md)
