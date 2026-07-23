# Network-Scale Experiment Report

## Answer

A bigger network is not uniformly better. Doubling width to `512×2` gives two
narrow benefits: PC improves SplitMNIST final accuracy and acquisition, while
BP improves mean Permuted-MNIST accuracy. The same width change produces only
small static gains and does not improve PC final Permuted-MNIST accuracy.
Doubling depth to `256×4` is harmful for both methods under the fixed protocol,
with a particularly large deep-PC Permuted-MNIST collapse. The evidence does
not justify replacing the `256×2` architecture as the project default.

## Protocol

- Architectures: `256×2` control, `512×2` width intervention, `256×4` depth
  intervention.
- Methods: matched BP and PC; PC relaxation fixed at `T=5`.
- Seeds: paired `7`, `42`, and `123`.
- Static: full MNIST, ten epochs.
- Continual: SplitMNIST and Permuted MNIST, one pass/task, no replay, no model
  task ID.
- Frozen: data, order, task streams, model objective, optimizer, learning
  rates, batch size, epochs, evaluation cadence, and metrics.
- Backend: CyberEngine ROCm GPU; exact runtime provenance in raw JSON.

Complete tables are in [`artifacts/tables.md`](artifacts/tables.md), with
evidence-level interpretation in [`04-evaluation.md`](04-evaluation.md).

## Main findings

| Architecture | Parameters | Static BP / PC | Split BP / PC | Permuted BP / PC |
|---|---:|---:|---:|---:|
| `256×2` | 269,322 | 98.267 / 98.193 | 18.541 / 18.329 | 91.544 / 91.326 |
| `512×2` | 669,706 | 98.490 / 98.447 | 18.917 / 18.857 | 92.534 / 91.257 |
| `256×4` | 400,906 | 98.107 / 97.533 | 12.173 / 11.750 | 83.622 / 51.385 |

Values are mean accuracy percentages across three seeds.

### Width helps selectively

**Established fact:** Wide PC passes the registered SplitMNIST gate with
`+0.528` pp final accuracy, `+2.575` pp adaptation, and positive final changes
in all seeds. Prequential accuracy rises `+7.914` pp, but forgetting worsens
and remains catastrophic.

**Established fact:** Wide BP gains `+0.990` pp mean Permuted-MNIST accuracy
and `+2.041` pp adaptation, but one paired seed is negative. Wide PC changes
Permuted final accuracy by `−0.069` pp and trails wide BP in all three seeds.

**Interpretation:** Capacity helps particular method–scenario cells, not the
whole evaluation suite. The `2.49×` parameter increase is not justified as a
general replacement by these narrow gains.

### Depth is a negative result under this configuration

**Established fact:** The deep architecture lowers final accuracy for BP and
PC in every scenario mean. Deep PC loses `39.941` pp on Permuted MNIST versus
the shallow PC baseline, consistently across seeds, and its forgetting rises
`45.733` pp.

**Interpretation:** Plain depth with the frozen optimizer/training budget is
optimization- and interference-prone. Because BP also degrades, this is not
solely a PC implementation failure. PC suffers an additional severe continual
penalty, but `T=5` is only minimally sufficient for four PC layers.

## Decision

- Retain `256×2` as the default experimental architecture.
- Keep `512×2` only as a targeted follow-up for PC–SplitMNIST or
  BP–Permuted-MNIST; do not claim a general width benefit.
- Reject `256×4` under the current protocol.
- If deeper PC remains scientifically important, register a separate
  depth×relaxation study (`T=5/7/10`) and a depth-appropriate optimization or
  residual-architecture study. Do not retroactively tune this artifact.

## Validity event

The first architecture run (`pc-network-scale-v1`) changed RNG consumption by
interleaving BP/PC construction. It is preserved and marked invalid. The final
v2 run restores BP-first construction; all three control-seed checksums match
the prior frozen study. No v1 metric appears in this report.

## Limitations

- `n=3`; the gates are descriptive rather than statistical superiority tests.
- Only one width and one depth intervention were tested.
- The same learning rate and training budget may disadvantage larger/deeper
  networks.
- `T=5` is the minimum propagation depth for the four-PC-layer model, not a
  matched relaxation-margin condition.
- Low SplitMNIST forgetting can reflect weak acquisition; adaptation,
  prequential, and final accuracy must be read jointly.
- Static, class-incremental, and domain-incremental scores are not pooled.
- Timings include training-loop work and are hardware/order-specific.
- Peak allocated memory is runtime-specific; parameter-count differences are
  architecture accounting facts.
- These operational plasticity metrics are not biological evidence.

## Reproduction

```bash
bash playground/step1-static-mnist/remote-run-architecture-sweep.sh \
  --seeds 7 42 123 \
  --pc-steps 5 \
  --output playground/step1-static-mnist/runs/architecture-sweep-v2-reproduction.json

python3 playground/step1-static-mnist/architecture_analysis.py \
  --input playground/step1-static-mnist/runs/architecture-sweep-v2-reproduction.json \
  --output-dir agents/research/pc-network-scale/artifacts/reproduction
```

## Research Graph

Graph results `R-width-split`, `R-width-permuted`, and `R-depth-negative`
support decision `D-keep-baseline`: retain `256×2`, use width only for targeted
follow-up, and reject depth under this frozen protocol. See
[`research_graph.mmd`](research_graph.mmd) and
[`research_graph.json`](research_graph.json).
