# CIFAR-10 reproduction design

## Frozen experiment table

| ID | Experiment | Conditions | Seeds | Primary gate | GPU cap |
|---|---|---|---|---|---:|
| H1.E1 | Paper smoke/calibration | RBP + PC; 1 pass; 40 train/class; 20 test/class | 1482555873 | finite metrics, matching architecture count, PC objective descent | 0.3 h |
| H1.E2 | Fig. 4i reproduction | RBP + PC; source winning LR per method/seed; 80 passes | paper 3 | each best accuracy within ±1 pp of archived target | 4.7 h |
| H2.E1 | Split smoke | RBP + PC; 40 train/eval per task | 42 | paired initialization/stream, finite metrics, PC objective descent | part of reserve |
| H2.E2 | SplitCIFAR-10 | RBP + PC; one pass over five tasks | 7,42,123 | valid paired descriptive comparison | 3.0 h |
| H2.E3 | Representation diagnostics | Observational fixed-anchor probe, then matched H2.E2 rerun | probe 42; full 7,42,123 | TensorBoard RDM/MDS/drift figures exist; behavior and checksums remain matched | <=0.3 h |

## Fairness and interpretation

- H1 reproduces the paper's own test-informed selection. It is not used as a
  leakage-free model-selection result.
- H2 freezes method-specific learning rates before launch using only archived
  H1 source data; no Split validation or test result changes them.
- BP/PC learned tensors are identical within each H2 seed. H1 reconstructs
  each method from the same paper seed and source RNG order.
- H2 reports plasticity and stability together. Lower forgetting caused by
  weaker acquisition is not a benefit.
- Static and class-incremental accuracies remain separate.
- H2.E3 diagnostics are evaluation-only: 20 fixed test anchors/class selected
  by an isolated RNG, feedforward learned-layer hooks in evaluation mode, no
  checkpoint selection, and no diagnostic used for optimizer updates.

## Calibration decision rule

After smoke, compute projected full cost from synchronized update time. Launch
H1.E2 only if its projected total plus H2.E2 remains inside ten GPU-hours. If
not, run the maximum preregistered whole-seed subset that fits and label H1 a
partial reproduction. Do not shorten H1 epochs after seeing accuracy.

## Design self-review

Verdict: pass with warnings.

- Pass: primary source target, source revision, architecture, update schedule,
  seeds, learning rates, data quirks, metrics, and stop rules are explicit.
- Warning: local library revision is newer than the nested paper copy.
- Warning: H1 deliberately retains test-informed selection and shuffled
  drop-last evaluation because those are part of the target protocol.
- Warning: H2 fixed learning rates are paper-derived rather than tuned for
  continual learning; conclusions apply to this frozen transfer.
