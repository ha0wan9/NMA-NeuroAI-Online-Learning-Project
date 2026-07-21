# H1 — Relaxation depth

Working hypothesis: increasing PC relaxation steps changes effective inference
quality and therefore learning speed, final accuracy, continual adaptation,
stability, and cost. The direction is not assumed to be uniformly beneficial.

Decision gate: report paired three-seed differences at every level; identify a
monotonic trend only when all adjacent means agree; any recommended default
must also disclose runtime and stability–plasticity trade-offs.

Status: evaluated; trade-off supported, but no global default promoted.

Experiments: `H1.E1` static MNIST, `H1.E2` SplitMNIST, and `H1.E3` Permuted MNIST.
