# H1 — Width

Hypothesis: doubling hidden width from 256 to 512 may improve capacity-limited
accuracy or plasticity for BP and PC, at higher parameter, memory, and runtime
cost.

Decision gate: compare `512×2` against the shared `256×2` baseline separately
for BP and PC using paired seeds and all three evaluation settings.

Status: partial support; only PC–SplitMNIST and BP–Permuted-MNIST meet the
descriptive mean gate. Experiments: H1.E1 baseline and H1.E2 wide.
