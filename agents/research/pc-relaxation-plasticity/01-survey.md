# Code-truth survey

## Established facts

- `experiment.py` and `continual_experiment.py` expose `pc_steps` directly to
  `PCTrainer(T=...)`; no other optimizer or architecture parameter depends on it.
- Both runners use the same 784–256–256–10 architecture, one-hot half squared
  error, Adam parameter learning rate `0.001`, and PC state SGD learning rate
  `0.01`.
- Static evaluation records validation accuracy after each epoch and final
  train/test accuracy. Continual evaluation records predict-before-update
  prequential accuracy and full task matrices after every boundary.
- Existing `T=20` continual evidence is informative but cannot substitute for
  this sweep because the shared BP control and all PC conditions must be
  freshly evaluated under the same timing and artifact protocol.

## Metric interpretation

- Static plasticity is operationalized as first-epoch validation accuracy and
  the gain from initialization to epoch 1; it measures early learning speed,
  not biological synaptic plasticity directly.
- Continual plasticity is operationalized as mean task adaptation gain and
  prequential accuracy. Forgetting/BWT is reported separately as stability.
- Absolute SplitMNIST and Permuted MNIST scores are not pooled because the
  benchmarks expose different data and represent different continual regimes.

## Risks carried into design

- Three seeds describe variability but are underpowered for broad superiority
  claims.
- Relaxation depth changes compute; a small accuracy gain may not justify cost.
- A lower forgetting number can arise from weaker initial task learning, so it
  must be interpreted jointly with adaptation gain and prequential accuracy.
