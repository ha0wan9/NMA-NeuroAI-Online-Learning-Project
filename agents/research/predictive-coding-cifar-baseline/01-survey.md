# Source and code-truth survey

## Primary source

Song et al., *Inferring neural activity before plasticity as a foundation for
learning beyond backpropagation*, Nature Neuroscience 27, 348–358 (2024),
Fig. 4i-j. Official reproduction repository revision:
`625d52678b3d4cab2dbc403b1d6ee0cfc004aea0`.

## Established code truth

- Fig. 4i uses CIFAR-10, a `3→64→128` stride-two convolutional stack and an
  `8192→512→10` classifier, ReLU, no data augmentation, and half squared error.
- Archived configuration: batch 200, `T=16`, latent SGD `lr=0.5`, Adam,
  weight decay `0.01`, three seeds, six parameter learning rates, and 80
  training passes. The Methods prose says 64 epochs generally; the executable
  Fig. 4i configuration says 80, so reproduction follows the figure-specific
  code and records the discrepancy.
- Both the RBP control and PC condition use `PCTrainer` with parameter updates
  at every one of 16 steps. RBP has no PC latent layers but still receives 16
  repeated Adam updates per minibatch. This is not ordinary one-update BP.
- `partial_num=5000/1000` is applied per class. The archived helper's equal-size
  branch omits one randomly selected example per class, producing 49,990 train
  and 9,990 test examples before `drop_last=True`.
- Test data is evaluated every pass; the reported value is minimum test error
  over passes and the learning-rate grid. Test loading is shuffled and drops
  the final incomplete batch, so each evaluation can cover a different 9,800
  example subset. This is reproduced but treated as test-informed selection.

## Archived target

Best accuracy after the paper's per-seed learning-rate selection:

| Rule | Seed 1482555873 | Seed 698841058 | Seed 2283198659 | Mean |
|---|---:|---:|---:|---:|
| PC | 71.286% | 71.949% | 71.561% | 71.599% |
| RBP | 69.214% | 69.306% | 68.816% | 69.112% |

Winning PC learning rates are `5e-5/1e-4/2.5e-5`; winning RBP rates are
`7.5e-5/1e-4/5e-5` in the seed order above. The source-data result is a target,
not evidence that the local reproduction has already succeeded.

## SplitCIFAR extension

SplitCIFAR-10 is not Fig. 4i. It reuses the architecture and paper update
semantics but uses a deterministic 90/10 train/validation split, five disjoint
class-pair tasks, a shared ten-way head, one pass per task, no replay, no model
task ID, and project seeds `7/42/123`. Fixed learning rates are chosen before
launch from the best cross-paper-seed mean: `2.5e-5` PC and `7.5e-5` RBP.

## Implementation route

The local pinned Bogacz Group PC library differs from the archived trainer in
additional optional features and defaults, but all behaviorally relevant Fig.
4i arguments are explicit. The local harness imports that submodule instead of
copying the paper's AGPL code into this repository.
