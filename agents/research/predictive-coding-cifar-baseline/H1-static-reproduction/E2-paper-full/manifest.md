# H1.E2 full paper reproduction

Command: `bash playground/step2-cifar10/remote-run.sh --output playground/step2-cifar10/runs/paper-reproduction-v1.json`.

Prepared protocol: three archived seeds, per-method/per-seed source-winning
learning rates, 80 passes, immutable raw JSON.

Completed artifact: `playground/step2-cifar10/runs/paper-reproduction-v1.json`.
All six method/seed best-test accuracies passed the registered ±1 percentage
point source-target gate. TensorBoard backfill is under
`playground/step2-cifar10/runs/tensorboard/paper-reproduction-v1/`.
