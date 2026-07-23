# Monitor record

- Backend: CyberEngine ROCm GPU in
  `localhost/nma-neuroai-step1-rocm:7.2.1`.
- Command: `bash playground/step1-static-mnist/remote-run-relaxation-sweep.sh`.
- Status: completed successfully; exit code `0`.
- Duration class: approximately thirteen minutes wall-clock for the combined
  BP control and four PC levels across all three settings.
- Health: no crash, non-finite value, checksum mismatch, or stream mismatch.
- Expected warning: upstream `PCTrainer` reported that `T=1` is below the
  two-PC-layer minimum propagation depth; retained as a boundary condition.
- Raw artifact: [`relaxation-sweep-v1.json`](../../../playground/step1-static-mnist/runs/relaxation-sweep-v1.json).
