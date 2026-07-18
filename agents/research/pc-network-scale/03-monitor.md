# Monitor record

## Invalid v1 pass

- Completed on CyberEngine, then invalidated by cross-study reproducibility audit.
- Cause: generalized model construction interleaved BP and PC modules, changing
  seeded baseline initialization relative to the frozen prior protocol.
- Disposition: raw artifact and derived summaries preserved under
  `pre-init-order-fix` names; all three ledger rows marked invalid and excluded.

## Corrected v2 pass

- Backend: CyberEngine ROCm GPU, rootless Podman image recorded in raw JSON.
- Command: `bash playground/step1-static-mnist/remote-run-architecture-sweep.sh`.
- Status: completed successfully, exit code `0`, approximately twelve minutes.
- Health: no non-finite value, checksum/stream mismatch, parameter mismatch,
  or diagnostic-objective failure.
- Reproducibility: all three `256×2` baseline seed checksums match the prior
  frozen relaxation-study baseline.
- Raw artifact: [`architecture-sweep-v2.json`](../../../playground/step1-static-mnist/runs/architecture-sweep-v2.json).
