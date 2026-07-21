# Step 1 Run Artifacts

These are raw, small-sample GPU checks from the CyberEngine ROCm backend. They
verify the experiment path and must not be presented as a full matched
comparison.

- `smoke-seed42-pre-accounting-fix.json`: first successful 256-sample GPU run.
  Its generic parameter count includes PC batch-latent state, which motivated
  the corrected accounting fields in later artifacts.
- `smoke-seed42-accounting-fix.json`: confirms equal learned parameter counts
  and separate PC latent-state accounting after one epoch with four PC steps.
- `sanity-overfit-seed42.json`: ten-epoch, 256-sample sanity run with 20 PC
  steps. Both methods reach 98.83% training accuracy and the first PC batch's
  total objective decreases across relaxation.
- `sanity-overfit-seed42-provenance.json`: repeats the overfit sanity after
  runtime provenance was added. This is the preferred smoke artifact because
  it records the CyberEngine host and exact derived-container image ID.

- `continual-smoke-20260717T212801Z.json`: first SplitMNIST and Permuted-MNIST
  pipeline smoke. Its short runtime comparison is superseded because BP paid
  the initial ROCm kernel warm-up and stream timing lacked explicit GPU
  synchronization; its accuracy matrices and metric-path checks remain useful.
- `continual-smoke-timing-fixed.json`: adds explicit GPU synchronization but
  predates discarded BP/PC update warm-ups.
- `continual-smoke-warmup-fixed.json`: accepted two-scenario smoke with
  symmetric discarded update warm-ups and aggregate output.
- `continual-reference-v1-pre-peak-latent-fix.json`: first complete three-seed
  reference. Its accuracy matrices and metrics are identical to the final
  reference, but its PC latent-state field records only the final batch.
- `continual-reference-v1.json`: preferred full-data, three-seed reference with
  final and peak latent-state accounting.
- `continual-reference-v1-summary.md`: claim-labelled human-readable summary.
- `continual-reference-v1-integration-reproduction.json`: clean-revision local
  RTX 4090/CUDA 12.6 rerun of the frozen three-seed protocol. Structural,
  stream, initialization, and objective-descent checks match; accuracy
  matrices differ from the archived ROCm/PyTorch 2.9 run, so this is retained
  as a cross-backend reproduction with an exact-numerics warning.

- `relaxation-sweep-smoke-20260717.json`: first full-path four-level smoke;
  superseded for timing because it predates diagnostic-batch exclusion.
- `relaxation-sweep-smoke-timing-fixed.json`: accepted four-level smoke with
  synchronized timed-sample accounting and `T=1` dynamics marked not assessable.
- `relaxation-sweep-v1.json`: full-data PC `T=1/5/10/20` ablation with one
  fresh BP control, paired seeds `7/42/123`, static MNIST, SplitMNIST, and
  Permuted MNIST. This is the preferred raw relaxation-sweep artifact.

- `architecture-sweep-smoke-v1.json`: initial architecture full-path smoke;
  predates restoration of the frozen BP-first initialization order.
- `architecture-sweep-v1-pre-init-order-fix.json`: preserved invalid full run.
  Interleaved BP/PC construction changed seeded initialization; do not use its
  metrics for comparison.
- `architecture-sweep-smoke-v2.json`: accepted smoke confirming the original
  `256×2` initialization checksum is restored.
- `architecture-sweep-v2.json`: preferred full three-seed architecture study
  comparing `256×2`, `512×2`, and `256×4` with BP and PC at `T=5`.

Preserve failed and superseded artifacts rather than overwriting them.
