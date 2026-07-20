# Execution monitor

Last updated: 2026-07-19T16:53Z.

## Budget

The user-approved ceiling is 10 CyberEngine GPU-hours. H1.E2 recorded
`20,687.858` synchronized update seconds (`5.7466` hours), split into `2.5789`
hours for BP and `3.1677` hours for PC. The completed H2.E2 run used
approximately `0.08` hours of wall-clock GPU occupancy. H2.E3 recorded
`0.0746` hours for the diagnostics-on run and `0.0752` hours for its
same-revision diagnostics-off control, counting stream processing plus
evaluation across all six method/seed runs. The documented total is therefore
approximately `5.98` GPU-hours, leaving about `4.02` hours before small smoke
overhead. Timing is not a complete energy measurement, and the ROCm runs did
not expose a usable peak-memory counter in these artifacts.

## Run status

| Run | Status | Evidence |
|---|---|---|
| H1.E1 smoke-001 | invalid, preserved | Zero batches because 160 samples were smaller than the paper batch size with `drop_last=True`. |
| H1.E1 smoke-002 | completed, promoted | Matching initialization/data checksums; 400 samples/method; PC objective 101.0466 to 76.8237. |
| H2.E1 smoke | completed, promoted | Matching paired setup and decreasing PC objective on all task captures. |
| H2.E2 full Split CIFAR-10 | completed | Three paired seeds, 45,000 stream samples/method/seed; artifact is `playground/step2-cifar10/runs/split-cifar10-v1.json`. |
| H1.E2 full static CIFAR-10 | completed; gate passed | Six runs completed at 2026-07-19T00:50Z. Every best-test accuracy is within the registered ±1 pp source target; artifact is `playground/step2-cifar10/runs/paper-reproduction-v1.json`. |
| H2.E3 representation smoke series | completed | Fixed-anchor diagnostics, 3D trajectories, and interactive RDM timelines passed CPU probes without changing behavioral metrics. |
| H2.E3 full representation run | completed; kept | Three paired seeds, 20 fixed anchors/class, four layers, and six task-boundary checkpoints; artifact is `playground/step2-cifar10/runs/split-cifar10-representations-v1.json`. |
| H2.E3 instrumentation control | completed; kept | Diagnostics-on and diagnostics-off behavior matches exactly across all six current-revision runs and all registered behavioral fields. |

H1.E2 exited normally and its immutable JSON was synchronized locally.
Independent result-skeptic reviews accepted both the registered static
reproduction and the H2.E3 observational result with the claim ceilings in
`04-evaluation.md`.

## TensorBoard

TensorBoard 2.20.0 is provisioned in the persistent remote workspace without
overriding the ROCm image's NumPy 2.4.4 or Pillow 12.1.1. H2.E2 was backfilled
into six method/seed event streams and verified against the JSON artifact; PC
seed 42 reports final average accuracy `0.1714` in both representations.

The dashboard is served through an SSH tunnel at `http://127.0.0.1:6006`.
H1.E2 began before live instrumentation and was left untouched. Its complete
80-epoch histories were backfilled after completion into six method/seed event
streams under `runs/tensorboard/paper-reproduction-v1/`.

H2.E3 produced six method/seed event streams under
`runs/tensorboard/split-cifar10-representations-v1/`. Each includes scalar
representation diagnostics plus self-contained interactive 3D trajectories
and four-layer RDM timelines.
