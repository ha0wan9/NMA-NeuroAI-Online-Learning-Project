# CIFAR-10 evaluation

Primary raw sources:

- [`paper-reproduction-v1.json`](../../../playground/step2-cifar10/runs/paper-reproduction-v1.json)
- [`split-cifar10-v1.json`](../../../playground/step2-cifar10/runs/split-cifar10-v1.json)
- [`split-cifar10-representations-v1.json`](../../../playground/step2-cifar10/runs/split-cifar10-representations-v1.json)
- [`split-cifar10-instrumentation-control-v1.json`](../../../playground/step2-cifar10/runs/split-cifar10-instrumentation-control-v1.json)
- [`h2e3-representation-summary.json`](artifacts/h2e3-representation-summary.json)

Percentage differences below are percentage points. `E:` denotes measured
evidence, `I:` interpretation, and `H:` a follow-up hypothesis.

## H1.E2 — Song et al. Figure 4i reproduction

### Registered gate

Each RBP and PC best-test accuracy for the three archived seeds had to lie
within ±1.0 pp of its archived `fig4-i.csv` target. The run also required
finite values, matching BP/PC initialization and data checksums within each
seed, PC objective descent, and total study use below ten GPU-hours.

| Method | Seed | Reproduced best | Archived target | Signed difference | Best epoch | Pass |
|---|---:|---:|---:|---:|---:|---|
| BP | 1482555873 | 69.2143% | 69.2143% | −0.0000 pp | 10 | yes |
| PC | 1482555873 | 70.5000% | 71.2857% | −0.7857 pp | 72 | yes |
| BP | 698841058 | 68.5816% | 69.3061% | −0.7245 pp | 7 | yes |
| PC | 698841058 | 72.2143% | 71.9490% | +0.2653 pp | 64 | yes |
| BP | 2283198659 | 69.1020% | 68.8163% | +0.2857 pp | 24 | yes |
| PC | 2283198659 | 71.0714% | 71.5612% | −0.4898 pp | 65 | yes |

**E:** BP reproduced `68.9660 ± 0.3376%` best accuracy versus a `69.1122%`
archived mean. PC reproduced `71.2619 ± 0.8729%` versus `71.5986%`. Mean
absolute source-target error was `0.3367` pp for BP and `0.5136` pp for PC.
The descriptive reproduced PC−BP mean was `+2.2959` pp.

**E:** All six tolerance flags pass. Within every seed, BP and PC have equal
initial-parameter, train-index, and test-index checksums. Every metric is
finite, and the PC first-batch overall objective decreases for all three
seeds. Thus the gate is not carried by an unmatched stream or initialization.

**E:** H1.E2 recorded `5.7466` synchronized update hours. PC used `3.1677`
hours versus BP's `2.5789` hours (`1.228×`). Peak allocated memory was
`296.4 MB` for PC and `223.5 MB` for BP. Together with the approximately
`0.08`-hour H2.E2 run, the study remains below the ten-hour ceiling.

**I:** H1.E2 is a successful exact-as-practical reproduction of the archived
Figure 4i protocol and is promotable within that claim. It is not a bitwise
reproduction: the local PC library revision is newer than the nested archived
copy and the approved study tree is dirty/untracked.

**I:** The result reproduces **best** test accuracy under the paper's
test-informed learning-rate and epoch selection, shuffled/drop-last test
loader, and off-by-one full-class subset behavior. It is not leakage-free
model selection. The `+2.2959` pp descriptive PC−BP mean does not establish
statistical, general, continual-learning, or biological superiority.

Verdict: **kept and promotable within the registered claim ceiling**.

## H2.E2 — Replay-free SplitCIFAR-10 baseline

**E:** Across paired seeds `7`, `42`, and `123`, BP/PC final average accuracy
was `17.990/16.977%`, prequential accuracy `82.074/76.042%`, adaptation gain
`85.863/81.223%`, and forgetting `85.954/81.421%`. PC therefore changed these
metrics by `−1.013`, `−6.033`, `−4.640`, and `−4.533` pp respectively.

**I:** PC retains more old-task performance but acquires the stream less
effectively. Lower forgetting is coupled to lower adaptation and prequential
accuracy, so this is a stability–plasticity trade-off rather than PC
superiority. Both methods catastrophically forget under the shared-head,
one-pass, no-replay protocol.

Verdict: **kept as a paired descriptive continual-learning baseline**.

## H2.E3 — Representation stability and plasticity

**E:** The diagnostics-on artifact exactly matches the same-revision
diagnostics-off control for all six method/seed runs across test and validation
matrices, continual metrics, initialization checksum, stream checksums, and
samples seen. The fixed-anchor hooks are therefore observational for the
registered behavior. The older H2.E2 artifact differs slightly from both
same-revision reruns, so neutrality is established against the control rather
than by claiming bitwise equality to H2.E2.

Final mean old-task RDM drift at the fifth task boundary, averaged over tasks
0–3 and then summarized across the three independent seeds:

| Layer | BP mean drift | PC mean drift | PC−BP | Task pairs with lower PC drift |
|---|---:|---:|---:|---:|
| conv1 | 0.009335 | 0.00000035 | −0.009335 | 12/12 |
| conv2 | 0.025365 | 0.000483 | −0.024882 | 12/12 |
| fc1 | 0.362578 | 0.296496 | −0.066082 | 8/12 |
| logits | 0.870759 | 0.788836 | −0.081923 | 11/12 |

**E:** Every layer has lower PC drift for each of the three seed-level means.
The near-zero PC drift in `conv1` and very small `conv2` drift show that its
early geometry is almost frozen under this protocol. The twelve task rows are
nested within only three independent seeds and are not twelve replications.

The same current-revision runs preserve the behavioral trade-off:

| Metric | BP | PC | PC−BP |
|---|---:|---:|---:|
| Final average accuracy | 18.057% | 16.973% | −1.083 pp |
| Prequential accuracy | 81.913% | 76.050% | −5.863 pp |
| Mean adaptation gain | 86.077% | 81.223% | −4.853 pp |
| Average forgetting | 86.138% | 81.425% | −4.713 pp |

**E:** Initial-to-final label-alignment gains are also smaller for PC: BP/PC
gains are `0.021890/0.000008` at `conv1`, `0.054360/0.002808` at `conv2`,
`0.104883/0.074531` at `fc1`, and `0.074273/0.063124` at the logits. Lower
drift therefore co-occurs with weaker representational acquisition, especially
in the convolutional layers.

**I:** For research Question 2, the joint evidence is consistent with PC
shifting the system toward stability through reduced plasticity: it forgets
less and changes old-task geometry less, but it also acquires the stream less
effectively and develops less label-aligned geometry. RDM stability alone is
not a measure of useful continual learning.

**E:** A simple monotonic drift–forgetting account is not supported. For
`fc1`, within-method pooled Pearson correlations are `−0.722` for BP and
`−0.566` for PC, while the paired PC−BP task delta correlation is `−0.453`.
These values are descriptive and task-confounded; they neither estimate
mediation nor identify a mechanism.

Verdict: **kept as an observational representation-level result**. The claim
ceiling excludes causal drift-to-forgetting, monotonic mediation, superiority,
biological-mechanism, and general continual-learning claims.

## Independent result-skeptic review

The reviewer found no scientific blocker to the H1.E2 verdict. It independently
confirmed all six tolerance checks, paired checksums, finite values, PC descent,
and budget feasibility. It required the claim to remain limited to best-test
Figure 4i reproduction and flagged three reporting caveats incorporated here:
the artifact field `absolute_target_difference` stores a signed difference,
budget accounting is update-time rather than full energy accounting, and the
dirty/newer-library provenance prevents a bitwise-reproduction claim.

The H2.E3 reviewer independently verified all six diagnostics/control pairs,
zero non-finite numeric values, lower seed-mean PC drift at every layer, and
the joint behavioral/label-alignment trade-off. It accepted the observational
RQ2 answer and required the three-seed, nested-task, fixed-anchor,
boundary-only, single-architecture limitations stated above.

## Unresolved evidence

**H:** Within-task checkpoints may separate acquisition dynamics from
task-boundary geometry changes, but testing this requires a separately
registered causal-chunk extension. A Transformer/ViT comparison also remains
a separate study; neither is evidence from the completed baseline.
