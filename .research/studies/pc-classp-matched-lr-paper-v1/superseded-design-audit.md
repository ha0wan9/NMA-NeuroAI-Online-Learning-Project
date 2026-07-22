# Superseded `matched-lr-validation-v1` design audit

- **Status:** superseded before valid data collection
- **Superseded by:** [`matched-lr-paper-v1`](protocol.md)
- **Original draft date:** 2026-07-22
**Original draft SHA-256:**
`e0370f77d70b9585e5515bf8c782b298657cf9dac1209e34aa0234afd8e99b82`

The stopped agent's original registration remains unmodified at the local
audit path
`playground/continual-mnist/results/matched-lr-validation-v1/REGISTERED_STUDY_DESIGN.md`.
That path is ignored run state; this tracked record preserves its identity,
scientific intent, and reasons for supersession without presenting it as the
active protocol.

## What the draft proposed

The draft registered a matched-learning-rate 2×2 comparison of BP/PC and
Adam/CLASSP on 20-task Permuted MNIST. It proposed `lr=3e-4`, PC `T=20`,
CLASSP `p=2` and threshold `1e-5`, multiple seeds, a full accuracy matrix,
adaptation, forgetting, BWT, provenance, a claim ceiling, and a human approval
gate. Its central interaction was:

```text
ΔΔ = (PC+CLASSP − PC+Adam) − (BP+CLASSP − BP+Adam)
```

Those control and claim-discipline goals are retained in the replacement.

## Why it could not govern a valid run

The draft runner and registration had the following unresolved defects:

1. It instantiated the parameter optimizer and `PCTrainer` inside the task
   loop, resetting Adam/CLASSP history at every boundary.
2. It inherited the defective accumulator that cleared prior history on
   threshold-passing coordinates before adding the current contribution.
3. It matched LR but not the objective: BP used cross-entropy and PC used
   one-hot half-squared error.
4. Its permutations used the experimental seed rather than the corrected
   fixed permutation seed `0`.
5. Its "prequential" metric evaluated each task's test set before acquisition,
   not predict-before-update accuracy on each training batch.
6. Its BWT used final minus pre-acquisition accuracy instead of final minus
   post-acquisition accuracy over tasks `0..N-2`.
7. Its forgetting used an immediate post-task value and excluded task 0 rather
   than maximum post-acquisition minus final over tasks `0..N-2`.
8. It described an `N×N` matrix while the intended canonical matrix is
   `(N+1)×N` including the pre-training row.
9. It included seed `42` in the five-seed primary set even though that seed had
   informed hyperparameter choice, omitted `271828`, and therefore did not
   implement the corrected five-held-out-plus-bridge design.
10. It allowed CPU full runs, implicit multi-cell execution, existing-file
    skipping, and non-atomic writes.
11. Its smoke only reduced task count; it did not explicitly truncate samples
    or require CUDA.
12. It did not document the discrepancy between the paper equation and the
    official reference code.

No JSON produced by that draft is accepted as a `matched-lr-paper-v1` cell.
The original document is retained solely as process evidence showing why the
protocol was corrected rather than silently rewritten.
