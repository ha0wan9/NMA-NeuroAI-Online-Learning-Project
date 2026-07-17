# Dispatch: Continual-Learning Protocol Research

**Assigned:** 2026-07-17, by the end of the one-hour working session
**Submission deadline:** No later than 15 minutes before the 2026-07-20
three-hour working session
**Decision briefing:** First 20 minutes of the 2026-07-20 session
**Protocol decision deadline:** T+30 minutes on 2026-07-20
**Team:** Three people: Protocol DRI, Metrics DRI, and Evidence/Feasibility DRI
**Output:** One-page decision memo; compact protocol/metric tables and
references may follow as an appendix

## Objective

Recommend the smallest fair continual-learning protocol that can compare the
Backpropagation (BP) baseline with Predictive Coding (PC) within the remaining
project time. The output must be specific enough that the implementation team
can construct the stream and evaluator immediately after the Monday decision.

The working default to test is **domain-incremental MNIST with a controlled
rotation or noise shift**, because it preserves a fixed ten-class output space.
The group may reject this default only by showing that another regime is both
more informative for the research questions and feasible within the same time
and compute budget.

## Questions that must be answered

1. Which single regime should be used: task-, domain-, or class-incremental?
2. How are stages, transformations, boundaries, class exposure, sample order,
   and train/validation/test partitions constructed?
3. Are task identity, task boundaries, revisits, replay, or persistent
   per-example state available? State each allowance explicitly.
4. Is training single-pass, and how often are updates and evaluations made?
5. Which minimum metrics answer plasticity, stability, robustness, and resource
   cost? Give the formula or unambiguous implementation definition for each.
6. How will BP and PC receive matched streams, information, evaluation cadence,
   tuning budget, seeds, and result-inclusion rules?
7. How will PC-specific inference steps, runtime, memory, and persistent latent
   state be disclosed?
8. What is the simplest fallback if the recommended protocol cannot pass a
   smoke test on Monday?

## Required deliverable

Submit one file or shared document containing:

1. **Recommendation:** one primary protocol and one fallback, with a short
   feasibility rationale and explicit limitations.
2. **Protocol table:** regime, dataset, stages/shifts, sample counts, ordering,
   task-ID access, replay, preprocessing, split policy, seeds, cadence, compute
   cap, and stopping rule. Unknown values must be marked `decision required`.
3. **Metric table:** metric name, exact definition, research question answered,
   evaluation point, direction of improvement, and important failure mode.
4. **Fairness table:** information, memory/replay, parameter capacity, update
   frequency, tuning budget, compute reporting, and result inclusion for BP and
   PC.
5. **Implementation handoff:** pseudocode for stream/evaluation order, required
   evaluator outputs, leakage checks, and a minimal smoke-test specification.
6. **Evidence:** primary or authoritative sources for protocol and metric
   choices. Search the existing paper registry and claim ledger first; label
   any unsupported choice as an interpretation or working hypothesis.

The minimum proposed metric set should cover online/prequential performance,
final or average accuracy, forgetting or backward transfer when meaningful,
robustness to the chosen shift, and runtime/memory cost. Include signal-to-noise
ratio only if it receives an operational definition tied to a research
question.

## Scope limits

- Do not conduct another broad literature survey.
- Do not implement or modify experiment code during this assignment.
- Do not claim that PC is biologically superior or outperforms BP.
- Do not introduce multiple continual-learning regimes into the required path.
- Do not assume replay, task IDs, future boundaries, or test-stream tuning.
- Keep optional ablations separate from the minimum protocol.

## Monday briefing format and acceptance criteria

The Protocol DRI has five minutes to present the recommendation; the Metrics
DRI has five minutes to present definitions; the Evidence/Feasibility DRI has
five minutes to present fairness, evidence, and risks. Reserve five minutes for
questions. The project lead makes or assigns the final decision by T+30.

The dispatch is complete only when the implementation team can answer, without
further research: what stream to build, what information each method receives,
when to evaluate, what to log, how success is measured, and what fallback to
use. Remaining decisions must have an owner and a T+30 deadline.
