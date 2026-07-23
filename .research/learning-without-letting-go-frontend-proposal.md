# Front-End Build Proposal

## Learning Without Letting Go · Continual-Learning Atlas

**Status:** approved direction; implementation in progress
**Primary artifact:** [`learning-without-letting-go.html`](learning-without-letting-go.html)
**Research frame:** canonical Q1–Q3 in [`../research-project-proposal.md`](../research-project-proposal.md)

## 1. Objective

Turn the front end into a layered research publication: a project-level Atlas
for orientation and one complete reading page for every experiment.

The first viewport should let a reader answer three questions quickly:

1. What scientific space did the project explore?
2. Why did each experiment become necessary?
3. What does each experiment support, reject, or leave unresolved?

The page must preserve negative results and claim ceilings. It must not present
an observed trade-off, retrospective explanation, or proposed diagnostic as an
established finding.

## 2. Problems in the Previous Front End

- The top of the page gave equal prominence to lifecycle status, proposal
  wording, paper audit details, models, protocols, and results.
- Experiments appeared in several sections, but their hypotheses and scientific
  roles were not linked into one narrative.
- The protocol selector and learning curves were useful but detached from the
  question that each experiment was designed to answer.
- A linear roadmap could show order, but not where an experiment sits in the
  wider classic/continual and behavior/mechanism research space.
- Repeated prose across sections risked future drift between the roadmap,
  experiment descriptions, results, and next steps.
- A single long HTML document could summarize the project but could not support
  a complete, self-contained reading session for one experiment.

## 3. Approved Information Architecture

The page uses one research graph and two synchronized views.

### 3.1 Journey View

A primary-parent research tree answers:

> Why did this result make the next experiment necessary?

The root is the project thesis. Its main branches are:

- the paper/source-validity foundation;
- Q1 · classic learning;
- Q2 · continual learning;
- Q3 · inference and scale;
- the mechanistic frontier that refines Q2 and Q3.

Research questions are parent nodes. Completed and proposed experiments are
child nodes. Cross-branch dependencies remain part of the shared research graph
and are explained in the card rather than duplicating an experiment under
multiple parents.

### 3.2 Research Space View

A two-axis map answers:

> Where does this experiment sit in the space of questions?

- Horizontal axis: **classic/stationary → continual/non-stationary**.
- Vertical axis: **behavioral outcome → mechanistic explanation**.

The four regions are:

| Region | Governing question |
|---|---|
| Classic × behavior | Does PC learn and reproduce the static baseline? |
| Continual × behavior | Does PC adapt while retaining prior tasks? |
| Classic × mechanism | How do inference budget and architecture change learning? |
| Continual × mechanism | Why does the observed stability–plasticity trade-off occur? |

The coordinates are conceptual categories, not quantitative scores. Visual
distance must not imply a measured similarity or effect size.

### 3.3 Shared Story Panel

Journey and Research Space render the same node IDs from one registry. Changing
views preserves the selected node and opens the same card.

Every experiment card contains:

1. hypothesis;
2. experimental design;
3. goal;
4. controls and rigor;
5. observed result;
6. verdict;
7. why the next experiment is necessary.

Every research card contains:

1. research question;
2. motivation;
3. current evidence-backed answer;
4. claim ceiling;
5. unresolved mechanism or next test.

Cards link directly to the relevant protocol, learning curve, result chart, or
claim audit. Detailed evidence remains below the map and is shown on demand
rather than competing with the first-view narrative.

### 3.4 Individual Experiment Pages

Every experiment node has a stable, separate HTML page under `experiments/`.
The Atlas remains the summary and navigation layer; it is not the only place
where experiment content is rendered.

Each completed experiment page contains four required reading blocks:

1. model settings and experiment settings;
2. achieved and non-achieved results;
3. learning curves and experiment-owned visualizations;
4. explanation, hypothesis, verified observations, interpretation, verdict,
   and unresolved questions.

Visualizations belong to the experiment that produced them. For example,
SplitCIFAR representation diagnostics own the representation-drift,
label-alignment and behavioral-context plots. Width and depth interventions own
their learning trajectories and cross-regime comparison plots.

Proposed experiments use the same four-block structure, but their Results and
Curves blocks must state `Not run`. Planned hypotheses, settings and falsifying
outcomes are visible; synthetic or anticipated outcomes are not drawn as data.

Shared CSS, rendering logic and the experiment registry may be reused across
pages, but every experiment must have its own stable URL and HTML entry point.

### 3.5 Source-Backed Research Wiki

Domain-specific methods, datasets, continual-learning regimes, metrics,
representation diagnostics and experimental-design terms link to one
source-backed Wiki page under `wiki/`. Each term has a stable fragment URL,
for example `wiki/index.html#prequential-evaluation`.

Every entry contains:

1. a rigorous prose definition;
2. mathematical notation rendered from LaTeX, plus copyable LaTeX source;
3. the exact operational meaning used by this project;
4. primary, official or project-implementation sources;
5. related-term navigation.

Term links carry the originating page URL, including the selected Atlas node.
The Wiki exposes a persistent `Back to source page` link and retains native
browser-back behavior. Returning restores the original experiment/Atlas state
and, when session state is available, the prior scroll position.

## 4. Experiment Coverage

The shared registry contains eight completed experiment stories:

1. static CIFAR-10 Figure 4i reproduction;
2. static MNIST relaxation sweep;
3. Permuted-MNIST relaxation sweep;
4. SplitMNIST relaxation sweep;
5. network-width intervention;
6. network-depth intervention;
7. replay-free SplitCIFAR-10 behavior;
8. SplitCIFAR-10 representation diagnostics.

It also contains four explicitly proposed experiments:

1. fair-budget conventional BP / repeated-update RBP / PC comparison;
2. update-alignment and update-SNR diagnostics;
3. acquisition-matched representation causality study;
4. depth × relaxation interaction study.

Smoke runs, invalid attempts, and controls remain visible within the rigor field
of their owning experiment rather than becoming equal-weight scientific nodes.
For example, the invalid network-scale v1 run is preserved and excluded in the
depth/scale story instead of being silently erased.

## 5. Evidence and Claim Rules

- Separate hypothesis, established observation, interpretation, and open
  question.
- Use registered H/E language where it exists. Do not fabricate retrospective
  preregistration.
- Read forgetting jointly with adaptation, prequential accuracy, and final
  accuracy.
- Keep static, domain-incremental, and class-incremental conclusions separate.
- Keep the paper's static CIFAR reproduction separate from the project's
  SplitCIFAR extension.
- Label the CIFAR comparator as paper-protocol repeated-update BP/RBP where
  relevant.
- Treat three-seed comparisons as descriptive unless a registered gate supports
  a narrower promotion decision.
- Preserve negative results, invalid attempts, resource cost, and biological-
  plausibility limitations.
- Display proposed diagnostics as proposed; never include them in backed claims
  before execution.

## 6. Interaction Contract

- `Journey` and `Research Space` are the only top-level view controls.
- Selecting a node updates one shared Story Panel.
- Switching views preserves the selected node.
- Selected-node identity is reflected in the URL when possible.
- Card actions select the corresponding protocol or learning curve and scroll
  to its detailed evidence.
- Every experiment card provides a prominent `Read full experiment` link to its
  dedicated page.
- Dedicated pages link back to the same selected Atlas node and provide
  previous/next experiment navigation.
- Technical terms link to their anchored Wiki definition without losing the
  originating page, selected Atlas node or browser history.
- The graph uses native buttons and visible focus states for keyboard access.
- Proposed nodes use dashed borders; evidence status is always stated in text
  and never encoded by color alone.

## 7. Page Hierarchy

The Atlas order is:

1. project title and one-sentence thesis;
2. Journey / Research Space switch;
3. active research map;
4. shared Story Panel;
5. detailed paper, model, protocol, result, and learning-curve evidence;
6. next refinements;
7. audit boundaries and evidence sources.

Each experiment page uses this order:

1. experiment title, status, short answer and claim ceiling;
2. model settings and experiment settings;
3. achieved and non-achieved results;
4. learning curves and experiment-owned visualizations;
5. explanation, hypothesis and verification labels;
6. evidence/reproduction sources and adjacent-experiment navigation.

The Wiki is a separate reference layer rather than another Atlas view. It uses
one searchable term index followed by anchored definitions, LaTeX formulas,
project-specific readings and sources.

The prior lifecycle ribbon, top-level status panel, lifecycle card grid, and
duplicate Q1–Q3 summary cards are removed from the first-view hierarchy.

## 8. Responsive Behavior

- Desktop: research map and Story Panel appear side by side.
- Medium widths: the Story Panel moves below the map; the Journey tree becomes
  two columns.
- Narrow widths: Journey branches stack; Research Space becomes four labeled
  regions instead of forcing an unreadable coordinate canvas.
- Detailed tables and existing charts retain their current responsive behavior.
- Individual experiment pages keep settings, results, curves and explanations
  readable without document-level horizontal overflow.
- Wiki formulas scroll within their own panels on narrow screens, while term
  definitions and sources remain single-column and readable.

## 9. Acceptance Criteria

The build is complete when:

- both views render every registered node without duplicated content sources;
- selecting a node in either view opens the same card;
- view switching preserves selection;
- every completed experiment has hypothesis, design, goal, rigor, result,
  verdict, and next-step text;
- proposed experiments cannot be mistaken for completed evidence;
- protocol and learning-curve actions open the correct detailed content;
- every experiment node opens a separate, stable HTML page;
- all eight completed pages contain the four required reading blocks with at
  least one evidence visualization owned by that experiment;
- the representation-diagnostics page includes representation drift,
  label-alignment gain and behavioral context together;
- all four proposed pages retain the four-block structure while marking results
  and curves as not run;
- experiment pages link back to their selected Atlas node and between adjacent
  experiments;
- every registered technical term has one anchored Wiki entry with a LaTeX
  formula, project-specific interpretation and at least one source;
- linked terms on the Atlas and experiment pages reach the correct fragment;
- the Wiki return link restores the originating page and preserves the Atlas
  node encoded in its URL;
- the first viewport no longer leads with lifecycle administration;
- the document works in light and dark themes at desktop and narrow widths;
- inline JavaScript parses, repository validation passes, and browser checks
  find no runtime or interaction errors.

## 10. Non-Goals

- No new experiment is run by this front-end change.
- No raw artifact, result, canonical research question, or frozen protocol is
  rewritten.
- No synthetic capability score or radar chart is introduced.
- No public deployment, new framework, dependency, or backend is required.
