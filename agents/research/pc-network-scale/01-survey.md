# Code and prior-evidence survey

## Established facts

- The preceding relaxation study found that `T=5` reaches BP-like static
  learning for the `256×2` model and is the cheapest effective tested point.
- The upstream PC trainer requires at least `number of PC layers + 1`
  iterations. `T=5` therefore meets the propagation minimum for the deepest
  planned network (`256×4`).
- The shared runner already fixes data, optimizer, objective, seed, replay,
  and evaluation. Generalizing only the repeated hidden blocks preserves those
  interfaces and permits a controlled architecture intervention.

## Interpretation carried into design

- Width and depth answer different questions and must be separate tracks.
- A larger model can improve both BP and PC simply through capacity; the
  relevant evidence includes within-method architecture deltas and within-
  architecture PC−BP deltas.
- Parameter count, memory, and synchronized processing time are required
  complexity penalties, not optional metadata.

## Open risks

- `T=5` is only minimally sufficient for four PC layers and may disadvantage
  the deeper PC network relative to a separately tuned relaxation budget.
- A fixed learning rate may not be optimal across capacities; tuning is out of
  scope, so negative results apply to this matched configuration only.
- Three seeds support characterization, not general architecture superiority.
