# Development notes

A record of the decisions taken in Phase I, the reasoning behind them, the
known limitations, and the next research iteration.

## Phase I decisions

**Propositional STRIPS as the only representation.** Ground boolean variables
with add/delete effects are expressive enough to encode Blocksworld exactly and
simple enough to keep the engine small. Lifted representations, numeric fluents
and conditional effects are all deferred; none of them change the shape of the
discovery loop, and each would enlarge the engine before the loop was known to
work. The epistemic setting that motivates the project is a later extension,
reached by adding a domain type that satisfies the same `Domain` concept.

**States are fixed-capacity bitsets.** `hd::Bitset<4>` holds 256 propositions
in four machine words: trivially copyable, allocation-free, hashable in a few
instructions, and with subset tests (applicability, goal checking) compiled to
a handful of AND/CMP pairs. The capacity is a compile-time constant in
`strips.hpp`; raising it changes one line and nothing else. A dynamic bitset
would have made every successor a heap allocation, which is exactly the cost
that dominates when a search expands millions of nodes.

**Concepts rather than virtual interfaces.** `Domain` and `Heuristic` are C++20
concepts, so search algorithms are templates instantiated per (domain,
heuristic) pair. Heuristic evaluation is inlined into the expansion loop and no
vtable is consulted per node. The cost is compile time and slightly harder
error messages; the benefit is that the measured expansion rate reflects the
heuristic rather than the dispatch mechanism. Adding a heuristic requires only
a callable, with no base class to inherit.

**Features dispatched by enum switch, computed lazily.** Features are named
values in a single enum with a parallel name table shared with Python (an
integration test asserts the two agree). `LinearHeuristic` evaluates only terms
with a non-zero weight, so an expensive feature costs nothing when unused. The
relaxed planning graph, which both relaxed features need, is built once per
state and cached on the last-evaluated state, so a candidate using both
features pays for one graph.

**One process invocation per candidate, not per instance.** `hd_plan` accepts
repeated `--instance` arguments and emits one JSON document covering all runs.
A candidate evaluation over the 20-instance suite is therefore a single
subprocess call, and process startup is amortised rather than repeated. This
also keeps the crossing points between Python and C++ at exactly two per
evaluation.

**A hand-written JSON writer, no parser.** The engine only emits JSON; it never
reads it. Task files use a line-oriented format that is trivial to parse in
C++ and to generate from Python. This avoids a third-party dependency in the
engine for the sake of a few dozen lines of writer.

**Per-instance normalisation against a fixed reference.** The objective
normalises each instance against the same reference run rather than comparing
totals. Totals are dominated by the largest instance, which makes the objective
a proxy for performance on that instance alone. Per-instance ratios weight
every problem equally and make `J = 1` mean "reproduces the reference"
regardless of suite composition.

**Coverage is priced explicitly.** Ratios of search effort reward failure: a
heuristic that gives up expands nothing. The `δ (1 − coverage)` term makes an
unsolved instance more expensive than any plausible expansion ratio, with `δ`
configurable.

**Deterministic tie-breaking.** The open list orders by `(key, tiebreak,
insertion counter)`. Without the counter, ties would be resolved by heap
internals and two runs of the same configuration could differ, making the
objective noisy for reasons unrelated to the heuristic. A unit test asserts
run-to-run identity.

**Weak optimisers on purpose.** Random search, randomised local search and grid
search are all included, and all are weak. Their role is to be the control
condition: a learned method that does not beat random search under an equal
evaluation budget has demonstrated nothing. Phase II results should always be
reported against them.

**Ground truth by enumeration, not by regression.** `h*` is computed by
enumerating the reachable state space forward and then running a backward
Dijkstra from the goal states over the reversed transitions. Regression over
partial states would reach further, but it needs a second, backward
implementation of the domain semantics, and any disagreement between the two
would silently corrupt the ground truth. Forward enumeration reuses the same
`apply` the search uses, and an integration test cross-checks `h*` of the
initial state against the cost A\* reports. The price is that only small
instances can be checked: eight Blocksworld blocks enumerate in seconds, nine
exceed a ceiling of $10^6$ states.

**A truncated state space yields no verdict.** When enumeration hits its
ceiling the missing transitions can only make `h*` look larger, which would
hide violations rather than produce false ones. `hd_verify` therefore computes
no distances at all for a truncated instance, and the Python layer reports it
as skipped. A heuristic is never recorded as passing an instance that was not
actually checked.

**Verification is falsification.** A violating state is a certificate and is
reported with the propositions true in it; a clean run over the enumerable part
of a suite is evidence and is labelled "not falsified". Nothing in the
reporting path is permitted to phrase the latter as proof.

**Landmarks are regenerated per state, not maintained along a path.** The
`Heuristic` concept is `h : State -> double`, so a landmark set carried along
the search path (which is what LM-A\* does, and what makes landmark heuristics
cheap) has nowhere to live. Landmarks are therefore recomputed from the state
alone: sound, deterministic, reproducible from a state in isolation, and
expensive, at one relaxed reachability test per candidate proposition. The
alternative would change the interface every search algorithm and the verifier
depend on, in exchange for a heuristic that is no longer a function of the
state, and it was not worth that before the component had been shown to pay for
itself in expansions.

**Landmarks are priced, not counted.** Counting unachieved landmarks is
inadmissible whenever one action achieves several at once, which Blocksworld
does constantly. The uniform cost partition (each action divides its cost among
the landmarks it can achieve; each landmark takes the cheapest share offered)
restores admissibility.

**A component carries its own cost function.** `LandmarkFactory` takes an
`ActionCosts` (`costs.hpp`) and prices landmarks with it, defaulting to the
task's own costs, and `enumerate_state_space` takes one too. That pairing is
what makes the property testable: the bound a component computes under `c_i`
has to be admissible against the `h*` of the same task under `c_i`, and both
sides of that statement now accept the same cost vector.

Generation is untouched by this, because a landmark is determined by
reachability and reachability does not depend on costs. Only the pricing
changes, so a component of a partition is the same factory constructed with a
different share, and the landmarks it finds are still the landmarks of the
task. `is_cost_partition` checks the condition the whole scheme rests on, that
the shares of an action never exceed what the action costs; leaving cost
unspent is legal and only overspending breaks the bound.

**A projection is enumerated past its goal states.** `enumerate_state_space`
stops at a goal state, because a search does, and verification has to see
exactly the states a search could evaluate. An abstraction cannot afford that:
a concrete state that is not a goal can project onto an abstract goal state,
and every state beyond it would be missing from the table and read as a dead
end. The enumeration therefore takes a flag, false for verification and true
for building a database.

**Patterns are grown by bounded causal closure.** A pattern seeded with a goal
condition alone is worthless: every achiever projects to an empty precondition,
so the database can only count. Adding the preconditions of the achievers, then
of their achievers, up to a fixed budget, is what makes the abstraction see why
a proposition is hard to reach, and it raises informedness on `blocks-05-00`
from 0.370 to 0.655 at a budget of twelve. The budget is a memory and
build-time ceiling, not a tuned value: informedness keeps rising with it.
Pattern selection is a research problem in its own right and this is a default
to build components from, not an attempt at solving it.

**An oversized pattern is an error, not a truncation.** A database with missing
entries reads them as dead ends, which is silently wrong rather than loudly
wrong, so `PatternDatabase` refuses a pattern whose abstract state space
exceeds its ceiling.

**`pdb` takes the maximum, not the sum.** The maximum of admissible components
is admissible and the sum is not, and the maximum is precisely the control a
cost partition over the same components has to beat. Building it as a baseline
now means the control exists before the thing it controls for.

**Blocksworld, one domain.** A single domain family with a size parameter gives
a difficulty gradient (the committed suite spans 5 to 9 blocks; the zero
heuristic solves 15 of 20 within budget, goal count solves all) without the
maintenance cost of several encodings. A second domain is needed before any
claim about generality, which is the point at which one should be added.

## Known limitations

- **Single domain family.** Nothing reported here separates a heuristic that
  captures planning structure from one that captures Blocksworld structure.
  This is the most important limitation and the first thing Phase II should
  address.
- **Selection on the evaluation set.** By default the optimiser is scored on
  the same instances that are reported. `--holdout-stride N` splits the suite
  and reports the held-out part separately; it is off by default to keep the
  headline command simple, and it should be on for any reported result.
- **Admissibility is checked, not guaranteed.** The verifier reaches eight
  Blocksworld blocks; the hypothesis class itself offers no guarantee, so
  larger instances are simply unverified. See item 1 below.
- **Linear hypothesis class.** `h_θ` cannot express interactions between
  features. This is a deliberate constraint in service of interpretability, and
  the natural first relaxation (feature products, piecewise forms) is cheap to
  try.
- **The feature set is small and hand-designed.** Seven features, of which
  three are already close to standard heuristics. Discovering features rather
  than weights is a separate and harder problem.
- **`landmark_cost` is expensive and inconsistent.** It regenerates its
  landmarks at every state, costing one relaxed reachability test per candidate
  proposition, and the verifier reports it as admissible but not consistent, so
  A\* reopens states. Both are consequences of `h` having to be a pure function
  of the state; a path-dependent landmark set would fix both and would not fit
  the `Heuristic` concept.
- **`reopened` is not an inconsistency signal.** It counts every state
  rediscovered along a cheaper path, which a consistent heuristic also causes:
  `pdb` is consistent by construction and records 882 of them on the four
  seven-block instances. It tracks how depth-first the search order is, and the
  name in the JSON schema is more suggestive than the quantity deserves.
- **Pattern selection is arbitrary.** The patterns are one per goal condition,
  closed over preconditions to a fixed budget. Nothing chose them, and the
  maximum over them is weaker than the landmark bound on this domain.
- **Search effort is measured, plan quality mostly is not.** The default
  objective weights expansions only (`γ = 0`). Greedy best-first search returns
  suboptimal plans, and a candidate can trade cost for speed unnoticed unless
  `γ` is raised.
- **`peak_memory_kb` is process-wide.** It comes from `getrusage` and reflects
  the whole invocation, so with several instances per invocation it is an upper
  bound rather than a per-instance figure.
- **No parallel evaluation.** Candidates are evaluated sequentially, although
  the loop is embarrassingly parallel. This is a throughput limit, not a
  correctness one.

## Next research iteration

In rough order of expected value per unit of work:

1. **A ceiling for the partition.** The scheme is done and is written up in
   `docs/cost_partitioning.pdf`. `pdb_uniform` is the control the previous
   version of this list asked for and is the worst heuristic on the benchmark;
   `scp` is a saturated chain under a fixed order and ties the maximum it was
   meant to beat; `scp_rotations` is the maximum over the `k` cyclic rotations
   of that order and beats it by 24% of informedness and 29% of expansions,
   with a guarantee that holds at every state; `scp_mixed` puts the databases
   and the landmark bound under one partition and dominates both controls by
   construction.

   The previous version of this item said the order the components are paid in
   was "the right object for this project to discover". That is falsified.
   `hd_orders` enumerates every ordering at the sizes where `k!` is tractable,
   and §6.6 of the README reports the result: the best ordering of all 120 at
   five blocks, chosen with full knowledge of the state space, is about 20%
   below the rotation family and beats the plain maximum over the components on
   none of the four instances, so no search over permutations can replace the
   family; and the rotations already recover 98.6% to 99.2% of the maximum over
   all `k!` orderings, so no larger family can add more than about one per cent.
   The value is in taking a maximum over a family that covers every component,
   and the cheapest such family gets almost all of it.

   What remains is a ceiling and better components, in that order.

   1. **Optimal cost partitioning by LP**, per state, over the shares of the
      same components. Without it, 0.420 and 0.530 are comparisons against the
      controls and not against what the components could give, and there is no
      way to tell whether saturation is leaving anything on the table. The
      variables are one share per component and action, plus the component
      values; the constraints are the partition inequalities together with,
      per component, the requirement that its stored table remains a bound.
      Two decisions come first: the repository has no external dependencies
      today and an LP solver is one, and the LP is per state, so with 263958
      states it has to be a stratified sample.
   2. **Pattern selection.** §6.2 already records that raising the closure
      budget from twelve to eighteen propositions moves informedness from 0.655
      to 0.745 on `blocks-05-00`, which is a larger effect than anything cost
      partitioning bought. With the order exhausted, the components are where
      the remaining accuracy has to be.

2. **A second domain.** Gripper or Logistics, added as a generator alongside
   `hd.domains.blocksworld`. Then re-run the Phase I experiment and report
   cross-domain transfer: does `θ*` learned on one domain help on the other?
   This is the first result the framework can produce that is about planning
   rather than about the framework.
3. **Held-out evaluation as the default protocol**, with several optimiser
   seeds per configuration and the variance reported. The present numbers are
   single-seed point estimates.
4. **Evolutionary search over the weight vector** (CMA-ES or a simple
   population method), compared against random search under an equal budget of
   planner invocations. This is the smallest step that tests whether a genuine
   optimiser buys anything over sampling.
5. **Feature discovery rather than weight fitting**, via a small grammar over
   features (products, thresholds, goal-restricted counts) searched by genetic
   programming. This is the first point at which "program synthesis" is a fair
   description of what is happening, and where interpretability starts to cost
   something.
6. **Reinforcement learning.** Two distinct formulations are worth separating:
   (a) treat `θ` as a policy parameter and the objective as the return, which is
   contextual-bandit-shaped and directly comparable to the optimisers here;
   (b) treat state expansion as the action and learn a value function online,
   which changes the engine interface substantially and needs the search loop to
   call back into a learned model. Attempt (a) first; it reuses the entire
   present infrastructure.
7. **Epistemic planning.** The motivating setting. It requires a domain type
   with epistemic states satisfying the `Domain` concept, plus features over
   knowledge and belief. Worth beginning only once the discovery methods have
   been discriminated on classical domains.

Two invariants should survive all of the above: discovered heuristics remain
interpretable and serialisable, and every reported number remains reproducible
from its experiment record alone.
