# Automatic Discovery of Planning Heuristics

A research framework for investigating whether effective search heuristics for
classical and epistemic planning can be discovered automatically rather than
designed by hand.

## 1. Motivation

Heuristic search is the dominant paradigm in automated planning, and the
heuristic function is the component that decides whether a planner is usable.
The heuristics in current use — delete relaxation, landmarks, abstractions,
critical paths — are human artefacts: each is the product of a research
programme, each encodes a particular structural insight, and each is fixed once
published. Two consequences follow. Progress is bounded by the rate at which
researchers invent new relaxations, and a heuristic that is excellent on one
domain is frequently mediocre on another, because the insight it encodes is not
the one that domain rewards.

The alternative pursued here is to treat the heuristic itself as the object of
search. Given a planner, a set of state features, and a benchmark suite, an
outer optimisation loop proposes heuristics, measures the search effort they
induce, and uses that measurement to propose better ones. The question this
repository exists to answer is empirical: *how far does that get, and under
which method?*

The intended trajectory is to compare families of discovery methods —
derivative-free optimisation, evolutionary search, reinforcement learning, and
program synthesis — on the same benchmark, under the same objective, with the
same execution engine. This requires infrastructure that keeps the discovery
method interchangeable and the measurement trustworthy. That infrastructure is
what the present phase provides; the discovery methods themselves are the
subject of later phases.

A methodological commitment runs through the design: **the discovered heuristic
must remain interpretable**. A weight vector over named features can be read,
compared against known heuristics, and reasoned about. This constrains what can
be discovered, and that is accepted deliberately: a result that cannot be
explained teaches less than a weaker result that can.

## 2. Problem formulation

### 2.1 Planning tasks

A planning task is a tuple

```math
T \;=\; \langle P,\; A,\; s_0,\; G \rangle
```

where $P$ is a finite set of propositions, $A$ a set of actions, $s_0$ the
initial state and $G$ the goal condition. In Phase I tasks are propositional
STRIPS: a state is a subset $s \subseteq P$, and an action

```math
a \;=\; \langle \mathrm{pre}(a),\; \mathrm{add}(a),\; \mathrm{del}(a) \rangle,
\qquad \mathrm{pre}(a),\, \mathrm{add}(a),\, \mathrm{del}(a) \subseteq P,
\qquad c(a) > 0
```

is applicable in $s$ iff $\mathrm{pre}(a) \subseteq s$, in which case it induces
the transition

```math
s' \;=\; \bigl(s \setminus \mathrm{del}(a)\bigr) \,\cup\, \mathrm{add}(a).
```

A state $s$ is a goal state iff $G \subseteq s$. A plan is a sequence
$\pi = \langle a_1, \dots, a_k \rangle$ leading from $s_0$ to a goal state, of
cost $c(\pi) = \sum_{i=1}^{k} c(a_i)$.

### 2.2 The hypothesis class

Let $f_1, \dots, f_n$ with $f_i : S \to \mathbb{R}$ be state features. The
candidate heuristics are their linear combinations,

```math
h_\theta(s) \;=\; \sum_{i=1}^{n} \theta_i \, f_i(s) \;=\; \theta^{\top} f(s),
\qquad \theta \in \mathbb{R}^n .
```

The class is deliberately small. Every element of it is a readable equation over
named quantities, so a discovered $\theta$ can be compared with heuristics that
were designed rather than found.

### 2.3 The objective

Fix a search algorithm $\mathcal{A}$ (greedy best-first search unless stated
otherwise) and a benchmark suite $I = \{T_1, \dots, T_m\}$. Running
$\mathcal{A}$ with $h_\theta$ on $T_j$ yields the measured quantities
$\mathrm{exp}(T_j, \theta)$, $\mathrm{time}(T_j, \theta)$ and
$\mathrm{cost}(T_j, \theta)$ — expansions, runtime and solution cost — together
with an indicator $\mathrm{solved}(T_j, \theta) \in \{0, 1\}$. Heuristic
discovery is the optimisation problem

```math
\theta^{\star} \;=\; \arg\min_{\theta} \; J(\theta),
```

```math
J(\theta) \;=\;
\alpha \, \frac{1}{m}\sum_{j=1}^{m} \frac{\mathrm{exp}(T_j, \theta)}{\mathrm{exp}(T_j, \mathrm{ref})}
\;+\; \beta \, \frac{1}{m}\sum_{j=1}^{m} \frac{\mathrm{time}(T_j, \theta)}{\mathrm{time}(T_j, \mathrm{ref})}
\;+\; \gamma \, \frac{1}{m}\sum_{j=1}^{m} \frac{\mathrm{cost}(T_j, \theta)}{\mathrm{cost}(T_j, \mathrm{ref})}
\;+\; \delta \, \bigl(1 - \mathrm{cov}(\theta)\bigr),
```

```math
\mathrm{cov}(\theta) \;=\; \frac{1}{m} \sum_{j=1}^{m} \mathrm{solved}(T_j, \theta),
```

where $\mathrm{ref}$ is a fixed reference configuration and
$\alpha, \beta, \gamma, \delta$ are configurable. The cost term ranges over
solved instances only, since an unsolved instance has no cost.

Normalising per instance against a fixed reference makes the objective
scale-free — $J = 1$ reproduces the reference and $J < 1$ improves on it — and
prevents large instances from dominating the mean. The coverage term prices
failure, which ratios of search effort cannot express: a heuristic that solves
nothing expands few nodes.

### 2.4 Properties of the problem

Two properties matter for what follows. First, $J$ is not differentiable in
$\theta$ and has no useful analytic structure: it is defined by the behaviour of
a search algorithm, so only zeroth-order methods apply. Second, greedy
best-first search orders nodes by $h$ alone, so for any $\lambda > 0$

```math
h_{\lambda\theta}(s) \;=\; \lambda \, h_{\theta}(s)
\qquad\Longrightarrow\qquad
J(\lambda\theta) \;=\; J(\theta),
```

that is, $\theta$ is identified only up to a positive scalar. The weight space
is bounded accordingly, and candidates are compared after normalising
$\lVert \theta \rVert_\infty = 1$.

## 3. Architecture

The system is separated into an execution engine and a research layer.

**The execution engine (C++20)** implements states, actions, tasks, the search
algorithms (breadth-first, greedy best-first, A\*), the feature evaluators, and
the heuristics. It reads a task file and a heuristic specification, runs the
search, and writes a JSON metrics document. It performs no learning and holds
no experiment state.

**The research layer (Python)** implements everything outside the search loop:
candidate representation, the objective, the optimisers, benchmark suites,
experiment records, and reporting. It treats the engine as a black box invoked
once per candidate evaluation.

The interface between them is deliberately narrow: a heuristic specification
string in, a JSON document out. Nothing else crosses.

### 3.1 Why the two are separated

The two components have incompatible requirements. Search is a tight loop over
millions of states in which the cost of a state expansion determines what is
measurable at all; the outer loop performs a few dozen iterations, is dominated
by the planner runs it triggers, and changes shape with every new research
idea. Implementing the inner loop in a language with fast, predictable state
representations and the outer loop in one that makes experimentation cheap
gives each what it needs.

The separation also protects the measurement. If Python were callable from
inside the search loop, per-expansion overhead would depend on the interpreter,
and expansions and runtime would no longer be comparable across candidates.
Because the boundary is crossed exactly twice per evaluation — once to specify
the heuristic, once to return the metrics — reported search effort is a
property of the heuristic, not of the harness.

Consequences of this choice are enforced in the engine design: heuristics and
domains reach the search algorithms through C++20 concepts rather than virtual
interfaces, so evaluation is inlined and no dynamic dispatch occurs per node; a
state is a fixed-capacity bitset, trivially copyable and allocation-free; and
features that require expensive computation are evaluated only when they carry
a non-zero weight.

## 4. Phase I scope

Phase I establishes the infrastructure and the baseline result. It provides:

- a propositional STRIPS engine with breadth-first search, greedy best-first
  search, and A\*;
- seven interpretable state features and four baseline heuristics (zero,
  goal count, delete-relaxed layers, and the landmark bound of §6.1);
- a Blocksworld benchmark generator and a fixed 20-instance suite;
- structured JSON metrics for every planner execution;
- an exact-`h*` oracle and an admissibility verifier for instances small
  enough to enumerate;
- a Python research layer with the objective, three derivative-free optimisers
  (random search, randomised local search, exhaustive grid), and reproducible
  experiment records;
- unit tests for every component and an integration test covering the full
  Python → C++ → JSON → Python loop.

It deliberately excludes reinforcement learning, neural function approximation,
and program synthesis. Those are the subject of the next phase, and admitting
them before the measurement infrastructure is trustworthy would make their
results uninterpretable. The derivative-free optimisers included here are weak
by design: they are the baseline any learned method must beat before it can
claim to have discovered anything.

### 4.1 Features

Write $s^{+}$ for the set of propositions reachable from $s$ in the delete
relaxation, and $\ell(p, s)$ for the layer of the relaxed planning graph at
which $p$ first appears.

| Feature | Definition |
| --- | --- |
| `unsatisfied_goals` | $\lvert G \setminus s \rvert$ |
| `achieved_goals` | $\lvert G \cap s \rvert$ |
| `applicable_actions` | $\lvert \{\, a \in A : \mathrm{pre}(a) \subseteq s \,\} \rvert$ |
| `true_propositions` | $\lvert s \rvert$ |
| `relaxed_layers` | $\max_{p \in G} \ell(p, s)$, an $h_{\max}$-like distance |
| `relaxed_sum` | $\sum_{p \in G} \ell(p, s)$, an $h_{\mathrm{add}}$-like distance |
| `landmark_cost` | uniform cost partition over the landmarks of $s$ (§6.1) |

A goal proposition $p \notin s^{+}$ is unreachable even under the relaxation, so
$s$ is a proven dead end; such a $p$ contributes a large finite constant, which
keeps every heuristic total and comparable without saturating the arithmetic.

### 4.2 Baselines

| Heuristic | Definition |
| --- | --- |
| `zero` | $h(s) = 0$; reduces A\* to uniform-cost search, the control condition |
| `goal_count` | $h(s) = \lvert G \setminus s \rvert$ |
| `relaxed_layers` | $h(s) = \max_{p \in G} \ell(p, s)$; domain-independent and admissible |
| `landmark_cost` | the landmark bound of §6.1; admissible, and the strongest baseline here |

These are the reference points against which discovered heuristics are reported.

## 5. Phase I results

The committed suite holds $m = 20$ Blocksworld instances of five sizes, five to
nine blocks, generated from a fixed base seed. Under greedy best-first search
with a budget of $2 \times 10^{5}$ expansions per instance, the baselines
perform as follows. This section predates `landmark_cost`, and both the table
and the optimisation below range over the six features that existed then; the
search space is now seven-dimensional.

| Heuristic | Solved | Expanded | Generated | Cost | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zero` | 15/20 | 1476582 | 2586793 | 160 | 1.72 s |
| `goal_count` | 20/20 | 2602 | 7579 | 334 | 0.00 s |
| `relaxed_layers` | 20/20 | 19720 | 79271 | 262 | 0.24 s |

Taking `goal_count` as the reference configuration and minimising $J$ with
$\alpha = 1$, $\beta = \gamma = 0$, $\delta = 10$ — that is, scoring expansions
alone under a coverage penalty — random search followed by local refinement
returns, after 25 planner invocations,

```math
h_{\theta^{\star}}(s) \;=\;
2.34\, f_{\mathrm{unsat}}(s) \;+\; 3.62\, f_{\mathrm{ach}}(s) \;+\; 2.73\, f_{\mathrm{app}}(s)
\;+\; 3.72\, f_{\mathrm{true}}(s) \;+\; 3.43\, f_{\mathrm{layers}}(s) \;+\; 3.96\, f_{\mathrm{sum}}(s).
```

Every fourth instance was held out from the optimiser and scored separately.

| Split | Instances | Baseline expanded | Discovered expanded | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Training | 15 | 1838 | 465 | 74.7 % |
| Held out | 5 | 764 | 112 | 85.3 % |

Coverage remained complete on both splits and solution cost did not regress
(280 against 278 on the training split), although the objective did not reward
plan quality.

The reduction should be read narrowly: it is one domain, one search algorithm,
one optimiser seed, and an objective weighting expansions only. What it
establishes is that the loop closes and that $J$ is optimisable at all, which is
what Phase I set out to show. The improvement on held-out instances indicates
that the fitted weights are not purely an artefact of the instances the
optimiser observed, but a single split of five instances supports nothing
stronger than that.

## 6. Admissibility

Section 5 says nothing about admissibility, and no search run can. A heuristic
that overestimates still returns plans, often quickly; A\* simply returns
suboptimal ones and reports nothing unusual. The property is a statement about
`h*`, which on instances small enough to enumerate is computable exactly:
expand every reachable state, then run a backward Dijkstra from the goal states
over the reversed transition relation. `hd_verify` does this and checks each
heuristic against the result.

```
hd_verify --instance instances/blocksworld/blocks-05-00.task \
          --heuristic goal_count --heuristic relaxed_layers
python -m hd.experiments.verify_admissibility --max-states 1000000
```

Three properties are reported per instance: admissibility (`h(s) <= h*(s)` at
every reachable state), consistency (`h(u) <= c(u,v) + h(v)` across every
transition, which is what lets A\* close a state on first expansion), and
informedness, the mean `h/h*` over states with a finite non-zero goal distance.
Among admissible candidates informedness is the quantity that predicts A\*
expansions, and unlike expansions it is dense, deterministic, and obtained
without running a search.

![Admissibility and informedness](docs/figures/admissibility.svg)

**Figure 1.** Where a heuristic is allowed to lie. Admissibility confines it to
the region below $h = h^{*}$, and informedness measures how close to that
boundary it gets. Each ray has the gradient of one heuristic's mean $h/h^{*}$
over the twelve instances of the table below; `zero` is the horizontal axis
itself. The $\theta^{\star}$ of §5 leaves the region immediately.

Enumeration reaches eight blocks: at a ceiling of $10^6$ states, 16 of the 20
committed instances are enumerable — 695417 states for the largest — and the
three cheap baselines are checked over all of them in about 22 seconds. The
table below uses the twelve instances up to seven blocks, where every heuristic
including the expensive one is checked on the same 295648 states (42 seconds).

| Heuristic | Verdict | Violations | Informedness |
| --- | --- | ---: | ---: |
| `zero` | admissible, consistent | 0 | 0.000 |
| `goal_count` | admissible, consistent | 0 | 0.419 |
| `relaxed_layers` | admissible, consistent | 0 | 0.442 |
| `landmark_cost` | admissible, inconsistent | 0 | 0.601 |
| $\theta^{\star}$ of §5 | inadmissible | 3077314 | 12.657 |

The row for $\theta^{\star}$ is measured over the 16 enumerable instances: it
overestimates at every one of the 3077314 reachable states it was checked on,
by up to 289.29, and values the goal state itself at 59.77 rather than 0. This is not a defect of the optimiser: greedy
best-first search orders nodes by `h` alone, so the objective of §2.3 is
indifferent to admissibility, and §2.4 makes `θ` identifiable only up to a
positive scalar, which admissibility is not invariant to. The result is a
satisficing one and only ever was.

Two asymmetries should be kept in view. A violation is a certificate: the
witness state settles the question for good. A clean report establishes only
that no counterexample exists among the instances that could be enumerated,
and is reported as *not falsified* rather than as proof. And enumeration is
exponential in instance size, so the check is available exactly where search is
easy — which is why a hypothesis class whose members are admissible *by
construction* is worth more than a class that has to be tested.

The present class does not have that property. Admissibility forces `h` to
vanish on goal states, where `achieved_goals`, `true_propositions` and
`applicable_actions` are all non-zero, so three of the seven weights must be zero
before anything else is considered; and the sum of two admissible heuristics is
in general not admissible, which leaves little inside `h_θ` to discover.
Constructing an admissible class — cost partitioning over structurally
distinct admissible components — is recorded in `DEVELOPMENT.md`.

### 6.1 The landmark component

A landmark of a state $s$ is a proposition true at some point in every plan
from $s$. Landmarks are generated here by relaxed reachability: $p$ is a
landmark iff the goal is unreachable in the delete relaxation from $s$ once
every action adding $p$ is removed. The test is sound — every real plan is also
a relaxed plan, so a fact all relaxed plans need is a fact all plans need — and
incomplete in two ways that cost informedness but not admissibility: it finds
only single-fact landmarks, and it misses those whose necessity depends on
delete effects.

![Landmark generation by relaxed reachability](docs/figures/landmarks.svg)

**Figure 2.** The test that generates a landmark. Both panels live in the
delete relaxation, where reachability is cheap to decide and monotone, so the
question "is the goal still reachable without $p$?" is answered by one fixpoint
per candidate proposition.

*Counting* the unachieved landmarks is not admissible, because one action may
achieve several at once. The value used is the uniform cost partition over
landmarks (Karpas and Domshlak, 2009): each action divides its cost equally
among the landmarks it can achieve, and each landmark contributes the cheapest
share any of its achievers assigns it,

```math
h_{\mathrm{LM}}(s) \;=\; \sum_{L \,\in\, \mathrm{LM}(s)} \;
\min_{a \,:\, L \in \mathrm{add}(a)}
\frac{c(a)}{\lvert \mathrm{add}(a) \cap \mathrm{LM}(s) \rvert}.
```

Every plan achieves every landmark, and the shares one action hands out sum to
at most its own cost, so the total is a lower bound.

![Counting against pricing landmarks](docs/figures/partition.svg)

**Figure 3.** The case that separates the two, and the task the unit tests use.
`both` achieves the two goals at once, and each goal also has a single-goal
achiever of its own. Counting the unachieved landmarks returns 2 against an
optimal cost of 1, and the verifier falsifies it; dividing the cost of `both`
between the landmarks it achieves returns exactly 1.

Under A\* on the eight instances of seven and eight blocks, all three
admissible baselines return optimal plans of total cost 114:

| Heuristic | Expanded | Reopened | Time |
| --- | ---: | ---: | ---: |
| `goal_count` | 39607 | 0 | 0.05 s |
| `relaxed_layers` | 113684 | 0 | 0.97 s |
| `landmark_cost` | 7074 | 669 | 1.34 s |

The bound is much better informed per node and much more expensive per node:
landmarks are regenerated from scratch at every state, since the framework
requires $h$ to be a function of the state alone, and that costs one relaxed
reachability test per candidate proposition. Making it competitive in runtime,
rather than only in expansions, is a separate problem from making it admissible.

The reopenings are the visible consequence of the one property the verifier
denies it: `landmark_cost` is admissible but *not consistent*. State-generated
landmark sets do not vary monotonically along a transition, so $h$ can fall by
more than the cost of the edge, and A\* must reopen closed states. It remains
correct; it loses the guarantee that a state is closed once.

Because $h_{\mathrm{LM}}$ is linear in the action costs, evaluating it under
costs $w \cdot c$ multiplies it by exactly $w$. That is precisely what a cost
partition over several components requires, and the linear class already
supplies the weight — which is why this is the first component of the class
described in `DEVELOPMENT.md`, and not merely another baseline.

## 7. Reproducibility

Every experiment writes a single self-describing JSON record containing the
random seed, the search algorithm and resource budgets, the objective weights,
the search space, the heuristic definition and its weights, the benchmark
instances, the complete optimisation trajectory, the per-instance metrics, the
planner build provenance (compiler, C++ standard, build type, git commit) and
the environment and timestamp.

The engine itself is deterministic: tie-breaking in the open list is by
insertion order, so repeated runs of the same configuration return identical
metrics. Optimisers take an explicit seed and never touch global random state,
and benchmark instances are a pure function of their generator seed. A
discovered heuristic serialises to JSON or YAML and, reloaded, reproduces its
metrics exactly.

## 8. Repository conventions

The C++ engine lives under `cpp/` (headers in `cpp/include/hd/`, the
executables `hd_plan` and `hd_verify` with the platform code in `cpp/src/`,
unit tests in `cpp/tests/`), the
Python research layer under `python/hd/` with its tests in `python/tests/` and
runnable experiments in `python/hd/experiments/`. Generated instances are in
`instances/`, suite manifests in `benchmarks/`, experiment records in
`results/`, and file-format documentation in `docs/`. The figures are TikZ
sources in `docs/figures/`, with the SVGs they generate committed beside them
so that the README renders without a LaTeX toolchain; `docs/figures/build.sh`
regenerates them.

Design decisions, their justifications, and the next research iteration are
recorded in `DEVELOPMENT.md`. The task file format and the JSON schemas are
specified in `docs/format.md`.

## 9. Licence

MIT. See `LICENSE`.
