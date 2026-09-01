# File formats

Four formats cross component boundaries: the task file read by the engine, the
metrics document it emits, the verification document produced when a heuristic
is checked against exact goal distances, and the experiment record written by
the research layer. All are versioned by a `schema` field where they are JSON.

## 1. Task files (`instances/**/*.task`)

Line-oriented text, one directive per line. Everything after `#` is a comment.
Propositions are referred to by index; indices must be below the declared
count. Directives may repeat, and their effects accumulate.

| Directive | Meaning |
| --- | --- |
| `name NAME` | task name, reported in the metrics |
| `propositions N` | declares `N` propositions; must precede any index |
| `prop I NAME` | names proposition `I` (documentation only) |
| `init I J ...` | propositions true in the initial state |
| `goal I J ...` | goal conditions |
| `action NAME [COST]` | begins an action; cost defaults to `1` |
| `pre I J ...` | preconditions of the current action |
| `add I J ...` | add effects of the current action |
| `del I J ...` | delete effects of the current action |
| `end` | ends the current action |

Semantics follow propositional STRIPS: action `a` is applicable in `s` when
`pre(a) ⊆ s`, and the successor is `(s \ del(a)) ∪ add(a)` — deletes are
applied before adds. A state `s` is a goal state when `goal ⊆ s`.

```
name corridor
propositions 3
prop 0 at_a
prop 1 at_b
prop 2 at_c
init 0
goal 2
action move_a_b 1
pre 0
add 1
del 0
end
```

The compiled proposition capacity is 256 (`hd::kStateWords = 4` in
`cpp/include/hd/strips.hpp`); a task declaring more is rejected.

## 2. Planner output (`schema: hd.planner_result/1`)

Written to stdout, or to `--output`. One document covers every `--instance`
given in the invocation.

```json
{
  "schema": "hd.planner_result/1",
  "timestamp": "2024-01-01T00:00:00Z",
  "search": "gbfs",
  "heuristic": {
    "kind": "linear",
    "expression": "1.82*unsatisfied_goals + 0.37*applicable_actions",
    "weights": { "unsatisfied_goals": 1.82, "applicable_actions": 0.37 }
  },
  "seed": 0,
  "limits": { "max_expansions": 200000, "time_limit_seconds": 30 },
  "build": {
    "version": "0.1.0", "compiler": "g++ 11.4.0", "cxx_standard": "202002",
    "build_type": "Release", "git_commit": "abcdef0", "compiled_at": "..."
  },
  "runs": [
    {
      "instance": "blocks-07-00",
      "instance_path": "instances/blocksworld/blocks-07-00.task",
      "num_propositions": 64, "num_actions": 98, "num_goal_conditions": 7,
      "metrics": {
        "solved": true,
        "status": "solved",
        "solution_cost": 12.0,
        "solution_length": 12,
        "expanded": 143,
        "generated": 512,
        "evaluated": 512,
        "reopened": 0,
        "max_depth": 12,
        "peak_nodes": 512,
        "runtime_seconds": 0.0021,
        "peak_memory_kb": 4096
      },
      "plan": ["unstack_2_1", "putdown_2"]
    }
  ]
}
```

`status` is one of `solved`, `unsolvable`, `expansion_limit`, `time_limit`.
`solution_cost` is `null` unless `solved` is true. `plan` is present only with
`--emit-plan`. `evaluated` counts heuristic evaluations, `reopened` counts
states rediscovered with a lower `g` (A\* only), and `peak_memory_kb` is the
resident-set peak of the whole process, hence an upper bound when several
instances share one invocation.

`hd_plan --list-features` and `hd_plan --build-info` emit the feature registry
and the build provenance respectively, and exit.

## 3. Verifier output (`schema: hd.verification/1`)

Written by `hd_verify` to stdout, or to `--output`. One document covers every
`--instance` given, and each instance carries one report per `--heuristic`, in
the order the heuristics were given.

```json
{
  "schema": "hd.verification/1",
  "timestamp": "2024-01-01T00:00:00Z",
  "limits": { "max_states": 200000, "time_limit_seconds": 0, "tolerance": 1e-09, "witnesses": 3 },
  "build": { "version": "0.1.0", "...": "..." },
  "runs": [
    {
      "instance": "blocks-05-00",
      "instance_path": "instances/blocksworld/blocks-05-00.task",
      "num_propositions": 36, "num_actions": 50, "num_goal_conditions": 5,
      "state_space": {
        "status": "complete",
        "complete": true,
        "states": 866,
        "transitions": 2087,
        "goal_states": 1,
        "dead_end_states": 0,
        "max_h_star": 12,
        "initial_h_star": 12,
        "runtime_seconds": 0.00035
      },
      "heuristics": [
        {
          "heuristic": { "kind": "goal_count", "expression": "goal_count", "weights": {} },
          "report": {
            "admissible": true,
            "consistent": true,
            "goal_aware": true,
            "non_negative": true,
            "states_checked": 866,
            "transitions_checked": 2087,
            "dead_end_states": 0,
            "goal_states": 1,
            "admissibility_violations": 0,
            "consistency_violations": 0,
            "max_excess": 0,
            "max_ratio": 0,
            "max_consistency_excess": 0,
            "max_goal_value": 0,
            "min_value": 0,
            "max_value": 5,
            "mean_informedness": 0.4658,
            "min_informedness": 0.1667,
            "mean_error": 4.3538,
            "admissibility_witnesses": [],
            "consistency_witnesses": []
          }
        }
      ]
    }
  ]
}
```

`state_space.status` is one of `complete`, `state_limit`, `time_limit`. Goal
distances are computed, and `heuristics` is populated, only when the status is
`complete`: a truncated space is missing transitions, which can only make `h*`
look larger than it is and would hide violations rather than report them. An
instance whose space could not be enumerated therefore carries an empty
`heuristics` array and must be reported as unchecked, never as passing.

`initial_h_star` is the optimal cost of the instance, and is `null` when the
task is unsolvable. `max_h_star` is the maximum over reachable states with a
finite distance. `dead_end_states` counts states from which no goal is
reachable; every finite heuristic value is admissible at such a state, so they
are excluded from `states_checked` and from the informedness statistics.

| Field | Meaning |
| --- | --- |
| `admissible` | no reachable state has `h(s) > h*(s)` |
| `consistent` | no transition has `h(u) > c(u,v) + h(v)` |
| `goal_aware` | `h` vanishes on every goal state |
| `non_negative` | `min_value >= 0` |
| `max_excess` | worst `h - h*`; `max_ratio` is the worst `h / h*` over states with `h* > 0` |
| `mean_informedness` | mean `h / h*` over states with a finite, non-zero `h*` |
| `mean_error` | mean `h* - h` over the same states |
| `*_witnesses` | up to `--witnesses` worst violations, each with the propositions true at the offending state |

Verdicts are asymmetric. A violation is a certificate: the witness state proves
the heuristic inadmissible. A clean report proves nothing beyond the instances
that were enumerable, and is reported as "not falsified".

## 4. Benchmark manifests (`schema: hd.benchmark_suite/1`)

```json
{
  "schema": "hd.benchmark_suite/1",
  "name": "blocksworld",
  "description": "...",
  "metadata": { "domain": "blocksworld", "base_seed": 20240101, "...": "..." },
  "instances": ["instances/blocksworld/blocks-05-00.task", "..."]
}
```

Paths are relative to the repository root. The order of `instances` is
significant: it defines the positional train/holdout split.

## 5. Experiment records (`schema: hd.experiment/1`)

Written to `results/NAME-TIMESTAMP.json` by `hd.experiment.build_record`.

| Field | Contents |
| --- | --- |
| `name`, `timestamp`, `seed` | identity of the run |
| `config` | suite, search algorithm, budgets, objective weights, reference heuristic |
| `environment` | Python version, platform, git revision, planner binary and its build info |
| `instances` | the exact instance list used |
| `search_space` | per-feature weight bounds, when optimising |
| `optimization` | optimiser name, seed, best candidate, and the full trial history |
| `evaluations` | per-configuration candidate, objective, summary and per-instance metrics |

## 6. Candidate files

A candidate serialises to JSON, or to YAML when the path ends in `.yaml` or
`.yml` (PyYAML is used when installed; a minimal fallback handles this flat
schema otherwise).

```json
{
  "name": "discovered",
  "weights": { "unsatisfied_goals": 1.82, "relaxed_sum": 0.37 },
  "metadata": {}
}
```

Reloading a candidate and re-running it reproduces its metrics exactly, since
the engine is deterministic.
