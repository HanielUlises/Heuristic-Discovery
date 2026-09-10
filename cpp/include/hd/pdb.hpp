// Pattern databases: exact goal distances of a projection.
//
// A pattern is a subset P of the propositions. Projecting a task onto P keeps
// only the part of every state, precondition and effect that lies inside P:
//
//     proj(s) = s cap P,
//     proj(a) = < pre(a) cap P, add(a) cap P, del(a) cap P >, cost unchanged.
//
// The projection is a homomorphism. Applying a projected action to a projected
// state gives the projection of the concrete successor, because intersection
// distributes over the STRIPS transition, and a projected precondition is
// weaker than the concrete one, so the abstract action is applicable whenever
// the concrete one is. Every concrete plan therefore maps to an abstract plan
// of the same cost, the abstract goal distance is at most the concrete one,
// and reading it off is admissible.
//
// The table is the exact goal distance of every abstract state, computed by
// the same enumeration and backward Dijkstra that oracle.hpp runs on the
// concrete task. The cost of the abstraction is exponential in |P|, so a
// pattern has to stay small; what makes a database worth building is that the
// cost is paid once, before search, and every later evaluation is a mask and a
// lookup.
//
// A pattern database is a component of a cost partition in the same sense the
// landmark bound is: it is built under a supplied cost function, and its value
// is a lower bound on plan cost under those costs.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hd/costs.hpp"
#include "hd/oracle.hpp"
#include "hd/strips.hpp"

namespace hd {

struct Projection {
  StripsTask task;
  ActionCosts costs;
  // Which action of the original task each kept action came from. Projection
  // drops the actions that touch nothing inside the pattern, so the indices of
  // the abstract task are not those of the concrete one, and a cost computed
  // in the abstraction cannot be returned to the task without this. Nothing
  // needed it until saturation did.
  std::vector<std::size_t> origin;
};

// Whether the projection onto `pattern` keeps `action`, which it does exactly
// when the action changes something the pattern can see. Stated separately
// from `project` because a cost partition has to decide which components can
// use an action before any of them is built.
inline bool touches(const StripsAction& action, const StripsState& pattern) {
  return action.add.count_common(pattern) > 0 || action.del.count_common(pattern) > 0;
}

// Projects a task and its costs onto a pattern. Actions that neither add nor
// delete anything inside the pattern become self-loops in the abstraction and
// are dropped: they cannot shorten an abstract distance, and keeping them
// would multiply the transitions to enumerate.
inline Projection project(const StripsTask& task, const StripsState& pattern,
                          const ActionCosts& costs) {
  StripsTask abstracted;
  abstracted.set_name(task.name() + "|projection");
  abstracted.set_num_propositions(task.num_propositions());
  for (std::size_t p = 0; p < task.num_propositions(); ++p) {
    if (pattern.test(p)) abstracted.set_proposition_name(p, task.proposition_name(p));
  }

  StripsState init = task.initial_state();
  init.and_with(pattern);
  abstracted.set_initial_state(init);

  StripsState goal = task.goal();
  goal.and_with(pattern);
  abstracted.set_goal(goal);

  std::vector<double> kept;
  std::vector<std::size_t> origin;
  kept.reserve(task.num_actions());
  origin.reserve(task.num_actions());
  for (std::size_t a = 0; a < task.num_actions(); ++a) {
    if (!touches(task.action(a), pattern)) continue;
    StripsAction act = task.action(a);
    act.pre.and_with(pattern);
    act.add.and_with(pattern);
    act.del.and_with(pattern);
    act.cost = costs[a];
    abstracted.add_action(std::move(act));
    kept.push_back(costs[a]);
    origin.push_back(a);
  }
  return Projection{std::move(abstracted), ActionCosts(std::move(kept)), std::move(origin)};
}

class PatternDatabase {
 public:
  // Builds the database. `max_states` bounds the abstract state space; a
  // pattern that exceeds it is an error rather than a silently truncated
  // table, since a missing entry would be read as a dead end.
  PatternDatabase(const StripsTask& task, const StripsState& pattern, const ActionCosts& costs,
                  std::size_t max_states = 1u << 20)
      : pattern_(pattern) {
    const Projection projection = project(task, pattern, costs);
    OracleLimits limits;
    limits.max_states = max_states;
    const StateSpace space =
        enumerate_state_space(projection.task, limits, projection.costs, true);
    if (!space.complete()) {
      throw std::runtime_error("pattern too large: the abstract state space exceeds the ceiling");
    }
    table_.reserve(space.size());
    for (std::size_t i = 0; i < space.size(); ++i) table_.emplace(space.states[i], space.h_star[i]);
    abstract_transitions_ = space.transitions.size();
    saturated_ = saturate(space, projection, task.num_actions(), costs);
  }

  // Satisfies the Heuristic concept, so a database can be searched with and
  // verified exactly like any other heuristic.
  double operator()(const StripsState& s) const {
    StripsState key = s;
    key.and_with(pattern_);
    const auto it = table_.find(key);
    // A state whose projection was never enumerated is unreachable in the
    // abstraction and therefore unreachable in the task, so no search will
    // ever evaluate it. Reporting it as unbounded is safe.
    return it == table_.end() ? kUnboundedCost : it->second;
  }

  const StripsState& pattern() const { return pattern_; }
  std::size_t size() const { return table_.size(); }
  std::size_t abstract_transitions() const { return abstract_transitions_; }

  // The least cost function under which this database keeps every estimate it
  // has, expressed over the actions of the original task.
  //
  // This is what makes a partition worth more than a maximum. A database
  // charges an action only for the drop in abstract goal distance the action
  // can produce; whatever the action costs beyond that is cost this database
  // is not using, and it can go to another component without either of them
  // charging for the same thing twice.
  const ActionCosts& saturated_costs() const { return saturated_; }

  // What is left of `costs` once this database has taken its saturated share.
  ActionCosts remainder(const ActionCosts& costs) const {
    std::vector<double> left(costs.size());
    for (std::size_t a = 0; a < costs.size(); ++a) {
      left[a] = std::max(0.0, costs[a] - saturated_[a]);
    }
    return ActionCosts(std::move(left));
  }

 private:
  // scf(a) = max over abstract transitions on a of (h*(from) - h*(to)),
  // clipped below at zero and above at what the action was given.
  //
  // The clip at the given cost is what keeps the result a share rather than a
  // demand: the drop can exceed the cost only where several actions were
  // needed to produce it, and the projection has folded them together.
  //
  // Two kinds of transition impose no constraint and are skipped. One out of a
  // dead end: h*(from) is infinite, no search will expand the state, and the
  // difference says nothing. One into a dead end: h*(to) is infinite, so
  // h*(from) - h*(to) is negative without bound, and the requirement that
  // scf(a) be at least that is met by every non-negative number. Charging for
  // either would take cost from the components that come after this one for
  // no gain here.
  //
  // What the survivors say is that this database keeps its whole table under
  // scf. Every transition then satisfies h*(from) <= scf(a) + h*(to) and every
  // goal state has h* = 0, so the table is consistent and goal-aware under
  // scf; along any scf-optimal path to a goal every state has finite h*, and
  // telescoping the inequality along it gives h* <= h*_scf. The database is
  // therefore admissible under a cost function that may be far smaller than
  // the one it was built with, and the difference is what the next component
  // gets to spend.
  static ActionCosts saturate(const StateSpace& space, const Projection& projection,
                              std::size_t num_actions, const ActionCosts& given) {
    std::vector<double> scf(num_actions, 0.0);
    for (const Transition& t : space.transitions) {
      const double from = space.h_star[t.from];
      const double to = space.h_star[t.to];
      if (!std::isfinite(from) || !std::isfinite(to)) continue;
      const double drop = from - to;
      if (!(drop > 0.0)) continue;
      const std::size_t a = projection.origin[t.action];
      if (drop > scf[a]) scf[a] = drop;
    }
    for (std::size_t a = 0; a < num_actions; ++a) {
      if (scf[a] > given[a]) scf[a] = given[a];
    }
    return ActionCosts(std::move(scf));
  }


  StripsState pattern_;
  std::unordered_map<StripsState, double> table_;
  std::size_t abstract_transitions_ = 0;
  ActionCosts saturated_{std::vector<double>{}};
};

// A pattern seeded by one proposition and grown over the preconditions of the
// actions that achieve it, then over the preconditions of the actions that
// achieve those, until `max_size` propositions have been collected. The
// closure is what makes the abstraction see why a proposition is hard to
// reach: with the seed alone, every achiever has an empty precondition and the
// database can only count.
//
// Propositions are added in index order and the seed is processed first, so
// the result is a deterministic function of the task and the budget.
//
// Pattern selection is a research problem of its own and this is not an
// attempt at solving it. It is a defensible default to build components from.
inline StripsState causal_pattern(const StripsTask& task, std::size_t seed,
                                  std::size_t max_size = 12) {
  StripsState pattern;
  if (max_size == 0) return pattern;
  pattern.set(seed);
  std::vector<std::size_t> frontier{seed};
  std::size_t size = 1;

  for (std::size_t i = 0; i < frontier.size() && size < max_size; ++i) {
    const std::size_t target = frontier[i];
    for (std::size_t a = 0; a < task.num_actions() && size < max_size; ++a) {
      const StripsAction& act = task.action(a);
      if (!act.add.test(target)) continue;
      for (std::size_t p = 0; p < task.num_propositions() && size < max_size; ++p) {
        if (!act.pre.test(p) || pattern.test(p)) continue;
        pattern.set(p);
        frontier.push_back(p);
        ++size;
      }
    }
  }
  return pattern;
}

// One pattern per goal condition.
inline std::vector<StripsState> goal_patterns(const StripsTask& task, std::size_t max_size = 12) {
  std::vector<StripsState> patterns;
  for (std::size_t p = 0; p < task.num_propositions(); ++p) {
    if (task.goal().test(p)) patterns.push_back(causal_pattern(task, p, max_size));
  }
  return patterns;
}

}  // namespace hd
