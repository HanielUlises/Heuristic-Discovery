// Checking a candidate heuristic against exact goal distances.
//
// Given the enumerated state space of a task (oracle.hpp), a heuristic is
// checked for the three properties that decide whether it can be used for
// optimal planning:
//
//   admissibility   h(s) <= h*(s) for every reachable s
//   consistency     h(u) <= c(u, v) + h(v) for every transition u -> v
//   non-negativity  h(s) >= 0
//
// Consistency implies admissibility when h vanishes on goal states, and is
// what allows A* to close a state permanently on first expansion, so the two
// are reported separately.
//
// Alongside the verdict the report carries informedness, mean h/h* over the
// states with a finite goal distance. Among admissible candidates this is the
// quantity that predicts A* expansions, and unlike expansions it is dense,
// noise-free, and obtainable without running a search.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "hd/json.hpp"
#include "hd/oracle.hpp"
#include "hd/strips.hpp"

namespace hd {

// Floating-point slack. Weights come from an optimiser and features are
// integer-valued counts, so a violation of interest is never this small.
inline constexpr double kVerifyTolerance = 1e-9;

struct VerifyOptions {
  std::size_t max_witnesses = 3;
  double tolerance = kVerifyTolerance;
};

// A state where h overestimates the true goal distance.
struct AdmissibilityWitness {
  std::uint32_t state = 0;
  double h = 0.0;
  double h_star = 0.0;
  double excess = 0.0;  // h - h*
  double ratio = 0.0;   // h / h*, infinite when h* = 0 and h > 0
};

// A transition along which h drops by more than the cost of the edge.
struct ConsistencyWitness {
  std::uint32_t from = 0;
  std::uint32_t to = 0;
  std::uint32_t action = 0;
  double h_from = 0.0;
  double h_to = 0.0;
  double cost = 0.0;
  double excess = 0.0;  // h(u) - (c + h(v))
};

struct HeuristicReport {
  // Population
  std::size_t states = 0;             // reachable states
  std::size_t goal_states = 0;
  std::size_t dead_end_states = 0;    // h* infinite; any finite h is admissible
  std::size_t states_checked = 0;     // states with a finite h*
  std::size_t transitions_checked = 0;

  // Verdict
  std::size_t admissibility_violations = 0;
  std::size_t consistency_violations = 0;
  double max_excess = 0.0;            // worst h - h*
  double max_ratio = 0.0;             // worst h / h* over states with h* > 0
  double max_consistency_excess = 0.0;

  // Diagnostics
  double max_goal_value = 0.0;        // max |h| over goal states; must be 0
  double min_value = 0.0;             // min h over all reachable states
  double max_value = 0.0;

  // Quality, over states with h* > 0 (finite, non-goal)
  double mean_informedness = 0.0;     // mean h / h*
  double min_informedness = 0.0;
  double mean_error = 0.0;            // mean h* - h

  std::vector<AdmissibilityWitness> admissibility_witnesses;
  std::vector<ConsistencyWitness> consistency_witnesses;

  bool admissible() const { return admissibility_violations == 0; }
  bool consistent() const { return consistency_violations == 0; }
  bool goal_aware() const { return max_goal_value <= kVerifyTolerance; }
  bool non_negative() const { return min_value >= -kVerifyTolerance; }
};

namespace detail {

// Keeps the k worst witnesses without retaining the rest: violations can
// number in the millions and only the extremes are diagnostic.
template <class W>
void keep_worst(std::vector<W>& kept, W candidate, std::size_t k) {
  if (k == 0) return;
  if (kept.size() < k) {
    kept.push_back(candidate);
  } else {
    auto worst = std::min_element(kept.begin(), kept.end(),
                                  [](const W& a, const W& b) { return a.excess < b.excess; });
    if (candidate.excess <= worst->excess) return;
    *worst = candidate;
  }
  std::sort(kept.begin(), kept.end(),
            [](const W& a, const W& b) { return a.excess > b.excess; });
}

}  // namespace detail

// Evaluates `h` on every reachable state and checks it against the oracle.
// The state space must be complete: a truncated one is missing transitions,
// which inflates h* and would hide violations rather than reporting them.
template <class H>
HeuristicReport verify_heuristic(const StateSpace& space, const H& h,
                                 const VerifyOptions& options = {}) {
  HeuristicReport r;
  r.states = space.size();
  r.goal_states = space.num_goal_states();
  r.dead_end_states = space.num_dead_ends();
  if (space.size() == 0) return r;

  std::vector<double> values(space.size());
  for (std::size_t i = 0; i < space.size(); ++i) values[i] = h(space.states[i]);

  r.min_value = values[0];
  r.max_value = values[0];
  double informedness_sum = 0.0;
  double error_sum = 0.0;
  std::size_t informedness_count = 0;
  r.min_informedness = kInfiniteDistance;

  for (std::size_t i = 0; i < space.size(); ++i) {
    const double hi = values[i];
    const double star = space.h_star[i];
    r.min_value = std::min(r.min_value, hi);
    r.max_value = std::max(r.max_value, hi);
    if (space.is_goal[i]) r.max_goal_value = std::max(r.max_goal_value, std::fabs(hi));
    if (star == kInfiniteDistance) continue;  // dead end: nothing to overestimate

    ++r.states_checked;
    const double excess = hi - star;
    if (excess > options.tolerance) {
      ++r.admissibility_violations;
      r.max_excess = std::max(r.max_excess, excess);
      // A goal state has h* = 0, where no ratio is defined; that case is
      // reported by max_goal_value and max_excess instead.
      const double ratio = star > 0.0 ? hi / star : kInfiniteDistance;
      if (star > 0.0) r.max_ratio = std::max(r.max_ratio, ratio);
      detail::keep_worst(r.admissibility_witnesses,
                         AdmissibilityWitness{static_cast<std::uint32_t>(i), hi, star, excess,
                                              ratio},
                         options.max_witnesses);
    }
    if (star > 0.0) {
      informedness_sum += hi / star;
      error_sum += star - hi;
      r.min_informedness = std::min(r.min_informedness, hi / star);
      ++informedness_count;
    }
  }

  if (informedness_count > 0) {
    r.mean_informedness = informedness_sum / static_cast<double>(informedness_count);
    r.mean_error = error_sum / static_cast<double>(informedness_count);
  } else {
    r.min_informedness = 0.0;
  }

  for (const Transition& t : space.transitions) {
    ++r.transitions_checked;
    const double excess = values[t.from] - (t.cost + values[t.to]);
    if (excess > options.tolerance) {
      ++r.consistency_violations;
      r.max_consistency_excess = std::max(r.max_consistency_excess, excess);
      detail::keep_worst(r.consistency_witnesses,
                         ConsistencyWitness{t.from, t.to, t.action, values[t.from], values[t.to],
                                            t.cost, excess},
                         options.max_witnesses);
    }
  }
  return r;
}

// --- serialisation ------------------------------------------------------

inline std::string witness_json(const StripsTask& task, const StateSpace& space,
                                const AdmissibilityWitness& w) {
  json::Object o;
  o.set("h", w.h);
  o.set("h_star", w.h_star);
  o.set("excess", w.excess);
  o.set("ratio", w.ratio);
  o.set("state", state_proposition_names(task, space.states[w.state]));
  return o.str();
}

inline std::string witness_json(const StripsTask& task, const StateSpace& space,
                                const ConsistencyWitness& w) {
  json::Object o;
  o.set("action", task.action(w.action).name);
  o.set("cost", w.cost);
  o.set("h_from", w.h_from);
  o.set("h_to", w.h_to);
  o.set("excess", w.excess);
  o.set("from_state", state_proposition_names(task, space.states[w.from]));
  o.set("to_state", state_proposition_names(task, space.states[w.to]));
  return o.str();
}

template <class W>
std::string witness_array_json(const StripsTask& task, const StateSpace& space,
                               const std::vector<W>& witnesses) {
  std::string arr = "[";
  for (std::size_t i = 0; i < witnesses.size(); ++i) {
    if (i) arr += ",";
    arr += witness_json(task, space, witnesses[i]);
  }
  return arr + "]";
}

inline std::string state_space_json(const StateSpace& space) {
  json::Object o;
  o.set("status", to_string(space.status));
  o.set("complete", space.complete());
  o.set("states", space.size());
  o.set("transitions", space.transitions.size());
  o.set("goal_states", space.num_goal_states());
  o.set("dead_end_states", space.num_dead_ends());
  o.set("max_h_star", space.complete() ? space.max_finite_h_star() : 0.0);
  o.set("initial_h_star", space.complete() ? space.initial_h_star() : kInfiniteDistance);
  o.set("runtime_seconds", space.runtime_seconds);
  return o.str();
}

inline std::string report_json(const StripsTask& task, const StateSpace& space,
                               const HeuristicReport& r) {
  json::Object o;
  o.set("admissible", r.admissible());
  o.set("consistent", r.consistent());
  o.set("goal_aware", r.goal_aware());
  o.set("non_negative", r.non_negative());
  o.set("states_checked", r.states_checked);
  o.set("transitions_checked", r.transitions_checked);
  o.set("dead_end_states", r.dead_end_states);
  o.set("goal_states", r.goal_states);
  o.set("admissibility_violations", r.admissibility_violations);
  o.set("consistency_violations", r.consistency_violations);
  o.set("max_excess", r.max_excess);
  o.set("max_ratio", r.max_ratio);
  o.set("max_consistency_excess", r.max_consistency_excess);
  o.set("max_goal_value", r.max_goal_value);
  o.set("min_value", r.min_value);
  o.set("max_value", r.max_value);
  o.set("mean_informedness", r.mean_informedness);
  o.set("min_informedness", r.min_informedness);
  o.set("mean_error", r.mean_error);
  o.raw("admissibility_witnesses",
        witness_array_json(task, space, r.admissibility_witnesses));
  o.raw("consistency_witnesses", witness_array_json(task, space, r.consistency_witnesses));
  return o.str();
}

}  // namespace hd
