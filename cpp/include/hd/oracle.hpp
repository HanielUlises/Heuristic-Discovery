// Exact goal distances by exhaustive enumeration of the reachable state space.
//
// Admissibility is a statement about h*(s), and no search over a single
// instance reports it. On instances small enough to enumerate, h* is
// computable exactly: expand every reachable state, then run a backward
// Dijkstra from the goal states over the reversed transition relation. The
// result is the ground truth against which a candidate heuristic is checked.
//
// What this establishes is falsification, not proof. A heuristic that is
// admissible on every enumerated instance may still overestimate on an
// instance too large to enumerate; a single violating state, by contrast, is
// a certificate that settles the question.
#pragma once

#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hd/search/result.hpp"
#include "hd/strips.hpp"

namespace hd {

// Infinity marks a state from which no goal state is reachable. Such a state
// is a true dead end, and every finite heuristic value is admissible there.
inline constexpr double kInfiniteDistance = std::numeric_limits<double>::infinity();

enum class OracleStatus { kComplete, kStateLimit, kTimeLimit };

inline const char* to_string(OracleStatus s) {
  switch (s) {
    case OracleStatus::kComplete: return "complete";
    case OracleStatus::kStateLimit: return "state_limit";
    case OracleStatus::kTimeLimit: return "time_limit";
  }
  return "unknown";
}

struct OracleLimits {
  // Enumeration is quadratic in nothing but still stores every state and every
  // transition, so the ceiling is memory. 200k states of a Blocksworld task
  // cost a few tens of megabytes including the transition relation.
  std::size_t max_states = 200000;
  double time_limit_seconds = 0.0;  // <= 0 = unlimited
};

struct Transition {
  std::uint32_t from = 0;
  std::uint32_t to = 0;
  std::uint32_t action = 0;
  double cost = 1.0;
};

// The reachable state space of one task together with exact goal distances.
//
// Goal states are terminal: they are recorded but never expanded, matching
// what a search does when it reaches one. Consequently `states` is the set of
// states an A* run could evaluate, which is exactly the set over which
// admissibility has to hold.
struct StateSpace {
  std::vector<StripsState> states;
  std::vector<double> h_star;         // kInfiniteDistance where no goal is reachable
  std::vector<std::uint8_t> is_goal;  // vector<bool> is not worth the packing here
  std::vector<Transition> transitions;
  OracleStatus status = OracleStatus::kComplete;
  double runtime_seconds = 0.0;

  // Distances are trustworthy only if enumeration ran to completion: a
  // truncated space is missing transitions, and a missing transition can only
  // make h* look larger than it is, which would hide violations.
  bool complete() const { return status == OracleStatus::kComplete; }

  std::size_t size() const { return states.size(); }

  std::size_t num_goal_states() const {
    std::size_t n = 0;
    for (const std::uint8_t g : is_goal) n += g;
    return n;
  }

  std::size_t num_dead_ends() const {
    std::size_t n = 0;
    for (const double d : h_star) n += (d == kInfiniteDistance) ? 1 : 0;
    return n;
  }

  // The optimal cost of the instance itself: index 0 is the initial state,
  // which is interned before any other.
  double initial_h_star() const { return h_star.empty() ? kInfiniteDistance : h_star[0]; }

  double max_finite_h_star() const {
    double m = 0.0;
    for (const double d : h_star) {
      if (d != kInfiniteDistance && d > m) m = d;
    }
    return m;
  }
};

namespace detail {

// Backward Dijkstra from every goal state over the reversed transitions.
// Action costs are positive, so a settled distance is final.
inline void compute_goal_distances(StateSpace& space) {
  const std::size_t n = space.size();
  space.h_star.assign(n, kInfiniteDistance);
  if (n == 0) return;

  // Predecessor lists in compressed form: pred_begin[v]..pred_begin[v+1]
  // indexes the transitions entering v.
  std::vector<std::uint32_t> pred_begin(n + 1, 0);
  for (const Transition& t : space.transitions) ++pred_begin[t.to + 1];
  std::partial_sum(pred_begin.begin(), pred_begin.end(), pred_begin.begin());

  std::vector<std::uint32_t> pred(space.transitions.size());
  std::vector<double> pred_cost(space.transitions.size());
  std::vector<std::uint32_t> cursor(pred_begin.begin(), pred_begin.end() - 1);
  for (const Transition& t : space.transitions) {
    const std::uint32_t slot = cursor[t.to]++;
    pred[slot] = t.from;
    pred_cost[slot] = t.cost;
  }

  using Entry = std::pair<double, std::uint32_t>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  for (std::uint32_t id = 0; id < n; ++id) {
    if (space.is_goal[id]) {
      space.h_star[id] = 0.0;
      open.emplace(0.0, id);
    }
  }

  while (!open.empty()) {
    const auto [d, id] = open.top();
    open.pop();
    if (d > space.h_star[id]) continue;  // stale entry
    for (std::uint32_t i = pred_begin[id]; i < pred_begin[id + 1]; ++i) {
      const std::uint32_t p = pred[i];
      const double candidate = d + pred_cost[i];
      if (candidate < space.h_star[p]) {
        space.h_star[p] = candidate;
        open.emplace(candidate, p);
      }
    }
  }
}

}  // namespace detail

// Enumerates every state reachable from the initial state and annotates each
// with its exact cost to the nearest goal state.
inline StateSpace enumerate_state_space(const StripsTask& task, const OracleLimits& limits = {}) {
  Timer timer;
  StateSpace space;
  std::unordered_map<StripsState, std::uint32_t> index;

  const auto intern = [&](const StripsState& s) -> std::uint32_t {
    const auto [it, inserted] =
        index.emplace(s, static_cast<std::uint32_t>(space.states.size()));
    if (inserted) {
      space.states.push_back(s);
      space.is_goal.push_back(task.is_goal(s) ? 1 : 0);
    }
    return it->second;
  };

  intern(task.initial_state());

  // The state vector is its own worklist: successors are appended as they are
  // discovered, so visiting indices in order reaches every reachable state.
  for (std::size_t id = 0; id < space.states.size(); ++id) {
    if (limits.max_states && space.states.size() > limits.max_states) {
      space.status = OracleStatus::kStateLimit;
      break;
    }
    if (limits.time_limit_seconds > 0.0 && (id % 1024) == 0 &&
        timer.seconds() > limits.time_limit_seconds) {
      space.status = OracleStatus::kTimeLimit;
      break;
    }
    if (space.is_goal[id]) continue;  // terminal: a search stops here too

    const StripsState s = space.states[id];  // by value; interning may reallocate
    for (std::size_t a = 0; a < task.num_actions(); ++a) {
      const StripsAction& act = task.action(a);
      if (!task.applicable(s, act)) continue;
      const std::uint32_t to = intern(task.apply(s, act));
      space.transitions.push_back({static_cast<std::uint32_t>(id), to,
                                   static_cast<std::uint32_t>(a), task.cost(act)});
    }
  }

  if (space.complete()) detail::compute_goal_distances(space);
  space.runtime_seconds = timer.seconds();
  return space;
}

// The propositions true in a state, by name: how a witness is reported.
inline std::vector<std::string> state_proposition_names(const StripsTask& task,
                                                        const StripsState& s) {
  std::vector<std::string> names;
  for (std::size_t p = 0; p < task.num_propositions(); ++p) {
    if (s.test(p)) names.push_back(task.proposition_name(p));
  }
  return names;
}

}  // namespace hd
