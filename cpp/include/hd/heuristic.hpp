// Heuristics satisfying the Heuristic<State> concept.
//
// A candidate heuristic in this framework is the linear form
//     h_theta(s) = sum_i w_i f_i(s)
// over the features declared in features.hpp. Baselines are provided for
// reference points in experiments.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "hd/costs.hpp"
#include "hd/features.hpp"
#include "hd/landmarks.hpp"
#include "hd/pdb.hpp"
#include "hd/strips.hpp"

namespace hd {

// h(s) = 0: turns A* into uniform-cost search and GBFS into an arbitrary
// tie-broken traversal. The control condition for every experiment.
struct ZeroHeuristic {
  double operator()(const StripsState&) const { return 0.0; }
};

// h(s) = number of unsatisfied goal conditions.
class GoalCountHeuristic {
 public:
  explicit GoalCountHeuristic(const StripsTask& task) : goal_(task.goal()) {}
  double operator()(const StripsState& s) const {
    return static_cast<double>(s.count_missing(goal_));
  }

 private:
  StripsState goal_;
};

// h(s) = number of layers of the delete-relaxed planning graph needed to reach
// all goals. Domain-independent, admissible, and the strongest baseline here.
class RelaxedLayersHeuristic {
 public:
  explicit RelaxedLayersHeuristic(const StripsTask& task) : eval_(task) {}
  double operator()(const StripsState& s) const {
    return eval_.evaluate(FeatureId::kRelaxedLayers, s);
  }

 private:
  FeatureEvaluator eval_;
};

// h(s) = the uniform cost partition over the landmarks of s. Admissible and,
// unlike the relaxed-graph baselines, a lower bound that prices actions rather
// than counting layers. It is also the first component intended to enter a
// cost partition over several heuristics, where its weight is its share of the
// action costs.
class LandmarkCostHeuristic {
 public:
  explicit LandmarkCostHeuristic(const StripsTask& task) : eval_(task) {}
  double operator()(const StripsState& s) const {
    return eval_.evaluate(FeatureId::kLandmarkCost, s);
  }

 private:
  FeatureEvaluator eval_;
};

// h(s) = max over a set of pattern databases. The maximum of admissible
// heuristics is admissible, and it is the control a cost partition over the
// same components has to beat: a partition earns its keep only by exceeding
// what simply taking the best of its parts already gives.
//
// The databases are built once, when the heuristic is constructed, and every
// evaluation afterwards is a mask and a lookup per pattern.
class MaxPatternDatabaseHeuristic {
 public:
  explicit MaxPatternDatabaseHeuristic(const StripsTask& task, std::size_t max_pattern_size = 12)
      : MaxPatternDatabaseHeuristic(task, ActionCosts::of(task), max_pattern_size) {}

  MaxPatternDatabaseHeuristic(const StripsTask& task, const ActionCosts& costs,
                              std::size_t max_pattern_size) {
    for (const StripsState& pattern : goal_patterns(task, max_pattern_size)) {
      databases_.emplace_back(task, pattern, costs);
    }
  }

  double operator()(const StripsState& s) const {
    double best = 0.0;
    for (const PatternDatabase& pdb : databases_) {
      const double v = pdb(s);
      if (v == kUnboundedCost) return kUnreachableValue;  // a proven dead end
      if (v > best) best = v;
    }
    return best;
  }

  std::size_t num_databases() const { return databases_.size(); }

 private:
  std::vector<PatternDatabase> databases_;
};

// h(s) = sum over a set of pattern databases built under the uniform cost
// partition: every action divides its cost equally among the components whose
// projection keeps it, and gives nothing to the ones that drop it.
//
// This is the control for the saturated construction, and the simplest scheme
// that is a partition in the general sense rather than a scaling. Unlike a
// per-component weight it can exceed the maximum, because the shares are
// disjoint; unlike the saturated one it is blind to what a component can
// actually spend, so a component that needs the whole of an action's cost gets
// a fraction of it and every component pays for that.
class UniformPatternDatabaseHeuristic {
 public:
  explicit UniformPatternDatabaseHeuristic(const StripsTask& task,
                                           std::size_t max_pattern_size = 12)
      : UniformPatternDatabaseHeuristic(task, goal_patterns(task, max_pattern_size)) {}

  UniformPatternDatabaseHeuristic(const StripsTask& task,
                                  const std::vector<StripsState>& patterns) {
    const ActionCosts full = ActionCosts::of(task);
    std::vector<double> users(task.num_actions(), 0.0);
    for (const StripsState& pattern : patterns) {
      for (std::size_t a = 0; a < task.num_actions(); ++a) {
        if (touches(task.action(a), pattern)) users[a] += 1.0;
      }
    }
    shares_.reserve(patterns.size());
    databases_.reserve(patterns.size());
    for (const StripsState& pattern : patterns) {
      std::vector<double> share(task.num_actions(), 0.0);
      for (std::size_t a = 0; a < task.num_actions(); ++a) {
        // users[a] is at least one whenever the test passes, so the division
        // is guarded by the same condition that makes the share meaningful.
        if (touches(task.action(a), pattern)) share[a] = full[a] / users[a];
      }
      shares_.emplace_back(std::move(share));
      databases_.emplace_back(task, pattern, shares_.back());
    }
  }

  double operator()(const StripsState& s) const {
    double total = 0.0;
    for (const PatternDatabase& pdb : databases_) {
      const double v = pdb(s);
      if (v == kUnboundedCost) return kUnreachableValue;  // a proven dead end
      total += v;
    }
    return total;
  }

  std::size_t num_databases() const { return databases_.size(); }
  const std::vector<ActionCosts>& shares() const { return shares_; }

 private:
  std::vector<PatternDatabase> databases_;
  std::vector<ActionCosts> shares_;
};

// h(s) = sum over a set of pattern databases built under a saturated cost
// partition.
//
// The construction is Seipp and Helmert's. Take the components in some order;
// give the first the task's own costs; let it keep only the cost it can
// actually use, which is its saturated cost function; hand the remainder to
// the next, and so on. Every action's cost is then divided among the
// components with nothing counted twice, so the sum of what they report is a
// lower bound on plan cost. Where the maximum picks one component and discards
// the rest, this adds them, and the sum can exceed the maximum exactly because
// the shares are disjoint.
//
// The order matters and is the only free parameter. A component early in the
// order sees the full costs and saturates against them; one late in the order
// may find nothing left to charge for and contribute zero. That order is a
// permutation of the components: small, interpretable, and the object this
// project should be discovering in place of a weight vector.
//
// Admissibility does not depend on the order, on the patterns, or on the
// number of components. It depends only on the shares summing to at most the
// action's cost, which `is_cost_partition` checks and which the construction
// maintains by handing on remainders that are clipped at zero.
//
// What does depend on the order is whether this beats the maximum, and it does
// not always. A component that would have been the maximum can be starved by
// the components before it and return zero, and the sum is then below the
// maximum at that state. The `joint` fixture exhibits it: at {start, left} the
// first database has already reached its abstract goal and takes the cost of
// `both` anyway, which leaves the second database able to reach its own goal
// for nothing. `RotatedSaturatedPatternDatabaseHeuristic` is the answer to
// that, and this class is what it is built out of.
class SaturatedPatternDatabaseHeuristic {
 public:
  explicit SaturatedPatternDatabaseHeuristic(const StripsTask& task,
                                             std::size_t max_pattern_size = 12)
      : SaturatedPatternDatabaseHeuristic(task, goal_patterns(task, max_pattern_size)) {}

  // The patterns in the order they are to be saturated in.
  SaturatedPatternDatabaseHeuristic(const StripsTask& task,
                                    const std::vector<StripsState>& patterns)
      : remaining_(ActionCosts::of(task)) {
    shares_.reserve(patterns.size());
    for (const StripsState& pattern : patterns) {
      databases_.emplace_back(task, pattern, remaining_);
      shares_.push_back(databases_.back().saturated_costs());
      remaining_ = databases_.back().remainder(remaining_);
    }
  }

  double operator()(const StripsState& s) const {
    double total = 0.0;
    for (const PatternDatabase& pdb : databases_) {
      const double v = pdb(s);
      if (v == kUnboundedCost) return kUnreachableValue;  // a proven dead end
      total += v;
    }
    return total;
  }

  std::size_t num_databases() const { return databases_.size(); }

  // The shares handed to each component, in order. Exposed so that a run can
  // assert the partition property against the task it was built for instead of
  // trusting the construction, which is the whole point of having the check.
  const std::vector<ActionCosts>& shares() const { return shares_; }

  // What no database took. Every saturated share is at most what its component
  // was given, so the chain spends exactly `c - remaining()` and this is free
  // for a component of another kind to use.
  const ActionCosts& remaining() const { return remaining_; }

 private:
  std::vector<PatternDatabase> databases_;
  std::vector<ActionCosts> shares_;
  ActionCosts remaining_;
};

// h(s) = max over the cyclic rotations of the pattern order of the saturated
// cost partition over them.
//
// A maximum of admissible heuristics is admissible, so nothing about the
// bound has to be re-established. What the rotations buy is the domination
// that a single order does not give:
//
//     for every state s,  h(s) >= max_i PDB_i(s).
//
// In the rotation that begins at component i, component i is built under the
// task's own costs and therefore reports exactly what it reports on its own,
// and every component after it contributes a non-negative amount on top. So
// that rotation alone is at least PDB_i(s), and the maximum over rotations is
// at least the maximum over i. The starvation that makes a fixed order lose to
// the maximum is undone by the rotation in which the starved component goes
// first.
//
// The cost is k orders of k databases for k goal conditions, paid once before
// search. It is the smallest family of orders with that guarantee: dropping
// any rotation drops the only order in which its leading component is unpriced
// by anything else.
class RotatedSaturatedPatternDatabaseHeuristic {
 public:
  explicit RotatedSaturatedPatternDatabaseHeuristic(const StripsTask& task,
                                                    std::size_t max_pattern_size = 12)
      : RotatedSaturatedPatternDatabaseHeuristic(task, goal_patterns(task, max_pattern_size)) {}

  RotatedSaturatedPatternDatabaseHeuristic(const StripsTask& task,
                                           const std::vector<StripsState>& patterns) {
    const std::size_t k = patterns.size();
    orders_.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
      std::vector<StripsState> rotated;
      rotated.reserve(k);
      for (std::size_t j = 0; j < k; ++j) rotated.push_back(patterns[(i + j) % k]);
      orders_.emplace_back(task, rotated);
    }
  }

  double operator()(const StripsState& s) const {
    double best = 0.0;
    for (const SaturatedPatternDatabaseHeuristic& order : orders_) {
      const double v = order(s);
      if (v == kUnreachableValue) return kUnreachableValue;  // a proven dead end
      if (v > best) best = v;
    }
    return best;
  }

  std::size_t num_orders() const { return orders_.size(); }
  const std::vector<SaturatedPatternDatabaseHeuristic>& orders() const { return orders_; }

 private:
  std::vector<SaturatedPatternDatabaseHeuristic> orders_;
};

// h(s) = max over a family of partitions, each of which gives part of the
// action costs to a chain of pattern databases and everything left over to the
// landmark bound.
//
// The two components of chapter 6 are of different kinds and are good at
// different things: the databases are consistent and cheap and see the
// interactions inside a pattern, the landmark bound is better informed and
// sees facts every plan needs. Taking the maximum of the two discards one of
// them at every state. A partition does not have to.
//
// A landmark bound cannot be saturated the way an abstraction can. An
// abstraction stores a whole function, so the least cost preserving it is a
// property of the table; the landmarks of a state are recomputed at that state
// and there is no stored function to preserve. So the chain runs in the only
// order available: the databases saturate first, and the landmark bound is
// priced with what they did not spend, which needs no saturation because
// nothing comes after it.
//
// The family has k + 1 members for k goal conditions: the k rotations of the
// database order, and the degenerate partition in which the landmark bound
// takes the whole cost function and no database is paid at all. That last
// member is `landmark_cost`, and including it is what makes the maximum
// dominate it. Dominating the maximum over databases comes from the rotations
// exactly as before, since the landmark term is never negative. So this is at
// least both of the controls of chapter 6 at every state, and the measurements
// say by how much.
class SaturatedMixedHeuristic {
 public:
  explicit SaturatedMixedHeuristic(const StripsTask& task, std::size_t max_pattern_size = 12)
      : landmarks_(task), rotations_(task, max_pattern_size), full_(ActionCosts::of(task)) {
    leftovers_.reserve(rotations_.num_orders());
    for (const SaturatedPatternDatabaseHeuristic& order : rotations_.orders()) {
      leftovers_.push_back(order.remaining());
    }
  }

  double operator()(const StripsState& s) const {
    // The landmarks of s are generated once and priced k + 1 times: the set is
    // a question about relaxed reachability and does not depend on the costs,
    // and the factory caches it against the state it was asked about.
    const double alone = landmarks_.value_under(full_, s);
    if (alone == kUnboundedCost) return kUnreachableValue;  // a proven dead end
    double best = alone;
    const std::vector<SaturatedPatternDatabaseHeuristic>& orders = rotations_.orders();
    for (std::size_t i = 0; i < orders.size(); ++i) {
      const double abstraction = orders[i](s);
      if (abstraction == kUnreachableValue) return kUnreachableValue;
      const double landmark = landmarks_.value_under(leftovers_[i], s);
      if (landmark == kUnboundedCost) return kUnreachableValue;
      const double total = abstraction + landmark;
      if (total > best) best = total;
    }
    return best;
  }

  std::size_t num_orders() const { return rotations_.num_orders() + 1; }

  // The partition used by rotation `i`: the databases' saturated shares
  // followed by the landmark bound's leftover. Together they spend exactly the
  // task's costs.
  std::vector<ActionCosts> partition(std::size_t i) const {
    std::vector<ActionCosts> parts = rotations_.orders()[i].shares();
    parts.push_back(leftovers_[i]);
    return parts;
  }

 private:
  LandmarkFactory landmarks_;
  RotatedSaturatedPatternDatabaseHeuristic rotations_;
  ActionCosts full_;
  std::vector<ActionCosts> leftovers_;
};

// h_theta(s) = sum_i w_i f_i(s), the object the discovery loop searches over.
// Only features with a non-zero weight are evaluated, so an unused expensive
// feature costs nothing.
class LinearHeuristic {
 public:
  struct Term {
    FeatureId feature;
    double weight;
  };

  LinearHeuristic(const StripsTask& task, std::vector<Term> terms)
      : eval_(task), terms_(std::move(terms)) {}

  double operator()(const StripsState& s) const {
    double h = 0.0;
    for (const Term& t : terms_) h += t.weight * eval_.evaluate(t.feature, s);
    return h;
  }

  const std::vector<Term>& terms() const { return terms_; }

  // "1.82*unsatisfied_goals + 0.37*applicable_actions"
  std::string to_string() const {
    std::string out;
    for (std::size_t i = 0; i < terms_.size(); ++i) {
      if (i) out += " + ";
      out += std::to_string(terms_[i].weight);
      out += "*";
      out += std::string(feature_name(terms_[i].feature));
    }
    return out.empty() ? "0" : out;
  }

 private:
  FeatureEvaluator eval_;
  std::vector<Term> terms_;
};

}  // namespace hd
