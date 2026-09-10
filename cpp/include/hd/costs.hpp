// Cost functions over the actions of a task.
//
// A cost partition gives every component of a heuristic its own cost function,
// subject to the shares of an action summing to at most what the action costs
// in the task:
//
//     c_1, ..., c_k >= 0,   sum_i c_i(a) <= c(a)  for every action a.
//
// Under that condition the sum of the components' estimates is itself a lower
// bound on plan cost, which is what makes partitioning stronger than taking
// the maximum: no two components can charge for the same action twice. A
// component therefore has to be evaluable under supplied costs rather than the
// task's own, and this type is what carries them.
//
// Scaling every action by one factor is the degenerate case. It is admissible,
// but a sum of uniformly scaled components is a convex combination and cannot
// exceed the maximum of those components, so it is useful for testing the
// machinery rather than for building a bound.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hd/strips.hpp"

namespace hd {

class ActionCosts {
 public:
  explicit ActionCosts(std::vector<double> costs) : costs_(std::move(costs)) {
    for (const double c : costs_) {
      // Negative costs would break both admissibility and the Dijkstra that
      // computes h*, and NaN would propagate silently through both.
      if (!(c >= 0.0) || std::isinf(c)) {
        throw std::runtime_error("action costs must be finite and non-negative");
      }
    }
  }

  // The task's own costs: what every component uses when no partition is in
  // force.
  static ActionCosts of(const StripsTask& task) {
    std::vector<double> costs(task.num_actions());
    for (std::size_t a = 0; a < task.num_actions(); ++a) costs[a] = task.cost(task.action(a));
    return ActionCosts(std::move(costs));
  }

  static ActionCosts zero(const StripsTask& task) {
    return ActionCosts(std::vector<double>(task.num_actions(), 0.0));
  }

  double operator[](std::size_t action) const { return costs_[action]; }
  std::size_t size() const { return costs_.size(); }
  const std::vector<double>& values() const { return costs_; }

  ActionCosts scaled(double factor) const {
    if (!(factor >= 0.0)) throw std::runtime_error("a cost scaling must be non-negative");
    std::vector<double> out(costs_.size());
    for (std::size_t a = 0; a < costs_.size(); ++a) out[a] = costs_[a] * factor;
    return ActionCosts(std::move(out));
  }

 private:
  std::vector<double> costs_;
};

// Whether `parts` is a legal partition of the task's costs, which is the
// condition under which the sum of the components evaluated on them is a lower
// bound. Costs may be left unspent; only overspending breaks the bound.
inline bool is_cost_partition(const StripsTask& task, const std::vector<ActionCosts>& parts,
                              double tolerance = 1e-9) {
  for (const ActionCosts& part : parts) {
    if (part.size() != task.num_actions()) return false;
  }
  for (std::size_t a = 0; a < task.num_actions(); ++a) {
    double spent = 0.0;
    for (const ActionCosts& part : parts) spent += part[a];
    if (spent > task.cost(task.action(a)) + tolerance) return false;
  }
  return true;
}

}  // namespace hd
