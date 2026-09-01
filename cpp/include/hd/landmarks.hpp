// Landmarks, and the admissible lower bound built from them.
//
// A landmark of a state s is a proposition that is true at some point in every
// plan from s. Landmarks are generated here by relaxed reachability: p is a
// landmark iff the goal is unreachable in the delete relaxation of the task
// from s once every action adding p is removed. The test is sound, because
// every real plan is also a relaxed plan, so a fact that all relaxed plans need
// is a fact that all plans need. It is incomplete in two ways that matter: it
// finds only single-fact landmarks, and it misses those whose necessity depends
// on delete effects. Both cost informedness, neither costs admissibility.
//
// Counting the unachieved landmarks is *not* admissible: a single action may
// achieve several of them at once, and then the count exceeds the cost of the
// plan that does so. The value computed here is the uniform cost partition over
// landmarks (Karpas and Domshlak, 2009): each action divides its cost equally
// among the landmarks it can achieve, and each landmark contributes the
// cheapest share that any of its achievers assigns it,
//
//     h(s) = sum_{L unachieved} min_{a : L in add(a)} c(a) / |add(a) cap LM(s)|.
//
// Every plan achieves every landmark, and the shares one action hands out sum
// to at most its own cost, so the total is a lower bound on the cost of any
// plan from s.
//
// The value is linear in the action costs: evaluating this component under
// costs w*c multiplies it by exactly w. A cost partition over several
// components therefore needs no support here beyond a scalar weight, which the
// linear heuristic already provides.
#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "hd/strips.hpp"

namespace hd {

// Returned when the goal is unreachable even in the relaxation: the state is a
// proven dead end, and callers map this onto their own dead-end convention.
inline constexpr double kUnboundedCost = std::numeric_limits<double>::infinity();

struct LandmarkSet {
  // The goal conditions together with every fact the state must still pass
  // through. A fact already true in the state is trivially a landmark of it
  // and is never searched for: it cannot raise the bound, so the only such
  // facts here are goal conditions that are already satisfied.
  StripsState landmarks;
  StripsState unachieved;  // landmarks not yet true in the state
  bool dead_end = false;   // the goal is not relaxed-reachable

  std::size_t size() const { return landmarks.count(); }
  std::size_t num_unachieved() const { return unachieved.count(); }
};

// Generates the landmarks of a state and prices them. Scratch storage is held
// across calls, and the last state's result is cached, so a heuristic that asks
// for both the set and its value pays for one generation.
class LandmarkFactory {
 public:
  explicit LandmarkFactory(const StripsTask& task)
      : task_(&task),
        fired_(task.num_actions(), 0),
        reachable_actions_(task.num_actions(), 0),
        share_(task.num_propositions(), 0.0) {}

  const LandmarkSet& compute(const StripsState& s) const {
    if (cached_ && key_ == s) return set_;
    generate(s);
    key_ = s;
    cached_ = true;
    return set_;
  }

  // The uniform cost partition over the landmarks of s, or kUnboundedCost when
  // s is a proven dead end.
  double value(const StripsState& s) const {
    const LandmarkSet& lm = compute(s);
    if (lm.dead_end) return kUnboundedCost;
    if (lm.unachieved.empty()) return 0.0;

    const std::size_t np = task_->num_propositions();
    share_.assign(np, kUnboundedCost);
    for (std::size_t a = 0; a < task_->num_actions(); ++a) {
      if (!reachable_actions_[a]) continue;  // never applicable, even relaxed
      const StripsAction& act = task_->action(a);
      const std::size_t shares = act.add.count_common(lm.unachieved);
      if (shares == 0) continue;
      const double portion = task_->cost(act) / static_cast<double>(shares);
      for (std::size_t p = 0; p < np; ++p) {
        if (act.add.test(p) && lm.unachieved.test(p) && portion < share_[p]) share_[p] = portion;
      }
    }

    double total = 0.0;
    for (std::size_t p = 0; p < np; ++p) {
      if (!lm.unachieved.test(p)) continue;
      if (share_[p] == kUnboundedCost) return kUnboundedCost;  // no achiever at all
      total += share_[p];
    }
    return total;
  }

 private:
  // Delete-relaxed reachability from s, pretending that no action adding
  // `banned` exists (`banned < 0` bans nothing). Leaves the fixpoint in
  // `reached_` and the actions that fired in `fired_`.
  //
  // This is a plain reachability test, not the layered graph that features.hpp
  // builds for h_max and h_add: no layer numbers are needed, and the ban makes
  // the two loops different enough that sharing one would obscure both.
  bool relaxed_reaches_goal(const StripsState& s, int banned) const {
    const std::size_t na = task_->num_actions();
    reached_ = s;
    fired_.assign(na, 0);
    for (;;) {
      if (reached_.contains(task_->goal())) return true;
      bool grew = false;
      for (std::size_t a = 0; a < na; ++a) {
        if (fired_[a]) continue;
        const StripsAction& act = task_->action(a);
        if (banned >= 0 && act.add.test(static_cast<std::size_t>(banned))) continue;
        if (!reached_.contains(act.pre)) continue;
        fired_[a] = 1;
        const std::size_t before = reached_.count();
        reached_.or_with(act.add);
        grew = grew || reached_.count() != before;
      }
      if (!grew) return reached_.contains(task_->goal());
    }
  }

  void generate(const StripsState& s) const {
    set_ = LandmarkSet{};
    if (!relaxed_reaches_goal(s, -1)) {
      set_.dead_end = true;
      return;
    }
    // The unrestricted fixpoint fixes both the candidates worth testing and
    // the actions allowed to price a landmark: an action that never becomes
    // applicable cannot achieve anything, and dropping it only raises the
    // bound, which keeps it admissible.
    const StripsState reachable = reached_;
    reachable_actions_ = fired_;

    // Goal conditions are landmarks by definition and need no test.
    set_.landmarks = task_->goal();
    for (std::size_t p = 0; p < task_->num_propositions(); ++p) {
      if (!reachable.test(p) || s.test(p) || set_.landmarks.test(p)) continue;
      if (!relaxed_reaches_goal(s, static_cast<int>(p))) set_.landmarks.set(p);
    }
    set_.unachieved = set_.landmarks;
    set_.unachieved.and_not_with(s);
  }

  const StripsTask* task_;
  mutable std::vector<unsigned char> fired_;
  mutable std::vector<unsigned char> reachable_actions_;
  mutable std::vector<double> share_;
  mutable StripsState reached_{};
  mutable LandmarkSet set_{};
  mutable StripsState key_{};
  mutable bool cached_ = false;
};

}  // namespace hd
