// State features f_i : State -> double.
//
// Features are the vocabulary from which candidate heuristics are built. They
// are identified by a stable name (shared with the Python layer) and dispatched
// through a switch rather than a virtual call, so adding a feature never
// touches the search algorithms.
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "hd/landmarks.hpp"
#include "hd/strips.hpp"

namespace hd {

enum class FeatureId : std::size_t {
  kUnsatisfiedGoals = 0,
  kAchievedGoals,
  kApplicableActions,
  kTruePropositions,
  kRelaxedLayers,   // h_max under unit-cost relaxed planning graph layers
  kRelaxedSum,      // h_add-style sum of per-goal relaxed layers
  kLandmarkCost,    // uniform cost partition over relaxed-reachability landmarks
  kCount
};

inline constexpr std::size_t kNumFeatures = static_cast<std::size_t>(FeatureId::kCount);

inline constexpr std::array<std::string_view, kNumFeatures> kFeatureNames = {
    "unsatisfied_goals", "achieved_goals", "applicable_actions",
    "true_propositions", "relaxed_layers", "relaxed_sum",
    "landmark_cost",
};

inline constexpr std::array<std::string_view, kNumFeatures> kFeatureDescriptions = {
    "number of goal conditions not satisfied in s",
    "number of goal conditions satisfied in s",
    "number of actions applicable in s",
    "number of propositions true in s",
    "layers of the delete-relaxed planning graph until all goals appear (h_max-like)",
    "sum over goals of the layer at which each first appears (h_add-like)",
    "uniform cost partition over the landmarks of s; admissible on its own",
};

// Value substituted for a goal that is unreachable in the delete relaxation.
// The state is then a proven dead end; a large finite value keeps every
// heuristic total and comparable without saturating the arithmetic.
inline constexpr double kUnreachableValue = 1000.0;

inline std::optional<FeatureId> feature_from_name(std::string_view name) {
  for (std::size_t i = 0; i < kNumFeatures; ++i) {
    if (kFeatureNames[i] == name) return static_cast<FeatureId>(i);
  }
  return std::nullopt;
}

inline std::string_view feature_name(FeatureId id) {
  return kFeatureNames[static_cast<std::size_t>(id)];
}

// Evaluates features of a single task. Holds reusable scratch storage so that
// evaluation performs no allocation once warmed up.
class FeatureEvaluator {
 public:
  explicit FeatureEvaluator(const StripsTask& task)
      : task_(&task),
        landmarks_(task),
        level_(task.num_propositions(), -1),
        counter_(task.num_actions(), 0) {}

  double evaluate(FeatureId id, const StripsState& s) const {
    switch (id) {
      case FeatureId::kUnsatisfiedGoals:
        return static_cast<double>(s.count_missing(task_->goal()));
      case FeatureId::kAchievedGoals:
        return static_cast<double>(s.count_common(task_->goal()));
      case FeatureId::kApplicableActions:
        return static_cast<double>(task_->num_applicable(s));
      case FeatureId::kTruePropositions:
        return static_cast<double>(s.count());
      case FeatureId::kRelaxedLayers:
        build_relaxed_graph(s);
        return relaxed_max_;
      case FeatureId::kRelaxedSum:
        build_relaxed_graph(s);
        return relaxed_sum_;
      case FeatureId::kLandmarkCost: {
        // The factory reports a proven dead end as an unbounded cost; the
        // feature vocabulary expresses that as its own large finite constant,
        // so every feature stays total and comparable.
        const double v = landmarks_.value(s);
        return v == kUnboundedCost ? kUnreachableValue : v;
      }
      case FeatureId::kCount:
        break;
    }
    return 0.0;
  }

  // Fills `out` with the full feature vector of s, in FeatureId order.
  void evaluate_all(const StripsState& s, std::array<double, kNumFeatures>& out) const {
    for (std::size_t i = 0; i < kNumFeatures; ++i) {
      out[i] = evaluate(static_cast<FeatureId>(i), s);
    }
  }

  const StripsTask& task() const { return *task_; }

 private:
  // Delete-relaxed reachability: the layer at which each proposition first
  // becomes true when delete effects are ignored.
  void build_relaxed_graph(const StripsState& s) const {
    if (cached_ && cache_key_ == s) return;

    const std::size_t np = task_->num_propositions();
    const std::size_t na = task_->num_actions();
    level_.assign(np, -1);
    counter_.assign(na, 0);

    StripsState reached;
    for (std::size_t p = 0; p < np; ++p) {
      if (s.test(p)) {
        level_[p] = 0;
        reached.set(p);
      }
    }

    int layer = 0;
    while (!reached.contains(task_->goal())) {
      StripsState next = reached;
      bool grew = false;
      for (std::size_t a = 0; a < na; ++a) {
        const StripsAction& act = task_->action(a);
        if (counter_[a]) continue;               // already fired
        if (!reached.contains(act.pre)) continue;
        counter_[a] = 1;
        next.or_with(act.add);
      }
      for (std::size_t p = 0; p < np; ++p) {
        if (level_[p] < 0 && next.test(p)) {
          level_[p] = layer + 1;
          grew = true;
        }
      }
      if (!grew) break;  // fixpoint without reaching the goal: relaxed dead end
      reached = next;
      ++layer;
    }

    relaxed_max_ = 0.0;
    relaxed_sum_ = 0.0;
    for (std::size_t p = 0; p < np; ++p) {
      if (!task_->goal().test(p)) continue;
      const double v = (level_[p] < 0) ? kUnreachableValue : static_cast<double>(level_[p]);
      relaxed_sum_ += v;
      if (v > relaxed_max_) relaxed_max_ = v;
    }

    cached_ = true;
    cache_key_ = s;
  }

  const StripsTask* task_;
  LandmarkFactory landmarks_;
  mutable std::vector<int> level_;
  mutable std::vector<unsigned char> counter_;
  mutable StripsState cache_key_{};
  mutable bool cached_ = false;
  mutable double relaxed_max_ = 0.0;
  mutable double relaxed_sum_ = 0.0;
};

}  // namespace hd
