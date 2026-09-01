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

#include "hd/features.hpp"
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
