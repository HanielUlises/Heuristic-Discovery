// Features: individual evaluation and the name registry shared with Python.
#include <array>

#include "fixtures.hpp"
#include "hd/features.hpp"
#include "test_framework.hpp"

using hd::FeatureId;
using hdtest::parse;

TEST("features: names are unique and resolve back to their id") {
  for (std::size_t i = 0; i < hd::kNumFeatures; ++i) {
    const auto id = hd::feature_from_name(hd::kFeatureNames[i]);
    CHECK(id.has_value());
    CHECK_EQ(static_cast<std::size_t>(*id), i);
    CHECK(!hd::kFeatureDescriptions[i].empty());
    for (std::size_t j = i + 1; j < hd::kNumFeatures; ++j) {
      CHECK(hd::kFeatureNames[i] != hd::kFeatureNames[j]);
    }
  }
  CHECK(!hd::feature_from_name("not_a_feature").has_value());
}

TEST("features: goal counting features are complementary") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();
  CHECK_NEAR(eval.evaluate(FeatureId::kUnsatisfiedGoals, s0), 2.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kAchievedGoals, s0), 0.0, 1e-12);

  const hd::StripsState s1 = task.apply(s0, task.action(0));
  CHECK_NEAR(eval.evaluate(FeatureId::kUnsatisfiedGoals, s1), 1.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kAchievedGoals, s1), 1.0, 1e-12);
}

TEST("features: structural counts of a state") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();
  CHECK_NEAR(eval.evaluate(FeatureId::kApplicableActions, s0), 2.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kTruePropositions, s0), 2.0, 1e-12);
}

TEST("features: relaxed graph measures distance to the goal") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();
  // at_c appears in the second relaxed layer.
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, s0), 2.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedSum, s0), 2.0, 1e-12);

  const hd::StripsState s1 = task.apply(s0, task.action(0));
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, s1), 1.0, 1e-12);

  const hd::StripsState goal_state = task.apply(s1, task.action(1));
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, goal_state), 0.0, 1e-12);
}

TEST("features: relaxed sum aggregates over goals while layers take the max") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, s0), 1.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedSum, s0), 2.0, 1e-12);
}

TEST("features: relaxed dead ends receive the unreachable value") {
  const hd::StripsTask task = parse(hdtest::kUnsolvableTask);
  const hd::FeatureEvaluator eval(task);
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, task.initial_state()),
             hd::kUnreachableValue, 1e-12);
}

TEST("features: evaluate_all matches individual evaluation") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();
  std::array<double, hd::kNumFeatures> all{};
  eval.evaluate_all(s0, all);
  for (std::size_t i = 0; i < hd::kNumFeatures; ++i) {
    CHECK_NEAR(all[i], eval.evaluate(static_cast<FeatureId>(i), s0), 1e-12);
  }
}

TEST("features: the relaxed graph cache is invalidated between states") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();
  const hd::StripsState s1 = task.apply(s0, task.action(0));
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, s0), 2.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, s1), 1.0, 1e-12);
  CHECK_NEAR(eval.evaluate(FeatureId::kRelaxedLayers, s0), 2.0, 1e-12);
}
