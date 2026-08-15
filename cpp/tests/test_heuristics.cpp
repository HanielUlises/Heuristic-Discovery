// Heuristics: baselines, linear composition, and specification parsing.
#include "fixtures.hpp"
#include "hd/heuristic.hpp"
#include "hd/runner.hpp"
#include "test_framework.hpp"

using hd::FeatureId;
using hdtest::parse;

TEST("heuristic: zero is identically zero") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::ZeroHeuristic h;
  CHECK_NEAR(h(task.initial_state()), 0.0, 1e-12);
}

TEST("heuristic: goal count decreases along a solution path") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::GoalCountHeuristic h(task);
  const hd::StripsState s0 = task.initial_state();
  const hd::StripsState s1 = task.apply(s0, task.action(0));
  const hd::StripsState s2 = task.apply(s1, task.action(1));
  CHECK_NEAR(h(s0), 2.0, 1e-12);
  CHECK_NEAR(h(s1), 1.0, 1e-12);
  CHECK_NEAR(h(s2), 0.0, 1e-12);
}

TEST("heuristic: goal states have zero heuristic value under every baseline") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  hd::StripsState goal_state = task.apply(task.initial_state(), task.action(0));
  goal_state = task.apply(goal_state, task.action(1));
  CHECK(task.is_goal(goal_state));
  CHECK_NEAR(hd::GoalCountHeuristic(task)(goal_state), 0.0, 1e-12);
  CHECK_NEAR(hd::RelaxedLayersHeuristic(task)(goal_state), 0.0, 1e-12);
}

TEST("heuristic: linear combination equals the weighted feature sum") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::FeatureEvaluator eval(task);
  const hd::StripsState s0 = task.initial_state();

  const hd::LinearHeuristic h(task, {{FeatureId::kUnsatisfiedGoals, 1.5},
                                     {FeatureId::kApplicableActions, 0.25}});
  const double expected = 1.5 * eval.evaluate(FeatureId::kUnsatisfiedGoals, s0) +
                          0.25 * eval.evaluate(FeatureId::kApplicableActions, s0);
  CHECK_NEAR(h(s0), expected, 1e-12);
}

TEST("heuristic: an empty linear combination degenerates to zero") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::LinearHeuristic h(task, {});
  CHECK_NEAR(h(task.initial_state()), 0.0, 1e-12);
  CHECK_EQ(h.to_string(), std::string("0"));
}

TEST("heuristic: a unit weight on one feature reproduces that feature") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::LinearHeuristic h(task, {{FeatureId::kUnsatisfiedGoals, 1.0}});
  const hd::GoalCountHeuristic baseline(task);
  CHECK_NEAR(h(task.initial_state()), baseline(task.initial_state()), 1e-12);
}

TEST("heuristic: linear form is linear in its weights") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::StripsState s = task.initial_state();
  const hd::LinearHeuristic a(task, {{FeatureId::kUnsatisfiedGoals, 1.0}});
  const hd::LinearHeuristic b(task, {{FeatureId::kUnsatisfiedGoals, 3.0}});
  CHECK_NEAR(3.0 * a(s), b(s), 1e-12);
}

TEST("spec: baseline names parse to their kind") {
  CHECK(hd::parse_heuristic_spec("zero").kind == hd::HeuristicSpec::Kind::kZero);
  CHECK(hd::parse_heuristic_spec("goal_count").kind == hd::HeuristicSpec::Kind::kGoalCount);
  CHECK(hd::parse_heuristic_spec("relaxed_layers").kind ==
        hd::HeuristicSpec::Kind::kRelaxedLayers);
}

TEST("spec: linear specifications parse with and without the prefix") {
  for (const char* text : {"linear:unsatisfied_goals=1.82,applicable_actions=0.37",
                           "unsatisfied_goals=1.82,applicable_actions=0.37"}) {
    const hd::HeuristicSpec spec = hd::parse_heuristic_spec(text);
    CHECK(spec.kind == hd::HeuristicSpec::Kind::kLinear);
    CHECK_EQ(spec.terms.size(), 2u);
    CHECK(spec.terms[0].feature == FeatureId::kUnsatisfiedGoals);
    CHECK_NEAR(spec.terms[0].weight, 1.82, 1e-12);
    CHECK_NEAR(spec.terms[1].weight, 0.37, 1e-12);
  }
}

TEST("spec: negative weights are allowed") {
  const hd::HeuristicSpec spec = hd::parse_heuristic_spec("applicable_actions=-2.5");
  CHECK_NEAR(spec.terms[0].weight, -2.5, 1e-12);
}

TEST("spec: unknown features and malformed terms are rejected") {
  for (const char* text : {"not_a_feature=1.0", "unsatisfied_goals", "linear:"}) {
    bool threw = false;
    try {
      hd::parse_heuristic_spec(text);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }
}
