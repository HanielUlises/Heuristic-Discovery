// Landmark generation and the uniform cost partition built on it.
#include "fixtures.hpp"
#include "hd/features.hpp"
#include "hd/heuristic.hpp"
#include "hd/landmarks.hpp"
#include "hd/oracle.hpp"
#include "hd/verify.hpp"
#include "test_framework.hpp"

namespace {

hd::StripsState state_of(std::initializer_list<std::size_t> propositions) {
  hd::StripsState s;
  for (const std::size_t p : propositions) s.set(p);
  return s;
}

}  // namespace

TEST("landmarks: every unachieved goal condition is one") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::LandmarkFactory factory(task);
  const hd::LandmarkSet& lm = factory.compute(task.initial_state());

  CHECK(!lm.dead_end);
  CHECK(lm.landmarks.test(1));  // on_x
  CHECK(lm.landmarks.test(3));  // on_y
  CHECK_EQ(lm.num_unachieved(), std::size_t{2});
}

TEST("landmarks: a fact on the only route to the goal is one") {
  // Reaching at_c requires passing through at_b, which is not a goal.
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::LandmarkFactory factory(task);
  const hd::LandmarkSet& lm = factory.compute(task.initial_state());

  CHECK(lm.landmarks.test(1));  // at_b, found by the reachability test
  CHECK(lm.landmarks.test(2));  // at_c, a goal condition
  CHECK_EQ(lm.num_unachieved(), std::size_t{2});
}

TEST("landmarks: a fact with an alternative achiever is not one") {
  // Every route to the goals runs through `start`, but nothing else is
  // necessary: `both`, take_left and take_right make each goal achievable in
  // more than one way, so no intermediate fact is forced.
  const hd::StripsTask task = hdtest::parse(hdtest::kJointTask);
  const hd::LandmarkFactory factory(task);
  const hd::LandmarkSet& lm = factory.compute(task.initial_state());

  CHECK_EQ(lm.num_unachieved(), std::size_t{2});  // the two goals, nothing else
  // Facts already true in the state are trivially landmarks and are not
  // searched for, since they cannot raise the bound.
  CHECK(!lm.unachieved.test(0));  // start
}

TEST("landmarks: achieved landmarks contribute nothing") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::LandmarkFactory factory(task);

  CHECK_NEAR(factory.value(state_of({0})), 2.0, 1e-12);  // at_b and at_c to go
  CHECK_NEAR(factory.value(state_of({1})), 1.0, 1e-12);  // only at_c to go
  CHECK_NEAR(factory.value(state_of({2})), 0.0, 1e-12);  // the goal itself
}

TEST("landmarks: the value follows action costs") {
  // flip_x costs 1 and flip_y costs 2; each achieves one landmark.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::LandmarkFactory factory(task);
  CHECK_NEAR(factory.value(task.initial_state()), 3.0, 1e-12);
}

TEST("landmarks: one action achieving two landmarks pays for one") {
  // Counting the unachieved landmarks would return 2 against an optimal cost
  // of 1. Dividing the cost of `both` between them returns exactly 1.
  const hd::StripsTask task = hdtest::parse(hdtest::kJointTask);
  const hd::LandmarkFactory factory(task);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  CHECK_NEAR(factory.value(task.initial_state()), 1.0, 1e-12);
  CHECK_NEAR(space.initial_h_star(), 1.0, 1e-12);

  const hd::HeuristicReport counting =
      hd::verify_heuristic(space, [&](const hd::StripsState& s) {
        return static_cast<double>(factory.compute(s).num_unachieved());
      });
  CHECK(!counting.admissible());

  const hd::HeuristicReport partitioned =
      hd::verify_heuristic(space, hd::LandmarkCostHeuristic(task));
  CHECK(partitioned.admissible());
}

TEST("landmarks: a proven dead end has no bound") {
  const hd::StripsTask task = hdtest::parse(hdtest::kUnsolvableTask);
  const hd::LandmarkFactory factory(task);

  CHECK(factory.compute(task.initial_state()).dead_end);
  CHECK(factory.value(task.initial_state()) == hd::kUnboundedCost);
  // The feature vocabulary reports the same state with its finite constant.
  const hd::FeatureEvaluator eval(task);
  CHECK_NEAR(eval.evaluate(hd::FeatureId::kLandmarkCost, task.initial_state()),
             hd::kUnreachableValue, 1e-12);
}

TEST("landmarks: the baseline is admissible and consistent on the fixtures") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    const hd::HeuristicReport r = hd::verify_heuristic(space, hd::LandmarkCostHeuristic(task));
    CHECK(r.admissible());
    CHECK(r.consistent());
    CHECK(r.goal_aware());
    CHECK(r.non_negative());
  }
}

TEST("landmarks: repeated evaluation of one state is cached, not recomputed") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::LandmarkFactory factory(task);
  const hd::LandmarkSet& first = factory.compute(task.initial_state());
  const hd::LandmarkSet& second = factory.compute(task.initial_state());
  CHECK(&first == &second);
  CHECK_EQ(first.num_unachieved(), second.num_unachieved());
}
