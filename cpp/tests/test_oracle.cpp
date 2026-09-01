// Exact goal distances, and the admissibility / consistency checks built on
// them. The fixtures are small enough that every h* value can be stated by
// hand, which is what makes them useful as ground truth for the ground truth.
#include "fixtures.hpp"
#include "hd/heuristic.hpp"
#include "hd/oracle.hpp"
#include "hd/verify.hpp"
#include "test_framework.hpp"

namespace {

hd::StripsState state_of(std::initializer_list<std::size_t> propositions) {
  hd::StripsState s;
  for (const std::size_t p : propositions) s.set(p);
  return s;
}

// Index of a state in the enumerated space; fails the test if it is absent.
std::size_t index_of(const hd::StateSpace& space, const hd::StripsState& s) {
  for (std::size_t i = 0; i < space.size(); ++i) {
    if (space.states[i] == s) return i;
  }
  throw hdtest::Failure("state not present in the enumerated space");
}

double distance_of(const hd::StateSpace& space, std::initializer_list<std::size_t> propositions) {
  return space.h_star[index_of(space, state_of(propositions))];
}

}  // namespace

TEST("oracle enumerates the corridor and its goal distances") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  CHECK(space.complete());
  CHECK_EQ(space.size(), std::size_t{3});
  CHECK_EQ(space.num_goal_states(), std::size_t{1});
  CHECK_EQ(space.num_dead_ends(), std::size_t{0});
  CHECK_NEAR(distance_of(space, {0}), 2.0, 1e-12);
  CHECK_NEAR(distance_of(space, {1}), 1.0, 1e-12);
  CHECK_NEAR(distance_of(space, {2}), 0.0, 1e-12);
  CHECK_NEAR(space.max_finite_h_star(), 2.0, 1e-12);
}

TEST("goal distances follow action costs, not action counts") {
  // flip_x costs 1 and flip_y costs 2, so the goal is three units away
  // although two actions suffice.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  CHECK(space.complete());
  CHECK_EQ(space.size(), std::size_t{4});
  CHECK_NEAR(distance_of(space, {0, 2}), 3.0, 1e-12);  // off_x, off_y
  CHECK_NEAR(distance_of(space, {1, 2}), 2.0, 1e-12);  // on_x, off_y
  CHECK_NEAR(distance_of(space, {0, 3}), 1.0, 1e-12);  // off_x, on_y
  CHECK_NEAR(distance_of(space, {1, 3}), 0.0, 1e-12);  // goal
}

TEST("goal states are terminal and unreachable goals give infinite distance") {
  const hd::StripsTask task = hdtest::parse(hdtest::kUnsolvableTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  CHECK(space.complete());
  CHECK_EQ(space.num_goal_states(), std::size_t{0});
  CHECK_EQ(space.num_dead_ends(), space.size());
  for (const double d : space.h_star) CHECK(d == hd::kInfiniteDistance);
}

TEST("a truncated enumeration reports its truncation and computes no distances") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  hd::OracleLimits limits;
  limits.max_states = 1;
  const hd::StateSpace space = hd::enumerate_state_space(task, limits);

  CHECK(!space.complete());
  CHECK_EQ(hd::to_string(space.status), "state_limit");
  CHECK(space.h_star.empty());
}

TEST("the baselines are admissible and consistent on the fixtures") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);

    const hd::HeuristicReport layers =
        hd::verify_heuristic(space, hd::RelaxedLayersHeuristic(task));
    CHECK(layers.admissible());
    CHECK(layers.consistent());
    CHECK(layers.goal_aware());
    CHECK(layers.non_negative());
  }
}

TEST("an overestimating heuristic is caught with a witness state") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  // 5 * unsatisfied_goals overestimates everywhere but the goal: h(at_a) = 5
  // against a true distance of 2, and h(at_b) = 5 against 1.
  const hd::LinearHeuristic h(task, {{hd::FeatureId::kUnsatisfiedGoals, 5.0}});
  const hd::HeuristicReport r = hd::verify_heuristic(space, h);

  CHECK(!r.admissible());
  CHECK_EQ(r.admissibility_violations, std::size_t{2});
  CHECK_NEAR(r.max_excess, 4.0, 1e-12);  // at_b: 5 - 1
  CHECK_NEAR(r.max_ratio, 5.0, 1e-12);
  CHECK(!r.admissibility_witnesses.empty());
  CHECK_NEAR(r.admissibility_witnesses[0].excess, 4.0, 1e-12);
  CHECK_NEAR(r.admissibility_witnesses[0].h_star, 1.0, 1e-12);
  CHECK(r.goal_aware());  // the goal itself is still valued at zero
}

TEST("a heuristic that drops faster than the edge cost is inconsistent") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  // 2 * unsatisfied_goals falls by 2 across flip_x, which costs 1.
  const hd::LinearHeuristic h(task, {{hd::FeatureId::kUnsatisfiedGoals, 2.0}});
  const hd::HeuristicReport r = hd::verify_heuristic(space, h);

  CHECK(!r.consistent());
  CHECK_NEAR(r.max_consistency_excess, 1.0, 1e-12);
  CHECK(!r.consistency_witnesses.empty());
  CHECK_EQ(r.consistency_witnesses[0].cost, 1.0);

  // goal_count itself never falls faster than the cheapest action.
  const hd::HeuristicReport counted = hd::verify_heuristic(space, hd::GoalCountHeuristic(task));
  CHECK(counted.consistent());
  CHECK(counted.admissible());
}

TEST("informedness measures how much of h* a heuristic recovers") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  const hd::HeuristicReport zero = hd::verify_heuristic(space, hd::ZeroHeuristic{});
  CHECK(zero.admissible());
  CHECK_EQ(zero.states_checked, std::size_t{3});
  CHECK_NEAR(zero.mean_informedness, 0.0, 1e-12);
  CHECK_NEAR(zero.mean_error, 1.5, 1e-12);  // (2 + 1) / 2

  // goal_count returns 1 at both non-goal states, so it recovers all of h* at
  // at_b and half of it at at_a.
  const hd::HeuristicReport counted = hd::verify_heuristic(space, hd::GoalCountHeuristic(task));
  CHECK(counted.admissible());
  CHECK_NEAR(counted.mean_informedness, 0.75, 1e-12);
  CHECK_NEAR(counted.min_informedness, 0.5, 1e-12);
  CHECK_NEAR(counted.mean_error, 0.5, 1e-12);
}

TEST("a dead end admits any finite value and is excluded from the statistics") {
  const hd::StripsTask task = hdtest::parse(hdtest::kUnsolvableTask);
  const hd::StateSpace space = hd::enumerate_state_space(task);

  const hd::LinearHeuristic h(task, {{hd::FeatureId::kUnsatisfiedGoals, 1000.0}});
  const hd::HeuristicReport r = hd::verify_heuristic(space, h);

  CHECK(r.admissible());
  CHECK_EQ(r.states_checked, std::size_t{0});
  CHECK_EQ(r.dead_end_states, space.size());
}
