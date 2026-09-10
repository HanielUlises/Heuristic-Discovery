// Projections and the pattern databases built on them.
#include <vector>

#include "fixtures.hpp"
#include "hd/costs.hpp"
#include "hd/heuristic.hpp"
#include "hd/landmarks.hpp"
#include "hd/oracle.hpp"
#include "hd/pdb.hpp"
#include "hd/verify.hpp"
#include "test_framework.hpp"

namespace {

hd::StripsState pattern_of(std::initializer_list<std::size_t> propositions) {
  hd::StripsState p;
  for (const std::size_t i : propositions) p.set(i);
  return p;
}

hd::StripsState everything(const hd::StripsTask& task) {
  hd::StripsState p;
  for (std::size_t i = 0; i < task.num_propositions(); ++i) p.set(i);
  return p;
}

}  // namespace

TEST("pdb: a projection keeps only what the pattern mentions") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::Projection p = hd::project(task, pattern_of({2}), hd::ActionCosts::of(task));

  // move_a_b touches nothing inside {at_c} and is dropped; move_b_c survives
  // with an empty precondition.
  CHECK_EQ(p.task.num_actions(), std::size_t{1});
  CHECK_EQ(p.task.action(0).name, std::string("move_b_c"));
  CHECK(p.task.action(0).pre.empty());
  CHECK(p.task.action(0).add.test(2));
  CHECK(p.task.goal().test(2));
  CHECK(p.task.initial_state().empty());
  CHECK_EQ(p.costs.size(), std::size_t{1});
}

TEST("pdb: the pattern of everything reproduces h* exactly") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    const hd::PatternDatabase pdb(task, everything(task), hd::ActionCosts::of(task));

    for (std::size_t i = 0; i < space.size(); ++i) {
      CHECK_NEAR(pdb(space.states[i]), space.h_star[i], 1e-12);
    }
  }
}

TEST("pdb: a partial pattern underestimates rather than overestimates") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::PatternDatabase pdb(task, pattern_of({2}), hd::ActionCosts::of(task));

  // Forgetting at_b leaves one action between any state and the goal.
  hd::StripsState at_a;
  at_a.set(0);
  CHECK_NEAR(pdb(at_a), 1.0, 1e-12);
  CHECK_NEAR(hd::enumerate_state_space(task).initial_h_star(), 2.0, 1e-12);
}

TEST("pdb: databases are admissible and consistent on the fixtures") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    for (const hd::StripsState& pattern : hd::goal_patterns(task)) {
      const hd::PatternDatabase pdb(task, pattern, hd::ActionCosts::of(task));
      const hd::HeuristicReport r = hd::verify_heuristic(space, pdb);
      CHECK(r.admissible());
      CHECK(r.consistent());  // an abstraction is consistent by construction
      CHECK(r.goal_aware());
    }
  }
}

TEST("pdb: a state beyond an abstract goal state is still in the table") {
  // at_c is the abstract goal of the pattern {at_c}; enumerating the
  // abstraction has to look past it, or states reached afterwards would be
  // missing and read as dead ends.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::PatternDatabase pdb(task, pattern_of({1}), hd::ActionCosts::of(task));
  CHECK_EQ(pdb.size(), std::size_t{2});  // on_x true and false, both priced
}

TEST("pdb: the table follows the costs it was built with") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);
  const hd::PatternDatabase whole(task, everything(task), full);
  const hd::PatternDatabase half(task, everything(task), full.scaled(0.5));

  const hd::StripsState& s = task.initial_state();
  CHECK_NEAR(whole(s), 3.0, 1e-12);
  CHECK_NEAR(half(s), 1.5, 1e-12);
}

TEST("pdb: an oversized pattern is refused, not truncated") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  bool threw = false;
  try {
    hd::PatternDatabase pdb(task, everything(task), hd::ActionCosts::of(task), 2);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

TEST("pdb: a database and the landmark bound add up under a cost partition") {
  // The first partition across two components of different kinds: half of every
  // action's cost prices the landmarks, the other half builds the database, and
  // the sum of the two bounds is still a bound.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);
  const hd::ActionCosts to_landmarks = full.scaled(0.5);
  const hd::ActionCosts to_database = full.scaled(0.5);
  CHECK(hd::is_cost_partition(task, {to_landmarks, to_database}));

  const hd::StateSpace space = hd::enumerate_state_space(task, {}, full);
  const hd::LandmarkFactory landmarks(task, to_landmarks);
  const hd::PatternDatabase database(task, pattern_of({0, 1}), to_database);

  const hd::HeuristicReport r = hd::verify_heuristic(space, [&](const hd::StripsState& s) {
    const double a = landmarks.value(s);
    const double b = database(s);
    return (a == hd::kUnboundedCost || b == hd::kUnboundedCost) ? 0.0 : a + b;
  });
  CHECK(r.admissible());
  CHECK(r.states_checked > 0);
}

TEST("pdb: a saturated cost function is a share of what the database was given") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::ActionCosts full = hd::ActionCosts::of(task);
    for (const hd::StripsState& pattern : hd::goal_patterns(task)) {
      const hd::PatternDatabase pdb(task, pattern, full);
      const hd::ActionCosts& scf = pdb.saturated_costs();
      const hd::ActionCosts left = pdb.remainder(full);
      CHECK_EQ(scf.size(), task.num_actions());
      for (std::size_t a = 0; a < task.num_actions(); ++a) {
        CHECK(scf[a] <= full[a] + 1e-12);
        CHECK_NEAR(scf[a] + left[a], full[a], 1e-12);
      }
    }
  }
}

TEST("pdb: an action outside the pattern is charged nothing") {
  // flip_y touches neither off_x nor on_x, so the database over {off_x, on_x}
  // cannot be helped by it and must leave its whole cost to another component.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);
  const hd::PatternDatabase pdb(task, pattern_of({0, 1}), full);
  CHECK_NEAR(pdb.saturated_costs()[0], 1.0, 1e-12);  // flip_x
  CHECK_NEAR(pdb.saturated_costs()[1], 0.0, 1e-12);  // flip_y
  CHECK_NEAR(pdb.remainder(full)[1], 2.0, 1e-12);
}

TEST("pdb: a database keeps its whole table under its saturated costs") {
  // The property the partition rests on. The table is built under the full
  // costs and read under the saturated ones, so it is only a bound if nothing
  // in the table exceeds the goal distance the smaller costs induce.
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::ActionCosts full = hd::ActionCosts::of(task);
    for (const hd::StripsState& pattern : hd::goal_patterns(task)) {
      const hd::PatternDatabase pdb(task, pattern, full);
      const hd::StateSpace under_scf =
          hd::enumerate_state_space(task, {}, pdb.saturated_costs());
      const hd::HeuristicReport r = hd::verify_heuristic(under_scf, pdb);
      CHECK(r.admissible());
      CHECK(r.states_checked > 0);
    }
  }
}

TEST("scp: the shares handed to the components are a cost partition") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::SaturatedPatternDatabaseHeuristic h(task);
    CHECK_EQ(h.shares().size(), h.num_databases());
    CHECK(hd::is_cost_partition(task, h.shares()));
  }
}

TEST("scp: the sum of the components is admissible and consistent") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    const hd::HeuristicReport r =
        hd::verify_heuristic(space, hd::SaturatedPatternDatabaseHeuristic(task));
    CHECK(r.admissible());
    CHECK(r.consistent());
    CHECK(r.goal_aware());
    CHECK(r.states_checked > 0);
  }
}

TEST("scp: independent subgoals are added, and the maximum is not") {
  // The case the partition exists for. The two switches are independent, so
  // one database prices flip_x and the other flip_y with nothing in common.
  // The maximum returns the more expensive of the two and throws the other
  // away; the partition adds them and recovers h* exactly.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::StripsState& s = task.initial_state();
  CHECK_NEAR(hd::MaxPatternDatabaseHeuristic(task)(s), 2.0, 1e-12);
  CHECK_NEAR(hd::SaturatedPatternDatabaseHeuristic(task)(s), 3.0, 1e-12);
  CHECK_NEAR(hd::enumerate_state_space(task).initial_h_star(), 3.0, 1e-12);
}

TEST("scp: a fixed order can fall below the maximum it is supposed to beat") {
  // The counterexample, pinned so that it cannot be lost. At {start, left} the
  // first database has reached its abstract goal and reports 0, but it has
  // already taken the cost of `both`, and the second database can then reach
  // its own goal for nothing. The maximum keeps 1 and the sum returns 0. Both
  // are admissible; only one of them is informative.
  const hd::StripsTask task = hdtest::parse(hdtest::kJointTask);
  hd::StripsState start_left;
  start_left.set(0);
  start_left.set(1);
  CHECK_NEAR(hd::MaxPatternDatabaseHeuristic(task)(start_left), 1.0, 1e-12);
  CHECK_NEAR(hd::SaturatedPatternDatabaseHeuristic(task)(start_left), 0.0, 1e-12);
}

TEST("scp: rotating the order recovers the maximum, at every state") {
  // The domination the single order does not give. In the rotation that begins
  // at component i, component i is built under the task's own costs and
  // reports what it reports alone, so that rotation is already at least
  // PDB_i(s); the maximum over rotations is therefore at least the maximum
  // over components.
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    const hd::MaxPatternDatabaseHeuristic max_h(task);
    const hd::RotatedSaturatedPatternDatabaseHeuristic rot_h(task);
    CHECK_EQ(rot_h.num_orders(), hd::goal_patterns(task).size());
    for (std::size_t i = 0; i < space.size(); ++i) {
      if (space.h_star[i] == hd::kInfiniteDistance) continue;
      CHECK(rot_h(space.states[i]) >= max_h(space.states[i]) - 1e-12);
      CHECK(rot_h(space.states[i]) <= space.h_star[i] + 1e-12);
    }
  }
}

TEST("scp: every rotation is itself a cost partition") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::RotatedSaturatedPatternDatabaseHeuristic rot_h(task);
    for (const hd::SaturatedPatternDatabaseHeuristic& order : rot_h.orders()) {
      CHECK(hd::is_cost_partition(task, order.shares()));
    }
  }
}

TEST("scp: the order decides which component gets paid") {
  // The only free parameter, and it is visible in one evaluation. Whichever
  // database comes first saturates against the full costs; here neither can
  // use what the other took, so the sum is the same, but the shares are not.
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const std::vector<hd::StripsState> patterns = hd::goal_patterns(task);
  CHECK_EQ(patterns.size(), std::size_t{2});
  const std::vector<hd::StripsState> reversed{patterns[1], patterns[0]};

  const hd::SaturatedPatternDatabaseHeuristic forward(task, patterns);
  const hd::SaturatedPatternDatabaseHeuristic backward(task, reversed);
  CHECK(hd::is_cost_partition(task, backward.shares()));
  CHECK_NEAR(forward(task.initial_state()), backward(task.initial_state()), 1e-12);
  CHECK_NEAR(forward.shares()[0][0], 1.0, 1e-12);   // flip_x, to the first database
  CHECK_NEAR(backward.shares()[0][0], 0.0, 1e-12);  // which is now the other one
}

TEST("scp: an unsolvable task is reported as a dead end and not as a bound") {
  const hd::StripsTask task = hdtest::parse(hdtest::kUnsolvableTask);
  const hd::SaturatedPatternDatabaseHeuristic h(task);
  CHECK_NEAR(h(task.initial_state()), hd::kUnreachableValue, 1e-12);
}

TEST("pdb: an action is kept by the projection exactly when it touches the pattern") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  CHECK(hd::touches(task.action(0), pattern_of({0, 1})));   // flip_x adds on_x
  CHECK(!hd::touches(task.action(1), pattern_of({0, 1})));  // flip_y is invisible here
  const hd::Projection p = hd::project(task, pattern_of({0, 1}), hd::ActionCosts::of(task));
  CHECK_EQ(p.task.num_actions(), std::size_t{1});
  CHECK_EQ(p.origin.size(), std::size_t{1});
  CHECK_EQ(p.origin[0], std::size_t{0});
}

TEST("uniform: the shares are a cost partition and follow the number of users") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::UniformPatternDatabaseHeuristic h(task);
  CHECK(hd::is_cost_partition(task, h.shares()));
  // Each action is kept by exactly one of the two patterns here, so each gets
  // its whole cost and the uniform partition coincides with the saturated one.
  CHECK_NEAR(h.shares()[0][0], 1.0, 1e-12);
  CHECK_NEAR(h.shares()[1][1], 2.0, 1e-12);
  CHECK_NEAR(h(task.initial_state()), 3.0, 1e-12);
}

TEST("uniform: an action several components can use is divided between them") {
  // `both` is kept by both patterns of the joint task, so each receives half of
  // it. That is what makes the uniform partition weaker than the saturated one
  // here: each database then prices its own goal at 0.5.
  const hd::StripsTask task = hdtest::parse(hdtest::kJointTask);
  const hd::UniformPatternDatabaseHeuristic h(task);
  CHECK(hd::is_cost_partition(task, h.shares()));
  CHECK_NEAR(h.shares()[0][0], 0.5, 1e-12);
  CHECK_NEAR(h.shares()[1][0], 0.5, 1e-12);
  CHECK_NEAR(h(task.initial_state()), 1.0, 1e-12);
}

TEST("uniform: the sum is admissible on the fixtures") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    const hd::HeuristicReport r =
        hd::verify_heuristic(space, hd::UniformPatternDatabaseHeuristic(task));
    CHECK(r.admissible());
    CHECK(r.consistent());
    CHECK(r.goal_aware());
  }
}

TEST("mixed: the databases and the landmark bound spend the costs exactly once") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::SaturatedMixedHeuristic h(task);
    for (std::size_t i = 0; i + 1 < h.num_orders(); ++i) {
      const std::vector<hd::ActionCosts> parts = h.partition(i);
      CHECK(hd::is_cost_partition(task, parts));
      for (std::size_t a = 0; a < task.num_actions(); ++a) {
        double spent = 0.0;
        for (const hd::ActionCosts& part : parts) spent += part[a];
        CHECK_NEAR(spent, task.cost(task.action(a)), 1e-12);  // nothing is left unspent
      }
    }
  }
}

TEST("mixed: the sum dominates both controls and stays admissible") {
  for (const char* text : {hdtest::kCorridorTask, hdtest::kSwitchesTask, hdtest::kJointTask}) {
    const hd::StripsTask task = hdtest::parse(text);
    const hd::StateSpace space = hd::enumerate_state_space(task);
    const hd::MaxPatternDatabaseHeuristic max_h(task);
    const hd::LandmarkCostHeuristic lm_h(task);
    const hd::SaturatedMixedHeuristic mix_h(task);
    for (std::size_t i = 0; i < space.size(); ++i) {
      if (space.h_star[i] == hd::kInfiniteDistance) continue;
      const double v = mix_h(space.states[i]);
      CHECK(v >= max_h(space.states[i]) - 1e-12);
      CHECK(v >= lm_h(space.states[i]) - 1e-12);
      CHECK(v <= space.h_star[i] + 1e-12);
    }
  }
}
