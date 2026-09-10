// Cost functions, and the properties a component of a cost partition must have
// when it is evaluated on one.
#include <vector>

#include "fixtures.hpp"
#include "hd/costs.hpp"
#include "hd/landmarks.hpp"
#include "hd/oracle.hpp"
#include "hd/verify.hpp"
#include "test_framework.hpp"

namespace {

// h_LM under `costs`, checked against the h* of the same task under the same
// costs: the property every component of a partition has to have.
hd::HeuristicReport check_under(const hd::StripsTask& task, const hd::ActionCosts& costs) {
  const hd::StateSpace space = hd::enumerate_state_space(task, {}, costs);
  const hd::LandmarkFactory factory(task, costs);
  return hd::verify_heuristic(space, [&](const hd::StripsState& s) {
    const double v = factory.value(s);
    return v == hd::kUnboundedCost ? 0.0 : v;  // a dead end bounds nothing
  });
}

}  // namespace

TEST("costs: a task's own costs are read off its actions") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts costs = hd::ActionCosts::of(task);

  CHECK_EQ(costs.size(), task.num_actions());
  CHECK_NEAR(costs[0], 1.0, 1e-12);  // flip_x
  CHECK_NEAR(costs[1], 2.0, 1e-12);  // flip_y
}

TEST("costs: negative and non-finite costs are rejected") {
  bool threw = false;
  try {
    hd::ActionCosts bad({1.0, -0.5});
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    hd::ActionCosts::of(hdtest::parse(hdtest::kSwitchesTask)).scaled(-1.0);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

TEST("costs: a partition may leave costs unspent but not overspend them") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);

  CHECK(hd::is_cost_partition(task, {full.scaled(0.5), full.scaled(0.5)}));
  CHECK(hd::is_cost_partition(task, {full.scaled(0.25), full.scaled(0.25)}));  // unspent
  CHECK(!hd::is_cost_partition(task, {full, full}));
  CHECK(hd::is_cost_partition(task, {full, hd::ActionCosts::zero(task)}));
}

TEST("landmarks: the bound scales with a uniform scaling of the costs") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);
  const hd::LandmarkFactory whole(task, full);
  const hd::LandmarkFactory third(task, full.scaled(1.0 / 3.0));

  const hd::StripsState& s = task.initial_state();
  CHECK_NEAR(whole.value(s), 2.0, 1e-12);
  CHECK_NEAR(third.value(s), 2.0 / 3.0, 1e-12);
}

TEST("landmarks: generation does not depend on the costs") {
  const hd::StripsTask task = hdtest::parse(hdtest::kCorridorTask);
  const hd::LandmarkFactory priced(task);
  const hd::LandmarkFactory free_of_charge(task, hd::ActionCosts::zero(task));

  const hd::StripsState& s = task.initial_state();
  CHECK_EQ(priced.compute(s).num_unachieved(), free_of_charge.compute(s).num_unachieved());
  CHECK_NEAR(free_of_charge.value(s), 0.0, 1e-12);  // priced at nothing, still a bound
}

TEST("landmarks: the bound is admissible under the costs it was given") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);

  for (const double factor : {1.0, 0.5, 0.1, 0.0}) {
    const hd::HeuristicReport r = check_under(task, full.scaled(factor));
    CHECK(r.admissible());
    CHECK(r.non_negative());
  }
  // An uneven split is the case a partition actually produces.
  const hd::HeuristicReport uneven = check_under(task, hd::ActionCosts({0.75, 0.25}));
  CHECK(uneven.admissible());
}

TEST("landmarks: two components on a partition sum to a bound") {
  // The weakest possible partition, but it exercises the invariant: each half
  // of the costs prices the landmarks on its own, and the two bounds add.
  const hd::StripsTask task = hdtest::parse(hdtest::kJointTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);
  const hd::ActionCosts left({0.5, 0.5, 0.5});
  const hd::ActionCosts right({0.5, 0.5, 0.5});
  CHECK(hd::is_cost_partition(task, {left, right}));

  const hd::StateSpace space = hd::enumerate_state_space(task, {}, full);
  const hd::LandmarkFactory a(task, left);
  const hd::LandmarkFactory b(task, right);
  const hd::HeuristicReport r = hd::verify_heuristic(space, [&](const hd::StripsState& s) {
    const double va = a.value(s);
    const double vb = b.value(s);
    return (va == hd::kUnboundedCost || vb == hd::kUnboundedCost) ? 0.0 : va + vb;
  });
  CHECK(r.admissible());
  CHECK_NEAR(a.value(task.initial_state()) + b.value(task.initial_state()), 1.0, 1e-12);
}

TEST("oracle: goal distances follow the costs they are given") {
  const hd::StripsTask task = hdtest::parse(hdtest::kSwitchesTask);
  const hd::ActionCosts full = hd::ActionCosts::of(task);

  const hd::StateSpace whole = hd::enumerate_state_space(task, {}, full);
  const hd::StateSpace half = hd::enumerate_state_space(task, {}, full.scaled(0.5));
  CHECK_NEAR(whole.initial_h_star(), 3.0, 1e-12);
  CHECK_NEAR(half.initial_h_star(), 1.5, 1e-12);

  const hd::StateSpace free_of_charge =
      hd::enumerate_state_space(task, {}, hd::ActionCosts::zero(task));
  CHECK_NEAR(free_of_charge.initial_h_star(), 0.0, 1e-12);
  CHECK_EQ(free_of_charge.size(), whole.size());  // the state space is unchanged
}
