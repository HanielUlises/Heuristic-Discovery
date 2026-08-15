// Tasks: parsing, applicability, transitions and goal checking.
#include <sstream>

#include "fixtures.hpp"
#include "hd/strips.hpp"
#include "test_framework.hpp"

using hdtest::parse;

TEST("task: parsing recovers structure, names and costs") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  CHECK_EQ(task.name(), std::string("corridor"));
  CHECK_EQ(task.num_propositions(), 3u);
  CHECK_EQ(task.num_actions(), 2u);
  CHECK_EQ(task.proposition_name(2), std::string("at_c"));
  CHECK(task.initial_state().test(0));
  CHECK(!task.initial_state().test(1));
  CHECK_EQ(task.num_goal_conditions(), 1u);
  CHECK_EQ(task.action(0).name, std::string("move_a_b"));
  CHECK_NEAR(task.action(0).cost, 1.0, 1e-12);
}

TEST("task: non-unit action costs are parsed") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  CHECK_NEAR(task.cost(task.action(0)), 1.0, 1e-12);
  CHECK_NEAR(task.cost(task.action(1)), 2.0, 1e-12);
}

TEST("task: comments and blank lines are ignored") {
  const char* text = R"(
# a comment
propositions 2   # trailing comment

init 0
goal 1
action a 1
pre 0
add 1
end
)";
  const hd::StripsTask task = parse(text);
  CHECK_EQ(task.num_propositions(), 2u);
  CHECK_EQ(task.num_actions(), 1u);
}

TEST("task: malformed input is rejected") {
  bool threw = false;
  try {
    parse("propositions 2\naction a 1\npre 99\nend\n");
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    parse("frobnicate 3\n");
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

TEST("action: applicability tests preconditions only") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::StripsState s0 = task.initial_state();
  CHECK(task.applicable(s0, task.action(0)));
  CHECK(!task.applicable(s0, task.action(1)));
}

TEST("action: transitions apply deletes before adds and leave the source intact") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::StripsState s0 = task.initial_state();
  const hd::StripsState s1 = task.apply(s0, task.action(0));
  CHECK(s0.test(0));  // predecessor is unchanged
  CHECK(!s1.test(0));
  CHECK(s1.test(1));
  CHECK_EQ(s1.count(), 1u);
}

TEST("goal: satisfied only when every goal condition holds") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  hd::StripsState s = task.initial_state();
  CHECK(!task.is_goal(s));
  s = task.apply(s, task.action(0));
  CHECK(!task.is_goal(s));
  s = task.apply(s, task.action(1));
  CHECK(task.is_goal(s));
}

TEST("task: applicable action counting matches manual enumeration") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  CHECK_EQ(task.num_applicable(task.initial_state()), 2u);
  const hd::StripsState s = task.apply(task.initial_state(), task.action(0));
  CHECK_EQ(task.num_applicable(s), 1u);
}
