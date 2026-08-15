// Search algorithms: correctness, optimality guarantees, limits and metrics.
#include "fixtures.hpp"
#include "hd/heuristic.hpp"
#include "hd/search/astar.hpp"
#include "hd/search/bfs.hpp"
#include "hd/search/gbfs.hpp"
#include "test_framework.hpp"

using hd::FeatureId;
using hdtest::parse;

TEST("bfs: finds a shortest plan") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::SearchResult r = hd::breadth_first_search(task);
  CHECK(r.solved);
  CHECK(r.status == hd::SearchStatus::kSolved);
  CHECK_EQ(r.solution_length, 2u);
  CHECK_EQ(r.plan[0], std::string("move_a_b"));
  CHECK_EQ(r.plan[1], std::string("move_b_c"));
}

TEST("bfs: reports an unsolvable task after exhausting the state space") {
  const hd::StripsTask task = parse(hdtest::kUnsolvableTask);
  const hd::SearchResult r = hd::breadth_first_search(task);
  CHECK(!r.solved);
  CHECK(r.status == hd::SearchStatus::kUnsolvable);
}

TEST("gbfs: solves the task under every baseline heuristic") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::SearchResult zero = hd::greedy_best_first_search(task, hd::ZeroHeuristic{});
  const hd::SearchResult gc = hd::greedy_best_first_search(task, hd::GoalCountHeuristic(task));
  const hd::SearchResult rl = hd::greedy_best_first_search(task, hd::RelaxedLayersHeuristic(task));
  for (const hd::SearchResult* r : {&zero, &gc, &rl}) {
    CHECK(r->solved);
    CHECK_EQ(r->solution_length, 2u);
    CHECK(r->evaluated > 0u);
  }
}

TEST("gbfs: a learned linear heuristic is accepted through the concept") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::LinearHeuristic h(task, {{FeatureId::kUnsatisfiedGoals, 1.82},
                                     {FeatureId::kApplicableActions, 0.37}});
  const hd::SearchResult r = hd::greedy_best_first_search(task, h);
  CHECK(r.solved);
  CHECK_NEAR(r.solution_cost, 3.0, 1e-12);
}

TEST("astar: returns an optimal plan under an admissible heuristic") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::SearchResult optimal = hd::a_star(task, hd::ZeroHeuristic{});
  const hd::SearchResult informed = hd::a_star(task, hd::RelaxedLayersHeuristic(task));
  CHECK(optimal.solved);
  CHECK(informed.solved);
  CHECK_NEAR(informed.solution_cost, optimal.solution_cost, 1e-12);
  CHECK_NEAR(optimal.solution_cost, 3.0, 1e-12);  // flip_x (1) + flip_y (2)
}

TEST("astar: never expands more nodes when better informed") {
  const hd::StripsTask task = parse(hdtest::kCorridorTask);
  const hd::SearchResult blind = hd::a_star(task, hd::ZeroHeuristic{});
  const hd::SearchResult informed = hd::a_star(task, hd::RelaxedLayersHeuristic(task));
  CHECK(informed.expanded <= blind.expanded);
  CHECK_NEAR(informed.solution_cost, blind.solution_cost, 1e-12);
}

TEST("search: an initial state that is already a goal costs nothing") {
  const char* text = R"(
propositions 2
init 0 1
goal 1
action a 1
pre 0
add 1
end
)";
  const hd::StripsTask task = parse(text);
  for (const hd::SearchResult& r :
       {hd::breadth_first_search(task), hd::greedy_best_first_search(task, hd::ZeroHeuristic{}),
        hd::a_star(task, hd::ZeroHeuristic{})}) {
    CHECK(r.solved);
    CHECK_EQ(r.solution_length, 0u);
    CHECK_NEAR(r.solution_cost, 0.0, 1e-12);
  }
}

TEST("search: the expansion budget is honoured") {
  const hd::StripsTask task = parse(hdtest::kUnsolvableTask);
  hd::SearchLimits limits;
  limits.max_expansions = 1;
  const hd::SearchResult r = hd::greedy_best_first_search(task, hd::ZeroHeuristic{}, limits);
  CHECK(!r.solved);
  CHECK(r.status == hd::SearchStatus::kExpansionLimit);
  CHECK(r.expanded <= 1u);
}

TEST("search: metrics are internally consistent") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::SearchResult r = hd::greedy_best_first_search(task, hd::GoalCountHeuristic(task));
  CHECK(r.generated >= r.expanded);
  CHECK(r.peak_nodes == r.generated);
  CHECK(r.max_depth >= r.solution_length);
  CHECK(r.runtime_seconds >= 0.0);
}

TEST("search: repeated runs are deterministic") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::LinearHeuristic h(task, {{FeatureId::kUnsatisfiedGoals, 1.0},
                                     {FeatureId::kApplicableActions, 0.5}});
  const hd::SearchResult a = hd::greedy_best_first_search(task, h);
  const hd::SearchResult b = hd::greedy_best_first_search(task, h);
  CHECK_EQ(a.expanded, b.expanded);
  CHECK_EQ(a.generated, b.generated);
  CHECK(a.plan == b.plan);
}

TEST("search: plans are executable and reach a goal state") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::SearchResult r = hd::greedy_best_first_search(task, hd::GoalCountHeuristic(task));
  hd::StripsState s = task.initial_state();
  double cost = 0.0;
  for (const std::string& name : r.plan) {
    bool applied = false;
    for (std::size_t i = 0; i < task.num_actions(); ++i) {
      if (task.action(i).name != name) continue;
      CHECK(task.applicable(s, task.action(i)));
      cost += task.cost(task.action(i));
      s = task.apply(s, task.action(i));
      applied = true;
      break;
    }
    CHECK(applied);
  }
  CHECK(task.is_goal(s));
  CHECK_NEAR(cost, r.solution_cost, 1e-12);
}
