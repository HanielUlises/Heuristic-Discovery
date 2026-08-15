// Serialisation: the JSON writer and the run records consumed by Python.
#include <string>

#include "fixtures.hpp"
#include "hd/json.hpp"
#include "hd/runner.hpp"
#include "test_framework.hpp"

using hdtest::parse;

namespace {
bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}
}  // namespace

TEST("json: objects render typed fields") {
  hd::json::Object o;
  o.set("name", "corridor");
  o.set("solved", true);
  o.set("expanded", std::size_t{42});
  o.set("runtime", 0.5);
  CHECK_EQ(o.str(),
           std::string(R"({"name":"corridor","solved":true,"expanded":42,"runtime":0.5})"));
}

TEST("json: strings are escaped") {
  hd::json::Object o;
  o.set("path", "a\"b\\c\nd");
  CHECK_EQ(o.str(), std::string(R"({"path":"a\"b\\c\nd"})"));
}

TEST("json: non-finite numbers become null") {
  hd::json::Object o;
  o.set("cost", std::numeric_limits<double>::quiet_NaN());
  CHECK_EQ(o.str(), std::string(R"({"cost":null})"));
}

TEST("json: arrays of strings render as plans do") {
  hd::json::Object o;
  o.set("plan", std::vector<std::string>{"a", "b"});
  CHECK_EQ(o.str(), std::string(R"({"plan":["a","b"]})"));
  hd::json::Object empty;
  empty.set("plan", std::vector<std::string>{});
  CHECK_EQ(empty.str(), std::string(R"({"plan":[]})"));
}

TEST("json: nested objects compose") {
  hd::json::Object inner;
  inner.set("a", 1);
  hd::json::Object outer;
  outer.raw("inner", inner.str());
  CHECK_EQ(outer.str(), std::string(R"({"inner":{"a":1}})"));
}

TEST("json: metrics of a solved run carry every recorded quantity") {
  const hd::StripsTask task = parse(hdtest::kSwitchesTask);
  const hd::SearchResult r =
      hd::greedy_best_first_search(task, hd::GoalCountHeuristic(task), hd::SearchLimits{});
  const std::string text = hd::metrics_json(r);
  for (const char* key : {"solved", "status", "solution_cost", "solution_length", "expanded",
                          "generated", "evaluated", "reopened", "max_depth", "peak_nodes",
                          "runtime_seconds", "peak_memory_kb"}) {
    CHECK(contains(text, std::string("\"") + key + "\":"));
  }
  CHECK(contains(text, "\"solved\":true"));
  CHECK(contains(text, "\"status\":\"solved\""));
}

TEST("json: an unsolved run serialises a null cost") {
  const hd::StripsTask task = parse(hdtest::kUnsolvableTask);
  const hd::SearchResult r = hd::breadth_first_search(task);
  const std::string text = hd::metrics_json(r);
  CHECK(contains(text, "\"solved\":false"));
  CHECK(contains(text, "\"solution_cost\":null"));
  CHECK(contains(text, "\"status\":\"unsolvable\""));
}

TEST("json: a heuristic record round-trips its weights and expression") {
  const hd::HeuristicSpec spec =
      hd::parse_heuristic_spec("unsatisfied_goals=1.82,applicable_actions=0.37");
  const std::string text = hd::heuristic_json(spec);
  CHECK(contains(text, "\"kind\":\"linear\""));
  CHECK(contains(text, "\"unsatisfied_goals\":1.82"));
  CHECK(contains(text, "\"applicable_actions\":0.37"));
  CHECK(contains(text, "1.82*unsatisfied_goals + 0.37*applicable_actions"));
}

TEST("json: build provenance is recorded") {
  const std::string text = hd::build_info_json();
  for (const char* key : {"version", "compiler", "cxx_standard", "build_type", "git_commit"}) {
    CHECK(contains(text, std::string("\"") + key + "\":"));
  }
}
