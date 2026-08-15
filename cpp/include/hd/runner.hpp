// Glue between the command line and the search engine: heuristic specification
// parsing, algorithm dispatch, and JSON serialisation of a run.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "hd/build_info.hpp"
#include "hd/features.hpp"
#include "hd/heuristic.hpp"
#include "hd/json.hpp"
#include "hd/search/astar.hpp"
#include "hd/search/bfs.hpp"
#include "hd/search/gbfs.hpp"
#include "hd/strips.hpp"

namespace hd {

enum class SearchAlgorithm { kBfs, kGbfs, kAstar };

inline SearchAlgorithm algorithm_from_name(const std::string& name) {
  if (name == "bfs") return SearchAlgorithm::kBfs;
  if (name == "gbfs") return SearchAlgorithm::kGbfs;
  if (name == "astar" || name == "a*") return SearchAlgorithm::kAstar;
  throw std::runtime_error("unknown search algorithm '" + name + "'");
}

inline const char* algorithm_name(SearchAlgorithm a) {
  switch (a) {
    case SearchAlgorithm::kBfs: return "bfs";
    case SearchAlgorithm::kGbfs: return "gbfs";
    case SearchAlgorithm::kAstar: return "astar";
  }
  return "unknown";
}

// A heuristic is either one of the named baselines or an explicit linear
// combination of features, which is what the discovery loop proposes.
struct HeuristicSpec {
  enum class Kind { kZero, kGoalCount, kRelaxedLayers, kLinear };
  Kind kind = Kind::kZero;
  std::vector<LinearHeuristic::Term> terms;

  const char* kind_name() const {
    switch (kind) {
      case Kind::kZero: return "zero";
      case Kind::kGoalCount: return "goal_count";
      case Kind::kRelaxedLayers: return "relaxed_layers";
      case Kind::kLinear: return "linear";
    }
    return "unknown";
  }
};

// Accepted forms:
//   "zero" | "goal_count" | "relaxed_layers"
//   "linear:unsatisfied_goals=1.82,applicable_actions=0.37"
//   "unsatisfied_goals=1.82,applicable_actions=0.37"   (linear is implied)
inline HeuristicSpec parse_heuristic_spec(const std::string& spec) {
  HeuristicSpec out;
  if (spec.empty() || spec == "zero") return out;
  if (spec == "goal_count") {
    out.kind = HeuristicSpec::Kind::kGoalCount;
    return out;
  }
  if (spec == "relaxed_layers") {
    out.kind = HeuristicSpec::Kind::kRelaxedLayers;
    return out;
  }

  std::string body = spec;
  if (body.rfind("linear:", 0) == 0) body = body.substr(7);
  out.kind = HeuristicSpec::Kind::kLinear;

  std::size_t pos = 0;
  while (pos <= body.size()) {
    const std::size_t comma = body.find(',', pos);
    const std::string term = body.substr(pos, comma == std::string::npos ? std::string::npos
                                                                        : comma - pos);
    if (!term.empty()) {
      const std::size_t eq = term.find('=');
      if (eq == std::string::npos) {
        throw std::runtime_error("malformed heuristic term '" + term + "' (expected name=weight)");
      }
      const std::string name = term.substr(0, eq);
      const auto id = feature_from_name(name);
      if (!id) throw std::runtime_error("unknown feature '" + name + "'");
      out.terms.push_back({*id, std::stod(term.substr(eq + 1))});
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (out.terms.empty()) throw std::runtime_error("linear heuristic has no terms");
  return out;
}

inline std::string heuristic_to_string(const HeuristicSpec& spec) {
  if (spec.kind != HeuristicSpec::Kind::kLinear) return spec.kind_name();
  std::string out;
  for (std::size_t i = 0; i < spec.terms.size(); ++i) {
    if (i) out += " + ";
    out += json::number(spec.terms[i].weight) + "*" +
           std::string(feature_name(spec.terms[i].feature));
  }
  return out;
}

// Runs one (algorithm, heuristic) pair on one task.
inline SearchResult run_search(const StripsTask& task, SearchAlgorithm algorithm,
                               const HeuristicSpec& spec, const SearchLimits& limits) {
  if (algorithm == SearchAlgorithm::kBfs) return breadth_first_search(task, limits);

  const auto dispatch = [&](const auto& h) {
    return algorithm == SearchAlgorithm::kGbfs ? greedy_best_first_search(task, h, limits)
                                               : a_star(task, h, limits);
  };
  switch (spec.kind) {
    case HeuristicSpec::Kind::kZero: return dispatch(ZeroHeuristic{});
    case HeuristicSpec::Kind::kGoalCount: return dispatch(GoalCountHeuristic(task));
    case HeuristicSpec::Kind::kRelaxedLayers: return dispatch(RelaxedLayersHeuristic(task));
    case HeuristicSpec::Kind::kLinear: return dispatch(LinearHeuristic(task, spec.terms));
  }
  throw std::runtime_error("unreachable heuristic kind");
}

inline std::string heuristic_json(const HeuristicSpec& spec) {
  json::Object h;
  h.set("kind", spec.kind_name());
  h.set("expression", heuristic_to_string(spec));
  json::Object w;
  for (const auto& t : spec.terms) w.set(std::string(feature_name(t.feature)), t.weight);
  h.raw("weights", w.str());
  return h.str();
}

inline std::string metrics_json(const SearchResult& r) {
  json::Object m;
  m.set("solved", r.solved);
  m.set("status", to_string(r.status));
  m.set("solution_cost", r.solution_cost);
  m.set("solution_length", r.solution_length);
  m.set("expanded", r.expanded);
  m.set("generated", r.generated);
  m.set("evaluated", r.evaluated);
  m.set("reopened", r.reopened);
  m.set("max_depth", r.max_depth);
  m.set("peak_nodes", r.peak_nodes);
  m.set("runtime_seconds", r.runtime_seconds);
  m.set("peak_memory_kb", r.peak_memory_kb);
  return m.str();
}

inline std::string build_info_json() {
  const BuildInfo b = build_info();
  json::Object o;
  o.set("version", b.version);
  o.set("compiler", b.compiler);
  o.set("cxx_standard", b.cxx_standard);
  o.set("build_type", b.build_type);
  o.set("git_commit", b.git_commit);
  o.set("compiled_at", b.timestamp);
  return o.str();
}

}  // namespace hd
