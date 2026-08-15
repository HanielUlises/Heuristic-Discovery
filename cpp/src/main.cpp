// hd_plan: run one or more STRIPS instances under a given search algorithm and
// heuristic, and emit machine-readable metrics on stdout or to a file.
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hd/runner.hpp"

namespace {

const char* kUsage = R"(hd_plan - propositional STRIPS planner with pluggable heuristics

Usage:
  hd_plan --instance FILE [--instance FILE ...] [options]
  hd_plan --list-features
  hd_plan --build-info

Options:
  --instance FILE          Task file (repeatable; all runs share one JSON output).
  --search NAME            bfs | gbfs | astar            (default: gbfs)
  --heuristic SPEC         zero | goal_count | relaxed_layers
                           | [linear:]name=w,name=w,...  (default: zero)
  --max-expansions N       Expansion budget, 0 for unlimited (default: 1000000)
  --time-limit SECONDS     Wall-clock budget, 0 for unlimited (default: 60)
  --seed N                 Recorded in the output; the search itself is deterministic.
  --emit-plan              Include the action sequence in the output.
  --output FILE            Write JSON here instead of stdout.
  --list-features          Print the available features as JSON and exit.
  --build-info             Print build provenance as JSON and exit.
  -h, --help               Print this message.
)";

std::string iso_timestamp() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

std::string features_json() {
  std::string arr = "[";
  for (std::size_t i = 0; i < hd::kNumFeatures; ++i) {
    if (i) arr += ",";
    hd::json::Object o;
    o.set("name", std::string(hd::kFeatureNames[i]));
    o.set("description", std::string(hd::kFeatureDescriptions[i]));
    o.set("index", i);
    arr += o.str();
  }
  return arr + "]";
}

hd::StripsTask load_task(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open instance file '" + path + "'");
  return hd::StripsTask::parse(in);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> instances;
  std::string search_name = "gbfs";
  std::string heuristic_spec = "zero";
  std::string output_path;
  hd::SearchLimits limits{1000000, 60.0};
  long long seed = 0;
  bool emit_plan = false;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto value = [&](const char* what) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
        return argv[++i];
      };
      if (arg == "-h" || arg == "--help") {
        std::cout << kUsage;
        return 0;
      } else if (arg == "--list-features") {
        std::cout << features_json() << "\n";
        return 0;
      } else if (arg == "--build-info") {
        std::cout << hd::build_info_json() << "\n";
        return 0;
      } else if (arg == "--instance") {
        instances.push_back(value("--instance"));
      } else if (arg == "--search") {
        search_name = value("--search");
      } else if (arg == "--heuristic") {
        heuristic_spec = value("--heuristic");
      } else if (arg == "--output") {
        output_path = value("--output");
      } else if (arg == "--max-expansions") {
        limits.max_expansions = std::stoull(value("--max-expansions"));
      } else if (arg == "--time-limit") {
        limits.time_limit_seconds = std::stod(value("--time-limit"));
      } else if (arg == "--seed") {
        seed = std::stoll(value("--seed"));
      } else if (arg == "--emit-plan") {
        emit_plan = true;
      } else {
        throw std::runtime_error("unknown argument '" + arg + "'");
      }
    }

    if (instances.empty()) {
      std::cerr << kUsage;
      return 2;
    }

    const hd::SearchAlgorithm algorithm = hd::algorithm_from_name(search_name);
    const hd::HeuristicSpec spec = hd::parse_heuristic_spec(heuristic_spec);

    std::string runs = "[";
    for (std::size_t i = 0; i < instances.size(); ++i) {
      const hd::StripsTask task = load_task(instances[i]);
      const hd::SearchResult result = hd::run_search(task, algorithm, spec, limits);

      hd::json::Object run;
      run.set("instance", task.name());
      run.set("instance_path", instances[i]);
      run.set("num_propositions", task.num_propositions());
      run.set("num_actions", task.num_actions());
      run.set("num_goal_conditions", task.num_goal_conditions());
      run.raw("metrics", hd::metrics_json(result));
      if (emit_plan) run.set("plan", result.plan);
      if (i) runs += ",";
      runs += run.str();
    }
    runs += "]";

    hd::json::Object doc;
    doc.set("schema", "hd.planner_result/1");
    doc.set("timestamp", iso_timestamp());
    doc.set("search", hd::algorithm_name(algorithm));
    doc.raw("heuristic", hd::heuristic_json(spec));
    doc.set("seed", static_cast<double>(seed));
    hd::json::Object lim;
    lim.set("max_expansions", limits.max_expansions);
    lim.set("time_limit_seconds", limits.time_limit_seconds);
    doc.raw("limits", lim.str());
    doc.raw("build", hd::build_info_json());
    doc.raw("runs", runs);

    const std::string text = doc.str();
    if (output_path.empty()) {
      std::cout << text << "\n";
    } else {
      std::ofstream out(output_path);
      if (!out) throw std::runtime_error("cannot write to '" + output_path + "'");
      out << text << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "hd_plan: " << e.what() << "\n";
    return 1;
  }
}
