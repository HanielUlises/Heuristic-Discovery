// hd_verify: check heuristics against exact goal distances on instances small
// enough to enumerate, and emit the verdict as machine-readable JSON.
//
// The state space of an instance is enumerated once and every heuristic given
// on the command line is checked against it, so verifying a set of candidates
// costs one enumeration rather than one per candidate.
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hd/oracle.hpp"
#include "hd/runner.hpp"
#include "hd/verify.hpp"

namespace {

const char* kUsage = R"(hd_verify - admissibility and consistency of heuristics against exact h*

Enumerates the reachable state space of each instance, computes h* by backward
Dijkstra from the goal states, and reports whether each heuristic overestimates.
Only instances small enough to enumerate can be checked; a truncated state space
is reported as such and no verdict is given for it.

Usage:
  hd_verify --instance FILE [--instance FILE ...] [--heuristic SPEC ...] [options]

Options:
  --instance FILE          Task file (repeatable).
  --heuristic SPEC         zero | goal_count | relaxed_layers | landmark_cost
                           | [linear:]name=w,name=w,...   (repeatable; default: zero)
  --max-states N           Enumeration ceiling, 0 for unlimited (default: 200000)
  --time-limit SECONDS     Enumeration budget per instance, 0 for unlimited (default: 0)
  --witnesses K            Violating states reported per heuristic (default: 3)
  --tolerance EPS          Slack before a violation is counted (default: 1e-9)
  --output FILE            Write JSON here instead of stdout.
  -h, --help               Print this message.
)";

std::string iso_timestamp() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

hd::StripsTask load_task(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open instance file '" + path + "'");
  return hd::StripsTask::parse(in);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> instances;
  std::vector<std::string> heuristics;
  std::string output_path;
  hd::OracleLimits limits;
  hd::VerifyOptions options;

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
      } else if (arg == "--instance") {
        instances.push_back(value("--instance"));
      } else if (arg == "--heuristic") {
        heuristics.push_back(value("--heuristic"));
      } else if (arg == "--max-states") {
        limits.max_states = std::stoull(value("--max-states"));
      } else if (arg == "--time-limit") {
        limits.time_limit_seconds = std::stod(value("--time-limit"));
      } else if (arg == "--witnesses") {
        options.max_witnesses = std::stoull(value("--witnesses"));
      } else if (arg == "--tolerance") {
        options.tolerance = std::stod(value("--tolerance"));
      } else if (arg == "--output") {
        output_path = value("--output");
      } else {
        throw std::runtime_error("unknown argument '" + arg + "'");
      }
    }

    if (instances.empty()) {
      std::cerr << kUsage;
      return 2;
    }
    if (heuristics.empty()) heuristics.push_back("zero");

    std::vector<hd::HeuristicSpec> specs;
    specs.reserve(heuristics.size());
    for (const std::string& text : heuristics) specs.push_back(hd::parse_heuristic_spec(text));

    std::string runs = "[";
    for (std::size_t i = 0; i < instances.size(); ++i) {
      const hd::StripsTask task = load_task(instances[i]);
      const hd::StateSpace space = hd::enumerate_state_space(task, limits);

      std::string reports = "[";
      for (std::size_t k = 0; space.complete() && k < specs.size(); ++k) {
        const hd::HeuristicReport report = hd::with_heuristic(
            task, specs[k],
            [&](const auto& h) { return hd::verify_heuristic(space, h, options); });
        hd::json::Object entry;
        entry.raw("heuristic", hd::heuristic_json(specs[k]));
        entry.raw("report", hd::report_json(task, space, report));
        if (k) reports += ",";
        reports += entry.str();
      }
      reports += "]";

      hd::json::Object run;
      run.set("instance", task.name());
      run.set("instance_path", instances[i]);
      run.set("num_propositions", task.num_propositions());
      run.set("num_actions", task.num_actions());
      run.set("num_goal_conditions", task.num_goal_conditions());
      run.raw("state_space", hd::state_space_json(space));
      run.raw("heuristics", reports);
      if (i) runs += ",";
      runs += run.str();
    }
    runs += "]";

    hd::json::Object doc;
    doc.set("schema", "hd.verification/1");
    doc.set("timestamp", iso_timestamp());
    hd::json::Object lim;
    lim.set("max_states", limits.max_states);
    lim.set("time_limit_seconds", limits.time_limit_seconds);
    lim.set("tolerance", options.tolerance);
    lim.set("witnesses", options.max_witnesses);
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
    std::cerr << "hd_verify: " << e.what() << "\n";
    return 1;
  }
}
