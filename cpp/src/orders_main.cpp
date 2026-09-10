// hd_orders: how much of a saturated cost partition is decided by the order.
//
// The rotation family of the cost-partitioning report takes the maximum over
// the k cyclic rotations of one pattern order, because that is the smallest
// family in which every component leads once and is therefore the cheapest one
// carrying the domination guarantee. Whether it is also close to the best a
// family of orders can do is a separate question, and the report answers it by
// inference. This answers it by enumeration.
//
// For one instance it builds the saturated chain of every permutation of the
// patterns, evaluates each over the same set of states, and reports four
// quantities against exact goal distances:
//
//   max over components   the control of section 6.3
//   best single order     what a perfect choice of one order would give
//   rotation family       the maximum over the k rotations
//   all orders            the maximum over every permutation
//
// The gap between the last two bounds what any further diversification of the
// order family can buy. The gap between the second and the third says whether
// one searched order could replace the family at a k-th of the cost.
//
// The number of patterns is the number of goal conditions, which for
// Blocksworld is the number of blocks, so the enumeration is 5! at five blocks
// and 7! at seven. Past that `--max-orders` samples permutations instead, with
// the rotations always included.
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "hd/heuristic.hpp"
#include "hd/json.hpp"
#include "hd/oracle.hpp"
#include "hd/pdb.hpp"
#include "hd/runner.hpp"

namespace {

const char* kUsage = R"(hd_orders - the order dependence of a saturated cost partition

Enumerates the saturated chains of every permutation of the goal patterns of an
instance and reports what the best order, the cyclic rotations, and the maximum
over all orders are worth against exact goal distances.

Usage:
  hd_orders --instance FILE [--instance FILE ...] [options]

Options:
  --instance FILE        Task file (repeatable).
  --sample N             Evaluate on at most N states, taken at a fixed stride
                         through the enumerated space (default: 0, all of them).
  --max-orders N         Sample N permutations instead of enumerating k!, when
                         k! exceeds N. The rotations are always included.
                         (default: 5040)
  --max-pattern-size N   Proposition budget per pattern (default: 12)
  --max-states N         Enumeration ceiling (default: 1000000)
  --seed N               Seed for permutation sampling (default: 20260910)
  --output FILE          Write JSON here instead of stdout.
  -h, --help             Print this message.
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

std::size_t factorial(std::size_t n) {
  std::size_t f = 1;
  for (std::size_t i = 2; i <= n; ++i) f *= i;
  return f;
}

// A permutation is a cyclic rotation iff it advances by one from its first
// element, modulo k. Recognised instead of generated so that the rotations are
// scored by the same code path as every other order.
bool is_rotation(const std::vector<std::size_t>& p) {
  const std::size_t k = p.size();
  for (std::size_t j = 0; j < k; ++j) {
    if (p[j] != (p[0] + j) % k) return false;
  }
  return true;
}

std::string permutation_string(const std::vector<std::size_t>& p) {
  std::string out;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (i) out += " ";
    out += std::to_string(p[i]);
  }
  return out;
}

struct Options {
  std::size_t sample = 0;
  std::size_t max_orders = 5040;
  std::size_t max_pattern_size = 12;
  std::size_t max_states = 1000000;
  unsigned seed = 20260910;
};

// One instance, enumerated.
std::string run_instance(const std::string& path, const Options& opt) {
  const hd::StripsTask task = load_task(path);
  hd::OracleLimits limits;
  limits.max_states = opt.max_states;
  const hd::StateSpace space = hd::enumerate_state_space(task, limits);
  if (!space.complete()) throw std::runtime_error("state space of '" + path + "' is truncated");

  // Informedness is defined over states with a finite non-zero goal distance.
  // A stride keeps the sample spread over the whole space; taking a prefix
  // would take the states nearest the initial one.
  std::vector<std::uint32_t> ids;
  for (std::uint32_t i = 0; i < space.size(); ++i) {
    if (space.h_star[i] > 0.0 && std::isfinite(space.h_star[i])) ids.push_back(i);
  }
  if (opt.sample && ids.size() > opt.sample) {
    const std::size_t stride = (ids.size() + opt.sample - 1) / opt.sample;
    std::vector<std::uint32_t> thinned;
    for (std::size_t i = 0; i < ids.size(); i += stride) thinned.push_back(ids[i]);
    ids.swap(thinned);
  }
  const double n = static_cast<double>(ids.size());

  const std::vector<hd::StripsState> patterns = hd::goal_patterns(task, opt.max_pattern_size);
  const std::size_t k = patterns.size();

  // The control, and the components it is the maximum of.
  const hd::MaxPatternDatabaseHeuristic control(task, opt.max_pattern_size);
  double control_sum = 0.0;
  for (const std::uint32_t id : ids) control_sum += control(space.states[id]) / space.h_star[id];

  const std::size_t total_orders = factorial(k);
  const bool exhaustive = total_orders <= opt.max_orders;

  std::vector<double> best_over_all(ids.size(), 0.0);
  std::vector<double> best_over_rotations(ids.size(), 0.0);
  double best_single = -1.0, worst_single = -1.0;
  std::vector<std::size_t> best_order, worst_order;
  std::size_t scored = 0, rotations_scored = 0;

  // Chains are built and discarded one at a time. Holding k! of them would be
  // k! * k databases; holding one is k.
  const auto score = [&](const std::vector<std::size_t>& perm) {
    std::vector<hd::StripsState> ordered;
    ordered.reserve(k);
    for (const std::size_t i : perm) ordered.push_back(patterns[i]);
    const hd::SaturatedPatternDatabaseHeuristic chain(task, ordered);

    const bool rotation = is_rotation(perm);
    double sum = 0.0;
    for (std::size_t j = 0; j < ids.size(); ++j) {
      const double v = chain(space.states[ids[j]]);
      sum += v / space.h_star[ids[j]];
      if (v > best_over_all[j]) best_over_all[j] = v;
      if (rotation && v > best_over_rotations[j]) best_over_rotations[j] = v;
    }
    const double mean = sum / n;
    if (best_single < 0.0 || mean > best_single) { best_single = mean; best_order = perm; }
    if (worst_single < 0.0 || mean < worst_single) { worst_single = mean; worst_order = perm; }
    ++scored;
    rotations_scored += rotation ? 1 : 0;
  };

  std::vector<std::size_t> perm(k);
  std::iota(perm.begin(), perm.end(), 0);
  if (exhaustive) {
    do { score(perm); } while (std::next_permutation(perm.begin(), perm.end()));
  } else {
    // The rotations first, so that the family they define is scored whatever
    // the sample does, then permutations drawn without replacement in effect:
    // a repeat costs a rebuild and changes no maximum.
    for (std::size_t i = 0; i < k; ++i) {
      std::vector<std::size_t> rot(k);
      for (std::size_t j = 0; j < k; ++j) rot[j] = (i + j) % k;
      score(rot);
    }
    std::mt19937 rng(opt.seed);
    for (std::size_t i = k; i < opt.max_orders; ++i) {
      std::shuffle(perm.begin(), perm.end(), rng);
      score(perm);
    }
  }

  double all_sum = 0.0, rot_sum = 0.0;
  std::size_t all_beats_rotations = 0;
  for (std::size_t j = 0; j < ids.size(); ++j) {
    all_sum += best_over_all[j] / space.h_star[ids[j]];
    rot_sum += best_over_rotations[j] / space.h_star[ids[j]];
    if (best_over_all[j] > best_over_rotations[j] + 1e-9) ++all_beats_rotations;
  }

  hd::json::Object o;
  o.set("instance", task.name());
  o.set("instance_path", path);
  o.set("num_patterns", k);
  o.set("orders_possible", total_orders);
  o.set("orders_scored", scored);
  o.set("orders_exhaustive", exhaustive);
  o.set("rotations_scored", rotations_scored);
  o.set("states_enumerated", space.size());
  o.set("states_scored", ids.size());
  o.set("informedness_max_over_components", control_sum / n);
  o.set("informedness_worst_single_order", worst_single);
  o.set("informedness_best_single_order", best_single);
  o.set("informedness_rotation_family", rot_sum / n);
  o.set("informedness_all_orders", all_sum / n);
  o.set("states_where_all_orders_beats_rotations", all_beats_rotations);
  o.set("best_order", permutation_string(best_order));
  o.set("worst_order", permutation_string(worst_order));

  std::fprintf(stderr,
               "%-16s k=%zu  orders=%zu%s  states=%zu\n"
               "                 max over components %.4f\n"
               "                 worst single order  %.4f\n"
               "                 best single order   %.4f  (%s)\n"
               "                 rotation family     %.4f\n"
               "                 all orders          %.4f   [%zu states above the rotations]\n",
               task.name().c_str(), k, scored, exhaustive ? "" : " sampled", ids.size(),
               control_sum / n, worst_single, best_single, permutation_string(best_order).c_str(),
               rot_sum / n, all_sum / n, all_beats_rotations);
  return o.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> instances;
  std::string output_path;
  Options opt;

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
      } else if (arg == "--sample") {
        opt.sample = std::stoull(value("--sample"));
      } else if (arg == "--max-orders") {
        opt.max_orders = std::stoull(value("--max-orders"));
      } else if (arg == "--max-pattern-size") {
        opt.max_pattern_size = std::stoull(value("--max-pattern-size"));
      } else if (arg == "--max-states") {
        opt.max_states = std::stoull(value("--max-states"));
      } else if (arg == "--seed") {
        opt.seed = static_cast<unsigned>(std::stoul(value("--seed")));
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

    std::string runs = "[";
    for (std::size_t i = 0; i < instances.size(); ++i) {
      if (i) runs += ",";
      runs += run_instance(instances[i], opt);
    }
    runs += "]";

    hd::json::Object doc;
    doc.set("schema", "hd.order_dependence/1");
    doc.set("timestamp", iso_timestamp());
    doc.set("sample", opt.sample);
    doc.set("max_orders", opt.max_orders);
    doc.set("max_pattern_size", opt.max_pattern_size);
    doc.set("seed", static_cast<std::size_t>(opt.seed));
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
    std::cerr << "hd_orders: " << e.what() << "\n";
    return 1;
  }
}
