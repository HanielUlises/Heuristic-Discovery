// Search node storage, resource limits, and the metrics record produced by
// every planner execution.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace hd {

struct SearchLimits {
  std::size_t max_expansions = 0;      // 0 = unlimited
  double time_limit_seconds = 0.0;     // <= 0 = unlimited
};

enum class SearchStatus { kSolved, kUnsolvable, kExpansionLimit, kTimeLimit };

inline const char* to_string(SearchStatus s) {
  switch (s) {
    case SearchStatus::kSolved: return "solved";
    case SearchStatus::kUnsolvable: return "unsolvable";
    case SearchStatus::kExpansionLimit: return "expansion_limit";
    case SearchStatus::kTimeLimit: return "time_limit";
  }
  return "unknown";
}

// Metrics recorded for one planner execution. This is the unit of observation
// of every experiment, and maps one-to-one onto the emitted JSON.
struct SearchResult {
  SearchStatus status = SearchStatus::kUnsolvable;
  bool solved = false;
  double solution_cost = std::numeric_limits<double>::quiet_NaN();
  std::size_t solution_length = 0;
  std::size_t expanded = 0;
  std::size_t generated = 0;
  std::size_t evaluated = 0;   // heuristic evaluations
  std::size_t reopened = 0;    // states rediscovered with a lower g (A*)
  std::size_t max_depth = 0;   // deepest node reached
  std::size_t peak_nodes = 0;  // nodes held in memory
  double runtime_seconds = 0.0;
  std::size_t peak_memory_kb = 0;
  std::vector<std::string> plan;
};

// One search node. Parent links keep states shared-free and plan extraction
// O(depth); the state itself is a fixed-size, trivially copyable bitset.
template <class State>
struct SearchNode {
  State state;
  std::size_t parent = kNoParent;
  std::size_t action = 0;
  double g = 0.0;
  std::uint32_t depth = 0;

  static constexpr std::size_t kNoParent = static_cast<std::size_t>(-1);
};

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

// Resident set peak in kilobytes, or 0 where unavailable.
std::size_t peak_memory_kb();

namespace detail {

// Reconstructs the action-name plan by following parent links.
template <class D, class State>
void extract_plan(const D& domain, const std::vector<SearchNode<State>>& nodes, std::size_t goal_id,
                  SearchResult& out) {
  std::vector<std::string> rev;
  for (std::size_t id = goal_id; nodes[id].parent != SearchNode<State>::kNoParent;
       id = nodes[id].parent) {
    rev.push_back(domain.action(nodes[id].action).name);
  }
  out.plan.assign(rev.rbegin(), rev.rend());
  out.solution_length = out.plan.size();
  out.solution_cost = nodes[goal_id].g;
  out.solved = true;
  out.status = SearchStatus::kSolved;
}

// Checks the resource limits every `kLimitCheckInterval` expansions so the
// clock is not read on every node.
inline constexpr std::size_t kLimitCheckInterval = 256;

inline bool limits_exceeded(const SearchLimits& limits, const Timer& timer, std::size_t expanded,
                            SearchResult& out) {
  if (limits.max_expansions && expanded >= limits.max_expansions) {
    out.status = SearchStatus::kExpansionLimit;
    return true;
  }
  if (limits.time_limit_seconds > 0.0 && (expanded % kLimitCheckInterval) == 0 &&
      timer.seconds() > limits.time_limit_seconds) {
    out.status = SearchStatus::kTimeLimit;
    return true;
  }
  return false;
}

}  // namespace detail
}  // namespace hd
