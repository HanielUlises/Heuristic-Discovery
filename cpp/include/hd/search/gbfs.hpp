// Greedy best-first search: expands the node minimising h(s), ignoring g.
//
// This is the workhorse of the discovery loop: it is maximally sensitive to
// heuristic quality, so differences in h show up directly in expansions.
#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <vector>

#include "hd/domain.hpp"
#include "hd/search/result.hpp"

namespace hd {

namespace detail {

// (key, insertion order, node id). The insertion counter makes tie-breaking
// deterministic, which is required for reproducible experiments.
struct OpenEntry {
  double key;
  double tiebreak;
  std::uint64_t order;
  std::size_t node;

  bool operator>(const OpenEntry& o) const {
    if (key != o.key) return key > o.key;
    if (tiebreak != o.tiebreak) return tiebreak > o.tiebreak;
    return order > o.order;
  }
};

using OpenList = std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>>;

}  // namespace detail

template <Domain D, class H>
  requires Heuristic<H, typename D::State>
SearchResult greedy_best_first_search(const D& domain, const H& h,
                                      const SearchLimits& limits = {}) {
  using State = typename D::State;
  using Node = SearchNode<State>;

  Timer timer;
  SearchResult out;
  std::vector<Node> nodes;
  std::unordered_set<State> closed;
  detail::OpenList open;
  std::uint64_t order = 0;

  nodes.push_back(Node{domain.initial_state(), Node::kNoParent, 0, 0.0, 0});
  closed.insert(nodes[0].state);
  ++out.generated;
  ++out.evaluated;
  open.push({h(nodes[0].state), 0.0, order++, 0});

  while (!open.empty()) {
    if (detail::limits_exceeded(limits, timer, out.expanded, out)) break;
    const std::size_t id = open.top().node;
    open.pop();
    ++out.expanded;
    out.max_depth = std::max<std::size_t>(out.max_depth, nodes[id].depth);

    if (domain.is_goal(nodes[id].state)) {
      detail::extract_plan(domain, nodes, id, out);
      break;
    }

    const std::size_t n = domain.num_actions();
    for (std::size_t a = 0; a < n; ++a) {
      const auto& act = domain.action(a);
      if (!domain.applicable(nodes[id].state, act)) continue;
      State succ = domain.apply(nodes[id].state, act);
      if (!closed.insert(succ).second) continue;
      nodes.push_back(Node{succ, id, a, nodes[id].g + domain.cost(act), nodes[id].depth + 1});
      ++out.generated;
      ++out.evaluated;
      open.push({h(succ), 0.0, order++, nodes.size() - 1});
    }
  }

  out.peak_nodes = nodes.size();
  out.runtime_seconds = timer.seconds();
  out.peak_memory_kb = peak_memory_kb();
  return out;
}

}  // namespace hd
