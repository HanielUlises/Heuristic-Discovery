// A*: expands the node minimising f(s) = g(s) + h(s).
//
// Optimal whenever h is admissible; used here both as an optimal reference
// planner and as a second objective surface for heuristic discovery.
#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "hd/domain.hpp"
#include "hd/search/gbfs.hpp"  // detail::OpenList
#include "hd/search/result.hpp"

namespace hd {

template <Domain D, class H>
  requires Heuristic<H, typename D::State>
SearchResult a_star(const D& domain, const H& h, const SearchLimits& limits = {}) {
  using State = typename D::State;
  using Node = SearchNode<State>;

  Timer timer;
  SearchResult out;
  std::vector<Node> nodes;
  std::unordered_map<State, double> best_g;  // lowest g seen per state
  detail::OpenList open;
  std::uint64_t order = 0;

  nodes.push_back(Node{domain.initial_state(), Node::kNoParent, 0, 0.0, 0});
  best_g[nodes[0].state] = 0.0;
  ++out.generated;
  ++out.evaluated;
  const double h0 = h(nodes[0].state);
  open.push({h0, h0, order++, 0});

  while (!open.empty()) {
    if (detail::limits_exceeded(limits, timer, out.expanded, out)) break;
    const detail::OpenEntry entry = open.top();
    open.pop();
    const std::size_t id = entry.node;

    // Stale entry: a cheaper path to this state was expanded already.
    if (const auto it = best_g.find(nodes[id].state);
        it != best_g.end() && nodes[id].g > it->second) {
      continue;
    }

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
      const double g = nodes[id].g + domain.cost(act);

      const auto it = best_g.find(succ);
      if (it != best_g.end()) {
        if (g >= it->second) continue;
        it->second = g;
        ++out.reopened;
      } else {
        best_g.emplace(succ, g);
      }

      nodes.push_back(Node{succ, id, a, g, nodes[id].depth + 1});
      ++out.generated;
      ++out.evaluated;
      const double hs = h(succ);
      open.push({g + hs, hs, order++, nodes.size() - 1});
    }
  }

  out.peak_nodes = nodes.size();
  out.runtime_seconds = timer.seconds();
  out.peak_memory_kb = peak_memory_kb();
  return out;
}

}  // namespace hd
