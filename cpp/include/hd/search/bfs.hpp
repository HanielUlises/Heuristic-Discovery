// Breadth-first search: uninformed, optimal for unit-cost tasks.
#pragma once

#include <deque>
#include <unordered_set>
#include <vector>

#include "hd/domain.hpp"
#include "hd/search/result.hpp"

namespace hd {

template <Domain D>
SearchResult breadth_first_search(const D& domain, const SearchLimits& limits = {}) {
  using State = typename D::State;
  using Node = SearchNode<State>;

  Timer timer;
  SearchResult out;
  std::vector<Node> nodes;
  std::unordered_set<State> closed;
  std::deque<std::size_t> open;

  nodes.push_back(Node{domain.initial_state(), Node::kNoParent, 0, 0.0, 0});
  closed.insert(nodes[0].state);
  open.push_back(0);
  out.generated = 1;

  if (domain.is_goal(nodes[0].state)) {
    detail::extract_plan(domain, nodes, 0, out);
  } else {
    while (!open.empty()) {
      if (detail::limits_exceeded(limits, timer, out.expanded, out)) break;
      const std::size_t id = open.front();
      open.pop_front();
      ++out.expanded;
      out.max_depth = std::max<std::size_t>(out.max_depth, nodes[id].depth);

      const std::size_t n = domain.num_actions();
      for (std::size_t a = 0; a < n; ++a) {
        const auto& act = domain.action(a);
        if (!domain.applicable(nodes[id].state, act)) continue;
        State succ = domain.apply(nodes[id].state, act);
        if (!closed.insert(succ).second) continue;
        nodes.push_back(Node{succ, id, a, nodes[id].g + domain.cost(act),
                             nodes[id].depth + 1});
        ++out.generated;
        const std::size_t child = nodes.size() - 1;
        if (domain.is_goal(succ)) {
          detail::extract_plan(domain, nodes, child, out);
          out.max_depth = std::max<std::size_t>(out.max_depth, nodes[child].depth);
          open.clear();
          break;
        }
        open.push_back(child);
      }
    }
  }

  out.peak_nodes = nodes.size();
  out.runtime_seconds = timer.seconds();
  out.peak_memory_kb = peak_memory_kb();
  return out;
}

}  // namespace hd
