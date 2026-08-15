// Compile-time interfaces (concepts) shared by the search engine.
//
// Search algorithms are templates constrained by these concepts, so no virtual
// call occurs on the expansion / evaluation path.
#pragma once

#include <concepts>
#include <cstddef>

namespace hd {

// A deterministic, fully observable transition system with unit or weighted
// action costs.
template <class D>
concept Domain = requires(const D& d, const typename D::State& s, std::size_t i) {
  typename D::State;
  typename D::Action;
  { d.initial_state() } -> std::convertible_to<typename D::State>;
  { d.is_goal(s) } -> std::convertible_to<bool>;
  { d.num_actions() } -> std::convertible_to<std::size_t>;
  { d.action(i) } -> std::convertible_to<const typename D::Action&>;
  { d.applicable(s, d.action(i)) } -> std::convertible_to<bool>;
  { d.apply(s, d.action(i)) } -> std::convertible_to<typename D::State>;
  { d.cost(d.action(i)) } -> std::convertible_to<double>;
};

// h : State -> double, callable as a function object.
template <class H, class State>
concept Heuristic = requires(const H& h, const State& s) {
  { h(s) } -> std::convertible_to<double>;
};

}  // namespace hd
