// Propositional STRIPS task: boolean variables, actions with preconditions and
// add/delete effects, and a partial goal condition.
#pragma once

#include <cstddef>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hd/bitset.hpp"

namespace hd {

// 256 propositions. Sufficient for the Phase I benchmark family; raising it is
// a one-line change with no effect on any other component.
inline constexpr std::size_t kStateWords = 4;
using StripsState = Bitset<kStateWords>;

struct StripsAction {
  std::string name;
  StripsState pre;
  StripsState add;
  StripsState del;
  double cost = 1.0;
};

class StripsTask {
 public:
  using State = StripsState;
  using Action = StripsAction;

  StripsTask() = default;

  // --- Domain concept ---------------------------------------------------
  const State& initial_state() const { return init_; }
  bool is_goal(const State& s) const { return s.contains(goal_); }
  std::size_t num_actions() const { return actions_.size(); }
  const Action& action(std::size_t i) const { return actions_[i]; }

  bool applicable(const State& s, const Action& a) const { return s.contains(a.pre); }

  State apply(const State& s, const Action& a) const {
    State t = s;
    t.and_not_with(a.del);
    t.or_with(a.add);
    return t;
  }

  double cost(const Action& a) const { return a.cost; }

  // --- Task description -------------------------------------------------
  std::size_t num_propositions() const { return num_props_; }
  const State& goal() const { return goal_; }
  std::size_t num_goal_conditions() const { return goal_.count(); }
  const std::vector<Action>& actions() const { return actions_; }
  const std::string& name() const { return name_; }
  const std::string& proposition_name(std::size_t i) const { return prop_names_[i]; }

  std::size_t num_applicable(const State& s) const {
    std::size_t n = 0;
    for (const Action& a : actions_) n += applicable(s, a) ? 1 : 0;
    return n;
  }

  // --- Construction -----------------------------------------------------
  void set_name(std::string n) { name_ = std::move(n); }

  void set_num_propositions(std::size_t n) {
    if (n > StripsState::kBits) {
      throw std::runtime_error("task exceeds the compiled proposition capacity");
    }
    num_props_ = n;
    prop_names_.assign(n, std::string());
    for (std::size_t i = 0; i < n; ++i) prop_names_[i] = "p" + std::to_string(i);
  }

  void set_proposition_name(std::size_t i, std::string n) { prop_names_[i] = std::move(n); }
  void set_initial_state(State s) { init_ = s; }
  void set_goal(State g) { goal_ = g; }
  void add_action(Action a) { actions_.push_back(std::move(a)); }

  // Parses the line-oriented task format described in docs/format.md.
  static StripsTask parse(std::istream& in);

 private:
  std::string name_ = "unnamed";
  std::size_t num_props_ = 0;
  std::vector<std::string> prop_names_;
  State init_;
  State goal_;
  std::vector<Action> actions_;
};

inline StripsTask StripsTask::parse(std::istream& in) {
  StripsTask task;
  StripsAction current;
  bool in_action = false;
  std::string line;
  std::size_t lineno = 0;

  auto fail = [&](const std::string& msg) {
    throw std::runtime_error("task parse error on line " + std::to_string(lineno) + ": " + msg);
  };
  auto read_bits = [&](std::istringstream& ss, StripsState& out) {
    std::size_t idx = 0;
    while (ss >> idx) {
      if (idx >= task.num_props_) fail("proposition index out of range");
      out.set(idx);
    }
  };

  while (std::getline(in, line)) {
    ++lineno;
    if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
    std::istringstream ss(line);
    std::string kw;
    if (!(ss >> kw)) continue;

    if (kw == "name") {
      std::string n;
      ss >> n;
      task.set_name(n);
    } else if (kw == "propositions") {
      std::size_t n = 0;
      if (!(ss >> n)) fail("expected a proposition count");
      task.set_num_propositions(n);
    } else if (kw == "prop") {
      std::size_t i = 0;
      std::string n;
      if (!(ss >> i >> n)) fail("expected an index and a name");
      if (i >= task.num_props_) fail("proposition index out of range");
      task.set_proposition_name(i, n);
    } else if (kw == "init") {
      StripsState s = task.init_;
      read_bits(ss, s);
      task.set_initial_state(s);
    } else if (kw == "goal") {
      StripsState g = task.goal_;
      read_bits(ss, g);
      task.set_goal(g);
    } else if (kw == "action") {
      if (in_action) task.add_action(current);
      current = StripsAction{};
      if (!(ss >> current.name)) fail("expected an action name");
      double c = 1.0;
      if (ss >> c) current.cost = c;
      in_action = true;
    } else if (kw == "pre" || kw == "add" || kw == "del") {
      if (!in_action) fail("effect declared outside of an action");
      StripsState& target = (kw == "pre") ? current.pre : (kw == "add" ? current.add : current.del);
      read_bits(ss, target);
    } else if (kw == "end") {
      if (in_action) task.add_action(current);
      in_action = false;
    } else {
      fail("unknown keyword '" + kw + "'");
    }
  }
  if (in_action) task.add_action(current);
  if (task.num_props_ == 0) throw std::runtime_error("task declares no propositions");
  return task;
}

static_assert(sizeof(StripsState) == kStateWords * 8, "state must stay allocation-free");

}  // namespace hd
