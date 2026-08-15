// Small hand-built tasks used across the unit tests.
#pragma once

#include <sstream>
#include <string>

#include "hd/strips.hpp"

namespace hdtest {

// A three-room corridor: at(a) -> at(b) -> at(c), goal at(c).
// The optimal plan has length two and the state space has three states.
inline const char* kCorridorTask = R"(
name corridor
propositions 3
prop 0 at_a
prop 1 at_b
prop 2 at_c
init 0
goal 2
action move_a_b 1
pre 0
add 1
del 0
end
action move_b_c 1
pre 1
add 2
del 1
end
)";

// Two independent switches, both required by the goal. Any interleaving works,
// so the task exercises duplicate detection and goal counting.
inline const char* kSwitchesTask = R"(
name switches
propositions 4
prop 0 off_x
prop 1 on_x
prop 2 off_y
prop 3 on_y
init 0 2
goal 1 3
action flip_x 1
pre 0
add 1
del 0
end
action flip_y 2
pre 2
add 3
del 2
end
)";

// Goal proposition that no action ever adds: unsolvable, and a relaxed dead end.
inline const char* kUnsolvableTask = R"(
name unsolvable
propositions 3
init 0
goal 2
action noop 1
pre 0
add 1
end
)";

inline hd::StripsTask parse(const char* text) {
  std::istringstream in(text);
  return hd::StripsTask::parse(in);
}

}  // namespace hdtest
