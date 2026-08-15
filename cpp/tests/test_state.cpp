// States: the fixed-capacity bitset and its set-theoretic operations.
#include "hd/bitset.hpp"
#include "test_framework.hpp"

using State = hd::Bitset<4>;

TEST("bitset: default state is empty") {
  State s;
  CHECK(s.empty());
  CHECK_EQ(s.count(), 0u);
  for (std::size_t i = 0; i < State::kBits; ++i) CHECK(!s.test(i));
}

TEST("bitset: set, clear and test round-trip across word boundaries") {
  State s;
  const std::size_t bits[] = {0, 63, 64, 127, 128, 255};
  for (const std::size_t b : bits) s.set(b);
  CHECK_EQ(s.count(), 6u);
  for (const std::size_t b : bits) CHECK(s.test(b));
  CHECK(!s.test(62));
  s.clear(64);
  CHECK(!s.test(64));
  CHECK_EQ(s.count(), 5u);
}

TEST("bitset: containment is subset, not equality") {
  State a, b;
  a.set(1);
  a.set(2);
  b.set(1);
  CHECK(a.contains(b));
  CHECK(!b.contains(a));
  CHECK(a.contains(a));
  CHECK(a.contains(State{}));  // every state satisfies the empty condition
}

TEST("bitset: union and difference implement add and delete effects") {
  State s, add, del;
  s.set(0);
  s.set(1);
  del.set(0);
  add.set(5);
  s.and_not_with(del);
  s.or_with(add);
  CHECK(!s.test(0));
  CHECK(s.test(1));
  CHECK(s.test(5));
  CHECK_EQ(s.count(), 2u);
}

TEST("bitset: counting helpers agree with the goal semantics") {
  State s, goal;
  s.set(1);
  s.set(3);
  goal.set(1);
  goal.set(2);
  goal.set(4);
  CHECK_EQ(s.count_common(goal), 1u);
  CHECK_EQ(s.count_missing(goal), 2u);
  CHECK(s.intersects(goal));
  CHECK(!State{}.intersects(goal));
}

TEST("bitset: equality and hashing are consistent") {
  State a, b;
  a.set(7);
  b.set(7);
  CHECK(a == b);
  CHECK_EQ(a.hash(), b.hash());
  b.set(8);
  CHECK(!(a == b));
}
