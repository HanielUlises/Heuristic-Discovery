// Fixed-capacity bitset used as the propositional state representation.
//
// The capacity is a compile-time constant so that a state is trivially
// copyable, allocation-free, and hashable in a handful of machine words.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace hd {

template <std::size_t Words>
class Bitset {
 public:
  static constexpr std::size_t kWords = Words;
  static constexpr std::size_t kBits = Words * 64;

  constexpr Bitset() : words_{} {}

  constexpr bool test(std::size_t i) const {
    return (words_[i >> 6] >> (i & 63)) & 1ULL;
  }

  constexpr void set(std::size_t i) { words_[i >> 6] |= (1ULL << (i & 63)); }
  constexpr void clear(std::size_t i) { words_[i >> 6] &= ~(1ULL << (i & 63)); }

  constexpr void set_to(std::size_t i, bool v) { v ? set(i) : clear(i); }

  // s |= o
  constexpr void or_with(const Bitset& o) {
    for (std::size_t w = 0; w < Words; ++w) words_[w] |= o.words_[w];
  }

  // s &= ~o
  constexpr void and_not_with(const Bitset& o) {
    for (std::size_t w = 0; w < Words; ++w) words_[w] &= ~o.words_[w];
  }

  // Whether every bit of `o` is also set here (o is a subset of *this).
  constexpr bool contains(const Bitset& o) const {
    for (std::size_t w = 0; w < Words; ++w) {
      if ((words_[w] & o.words_[w]) != o.words_[w]) return false;
    }
    return true;
  }

  constexpr bool intersects(const Bitset& o) const {
    for (std::size_t w = 0; w < Words; ++w) {
      if (words_[w] & o.words_[w]) return true;
    }
    return false;
  }

  constexpr std::size_t count() const {
    std::size_t n = 0;
    for (std::size_t w = 0; w < Words; ++w) n += popcount(words_[w]);
    return n;
  }

  // |o \ *this|: elements of o that are not satisfied here.
  constexpr std::size_t count_missing(const Bitset& o) const {
    std::size_t n = 0;
    for (std::size_t w = 0; w < Words; ++w) n += popcount(o.words_[w] & ~words_[w]);
    return n;
  }

  constexpr std::size_t count_common(const Bitset& o) const {
    std::size_t n = 0;
    for (std::size_t w = 0; w < Words; ++w) n += popcount(o.words_[w] & words_[w]);
    return n;
  }

  constexpr bool empty() const {
    for (std::size_t w = 0; w < Words; ++w) {
      if (words_[w]) return false;
    }
    return true;
  }

  friend constexpr bool operator==(const Bitset& a, const Bitset& b) {
    return a.words_ == b.words_;
  }

  std::size_t hash() const {
    // FNV-1a over the words; cheap and adequate for a closed list.
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t w = 0; w < Words; ++w) {
      h ^= words_[w];
      h *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(h);
  }

  constexpr std::uint64_t word(std::size_t w) const { return words_[w]; }

 private:
  static constexpr std::size_t popcount(std::uint64_t x) {
    return static_cast<std::size_t>(__builtin_popcountll(x));
  }

  std::array<std::uint64_t, Words> words_;
};

}  // namespace hd

template <std::size_t Words>
struct std::hash<hd::Bitset<Words>> {
  std::size_t operator()(const hd::Bitset<Words>& b) const { return b.hash(); }
};
