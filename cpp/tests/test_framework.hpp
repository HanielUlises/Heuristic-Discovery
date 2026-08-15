// Dependency-free unit test harness: a static registry, a main() that runs
// every registered case, and two assertion macros.
#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace hdtest {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(std::string name, std::function<void()> body) {
    registry().push_back({std::move(name), std::move(body)});
  }
};

struct Failure : std::exception {
  std::string message;
  explicit Failure(std::string m) : message(std::move(m)) {}
  const char* what() const noexcept override { return message.c_str(); }
};

inline void check(bool condition, const char* expr, const char* file, int line) {
  if (condition) return;
  std::ostringstream ss;
  ss << file << ":" << line << ": assertion failed: " << expr;
  throw Failure(ss.str());
}

template <class A, class B>
void check_eq(const A& a, const B& b, const char* expr, const char* file, int line) {
  if (a == b) return;
  std::ostringstream ss;
  ss << file << ":" << line << ": " << expr << "\n    left  = " << a << "\n    right = " << b;
  throw Failure(ss.str());
}

inline void check_near(double a, double b, double eps, const char* expr, const char* file,
                       int line) {
  if (std::fabs(a - b) <= eps) return;
  std::ostringstream ss;
  ss << file << ":" << line << ": " << expr << "\n    left  = " << a << "\n    right = " << b;
  throw Failure(ss.str());
}

inline int run_all() {
  int failed = 0;
  for (const TestCase& t : registry()) {
    try {
      t.body();
      std::cout << "[ ok ] " << t.name << "\n";
    } catch (const std::exception& e) {
      ++failed;
      std::cout << "[FAIL] " << t.name << "\n" << e.what() << "\n";
    }
  }
  std::cout << registry().size() - failed << "/" << registry().size() << " tests passed\n";
  return failed == 0 ? 0 : 1;
}

}  // namespace hdtest

#define HD_CONCAT_INNER(a, b) a##b
#define HD_CONCAT(a, b) HD_CONCAT_INNER(a, b)

#define TEST(name)                                                            \
  static void HD_CONCAT(hd_test_fn_, __LINE__)();                             \
  static const hdtest::Registrar HD_CONCAT(hd_test_reg_, __LINE__)(           \
      name, &HD_CONCAT(hd_test_fn_, __LINE__));                               \
  static void HD_CONCAT(hd_test_fn_, __LINE__)()

#define CHECK(cond) hdtest::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ(a, b) hdtest::check_eq((a), (b), #a " == " #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, eps) hdtest::check_near((a), (b), (eps), #a " ~= " #b, __FILE__, __LINE__)
