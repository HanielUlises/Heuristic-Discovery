// Minimal JSON writer.
//
// The planner only ever emits JSON, never parses it, so a dependency-free
// writer of a few dozen lines is preferable to a third-party library.
#pragma once

#include <cmath>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace hd::json {

inline std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

inline std::string quote(const std::string& s) { return "\"" + escape(s) + "\""; }

inline std::string number(double v) {
  if (std::isnan(v) || std::isinf(v)) return "null";
  std::ostringstream ss;
  ss.precision(10);
  ss << v;
  return ss.str();
}

// Builds a flat or nested object incrementally. Values are emitted verbatim,
// so nested objects are composed by passing another Object's str().
class Object {
 public:
  Object& raw(const std::string& key, const std::string& value) {
    if (!fields_.empty()) fields_ += ",";
    fields_ += quote(key) + ":" + value;
    return *this;
  }

  Object& set(const std::string& key, const std::string& value) {
    return raw(key, quote(value));
  }
  Object& set(const std::string& key, const char* value) { return set(key, std::string(value)); }
  Object& set(const std::string& key, double value) { return raw(key, number(value)); }
  Object& set(const std::string& key, std::size_t value) { return raw(key, std::to_string(value)); }
  Object& set(const std::string& key, int value) { return raw(key, std::to_string(value)); }
  Object& set(const std::string& key, bool value) { return raw(key, value ? "true" : "false"); }

  Object& set(const std::string& key, const std::vector<std::string>& values) {
    std::string arr = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i) arr += ",";
      arr += quote(values[i]);
    }
    arr += "]";
    return raw(key, arr);
  }

  std::string str() const { return "{" + fields_ + "}"; }

 private:
  std::string fields_;
};

}  // namespace hd::json
