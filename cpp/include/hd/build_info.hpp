// Build provenance recorded in every result file, so that a metric can always
// be traced back to the binary that produced it.
#pragma once

#include <string>

namespace hd {

struct BuildInfo {
  std::string compiler;
  std::string cxx_standard;
  std::string build_type;
  std::string git_commit;
  std::string version;
  std::string timestamp;  // compile time
};

BuildInfo build_info();

}  // namespace hd
