// Platform-dependent bits kept out of the headers: peak resident memory and
// build provenance.
#include <string>

#include "hd/build_info.hpp"
#include "hd/search/result.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#ifndef HD_VERSION
#define HD_VERSION "0.1.0"
#endif
#ifndef HD_GIT_COMMIT
#define HD_GIT_COMMIT "unknown"
#endif
#ifndef HD_BUILD_TYPE
#define HD_BUILD_TYPE "unknown"
#endif

namespace hd {

std::size_t peak_memory_kb() {
#if defined(__unix__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss) / 1024;  // bytes on macOS
#else
  return static_cast<std::size_t>(usage.ru_maxrss);         // kilobytes on Linux
#endif
#else
  return 0;
#endif
}

BuildInfo build_info() {
  BuildInfo info;
#if defined(__clang__)
  info.compiler = std::string("clang++ ") + __clang_version__;
#elif defined(__GNUC__)
  info.compiler = "g++ " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
                  std::to_string(__GNUC_PATCHLEVEL__);
#else
  info.compiler = "unknown";
#endif
  info.cxx_standard = std::to_string(__cplusplus);
  info.build_type = HD_BUILD_TYPE;
  info.git_commit = HD_GIT_COMMIT;
  info.version = HD_VERSION;
  info.timestamp = std::string(__DATE__) + " " + __TIME__;
  return info;
}

}  // namespace hd
