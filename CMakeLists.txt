cmake_minimum_required(VERSION 3.20)
project(heuristic_discovery VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HD_BUILD_TESTS "Build the C++ unit tests" ON)
option(HD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Build provenance is compiled into the binary and echoed into every result file.
find_package(Git QUIET)
set(HD_GIT_COMMIT "unknown")
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE HD_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

add_library(hd_core STATIC cpp/src/platform.cpp)
target_include_directories(hd_core PUBLIC ${CMAKE_SOURCE_DIR}/cpp/include)
target_compile_definitions(hd_core PRIVATE
  HD_VERSION="${PROJECT_VERSION}"
  HD_GIT_COMMIT="${HD_GIT_COMMIT}"
  HD_BUILD_TYPE="$<CONFIG>")

if(MSVC)
  target_compile_options(hd_core PUBLIC /W4)
else()
  target_compile_options(hd_core PUBLIC -Wall -Wextra -Wpedantic)
endif()
if(HD_WARNINGS_AS_ERRORS)
  target_compile_options(hd_core PUBLIC $<IF:$<BOOL:${MSVC}>,/WX,-Werror>)
endif()

add_executable(hd_plan cpp/src/main.cpp)
target_link_libraries(hd_plan PRIVATE hd_core)

if(HD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()
