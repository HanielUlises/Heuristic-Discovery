add_executable(hd_tests
  main.cpp
  test_state.cpp
  test_strips.cpp
  test_features.cpp
  test_heuristics.cpp
  test_search.cpp
  test_oracle.cpp
  test_landmarks.cpp
  test_json.cpp)

target_link_libraries(hd_tests PRIVATE hd_core)
target_include_directories(hd_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME hd_tests COMMAND hd_tests)
