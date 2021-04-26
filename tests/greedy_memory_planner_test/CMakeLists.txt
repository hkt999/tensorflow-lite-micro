cmake_minimum_required(VERSION 3.12)

project(greedy_memory_planner_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(greedy_memory_planner_test "")

target_include_directories(greedy_memory_planner_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/greedy_memory_planner_test
)

target_compile_options(
  greedy_memory_planner_test
  PUBLIC
  -fno-exceptions
)

target_sources(greedy_memory_planner_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/greedy_memory_planner_test/greedy_memory_planner_test.cpp
)

target_link_libraries(
  greedy_memory_planner_test
  tensorflow-lite
  tensorflow-lite-test
)
