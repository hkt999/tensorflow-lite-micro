cmake_minimum_required(VERSION 3.12)

project(linear_memory_planner_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(linear_memory_planner_test "")

target_include_directories(linear_memory_planner_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/linear_memory_planner_test
)

target_compile_options(
  linear_memory_planner_test
  PUBLIC
  -fno-exceptions
)

target_sources(linear_memory_planner_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/linear_memory_planner_test/linear_memory_planner_test.cpp
)

target_link_libraries(
  linear_memory_planner_test
  tensorflow-lite
  tensorflow-lite-test
)

#pico_add_extra_outputs(linear_memory_planner_test)
