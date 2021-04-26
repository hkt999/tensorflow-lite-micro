cmake_minimum_required(VERSION 3.12)

project(memory_arena_threshold_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(memory_arena_threshold_test "")

target_include_directories(memory_arena_threshold_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/memory_arena_threshold_test
)

target_compile_options(
  memory_arena_threshold_test
  PUBLIC
  -fno-exceptions
)

target_sources(memory_arena_threshold_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/memory_arena_threshold_test/memory_arena_threshold_test.cpp
)

target_link_libraries(
  memory_arena_threshold_test
  tensorflow-lite
  tensorflow-lite-test
)
