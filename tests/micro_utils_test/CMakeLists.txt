cmake_minimum_required(VERSION 3.12)

project(micro_utils_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(micro_utils_test "")

target_include_directories(micro_utils_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_utils_test
)

target_compile_options(
  micro_utils_test
  PUBLIC
  -fno-exceptions
)

target_sources(micro_utils_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_utils_test/micro_utils_test.cpp
)

target_link_libraries(
  micro_utils_test
  tensorflow-lite
  tensorflow-lite-test
)
