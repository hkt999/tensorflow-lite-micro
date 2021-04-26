cmake_minimum_required(VERSION 3.12)

project(micro_allocator_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(micro_allocator_test "")

target_include_directories(micro_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_allocator_test
)

target_compile_options(
  micro_allocator_test
  PUBLIC
  -fno-exceptions
)

target_sources(micro_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/micro_allocator_test/micro_allocator_test.cpp
)

target_link_libraries(
  micro_allocator_test
  tensorflow-lite
  tensorflow-lite-test
)

#pico_add_extra_outputs(micro_allocator_test)
