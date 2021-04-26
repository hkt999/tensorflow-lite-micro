cmake_minimum_required(VERSION 3.12)

project(kernel_dequantize_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(kernel_dequantize_test "")

target_include_directories(kernel_dequantize_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_dequantize_test
)

target_compile_options(
  kernel_dequantize_test
  PUBLIC
  -fno-exceptions
)

target_sources(kernel_dequantize_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernel_dequantize_test/dequantize_test.cpp
)

target_link_libraries(
  kernel_dequantize_test
  tensorflow-lite
  tensorflow-lite-test
)

#pico_add_extra_outputs(kernel_dequantize_test)
