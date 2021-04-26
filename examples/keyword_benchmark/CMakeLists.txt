cmake_minimum_required(VERSION 3.12)

project(keyword_benchmark C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

add_executable(keyword_benchmark "")

target_include_directories(keyword_benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  keyword_benchmark
  PUBLIC
  -fno-exceptions
)

target_sources(keyword_benchmark
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/benchmarks/keyword_benchmark.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h
)

target_link_libraries(
  keyword_benchmark
  tensorflow-lite
  #  hardware_pwm
)

#pico_add_extra_outputs(keyword_benchmark)

