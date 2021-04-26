cmake_minimum_required(VERSION 3.12)

project(magic_wand C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(gesture_predictor_test "")

target_include_directories(gesture_predictor_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  gesture_predictor_test
  PUBLIC
  -fno-exceptions
)

target_sources(gesture_predictor_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/gesture_predictor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gesture_predictor_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/gesture_predictor.h
)

target_link_libraries(
  gesture_predictor_test
  tensorflow-lite
  tensorflow-lite-test
)


add_executable(gesture_output_handler_test "")

target_include_directories(gesture_output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  gesture_output_handler_test
  PUBLIC
  -fno-exceptions
  -nostdlib
)

target_sources(gesture_output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/output_handler_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  gesture_output_handler_test
  tensorflow-lite
  tensorflow-lite-test
)


add_executable(magic_wand "")

target_include_directories(magic_wand
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  magic_wand
  PUBLIC
  -fno-exceptions
  -nostdlib
)

target_sources(magic_wand
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/accelerometer_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gesture_predictor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/magic_wand_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/accelerometer_handler.h
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/gesture_predictor.h
  ${CMAKE_CURRENT_LIST_DIR}/magic_wand_model_data.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  magic_wand
  tensorflow-lite
)


add_executable(magic_wand_test "")

target_include_directories(magic_wand_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  magic_wand_test
  PUBLIC
  -fno-exceptions
)

target_sources(magic_wand_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/magic_wand_model_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/magic_wand_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ring_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/slope_micro_features_data.cpp
  ${CMAKE_CURRENT_LIST_DIR}/magic_wand_model_data.h
  ${CMAKE_CURRENT_LIST_DIR}/ring_micro_features_data.h
  ${CMAKE_CURRENT_LIST_DIR}/slope_micro_features_data.h
)

target_link_libraries(
  magic_wand_test
  tensorflow-lite
  tensorflow-lite-test
)


add_executable(gesture_accelerometer_handler_test "")

target_include_directories(gesture_accelerometer_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  gesture_accelerometer_handler_test
  PUBLIC
  -fno-exceptions
  -nostdlib
)

target_sources(gesture_accelerometer_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/accelerometer_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/accelerometer_handler_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/accelerometer_handler.h
)

target_link_libraries(
  gesture_accelerometer_handler_test
  tensorflow-lite
  tensorflow-lite-test
)

