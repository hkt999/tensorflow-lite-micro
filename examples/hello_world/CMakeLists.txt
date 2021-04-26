cmake_minimum_required(VERSION 3.12)

project(hello_world C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(hello_world_test "")

target_include_directories(hello_world_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  hello_world_test
  PUBLIC
  -fno-exceptions
)

target_sources(hello_world_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/hello_world_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.h
)

target_link_libraries(
  hello_world_test
  tensorflow-lite
  tensorflow-lite-test
)

add_executable(hello_world "")

target_include_directories(hello_world
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  hello_world
  PUBLIC
  -fno-exceptions
)

target_sources(hello_world
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/constants.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/linux/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/model.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  hello_world
  tensorflow-lite
)

add_executable(output_handler_test "")

target_include_directories(output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

target_compile_options(
  output_handler_test
  PUBLIC
  -fno-exceptions
)

target_sources(output_handler_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/output_handler_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/linux/output_handler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/constants.h
  ${CMAKE_CURRENT_LIST_DIR}/output_handler.h
)

target_link_libraries(
  output_handler_test
  tensorflow-lite
  tensorflow-lite-test
)

