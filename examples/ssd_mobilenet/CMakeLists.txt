cmake_minimum_required(VERSION 3.12)

project(ssd_mobilenet_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "-fno-rtti -fno-threadsafe-statics")

add_executable(ssd_mobilenet_detection_int8 "")

target_include_directories(ssd_mobilenet_detection_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
  ${CMAKE_CURRENT_LIST_DIR}/../tensorflow-lite-micro/src
)


target_compile_options(
  ssd_mobilenet_detection_int8
  PUBLIC
  -fno-exceptions
  -nostdlib
)

target_sources(ssd_mobilenet_detection_int8
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/nms.h
  ${CMAKE_CURRENT_LIST_DIR}/nms.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.h
)

target_link_directories(ssd_mobilenet_detection_int8
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/../tensorflow-lite-micro/build
)

target_link_libraries(
  ssd_mobilenet_detection_int8
  PUBLIC
  tensorflow-lite
)

