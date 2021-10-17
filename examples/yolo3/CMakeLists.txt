cmake_minimum_required(VERSION 3.12)

project(yolo3_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "-fexceptions -fno-threadsafe-statics")

find_package(OpenCV REQUIRED)
find_package(JPEG REQUIRED)

add_executable(yolo3_detection "")

target_include_directories(yolo3_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
  ${CMAKE_CURRENT_LIST_DIR}/../tensorflow-lite-micro/src
  ${OpenCV_INCLUDE_DIRS}
)


target_compile_options(
  yolo3_detection
  PUBLIC
  -nostdlib
  -fexceptions
)

target_sources(yolo3_detection
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mstick.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mstick.h
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
  ${CMAKE_CURRENT_LIST_DIR}/model_settings.h
  ${CMAKE_CURRENT_LIST_DIR}/jpeg2rgb.cpp
  ${CMAKE_CURRENT_LIST_DIR}/hexdump.c
)

target_link_directories(yolo3_detection
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/../tensorflow-lite-micro/build
)

target_link_libraries(
  yolo3_detection
  PUBLIC
  tensorflow-lite
  ${JPEG_LIBRARIES}
  ${OpenCV_LIBS}
)

