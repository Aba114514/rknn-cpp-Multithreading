cmake_minimum_required(VERSION 3.10)

project(rknn_yolo_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)

# rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${RKNN_API_PATH}/include
        ${CMAKE_SOURCE_DIR}/include/3rdparty
        ${RGA_PATH}/include
        ${CMAKE_SOURCE_DIR}/src
)

# opencv
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV_DIR: ${OpenCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件及其所有源文件
add_executable(rknn_yolo_demo
        src/main.cc
        src/postprocess.cc
        src/preprocess.cc
        src/Yolo11.cc
        src/Visualizer.cc
        src/dma_alloc.cpp  # <--- 修正：文件扩展名为 .cpp
)

# 链接所有需要的库
target_link_libraries(rknn_yolo_demo
        ${RKNN_RT_LIB}
        ${OpenCV_LIBS}
        ${RGA_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolo_demo_${CMAKE_SYSTEM_NAME})
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

install(TARGETS rknn_yolo_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)