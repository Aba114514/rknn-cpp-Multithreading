#!/bin/bash
set -e

# 设置交叉编译工具链，如果是在开发板本机编译，可以注释掉这些
# GCC_COMPILER=aarch64-linux-gnu
# export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
# export CC=${GCC_COMPILER}-gcc
# export CXX=${GCC_COMPILER}-g++

# 获取脚本所在的当前目录，这种写法更健壮
ROOT_PWD=$(cd "$(dirname "$0")" && pwd)

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux
make -j8
make install
cd ${ROOT_PWD} # 返回项目根目录，而不是之前的目录

echo "Build finished. Running demo..."

# 运行demo，使用 install 目录下的可执行文件
# 注意：路径是相对于项目根目录的
INSTALL_DIR=${ROOT_PWD}/install/rknn_yolo_demo_Linux
DEMO_EXECUTABLE=${INSTALL_DIR}/rknn_yolo_demo
MODEL_PATH=${INSTALL_DIR}/model/RK3588/yolo11m.rknn
VIDEO_PATH=${ROOT_PWD}/720p60hz.mp4

# 确保可执行文件存在并有执行权限
if [ -f "${DEMO_EXECUTABLE}" ]; then
    chmod +x ${DEMO_EXECUTABLE}
    cd ${INSTALL_DIR} && ./rknn_yolo_demo ./model/RK3588/yolo11m.rknn ../../720p60hz.mp4
else
    echo "Error: Demo executable not found at ${DEMO_EXECUTABLE}"
    exit 1
fi

# 使用摄像头的示例
# cd ${INSTALL_DIR} && ./rknn_yolo_demo ./model/RK3588/yolo11m.rknn 0
