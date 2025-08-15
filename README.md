
[](https://isocpp.org/)
[](https://www.rock-chips.com/a/en/products/index.html)
[](https://github.com/rockchip-linux/rknpu2)
[](https://gstreamer.freedesktop.org/)
[](https://www.google.com/search?q=%23)

## 简介

本项目是基于 Rockchip RK3588/RK3588s 平台的高性能 C++ YOLO 目标检测实现。项目 Fork 并深度修改自 [rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)，在保留其高效率多线程推理优势的基础上，引入了 **全流程 GStreamer 硬解硬编**、**零拷贝 (Zero-Copy) 数据通路** 等关键优化，极大地提升了处理效率和部署灵活性。
仓库中内置了一个适用于无人机航拍目标检测模型(test.rknn)和训练自COCO80数据集的yolo模型，如有需要，也可以更换自己训练的模型并同步更新target.txt

## 核心特性

与原项目相比，本项目进行了以下关键升级：

1.  **GStreamer 全流程硬件加速**

      * **视频输入**: 支持视频文件或 V4L2 摄像头，通过 GStreamer `mppvideodec` / `mppjpegdec` 插件实现硬件解码，将 CPU 从繁重的解码任务中解放出来。
      * **视频输出**: 支持多种输出后端，并可通过 `mpph264enc` 进行硬件编码，以极低的 CPU 占用率实现高质量的视频流输出。

2.  **零拷贝 (Zero-Copy) 数据通路**

      * 我们构建了一个高效的数据流：`GStreamer硬解 -> RGA硬件预处理 -> RKNN NPU推理`。
      * RGA（2D图形加速单元）直接将预处理（缩放、色彩空间转换）的结果写入预分配的 **DMA 物理连续内存**。
      * RKNN NPU 直接从该 DMA 缓冲区读取数据进行推理，全程**无需 CPU 进行内存拷贝**，显著降低了数据延迟，有效提升了端到端处理帧率。

3.  **高性能多线程推理**

      * 沿用并优化了成熟的线程池模型 (`rknnPool`)，能够创建多个 RKNN 上下文实例，将推理任务并发地分发到 RK3588 的三个 NPU 核心，最大化硬件利用率。
      * 移除了预处理模块中的全局 RGA 锁，允许多个线程并行调用 RGA 硬件，进一步提升了多线程吞吐量。

4.  **灵活的输出模式**

      * **桌面窗口显示**: 程序能自动检测桌面环境 (X11/Wayland)，使用 `xvimagesink` 在窗口中实时显示检测结果。
      * **KMS 全屏显示**: 在无桌面环境的嵌入式系统中，自动切换到 `kmssink` 进行全屏硬件显示（可能需要 `sudo` 权限）。
      * **RTSP 网络推流**: 支持将带有检测框的视频流通过 `rtspclientsink` 推送到指定的 RTSP 服务器，非常适合用于远程监控、视频分析等场景。

## 环境依赖

  * **硬件**: Rockchip RK3588 / RK3588s 开发板。
  * **系统**: 官方或第三方 Linux 系统，并已正确安装 RGA、MPP、RKNN 驱动和库 (`librknnrt.so`, `librga.so`)。
  * **软件**:
      * OpenCV 4.x (编译时需开启 GStreamer 支持)。
      * GStreamer 1.0 及相关插件 (gst-plugins-base, good, bad, ugly, rockchip-mpp)。
      * C++14 编译器 (如 G++ 7.5.0+)。
      * CMake (3.10+)。

## 编译与运行

1.  **克隆项目**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **准备模型和视频**

      * 将您的 `.rknn` 模型文件和类别标签文件 `target.txt` 放置到 `install/Aerial_detection_demo_Linux/model/` 目录下。
      * 将用于测试的视频文件放置到项目根目录。

3.  **一键编译和安装**
    项目提供了便捷的编译脚本。直接运行：

    ```bash
    ./build-linux_RK3588.sh
    ```

    该脚本会自动完成编译和链接，并将所有需要的文件（可执行文件、依赖库、模型）安装到 `install/Aerial_detection_demo_Linux` 目录。

4.  **运行程序**
    编译脚本最后会自动运行一个示例命令。您也可以手动进入 `install` 目录运行。

    **命令格式:**

    ```bash
    cd install/Aerial_detection_demo_Linux/
    ./Aerial_detection_demo <rknn模型路径> <视频源> [可选参数]
    ```

      * **`<视频源>`**: 可以是视频文件路径 (如 `../../my_video.mp4`) 或摄像头设备号 (如 `0`, `1` 等)。

## 使用示例

所有命令均在 `install/Aerial_detection_demo_Linux/` 目录下执行。

#### 1\. 从视频文件检测并在桌面窗口显示

```bash
./Aerial_detection_demo ./model/RK3588/yolo11m.rknn ../../1080p60hz.mov
```

#### 2\. 从摄像头检测并全屏显示 (KMS)

*此模式通常在无桌面的嵌入式环境下自动激活，可能需要 `sudo` 权限。*

```bash
sudo ./Aerial_detection_demo ./model/RK3588/yolo11m.rknn 0
```

#### 3\. 从摄像头检测并推送到 RTSP 服务器

*假设您的 RTSP 服务器地址为 `rtsp://192.168.1.100:8554/live`*

```bash
./Aerial_detection_demo ./model/RK3588/yolo11m.rknn 0 --stream rtsp://192.168.1.100:8554/live
```

您可以使用 VLC 等播放器打开网络串流 `rtsp://192.168.1.100:8554/live` 来查看实时画面。

## 性能提升建议

  * 为了获得稳定且可复现的性能数据，建议在运行前将 RK3588 的 CPU、GPU、NPU 锁定在较高频率。可以参考原项目提供的 `performance.sh` 脚本。
  * 本项目已使用 `uncached` DMA 内存，避免了 CPU 和 NPU 之间因缓存不一致而需要的额外同步操作，这是零拷贝方案实现高性能的关键。

## 致谢

  * [rockchip-linux/rknpu2](https://github.com/rockchip-linux/rknpu2)
  * [senlinzhan/dpool](https://github.com/senlinzhan/dpool)
  * [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
  * [airockchip/rknn\_model\_zoo](https://github.com/airockchip/rknn_model_zoo)

以下为原仓库Readme

# 简介
* 此仓库为c++实现, 大体改自[rknpu2](https://github.com/rockchip-linux/rknpu2), python快速部署见于[rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)
* 使用[线程池](https://github.com/senlinzhan/dpool)异步操作rknn模型, 提高rk3588/rk3588s的NPU使用率, 进而提高推理帧数
* [yolov5s](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)使用relu激活函数进行优化,提高推理帧率

# 更新说明
* 修复了cmake找不到pthread的问题
* 新增nosigmoid分支,使用[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models)下的模型以达到极限性能提升
* 将RK3588 NPU SDK 更新至官方主线1.5.0, [yolov5s-silu](https://github.com/rockchip-linux/rknn-toolkit2/tree/v1.4.0/examples/onnx/yolov5)将沿用1.4.0的旧版本模型, [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)更新至1.5.0版本, 弃用nosigmoid分支。
* 新增v1.5.0分支(向下兼容1.4.0), main分支更新至v1.5.2, 修改了项目结构, 将rknn模型线程池封装成类(include/rknnPool.hpp)

# 使用说明
### 演示
  * 系统需安装有**OpenCV**
  * 下载Releases中的测试视频于项目根目录,运行build-linux_RK3588.sh
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov5_demo **模型所在路径** **视频所在路径/摄像头序号**

### 部署应用
  * 参考include/rkYolov5s.hpp中的rkYolov5s类构建rknn模型类

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 
* [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)
* 测试视频可见于 [bilibili](https://www.bilibili.com/video/BV1zo4y1x7aE/?spm_id_from=333.999.0.0)

|  模型\线程数   | 1    |  2   | 3  |  4  | 5  | 6  | 9  | 12  |
|  ----  | ----  |  ----  | ----  |  ----  | ----  | ----  | ----  | ----  |
| Yolov5s - relu  | 41.6044 | 71.6037 | 98.6057 | 98.0068 | 104.6001 | 114.7454 | 129.5693 | 140.8788 |

# 补充
* 异常处理尚未完善, 目前仅支持rk3588/rk3588s下的运行

# Acknowledgements
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo