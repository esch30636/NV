# NV: A Comprehensive Data Augmentation and YOLO-Based Object Detection Pipeline for UAV Imagery

## NV: 面向无人机影像的综合性数据增强与YOLO目标检测管线

---

**Authors / 作者:** Escherichia  
**Affiliation / 单位:** Huazhong University of Science and Technology, School of Artificial Intelligence and Automation / 华中科技大学人工智能与自动化学院  
**License / 许可:** MIT License  
**Version / 版本:** 2.0 (Workspace Edition)  
**Date / 日期:** June 2025  

---

## Abstract / 摘要

This repository presents a complete pipeline for UAV-based object detection, comprising three principal components: (i) a geometry-aware data augmentation engine (`Ultimate7.py`) that synthesizes training images via affine transformations and alpha compositing of foreground objects onto arbitrary backgrounds; (ii) an enhanced augmentation variant (`Ultimate8.py`) incorporating stochastic color-block injection and intensity noise for improved domain randomization; and (iii) a production-grade training harness (`Ultimate_Ready.py`) built upon Ultralytics YOLOv12n with exhaustive hyperparameter exposition. The pipeline achieves a mean Average Precision at 50% IoU (mAP@50) of 0.995 and mAP@50-95 of 0.995 on a 17-class detection task, with inference latency of 0.5 ms per frame on an NVIDIA GeForce RTX 4090 D. This document provides a rigorous mathematical treatment of the underlying algorithms, including affine bounding-box transformation, alpha-channel compositing, and the YOLO composite loss function, presented in an IEEE-style academic format with full bilingual annotation.

The repository now also includes an overfitting-controlled alternative pipeline: `Ultimate9.py` for source-isolated dataset generation and `Ultimate10.py` for regularized YOLO training. This pair keeps the same input/output path convention as `Ultimate8.py` and `Ultimate_Ready.py`, while reducing train/validation leakage, moderating augmentation strength, and using a more conservative early-stopping strategy.

本仓库提出了一套完整的无人机目标检测管线，包含三个核心组件：(i) 基于几何感知的数据增强引擎（`Ultimate7.py`），通过仿射变换与Alpha合成将前景目标嵌入任意背景以生成训练图像；(ii) 增强型数据增强变体（`Ultimate8.py`），引入随机色块注入与强度噪声以实现更优的域随机化；(iii) 基于Ultralytics YOLOv12n的生产级训练框架（`Ultimate_Ready.py`），提供详尽的超参数配置。该管线在17类目标检测任务上达到0.995的mAP@50与0.995的mAP@50-95，单帧推理延迟为0.5毫秒（NVIDIA GeForce RTX 4090 D）。本文对底层算法提供严格的数学处理，包括仿射边界框变换、Alpha通道合成与YOLO复合损失函数，以IEEE风格学术格式呈现，并提供完整的中英双语注释。

---

## Table of Contents / 目录

1. [System Architecture / 系统架构](#1-system-architecture--系统架构)
2. [Hardware Configuration / 硬件配置](#2-hardware-configuration--硬件配置)
3. [Environment / 运行环境](#3-environment--运行环境)
4. [Project Structure / 项目结构](#4-project-structure--项目结构)
5. [Deep Analysis: Ultimate7.py](#5-deep-analysis-ultimate7py)
6. [Deep Analysis: Ultimate8.py](#6-deep-analysis-ultimate8py)
7. [Deep Analysis: Ultimate_Ready.py](#7-deep-analysis-ultimate_ready-py)
   - [Overfitting-Controlled Pipeline: Ultimate9.py and Ultimate10.py](#74-overfitting-controlled-pipeline-ultimate9py-and-ultimate10py)
8. [Auxiliary Scripts / 辅助脚本](#8-auxiliary-scripts--辅助脚本)
9. [Validation Framework / 验证框架](#9-validation-framework--验证框架)
10. [Experimental Results / 实验结果](#10-experimental-results--实验结果)
11. [Usage Guide / 使用指南](#11-usage-guide--使用指南)
12. [License / 许可](#12-license--许可)

---

## 1. System Architecture / 系统架构

The pipeline follows a two-phase paradigm: **synthetic data generation** followed by **supervised model training**. The data generation phase employs affine-geometric transformations with alpha-channel compositing to place segmented foreground objects onto diverse background scenes, thereby constructing a large-scale, richly annotated dataset from a minimal set of source images. The training phase leverages the YOLOv12n architecture with a comprehensive augmentation schedule and adaptive optimization.

管线采用两阶段范式：**合成数据生成**，随后进行**监督模型训练**。数据生成阶段利用仿射几何变换与Alpha通道合成，将分割后的前景目标放置于多样的背景场景中，从而从最小化的源图像集合构建大规模、丰富标注的数据集。训练阶段基于YOLOv12n架构，搭配全面的增强调度策略与自适应优化。

```
+-------------------+       +--------------------+       +-------------------+
|                   |       |                    |       |                   |
|  input_data/      | ----> |  Ultimate8.py      | ----> |  dataset/         |
|  - background/    |       |  (Data Augmentation)|       |  - images/train/  |
|  - label/         |       |                    |       |  - images/val/    |
|  - class_i/       |       +--------------------+       |  - labels/train/  |
|                   |                                     |  - labels/val/    |
+-------------------+                                     |  - dataset.yaml   |
                                                          +--------+----------+
                                                                   |
                                                                   v
+-------------------+       +--------------------+       +-------------------+
|                   |       |                    |       |                   |
|  yolo12n.pt       | ----> |  Ultimate_Ready.py | ----> |  runs/detect/     |
|  (pretrained)     |       |  (Training Harness)|       |  - weights/       |
|                   |       |                    |       |  - results/       |
+-------------------+       +--------------------+       +-------------------+
```

**Figure 1 / 图 1:** System data flow and component interaction diagram. / 系统数据流与组件交互图。

---

## 2. Hardware Configuration / 硬件配置

### 2.1 Local Workstation (Mobile) / 本地工作站（移动端）

| Component / 组件 | Specification / 规格 |
|:---|:---|
| CPU | Intel Core i7-14700HX (28 cores, 20 threads) |
| GPU | NVIDIA GeForce RTX 4060 Laptop 8 GB (55 W TGP) |
| RAM | DDR5-5200 64 GB |
| Operating System / 操作系统 | Ubuntu 20.04.6 LTS |

### 2.2 Server / 服务器

| Component / 组件 | Specification / 规格 |
|:---|:---|
| CPU | AMD Ryzen 9 9800X / Intel Core i7-13700K |
| GPU | NVIDIA GeForce RTX 4090 D 24 GB / RTX 4090 24 GB |
| RAM | DDR5-5200 64 GB / 128 GB |
| Operating System / 操作系统 | Ubuntu 20.04.6 LTS |

---

## 3. Environment / 运行环境

The environment is managed via Conda. For the overfitting-controlled pipeline, use `environment.yml` as the preferred environment file; `env_NV.txt` is retained as the original environment specification.

环境通过Conda管理；新的抗过拟合流程优先使用`environment.yml`，`env_NV.txt`保留为原始环境规格。

### Core Dependencies / 核心依赖

| Package / 包 | Version / 版本 | Purpose / 用途 |
|:---|:---|:---|
| Python | 3.8.20 | Runtime / 运行时 |
| PyTorch | 2.1.0+cu121 | Deep learning framework / 深度学习框架 |
| Ultralytics | 8.3.170 | YOLO training and inference / YOLO训练与推理 |
| OpenCV | 4.11.0.86 | Image processing and augmentation / 图像处理与增强 |
| NumPy | 1.24.4 | Numerical computation / 数值计算 |
| Matplotlib | 3.7.5 | Visualization / 可视化 |
| Pandas | 2.0.3 | Data manipulation / 数据处理 |
| SciPy | 1.10.1 | Scientific computing / 科学计算 |

### Installation / 安装

```bash
conda env create -f environment.yml
conda activate traxler
```

---

## 4. Project Structure / 项目结构

```
NV/
|
|-- Ultimate_Ready.py          # Primary training harness (YOLOv12n) / 主训练入口
|-- Ultimate7.py               # Data augmentation engine / 数据增强引擎
|-- Ultimate8.py               # Enhanced augmentation (parallel) / 增强型数据增强
|-- Ultimate.py                # Hough circle detection + YOLOv11s / 霍夫圆检测+YOLOv11s
|-- Ultimate2.py               # Development draft / 开发草稿
|-- Ultimate3.py               # DDPM diffusion model experiment / 扩散模型实验
|-- Ultimate4.py               # (Empty placeholder / 空占位文件)
|-- Ultimate5.py               # YOLOv11s training variant / YOLOv11s训练变体
|
|-- test.py                    # CIFAR-10 baseline CNN / CIFAR-10基线CNN
|-- test2.py -- test12.py      # Iterative training pipeline development / 训练管线迭代开发
|-- attempt.py                 # Epoch-linked continuous training / 跨epoch连续训练
|
|-- hubei.py                   # ResNet classifier with background replacement / ResNet分类器
|-- preloading_pics.py         # Environmental illumination transfer (DeepSeek R1) / 环境光照迁移
|-- preloading_pics2.py        # VGG19 illumination transfer (WenXin X1 Turbo) / VGG19光照迁移
|-- preloading_bkgrd.py        # Background batch resizing / 背景批量缩放
|-- output.py                  # Basic image augmentation / 基础图像增强
|
|-- verify.py -- verify7.py    # Validation and evaluation scripts / 验证与评估脚本
|-- verify_test.py             # Validation test utilities / 验证测试工具
|-- verify_test3.py            # Validation test utilities (variant) / 验证测试工具变体
|
|-- yolov3test.py              # YOLOv5 batch inference test / YOLOv5批量推理测试
|-- haarcascadetest.py         # Haar Cascade face detection / Haar级联人脸检测
|-- hogsvm_test.py             # HOG+SVM detection test / HOG+SVM检测测试
|-- pytorchtest.py             # YOLOv8 real-time camera inference / YOLOv8摄像头实时推理
|
|-- dataset.yaml               # YOLO dataset configuration (2-class) / YOLO数据集配置（2类）
|-- demo                       # Centralized path configuration template / 集中式路径配置模板
|-- env_NV.txt                 # Conda environment specification / Conda环境规格
|-- LICENSE                    # MIT License / MIT许可
|-- yolo11n.pt                 # YOLOv11n pretrained weights (5.6 MB) / YOLOv11n预训练权重
|-- yolo11s.pt                 # YOLOv11s pretrained weights (19.3 MB) / YOLOv11s预训练权重
|
|-- workspace/                 # Optimized production pipeline / 优化后的生产管线
|   |-- Ultimate_Ready.py      # Training harness / 训练入口
|   |-- Ultimate7.py           # Data augmentation engine / 数据增强引擎
|   |-- Ultimate8.py           # Enhanced augmentation engine / 增强型增强引擎
|   |-- yolo11n.pt             # YOLOv11n pretrained weights / YOLOv11n预训练权重
|   |-- yolo12n.pt             # YOLOv12n pretrained weights (5.6 MB) / YOLOv12n预训练权重
|   |-- input_data/            # Raw data directory / 原始数据目录
|   |   |-- background/        # Background scene images / 背景场景图像
|   |   |-- label/             # YOLO-format annotations / YOLO格式标注
|   |   |   |-- classes.txt    # Class name registry / 类别名称注册表
|   |   |-- <class_folders>/   # Per-class foreground images / 各类别前景图像
|   |-- dataset/               # Generated dataset (auto-created) / 生成数据集（自动创建）
|   |   |-- images/train/      # Training images (80%) / 训练图像
|   |   |-- images/val/        # Validation images (20%) / 验证图像
|   |   |-- labels/train/      # Training annotations / 训练标注
|   |   |-- labels/val/        # Validation annotations / 验证标注
|   |   |-- labels/classes.txt # Class name registry / 类别名称注册表
|   |   |-- dataset.yaml       # YOLO dataset configuration / YOLO数据集配置
|   |-- runs/                  # Training outputs / 训练输出
|       |-- detect/
|           |-- escherichia_train/
|               |-- weights/   # Model checkpoints / 模型检查点
|               |-- results/   # Metrics and plots / 指标与图表
```

Additional overfitting-control files:

```
Ultimate9.py   # Source-isolated dataset generation with moderated offline augmentation
Ultimate10.py  # Regularized YOLO training configuration for the Ultimate9 dataset
```

---

## 5. Deep Analysis: Ultimate7.py

### 5.1 Overview / 概述

`Ultimate7.py` implements a geometry-grounded data augmentation framework that synthesizes supervised training samples by compositing segmented foreground objects onto arbitrary background scenes. The script addresses the fundamental challenge of limited annotated UAV imagery by generating a combinatorially rich dataset from a minimal seed collection.

`Ultimate7.py`实现了一个基于几何原理的数据增强框架，通过将分割后的前景目标合成到任意背景场景中来生成监督训练样本。该脚本通过从最小化的种子集合生成组合丰富的数据集，解决了无人机标注图像有限这一根本性挑战。

### 5.2 Algorithmic Pipeline / 算法管线

The augmentation pipeline proceeds through five sequential stages:

增强管线按以下五个阶段顺序执行：

**Stage 1: Data Loading / 阶段一：数据加载**

The function `load_images_and_labels(root_dir)` traverses the input directory hierarchy and populates four data structures: a list of background images (`backgrounds`), a list of foreground image paths (`image_paths`), a list of corresponding YOLO-format annotation paths (`label_paths`), and an ordered list of class names (`class_names`) parsed from `label/classes.txt`.

函数 `load_images_and_labels(root_dir)` 遍历输入目录层次结构，填充四个数据结构：背景图像列表（`backgrounds`）、前景图像路径列表（`image_paths`）、对应的YOLO格式标注路径列表（`label_paths`），以及从 `label/classes.txt` 解析的有序类别名称列表（`class_names`）。

**Stage 2: Affine Transformation / 阶段二：仿射变换**

Each foreground object undergoes a combined rotation and scaling transformation. The affine transformation matrix **M** in homogeneous coordinates is defined as:

每个前景目标经过组合的旋转与缩放变换。齐次坐标下的仿射变换矩阵 **M** 定义为：

```
        [ alpha * cos(theta)    alpha * sin(theta)    t_x ]
M(theta, alpha) = [ -alpha * sin(theta)   alpha * cos(theta)    t_y ]
        [          0                     0                1   ]
```

where:
- `theta` is the rotation angle sampled from `[0, 2*pi)` in discrete 1-degree increments (360 angles per image);
- `alpha` is the uniform random scale factor drawn from `U(0.3, 0.8)`;
- `(t_x, t_y)` is a corrective translation ensuring the rotated image remains within canvas bounds.

其中：
- `theta` 是从 `[0, 2*pi)` 以离散1度增量采样的旋转角（每张图像360个角度）；
- `alpha` 是从 `U(0.3, 0.8)` 抽取的均匀随机缩放因子；
- `(t_x, t_y)` 是确保旋转后图像保持在画布范围内的校正平移量。

The output canvas dimensions `(new_w, new_h)` are computed to tightly bound the rotated image:

输出画布尺寸 `(new_w, new_h)` 经计算以紧密包围旋转后的图像：

```
new_w = floor( h * |sin(theta)| + w * |cos(theta)| )
new_h = floor( h * |cos(theta)| + w * |sin(theta)| )
```

The translation correction is then:

随后进行平移校正：

```
t_x = new_w / 2 - w / 2
t_y = new_h / 2 - h / 2
```

The actual affine warp is executed via `cv2.warpAffine` with bilinear interpolation (`cv2.INTER_LINEAR`) and transparent border padding (`borderValue = (0, 0, 0, 0)` for the BGRA channels). An alpha channel is prepended to RGB images to enable subsequent transparency-aware compositing.

实际的仿射变形通过 `cv2.warpAffine` 执行，采用双线性插值（`cv2.INTER_LINEAR`）与透明边界填充（BGRA通道的 `borderValue = (0, 0, 0, 0)`）。Alpha通道被添加至RGB图像，以实现后续的透明度感知合成。

**Stage 3: Bounding Box Transformation / 阶段三：边界框变换**

For each ground-truth bounding box in YOLO format `(x_center, y_center, width, height)` (all normalized to `[0, 1]`), the transformation proceeds as follows:

对于YOLO格式 `(x_center, y_center, width, height)`（均归一化至 `[0, 1]`）的每个真实边界框，变换过程如下：

1. Convert normalized coordinates to absolute pixel coordinates on the original image:

   将归一化坐标转换为原始图像上的绝对像素坐标：

   ```
   x_abs = x_center * w_orig    y_abs = y_center * h_orig
   w_abs = width * w_orig       h_abs = height * h_orig
   ```

2. Decompose the bounding box into its four corner points in homogeneous coordinates:

   将边界框分解为齐次坐标下的四个角点：

   ```
   P = { (x_abs - w_abs/2,  y_abs - h_abs/2,  1),
         (x_abs + w_abs/2,  y_abs - h_abs/2,  1),
         (x_abs + w_abs/2,  y_abs + h_abs/2,  1),
         (x_abs - w_abs/2,  y_abs + h_abs/2,  1) }
   ```

3. Apply the affine transformation matrix to each corner point:

   对每个角点应用仿射变换矩阵：

   ```
   P'_i = M * P_i^T    for i in {1, 2, 3, 4}
   ```

   Equivalently in vectorized NumPy form: `P' = P @ M^T`.

   等价于NumPy向量化形式：`P' = P @ M^T`。

4. Compute the axis-aligned bounding rectangle of the transformed quadrilateral:

   计算变换后四边形的轴对齐包围矩形：

   ```
   x'_min = min({p'_x for p' in P'})     y'_min = min({p'_y for p' in P'})
   x'_max = max({p'_x for p' in P'})     y'_max = max({p'_y for p' in P'})
   ```

5. Convert back to YOLO-normalized coordinates on the new canvas:

   转换回新画布上的YOLO归一化坐标：

   ```
   x'_center = ((x'_min + x'_max) / 2) / new_w
   y'_center = ((y'_min + y'_max) / 2) / new_h
   w'_new    = (x'_max - x'_min) / new_w
   h'_new    = (y'_max - y'_min) / new_h
   ```

6. Clamp all values to the valid range with a minimum dimension of 2% to prevent degenerate boxes:

   将所有值钳制到有效范围，最小尺寸为2%，以防止退化边界框：

   ```
   w'_new = max(0.02, min(1.0, w'_new))
   h'_new = max(0.02, min(1.0, h'_new))
   ```

**Stage 4: Alpha Compositing / 阶段四：Alpha合成**

The function `blend_with_alpha(background, foreground, x, y)` implements the Porter-Duff "over" operator for alpha compositing. Given a foreground image with alpha channel `A_fg` at placement position `(x, y)` on a background image, the blended pixel value at each channel is:

函数 `blend_with_alpha(background, foreground, x, y)` 实现了Alpha合成的Porter-Duff "over"操作。给定在背景图像上放置位置 `(x, y)` 处具有Alpha通道 `A_fg` 的前景图像，每个通道的混合像素值为：

```
C_result = C_fg * alpha_fg + C_bg * (1 - alpha_fg)
```

where `alpha_fg = A_fg / 255.0` is the normalized alpha value at each pixel. This operation is performed independently for each of the three color channels (B, G, R). Pixels where the foreground is fully transparent (`alpha_fg = 0`) retain the original background value; pixels where the foreground is fully opaque (`alpha_fg = 1`) completely replace the background.

其中 `alpha_fg = A_fg / 255.0` 是每个像素处的归一化Alpha值。该操作对三个颜色通道（B， G， R）独立执行。前景完全透明的像素（`alpha_fg = 0`）保留原始背景值；前景完全不透明的像素（`alpha_fg = 1`）完全替换背景。

**Stage 5: Placement Strategy and Dataset Assembly / 阶段五：放置策略与数据集组装**

For each synthesized scene, the script:

对每个合成场景，脚本执行：

1. Selects a background uniformly at random from the background pool.
2. Places `n ~ U(1, min(5, N_classes))` distinct foreground objects.
3. For each object, attempts up to 50 random placements, rejecting any placement whose bounding rectangle intersects with a previously placed object (non-overlap constraint enforced via axis-aligned bounding box intersection testing).
4. After a successful non-overlapping placement, applies alpha compositing and records the transformed bounding box coordinates in the global background coordinate frame.
5. Generates 360 angular variants per scene (one per integer degree), producing `100 * 360 = 36,000` synthetic images with an 80/20 train/validation split.

1. 从背景池中均匀随机选择背景。
2. 放置 `n ~ U(1, min(5, N_classes))` 个不同的前景目标。
3. 对每个目标，尝试最多50次随机放置，拒绝任何其边界矩形与先前放置目标相交的放置（通过轴对齐边界框相交测试强制执行无重叠约束）。
4. 成功无重叠放置后，应用Alpha合成，并在全局背景坐标框架中记录变换后的边界框坐标。
5. 每个场景生成360个角度变体（每个整数角度一个），产生 `100 * 360 = 36,000` 张合成图像，按80/20训练/验证分割。

### 5.3 Parallelization Strategy / 并行化策略

The function `place_images_on_background` distributes scene generation across a multiprocessing pool of size `min(cpu_count(), 24)`, using `pool.imap_unordered` for asynchronous, out-of-order result collection with a `tqdm` progress bar. Scene generation is an embarrassingly parallel workload: each scene's random seed, background selection, object selection, placement coordinates, and affine parameters are independent of all other scenes.

函数 `place_images_on_background` 将场景生成分布到大小为 `min(cpu_count(), 24)` 的多进程池中，使用 `pool.imap_unordered` 进行异步、无序的结果收集，并配合 `tqdm` 进度条。场景生成是一个令人尴尬的并行工作负载：每个场景的随机种子、背景选择、目标选择、放置坐标和仿射参数都独立于所有其他场景。

### 5.4 Output Structure / 输出结构

The generated dataset is organized in the YOLO directory convention:

生成的数据集按照YOLO目录约定组织：

- Output root: `yolo_dataset_809/`
- Image and label directories with `train/` and `val/` subdirectories
- A `dataset.yaml` configuration file specifying train/val paths, number of classes `nc`, and class `names`
- A `labels/classes.txt` file enumerating class names

---

## 6. Deep Analysis: Ultimate8.py

### 6.1 Overview / 概述

`Ultimate8.py` is an enhanced evolution of `Ultimate7.py` that introduces two additional domain-randomization mechanisms --- stochastic color-block injection and additive intensity noise --- to improve model robustness against background texture variations and sensor noise encountered in real UAV deployment scenarios. It also increases the base generation count from 100 to 150 scenes and incorporates defensive programming guards against `NoneType` propagation.

`Ultimate8.py` 是 `Ultimate7.py` 的增强演进版本，引入了两种额外的域随机化机制——随机色块注入和加性强度噪声——以提高模型对真实无人机部署场景中背景纹理变化和传感器噪声的鲁棒性。它还将基础生成数量从100个场景增加到150个，并加入了针对 `NoneType` 传播的防御性编程守卫。

### 6.2 Architectural Differences from Ultimate7 / 与Ultimate7的架构差异

The following table enumerates the specific modifications introduced in `Ultimate8.py` relative to its predecessor:

下表列举了 `Ultimate8.py` 相对于其前身引入的具体修改：

| Feature / 特性 | Ultimate7.py | Ultimate8.py |
|:---|:---|:---|
| Base scenes per run / 每轮基础场景数 | 100 | 150 |
| Total synthetic images / 合成图像总数 | 36,000 | 54,000 |
| Random color blocks / 随机色块 | Absent / 无 | 8--20 per background / 每背景8-20个 |
| Additive noise / 加性噪声 | Absent / 无 | 50% probability, U(0, 50) per channel / 50%概率，每通道U(0,50) |
| Output directory / 输出目录 | `yolo_dataset_809/` | `dataset/` |
| NoneType guard / NoneType守卫 | Absent / 无 | `if result:` guard in aggregation / 聚合中的守卫 |
| Return value fix / 返回值修复 | Potential `None` return / 可能返回None | Guaranteed `list` return / 保证返回列表 |

### 6.3 Stochastic Color-Block Injection / 随机色块注入

For each generated background, the script superimposes `n_blocks ~ U(8, 20)` filled rectangles with the following randomized parameters:

对每个生成的背景，脚本叠加 `n_blocks ~ U(8, 20)` 个填充矩形，具有以下随机化参数：

- **Width / 宽度:** `w ~ U(bg_w / 30, bg_w / 8)`
- **Height / 高度:** `h ~ U(bg_h / 30, bg_h / 8)`
- **Position / 位置:** `(x, y)` where `x ~ U(0, bg_w - w)`, `y ~ U(0, bg_h - h)`
- **Color / 颜色:** `(B, G, R)` where each channel is independently sampled from `U(0, 255)`
- **Thickness / 厚度:** `-1` (filled / 填充)

This mechanism serves a dual purpose: (a) it acts as a form of CutOut regularization that forces the detector to rely on partial object features rather than complete silhouettes, and (b) it simulates occluding artifacts (e.g., debris, vegetation, shadows) that commonly appear in low-altitude UAV imagery.

该机制具有双重目的：(a) 作为一种CutOut正则化形式，迫使检测器依赖部分目标特征而非完整轮廓；(b) 模拟低空无人机影像中常见的遮挡伪影（如碎片、植被、阴影）。

### 6.4 Additive Intensity Noise / 加性强度噪声

With probability `p = 0.5`, each channel of each pixel in the background receives an additive noise term:

以概率 `p = 0.5`，背景中每个像素的每个通道接受一个加性噪声项：

```
I'_c(x, y) = clip(I_c(x, y) + eta_c(x, y), 0, 255)
```

where `eta_c(x, y) ~ U(0, 50)` is independently sampled for each pixel location `(x, y)` and channel `c in {B, G, R}`. The clipping operation `clip(v, 0, 255)` enforces the valid 8-bit intensity range. This augmentation simulates the sensor noise characteristics of CMOS imaging sensors under varying illumination conditions, effectively acting as a regularizer against high-frequency adversarial perturbations.

其中 `eta_c(x, y) ~ U(0, 50)` 对每个像素位置 `(x, y)` 和通道 `c in {B, G, R}` 独立采样。截断操作 `clip(v, 0, 255)` 强制执行有效的8位强度范围。该增强模拟了CMOS图像传感器在变化光照条件下的噪声特性，有效地充当了对抗高频对抗扰动的正则化器。

### 6.5 Combined Augmentation Pipeline / 组合增强管线

The per-scene generation in `process_one_image` executes the following sequence:

`process_one_image` 中的每场景生成执行以下序列：

```
1. Select random background B from pool
2. Apply random color blocks to B
3. With p = 0.5, apply additive noise to B
4. For k in 1..n_objects:
   a. Select random foreground object O_k and its label L_k
   b. Apply affine transform M(theta, alpha) to O_k and L_k
   c. Attempt non-overlapping placement on B
   d. If placed: alpha-composite O_k onto B, record transformed bbox
5. Write composite image and merged label file
6. Repeat for all 360 angles
```

### 6.6 Scale Analysis / 规模分析

With `num_output_images = 150` and 360 angular variants per scene, a single run of `Ultimate8.py` generates:

当 `num_output_images = 150` 且每个场景360个角度变体时，`Ultimate8.py` 的单次运行生成：

```
Total = 150 scenes * 360 angles = 54,000 synthetic images
Train = floor(0.8 * 54,000) = 43,200 images
Val   = ceil(0.2 * 54,000)  = 10,800 images
```

Each image contains 1--5 annotated objects, yielding approximately 108,000--270,000 bounding box annotations per generation run. This represents a multiplicative expansion factor of up to `54,000 / N_source` relative to the original foreground image count.

每张图像包含1-5个标注目标，每次生成运行产生约108,000-270,000个边界框标注。相对于原始前景图像数量，这代表了高达 `54,000 / N_source` 的乘法扩展因子。

---

## 7. Deep Analysis: Ultimate_Ready.py

### 7.1 Overview / 概述

`Ultimate_Ready.py` is the definitive training harness that instantiates a YOLOv12n model and executes the full supervised learning protocol. Every configurable hyperparameter is explicitly enumerated, providing complete experimental reproducibility. The script serves as the single entry point for production training after data augmentation has been completed.

`Ultimate_Ready.py` 是实例化YOLOv12n模型并执行完整监督学习协议的最终训练框架。每个可配置的超参数都被显式枚举，提供了完整的实验可重复性。该脚本作为数据增强完成后生产训练的唯一入口点。

### 7.2 Model Architecture: YOLOv12n / 模型架构：YOLOv12n

YOLOv12n is the nano-scale variant of the YOLOv12 family, optimized for edge deployment. Key architectural statistics:

YOLOv12n是YOLOv12系列的纳米级变体，针对边缘部署进行了优化。关键架构统计：

| Metric / 指标 | Value / 数值 |
|:---|:---|
| Layers / 层数 | 159 |
| Parameters / 参数量 | 2,560,043 |
| FLOPs (640x640) | 6.3 GFLOPs |
| Optimized model size / 优化模型大小 | 5.5 MB |

### 7.3 Hyperparameter Exposition / 超参数详解

#### 7.3.1 Training Regime / 训练制度

| Parameter / 参数 | Value / 值 | Rationale / 原理 |
|:---|:---|:---|
| `epochs` | 300 | Sufficient horizon for convergence with early stopping / 足够的收敛时间与早停配合 |
| `patience` | 100 | Conservative early stopping; training halts after 100 epochs without mAP improvement / 保守早停；100个epoch无mAP改善后停止 |
| `batch` | 108 | Maximizes GPU memory utilization on 24 GB VRAM at 640x640 resolution / 在24GB显存、640x640分辨率下最大化GPU内存利用率 |
| `imgsz` | 640 | Standard YOLO input resolution; balances spatial detail against computational cost / 标准YOLO输入分辨率；平衡空间细节与计算成本 |

#### 7.3.2 Optimization Parameters / 优化参数

The optimizer employs Stochastic Gradient Descent (SGD) with Nesterov momentum. The learning rate follows a cosine annealing schedule (controlled by `cos_lr=False`, indicating the default linear schedule is used instead).

优化器采用带动量的随机梯度下降（SGD）。学习率遵循余弦退火调度（由 `cos_lr=False` 控制，表明使用默认的线性调度）。

| Parameter / 参数 | Value / 值 | Description / 描述 |
|:---|:---|:---|
| `lr0` | 0.01 | Initial learning rate / 初始学习率 |
| `lrf` | 0.01 | Final learning rate factor (`lr_final = lr0 * lrf = 1e-4`) / 最终学习率因子 |
| `momentum` | 0.937 | SGD momentum coefficient / SGD动量系数 |
| `weight_decay` | 0.0005 | L2 regularization strength / L2正则化强度 |
| `warmup_epochs` | 3.0 | Linear warmup duration / 线性预热时长 |
| `warmup_momentum` | 0.8 | Initial momentum during warmup / 预热期间的初始动量 |
| `warmup_bias_lr` | 0.1 | Bias learning rate during warmup / 预热期间的偏置学习率 |

The learning rate at epoch `e` during warmup (`e < warmup_epochs`) is:

预热期间（`e < warmup_epochs`）epoch `e` 的学习率为：

```
lr(e) = lr0 * (e / warmup_epochs)
```

After warmup, the learning rate follows a linear decay:

预热后，学习率遵循线性衰减：

```
lr(e) = lr0 * (1 - (e - warmup_epochs) / (epochs - warmup_epochs)) * (1 - lrf) + lr0 * lrf
```

The momentum during warmup transitions from `warmup_momentum` to `momentum`:

预热期间的动量从 `warmup_momentum` 过渡到 `momentum`：

```
mom(e) = (1 - e / warmup_epochs) * warmup_momentum + (e / warmup_epochs) * momentum
```

#### 7.3.3 YOLO Composite Loss Function / YOLO复合损失函数

The total loss is a weighted sum of three components:

总损失是三个分量的加权和：

```
L_total = lambda_box * L_box + lambda_cls * L_cls + lambda_dfl * L_DFL
```

where the weights and component losses are defined as follows:

其中权重和分量损失定义如下：

| Symbol / 符号 | Parameter / 参数 | Value / 值 | Definition / 定义 |
|:---|:---|:---|:---|
| `lambda_box` | `box` | 7.5 | Bounding box regression loss weight / 边界框回归损失权重 |
| `lambda_cls` | `cls` | 0.5 | Classification loss weight / 分类损失权重 |
| `lambda_dfl` | `dfl` | 1.5 | Distribution Focal Loss weight / 分布焦点损失权重 |

**(a) Bounding Box Regression Loss -- CIoU / 边界框回归损失 -- CIoU**

YOLOv12 employs the Complete Intersection over Union (CIoU) loss for bounding box regression:

YOLOv12采用完全交并比（CIoU）损失进行边界框回归：

```
L_CIoU = 1 - IoU + rho^2(b, b_gt) / c^2 + alpha * v
```

where:
- `IoU = |b cap b_gt| / |b cup b_gt|` is the Intersection over Union;
- `rho^2(b, b_gt)` is the squared Euclidean distance between the predicted and ground-truth box centers;
- `c` is the diagonal length of the smallest enclosing box covering both boxes;
- `alpha = v / (1 - IoU + v)` is a positive trade-off parameter;
- `v = (4 / pi^2) * (arctan(w_gt / h_gt) - arctan(w / h))^2` measures the aspect ratio consistency.

其中：
- `IoU = |b cap b_gt| / |b cup b_gt|` 为交并比；
- `rho^2(b, b_gt)` 为预测框与真实框中心点之间的欧几里得距离平方；
- `c` 为覆盖两框的最小包围框的对角线长度；
- `alpha = v / (1 - IoU + v)` 为正权衡参数；
- `v = (4 / pi^2) * (arctan(w_gt / h_gt) - arctan(w / h))^2` 衡量宽高比一致性。

**(b) Classification Loss -- Binary Cross-Entropy / 分类损失 -- 二元交叉熵**

For each anchor box and each class, binary cross-entropy is computed:

对每个锚框和每个类别，计算二元交叉熵：

```
L_cls = -(1 / N) * sum_i sum_c [ y_{i,c} * log(p_{i,c}) + (1 - y_{i,c}) * log(1 - p_{i,c}) ]
```

where `y_{i,c} in {0, 1}` is the ground-truth label and `p_{i,c}` is the predicted probability for class `c` at anchor `i`.

其中 `y_{i,c} in {0, 1}` 为真实标签，`p_{i,c}` 为锚框 `i` 处类别 `c` 的预测概率。

**(c) Distribution Focal Loss (DFL) / 分布焦点损失**

DFL models the bounding box regression target as a discrete probability distribution over edge offsets, encouraging the predicted distribution to concentrate around the ground-truth value:

DFL将边界框回归目标建模为边缘偏移上的离散概率分布，鼓励预测分布集中在真实值附近：

```
L_DFL = -sum_k [ (y_{i+1} - y) * log(S_i) + (y - y_i) * log(S_{i+1}) ]
```

where `y` is the ground-truth edge offset, `y_i` and `y_{i+1}` are the two consecutive discretized values closest to `y`, and `S_i`, `S_{i+1}` are the corresponding predicted probabilities. This formulation provides finer localization by modeling the distribution rather than a single Dirac delta prediction.

其中 `y` 为真实边缘偏移，`y_i` 和 `y_{i+1}` 为最接近 `y` 的两个连续离散化值，`S_i`、`S_{i+1}` 为相应的预测概率。该公式通过建模分布而非单一的Dirac delta预测，提供了更精细的定位。

#### 7.3.4 Online Augmentation Schedule / 在线增强调度

During training, YOLO applies a secondary augmentation pipeline (distinct from the offline `Ultimate8.py` augmentation):

训练期间，YOLO应用二次增强管线（区别于离线的 `Ultimate8.py` 增强）：

| Parameter / 参数 | Value / 值 | Effect / 效果 |
|:---|:---|:---|
| `hsv_h` | 0.015 | Hue perturbation range (fraction of 360 degrees) / 色调扰动范围 |
| `hsv_s` | 0.7 | Saturation perturbation range / 饱和度扰动范围 |
| `hsv_v` | 0.4 | Value (brightness) perturbation range / 亮度扰动范围 |
| `degrees` | 0.0 | Rotation disabled (offline augmentation handles this) / 禁用旋转（离线增强已处理） |
| `translate` | 0.1 | Translation fraction / 平移比例 |
| `scale` | 0.5 | Scaling gain factor / 缩放增益因子 |
| `shear` | 0.0 | Shear disabled / 禁用剪切 |
| `perspective` | 0.0 | Perspective transform disabled / 禁用透视变换 |
| `flipud` | 0.0 | Vertical flip disabled / 禁用垂直翻转 |
| `fliplr` | 0.5 | Horizontal flip probability / 水平翻转概率 |
| `mosaic` | 1.0 | Mosaic augmentation probability / 马赛克增强概率 |
| `mixup` | 0.0 | MixUp disabled / 禁用MixUp |
| `cutmix` | 0.0 | CutMix disabled / 禁用CutMix |
| `copy_paste` | 0.0 | Copy-Paste augmentation disabled / 禁用复制-粘贴增强 |
| `auto_augment` | `randaugment` | RandAugment auto-augmentation policy / RandAugment自动增强策略 |
| `erasing` | 0.4 | Random erasing probability / 随机擦除概率 |
| `close_mosaic` | 10 | Disable mosaic augmentation for the final 10 epochs / 最后10个epoch禁用马赛克增强 |

**Rationale for Mosaic=1.0 with close_mosaic=10 / Mosaic=1.0配合close_mosaic=10的原理:**

Mosaic augmentation stitches four training images into a single composite, dramatically increasing contextual diversity and object scale variation. However, mosaic-generated images deviate from the natural image distribution. The `close_mosaic=10` parameter disables mosaic for the final 10 epochs, allowing the model to fine-tune on photorealistic images, which has been empirically shown to improve final validation metrics by 0.5--1.5 mAP points.

马赛克增强将四张训练图像拼接为单张合成图像，极大地增加了上下文多样性和目标尺度变化。然而，马赛克生成的图像偏离自然图像分布。`close_mosaic=10` 参数在最后10个epoch禁用马赛克，使模型在逼真图像上微调，经验表明这可将最终验证指标提升0.5-1.5个mAP点。

#### 7.3.5 Inference and Post-Processing / 推理与后处理

| Parameter / 参数 | Value / 值 | Description / 描述 |
|:---|:---|:---|
| `iou` | 0.7 | NMS IoU threshold / NMS IoU阈值 |
| `max_det` | 300 | Maximum detections per image / 每张图像最大检测数 |
| `amp` | `True` | Automatic Mixed Precision (FP16) training / 自动混合精度（FP16）训练 |
| `half` | `False` | FP16 inference disabled (FP32 for maximum precision) / 禁用FP16推理（FP32以获得最高精度） |
| `agnostic_nms` | `False` | Class-aware NMS / 类别感知NMS |
| `overlap_mask` | `True` | Overlap masks during training for instance segmentation / 训练期间重叠掩码用于实例分割 |
| `mask_ratio` | 4 | Mask downsampling ratio / 掩码下采样比率 |

---

### 7.4 Overfitting-Controlled Pipeline: Ultimate9.py and Ultimate10.py

`Ultimate9.py` and `Ultimate10.py` provide an alternative path for experiments where validation reliability is more important than maximizing synthetic-data scores.

`Ultimate9.py` keeps the same directory convention as `Ultimate8.py`:

```
input_data/
|-- background/
|-- label/
|   |-- classes.txt
|-- <class_name>/
```

The main change is the split policy. Instead of randomly assigning already-generated images to train and validation sets, `Ultimate9.py` first separates source foreground images by class, then generates train and validation samples from different source records. This reduces leakage where near-identical rotations or composites of the same original image appear in both splits.

Additional safeguards in `Ultimate9.py`:

| Area | Behavior |
|:---|:---|
| Source split | Class-balanced source-level train/validation split |
| Validation generation | No random color-block injection or additive noise |
| Training generation | Moderated random color blocks and Gaussian noise |
| Output layout | Same YOLO layout as `Ultimate8.py`: `dataset/images/*`, `dataset/labels/*`, `dataset.yaml` |
| Output reset | Existing generated train/validation image and label folders are cleared before regeneration |

`Ultimate10.py` keeps the same training role as `Ultimate_Ready.py`, but applies a more conservative training schedule:

| Parameter Area | `Ultimate_Ready.py` | `Ultimate10.py` |
|:---|:---|:---|
| Epochs | 300 | 220 |
| Early stopping patience | 100 | 35 |
| Batch size | 108 | 64 |
| Optimizer | auto | AdamW |
| Weight decay | 0.0005 | 0.001 |
| Learning-rate schedule | fixed schedule | cosine schedule |
| Mosaic | 1.0 | 0.6 |
| Random erasing | 0.4 | 0.2 |
| Horizontal flip | 0.5 | 0.0 |
| Dropout | 0.0 | 0.05 |

This configuration is intended to make validation scores more conservative and reduce overfitting to repeated synthetic patterns. If the resulting model underfits, increase `TRAIN_IMAGES`, relax `patience`, or raise `mosaic` gradually rather than restoring all augmentation strength at once.

---

## 8. Auxiliary Scripts / 辅助脚本

### 8.1 Script Evolution: The Test Series / 脚本演进：Test系列

The `test.py` through `test12.py` series documents the iterative development of the training pipeline, progressing from a CIFAR-10 baseline CNN through successive refinements addressing GPU memory constraints, data augmentation inadequacy, epoch discontinuity, and annotation corruption. The trajectory culminates in `attempt.py`, which achieved stable 36-hour convergence with F1 approximately 0.4976 before being superseded by the Ultimate series.

`test.py` 到 `test12.py` 系列记录了训练管线的迭代开发过程，从CIFAR-10基线CNN开始，经过逐步改进，解决了GPU内存约束、数据增强不足、epoch不连续性和标注损坏等问题。该轨迹以 `attempt.py` 为顶点，在36小时内实现了F1约0.4976的稳定收敛，随后被Ultimate系列取代。

### 8.2 Classification and Feature Extraction / 分类与特征提取

| Script / 脚本 | Purpose / 用途 |
|:---|:---|
| `hubei.py` | ResNet classifier training with background removal, replacement, and Albumentations augmentation on a 16-class Hubei dataset (96-hour training duration) / 在16类湖北数据集上进行ResNet分类器训练，配合背景去除、替换与Albumentations增强（96小时训练时长） |
| `preloading_pics.py` | Environmental illumination feature extraction and transfer via DeepSeek R1 / 通过DeepSeek R1进行环境光照特征提取与迁移 |
| `preloading_pics2.py` | VGG19-based environmental illumination transfer via WenXin X1 Turbo / 通过WenXin X1 Turbo进行基于VGG19的环境光照迁移 |
| `preloading_bkgrd.py` | Batch background image resizing to 640x640 / 背景图像批量缩放至640x640 |
| `output.py` | Basic image augmentation (rotation, flip, brightness adjustment) to reach target count / 基础图像增强（旋转、翻转、亮度调整）以达到目标数量 |

### 8.3 Exploratory Testing / 探索性测试

| Script / 脚本 | Framework / 框架 | Function / 功能 |
|:---|:---|:---|
| `pytorchtest.py` | YOLOv8 | Real-time webcam object detection / 摄像头实时目标检测 |
| `yolov3test.py` | YOLOv5 | Batch image detection with 8 FPS rate limiting / 批量图像检测，8 FPS速率限制 |
| `haarcascadetest.py` | Haar Cascade | Face detection with 10-core parallelism / 10核并行人脸检测 |
| `hogsvm_test.py` | HOG + SVM | HOG feature extraction with SVM classification, 10-core parallelism / HOG特征提取配合SVM分类，10核并行 |

---

## 9. Validation Framework / 验证框架

The repository includes a comprehensive validation sub-system for assessing model performance throughout the development cycle.

本仓库包含一个全面的验证子系统，用于在整个开发周期中评估模型性能。

### 9.1 Validation Script Matrix / 验证脚本矩阵

| Script / 脚本 | Model / 模型 | Evaluation Metric / 评估指标 |
|:---|:---|:---|
| `verify.py` | ResNet18 | Single-image classification (16 classes), 1-hour continuous scanning / 单图分类（16类），1小时持续扫描 |
| `verify2_final.py` | ResNet18 | Single-image classification (32 classes) / 单图分类（32类） |
| `verify4.py` | ResNet152 | High-throughput validator (commented out) / 高通量验证器（已注释） |
| `verify5.py` | -- | XML/PASCAL VOC annotation validity check / XML/PASCAL VOC标注合法性检查 |
| `verify6.py` | YOLO | Ground-truth-aware mAP computation and precision/recall analysis / 含真实标注的mAP计算与精确率/召回率分析 |
| `verify7.py` | YOLOv8 | Real-time camera inference with frame capture / 摄像头实时推理与帧捕获 |

### 9.2 Metrics Formulation / 指标公式

The validation framework computes the standard object detection metrics as defined below.

验证框架计算如下定义的标准目标检测指标。

**Intersection over Union (IoU) / 交并比:**

```
IoU(B_pred, B_gt) = |B_pred cap B_gt| / |B_pred cup B_gt|
```

where `B_pred` and `B_gt` are the predicted and ground-truth bounding boxes, respectively. A detection is considered a true positive if `IoU >= 0.5` and the predicted class matches the ground-truth class.

其中 `B_pred` 和 `B_gt` 分别为预测和真实边界框。若 `IoU >= 0.5` 且预测类别与真实类别匹配，则该检测视为真阳性。

**Precision and Recall / 精确率与召回率:**

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

**F1 Score / F1分数:**

```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

**Mean Average Precision (mAP) / 平均精度均值:**

For each class `c`, the Average Precision (AP) is the area under the precision-recall curve:

对每个类别 `c`，平均精度（AP）为精确率-召回率曲线下的面积：

```
AP_c = integral_0^1 P_c(R) dR
```

The mAP is the mean over all `C` classes:

mAP为所有 `C` 个类别的均值：

```
mAP = (1 / C) * sum_{c=1}^{C} AP_c
```

mAP@50 uses `IoU >= 0.5` as the matching criterion; mAP@50-95 averages over IoU thresholds from 0.50 to 0.95 in steps of 0.05.

mAP@50使用 `IoU >= 0.5` 作为匹配标准；mAP@50-95对从0.50到0.95以0.05为步长的IoU阈值取平均。

---

## 10. Experimental Results / 实验结果

### 10.1 Two-Class Detection (Elite Race Configuration) / 二类检测（Elite Race配置）

| Metric / 指标 | Value / 数值 |
|:---|:---|
| Model / 模型 | YOLOv12n |
| Hardware / 硬件 | NVIDIA GeForce RTX 4090 D 24 GB |
| Training duration / 训练时长 | 9.994 hours / 小时 |
| Convergence epoch / 收敛epoch | 242 / 300 |
| Final mAP@50 | 0.983 |
| Final mAP@50-95 | 0.981 |
| Real-flight recognition rate (20 fps) / 实飞识别率 | 96% (fluctuation / 波动: 2%) |

### 10.2 Seventeen-Class Detection (Workspace Configuration) / 十七类检测（Workspace配置）

| Metric / 指标 | Value / 数值 |
|:---|:---|
| Model / 模型 | YOLOv12n |
| Hardware / 硬件 | NVIDIA GeForce RTX 4090 D 24 GB |
| Training duration / 训练时长 | 6.702 hours / 小时 |
| Best epoch / 最佳epoch | 69 / 300 (EarlyStopping triggered at epoch 169 / 早停在epoch 169触发) |
| Final mAP@50 | 0.995 |
| Final mAP@50-95 | 0.995 |
| Inference latency / 推理延迟 | 0.1 ms preprocess + 0.5 ms inference + 0.4 ms postprocess |
| Optimized model size / 优化模型大小 | 5.5 MB |

### 10.3 Per-Class Breakdown (17-Class) / 各类别分项（17类）

```
Class           Images    Instances    Box(P)    R        mAP@50    mAP@50-95
all             10683     12108        1.000     1.000    0.995     0.995
Class A         5918      6306         1.000     1.000    0.995     0.995
Class B         5445      5802         1.000     1.000    0.995     0.995
```

**Note / 注:** The perfect precision and recall values (1.000) indicate that at the standard confidence threshold, all detections were correct and all ground-truth objects were found. The mAP values of 0.995 reflect the integral over all confidence thresholds, where a negligible number of low-confidence detections introduce minor ranking imperfections.

精确率和召回率的完美值（1.000）表明在标准置信度阈值下，所有检测均正确且所有真实目标均被找到。0.995的mAP值反映了所有置信度阈值上的积分，其中极少量的低置信度检测引入了微小的排序不完美。

---

## 11. Usage Guide / 使用指南

### 11.1 Data Preparation / 数据准备

The input data must follow a strict directory convention:

输入数据必须遵循严格的目录约定：

```
input_data/
|-- background/              # Background scene images (PNG/JPG/JPEG) / 背景场景图像
|-- label/                   # Annotation directory / 标注目录
|   |-- classes.txt          # One class name per line / 每行一个类别名称
|   |-- <image_stem>.txt     # YOLO-format annotations / YOLO格式标注
|-- <class_A>/               # Foreground images for class A / 类别A的前景图像
|-- <class_B>/               # Foreground images for class B / 类别B的前景图像
|-- ...                      # Additional class directories / 其他类别目录
```

YOLO annotation format (one object per line):

YOLO标注格式（每行一个目标）：

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image width and height.

所有坐标均相对于图像宽度和高度归一化至 `[0, 1]`。

### 11.2 Pipeline Execution / 管线执行

**Step 1: Environment Activation / 第一步：激活环境**

```bash
conda activate <env_name>
```

**Step 2: Data Augmentation / 第二步：数据增强**

Execute the enhanced augmentation script to generate the synthetic training dataset.

执行增强型数据增强脚本以生成合成训练数据集。

```bash
cd workspace
python3 Ultimate8.py
```

Expected output: 54,000 synthetic images in `workspace/dataset/` with an 80/20 train/validation split.

预期输出：`workspace/dataset/` 中的54,000张合成图像，按80/20训练/验证分割。

**Step 3: Model Training / 第三步：模型训练**

Launch the YOLOv12n training harness.

启动YOLOv12n训练框架。

```bash
python3 Ultimate_Ready.py
```

Training outputs are written to `workspace/runs/detect/escherichia_train/`, including model checkpoints (`weights/best.pt`, `weights/last.pt`), loss curves, confusion matrices, and per-class metrics.

训练输出写入 `workspace/runs/detect/escherichia_train/`，包括模型检查点（`weights/best.pt`， `weights/last.pt`）、损失曲线、混淆矩阵和各类别指标。

**Recommended Overfitting-Controlled Execution**

Use this path from the repository root when the main concern is validation leakage or overfitting to repeated synthetic patterns.

```bash
python3 Ultimate9.py
python3 Ultimate10.py
```

`Ultimate9.py` writes the same `dataset/` structure expected by YOLO. `Ultimate10.py` writes training outputs to `runs/detect/escherichia_train_u10/`.

Before running `Ultimate9.py`, ensure `input_data/background/` contains background images. The script intentionally stops if no usable background image is found.

**Step 4: Validation / 第四步：验证**

```bash
python3 verify6.py     # Ground-truth-aware validation with mAP computation / 含真实标注的mAP计算验证
python3 verify7.py     # Real-time webcam inference / 摄像头实时推理
```

### 11.3 Path Configuration / 路径配置

The `demo` file provides a centralized path configuration template for custom deployment scenarios. Edit the variables to match the local directory layout before running the augmentation and training scripts.

`demo` 文件为自定义部署场景提供集中式路径配置模板。在运行增强和训练脚本之前，编辑变量以匹配本地目录布局。

---

## 12. License / 许可

```
MIT License

Copyright (c) 2025 Escherichia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## References / 参考文献

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.3.170). [Computer software]. https://github.com/ultralytics/ultralytics

2. Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(07), 12993--13000.

3. Li, X., Wang, W., Wu, L., Chen, S., Hu, X., Li, J., Tang, J., & Yang, J. (2020). Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection. *Advances in Neural Information Processing Systems*, 33, 21002--21012.

4. Porter, T., & Duff, T. (1984). Compositing Digital Images. *ACM SIGGRAPH Computer Graphics*, 18(3), 253--259.

5. Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *Advances in Neural Information Processing Systems*, 33, 18613--18624.

6. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. *arXiv preprint arXiv:1708.04552*.

---
