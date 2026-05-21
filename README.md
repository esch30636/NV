# NV

基于 Ultralytics YOLO / PyTorch 的计算机视觉模型训练与部署脚本集，面向无人机目标检测竞赛（Elite Race）。

## 硬件配置

### 本地 (移动端)
| 组件 | 型号 |
|------|------|
| CPU | Intel Core i7-14700HX (28C) |
| GPU | NVIDIA GeForce RTX 4060 Laptop 8GB (55W) |
| RAM | DDR5-5200 64GB |
| OS | Ubuntu 20.04.6 |

### 服务器
| 组件 | 型号 |
|------|------|
| CPU | AMD Ryzen 9 9800X / Intel Core i7-13700K |
| GPU | NVIDIA GeForce RTX 4090 D 24GB / RTX 4090 24GB |
| RAM | DDR5-5200 64GB / 128GB |
| OS | Ubuntu 20.04.6 |

## 环境

Conda 环境配置参见 `env_NV.txt`。核心依赖：

- Python 3.8.20
- PyTorch 2.1.0+cu121
- Ultralytics 8.3.170
- OpenCV, Albumentations, NumPy, Pandas, Matplotlib

```bash
conda create -n gjs -f env_NV.txt
conda activate gjs
```

## 项目结构

```
NV/
├── Ultimate_Ready.py      # 主训练入口 (YOLOv12n)
├── Ultimate7.py           # 数据增强脚本
├── Ultimate8.py           # 数据增强脚本 (并行版本)
├── Ultimate.py            # YOLOv11s 训练 (霍夫圆靶心检测)
├── Ultimate2.py           # Ultimate 代码草稿 / 测试 (已注释)
├── Ultimate3.py           # 扩散模型实验 - MNIST (已注释)
├── Ultimate4.py           # (空文件)
├── Ultimate5.py           # YOLOv11s 训练变体 (英文注释版)
├── attempt.py             # 数据验证与增强训练 (100 epochs / 100 小时)
├── hubei.py               # ResNet 分类训练 (湖北数据集, 背景替换增强)
├── preloading_pics.py     # 环境光照特征提取与迁移 (DeepSeek R1)
├── preloading_pics2.py    # 环境光照 VGG19 特征迁移 (WenXin X1 Turbo)
├── preloading_bkgrd.py    # 背景图片批量缩放
├── output.py              # 简单图像增强 (旋转/翻转/亮度)
├── yolov3test.py          # YOLOv5 检测测试
├── haarcascadetest.py     # Haar Cascade 人脸检测
├── hogsvm_test.py         # HOG+SVM 检测测试
├── pytorchtest.py         # YOLOv8 摄像头实时检测
├── verify.py              # ResNet18 验证脚本
├── verify2_final.py       # ResNet18 验证 (32 类)
├── verify4.py             # 高速验证器 (ResNet152, 已注释)
├── verify5.py             # XML/PASCAL VOC 标注校验
├── verify6.py             # YOLO 模型验证 (含 mAP 计算)
├── verify7.py             # YOLOv8 摄像头实时检测与结果保存
├── verify_test.py         # 验证测试
├── verify_test3.py        # 验证测试 3
├── test.py               # CIFAR-10 自定义 CNN 训练
├── test2.py               # ResNet152/EfficientNet 训练 (已注释)
├── test3.py               # ResNet152/EfficientNet 优化版
├── test4.py               # YOLOv8x 激进训练 + PASCAL VOC 标注
├── test5.py               # YOLOv8x 激进训练 (修正环境图片输入)
├── test6.py               # ResNet152 国赛初赛实验
├── test7.py               # 修复数据增强不足 (800-1000 张/类)
├── test8.py               # 修复 epochs 不连续训练
├── test8_blockII.py       # 修复检索异常 + 增强数据扩充
├── test9.py               # 修复参数错误 (yolov8x.pt 未正确加载)
├── test10.py              # 降低显存占用 (减小 batch_size)
├── test11.py              # 强化数据扩充 + 修复 bug
├── test12.py              # 测试脚本
├── dataset.yaml           # YOLO 数据集配置 (2 类: H, target)
├── demo                   # 路径配置模板
├── weights/               # 模型权重目录
├── yolo11n.pt             # YOLOv11n 预训练权重
├── yolo11s.pt             # YOLOv11s 预训练权重
├── env_NV.txt             # Conda 环境配置
└── LICENSE                # MIT License
```

## 脚本演进

### test 系列 — 训练管线迭代

| 脚本 | 目的 | 结果 |
|------|------|------|
| `test.py` | CIFAR-10 自定义 CNN, 验证 RTX 4060 移动端可行性 | 可用 |
| `test2.py` | ResNet152 / EfficientNet_b7 训练 | 不适用于 RTX 4060 |
| `test3.py` | ResNet152 优化版, 理论兼容 4060 | 稳定性问题, 未通过 |
| `test4.py` | YOLOv8x 最激进配置, PASCAL VOC 自动标注 | 前部打标逻辑错误 (半成品) |
| `test5.py` | 修正 test4 的环境图片输入 | 应当可用 |
| `test6.py` | ResNet152 国赛初赛实验 | 数据集扩充不足 |
| `test7.py` | 修复扩充不足 (800-1000 张/类), 26h 稳定运行 | epochs 不连续, 权重无效 |
| `test8.py` | 修复 epochs 问题, 增强扩充 | epoch4 检索异常导致终止 |
| `test8_blockII.py` | 修复检索异常 | 训练灾难性失败 (epoch2 起) |
| `test9.py` | 修复 yolov8x.pt 加载错误 | 完全失效 |
| `test10.py` | 降低显存占用 | 扩充可靠性存疑 |
| `test11.py` | 强化扩充 + bug 修复 | ~40% 扩充错误, epoch6 后 loss=0 |
| `attempt.py` | 全新思路, epoch 间链接训练, 100h 不中断 | 36h 收敛, F1≈0.4976 |

### Ultimate 系列 — 成熟训练管线

| 脚本 | 说明 |
|------|------|
| `Ultimate.py` | 霍夫圆靶心检测 + YOLOv11s, 26h 收敛, 精确率 1.0 但过拟合 |
| `Ultimate2.py` | 测试草稿 / Demo |
| `Ultimate3.py` | 扩散模型 (DDPM) MNIST 实验 |
| `Ultimate5.py` | Ultimate.py 的英文版变体 |
| `Ultimate7.py` | 数据增强: 加载原始图片+标签, 旋转/缩放/融合背景 |
| `Ultimate8.py` | 数据增强并行版, 与 Ultimate7 配合使用 |
| `Ultimate_Ready.py` | **主训练循环** — YOLOv12n, 300 epochs, batch=108 |

### 辅助脚本

| 脚本 | 功能 |
|------|------|
| `hubei.py` | ResNet 分类 (16 类), 背景去除+替换+Albumentations 增强, 96h 训练 |
| `output.py` | 图片增强至目标数量 (旋转/翻转/亮度) |
| `preloading_pics.py` | 环境光照特征提取与迁移 |
| `preloading_pics2.py` | VGG19 环境光照迁移 |
| `preloading_bkgrd.py` | 背景图片统一缩放至 640x640 |

### 验证脚本

| 脚本 | 模型 | 功能 |
|------|------|------|
| `verify.py` | ResNet18 | 单图分类验证 (16 类), 1h 持续扫描 |
| `verify2_final.py` | ResNet18 | 单图分类验证 (32 类) |
| `verify4.py` | ResNet152 | 高速验证器 (已注释) |
| `verify5.py` | — | XML/PASCAL VOC 标注合法性校验 |
| `verify6.py` | YOLO | 含 Ground Truth 的 mAP 计算验证 |
| `verify7.py` | YOLOv8 | 摄像头实时检测 + 帧保存 |
| `verify_test.py` / `verify_test3.py` | — | 验证测试辅助 |

### 探索性测试

| 脚本 | 内容 |
|------|------|
| `pytorchtest.py` | YOLOv8 摄像头实时检测 |
| `yolov3test.py` | YOLOv5 批量图片检测 (8 FPS 限制) |
| `haarcascadetest.py` | Haar Cascade 人脸检测 (10 核并行) |
| `hogsvm_test.py` | HOG+SVM 检测 (10 核并行) |
| `test12.py` | 测试脚本 |

## 最终训练结果 (Elite Race)

**模型**: YOLOv12n | **算力**: RTX 4090 D 24GB | **训练时长**: 9.994h (242 epochs 收敛)

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
242/300   21.6G     0.1061     0.09759     0.8029      4        640

Class     Images  Instances      Box(P)          R      mAP50  mAP50-95)
all       10799      12152      0.969        0.968     0.98      0.978
```

最终权重验证结果:

```
Class     Images  Instances      Box(P)          R      mAP50  mAP50-95)
all       10799      12152       0.98      0.965      0.983      0.981
H          5616       5942      0.978      0.966      0.983      0.981
target     5847       6210      0.983      0.965      0.983      0.981
```

实机飞行测试: 20fps 抖动下识别率 96%, 波动仅 2%.

## 全新训练管线 (Workspace)

`workspace/` 目录提供了优化后的训练管线，支持多类别（17 类）目标检测，适配新版 YOLOv12n 模型。

### 项目结构

```
workspace/
├── Ultimate_Ready.py      # 主训练入口 (YOLOv12n, 300 epochs, batch=108)
├── Ultimate8.py           # 增强数据增强脚本 (并行版, 含随机色块+噪声)
├── Ultimate7.py           # 基础数据增强脚本
├── yolo12n.pt             # YOLOv12n 预训练权重
├── yolo11n.pt             # YOLOv11n 预训练权重
├── input_data/            # 原始数据目录
│   ├── A/                 # 类别 A 目标图片
│   ├── B/                 # 类别 B 目标图片
│   ├── background/        # 背景图片 (17 张)
│   ├── label/             # YOLO 格式标签
│   │   └── classes.txt    # 类别名称 (17 类)
│   └── random/            # 随机杂项图片
├── dataset/               # 增强后数据集 (自动生成)
│   ├── images/
│   │   ├── train/         # 训练集 (80%)
│   │   └── val/           # 验证集 (20%)
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── classes.txt
│   └── dataset.yaml       # YOLO 数据集配置
└── runs/
    └── detect/
        └── escherichia_train/  # 训练输出 (权重/图表/日志)
```

### 路径配置 (demo)

`demo` 文件提供集中式路径配置模板，所有路径在一处管理：

```
# 目标图像目录
TARGET_DIR = "/home/legion/dataset/raw_data"
# 背景图像目录
BACKGROUND_DIR = "/home/legion/dataset/background_data"
# 输出数据集目录
OUTPUT_DIR = "/home/legion/dataset/aug_data"
# YOLO 预训练模型路径
PRETRAINED_MODEL = "/home/legion/dataset/yolov8x.pt"
# 最终模型保存路径
FINAL_MODEL_PATH = "/home/legion/dataset/trained_model.pt"
```

### 增强版数据增强 (Ultimate8.py v2)

相比根目录旧版 `Ultimate8.py`，`workspace/Ultimate8.py` 进行了以下增强：

| 改进项 | 旧版 | 新版 |
|--------|------|------|
| 随机色块 | 无 | 每张背景随机绘制 8-20 个彩色矩形块 |
| 随机噪声 | 无 | 50% 概率添加 0-50 像素强度的噪声 |
| 基础输出量 | 100 张/轮 | 150 张/轮 |
| 输出目录 | `yolo_dataset_809/` | `dataset/` |
| NoneType 防护 | 无 | 新增 `if result:` 空值守卫 |
| 返回修复 | — | 修复 `process_one_image` 返回值异常 |

增强后的数据生成量: 150 × 360（角度）= 54,000 张/轮，训练/验证按 80/20 自动拆分。

### 17 类多分类支持

旧版为 2 类（H, target）的无人机目标检测。workspace 管线扩展为 17 类通用目标检测：

```
dog, person, cat, tv, car, meatballs, marinara sauce, tomato soup,
chicken noodle soup, french onion soup, chicken breast, ribs,
pulled pork, hamburger, cavity, A, B
```

类别通过 `input_data/label/classes.txt` 定义，`Ultimate8.py` 自动读取并生成对应的 YAML 配置。

### 最终训练结果 (17 类)

**模型**: YOLOv12n | **算力**: RTX 4090 D 24GB | **训练时长**: 6.702h (169 epochs 收敛)

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
169/300      22G     0.1584     0.1189     0.7867         15        640

Class     Images  Instances      Box(P)          R      mAP50  mAP50-95)
all       10683      12108          1          1      0.995      0.995
```

最佳模型位于 epoch 69（EarlyStopping patience=100 触发），最终验证结果：

```
Class     Images  Instances      Box(P)          R      mAP50  mAP50-95)
all       10683      12108          1          1      0.995      0.995
A          5918       6306          1          1      0.995      0.995
B          5445       5802          1          1      0.995      0.995
```

推理速度: 0.1ms 预处理 / 0.5ms 推理 / 0.4ms 后处理，模型体积 5.5MB (optimized)。

### 训练启动

```bash
conda activate gjs

# 1. 准备数据: 将目标和背景图片放入 workspace/input_data/
# 2. 编辑 input_data/label/classes.txt 定义类别

# 3. 数据增强 (增强版)
cd workspace
python3 Ultimate8.py

# 4. 开始训练
python3 Ultimate_Ready.py
```

## 使用方法

### 数据准备

```
input_data/
├── background/          # 无人机航拍空场景图片 (10+ 张)
├── label/               # YOLO 格式标签 (LabelImg 标注)
│   └── classes.txt      # 类别名称列表
└── <类别文件夹>/         # 各类别目标图片 (透明 PNG 最佳)
```

### 训练流程

```bash
conda activate gjs

# 1. 数据增强
python3 Ultimate8.py

# 2. 开始训练
python3 Ultimate_Ready.py
```

### 验证

```bash
python3 verify6.py    # 含 mAP 的完整验证
python3 verify7.py    # 摄像头实时检测
```

## License

MIT License — Copyright (c) 2025 Escherichia
