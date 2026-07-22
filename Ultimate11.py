"""
Ultimate11.py — 数据增强生成 + YOLO训练 双功能单文件管线

功能一：抗过拟合数据增强管线（来自 Ultimate_Ready.py）
    - 背景融合、旋转缩放、Alpha 混合、多进程生成
功能二：YOLO 训练 + 三模块自适应早停（来自 Ultimate8.py）
    - 模块 A: Overfit Kill Switch（过拟合熔断）
    - 模块 B: Fully Learned Guard（充分学习保障）
    - 模块 C: Plateau Detector（收益递减判定）

用法:
    python Ultimate11.py train       # 仅训练
    python Ultimate11.py generate    # 仅生成数据集
    python Ultimate11.py all         # 先生成数据集，再训练（默认）
"""

import os
import sys
import cv2
import random
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import multiprocessing

# =====================================================================
#  第一部分：数据增强管线（Ultimate_Ready.py 数据生成）
# =====================================================================

def load_images_and_labels(root_dir):
    backgrounds = []
    image_paths = []
    label_paths = []
    class_names = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    required_subdirs = ["background", "label"]
    for subdir in required_subdirs:
        if not os.path.exists(os.path.join(root_dir, subdir)):
            raise FileNotFoundError(f"必需的子目录 '{subdir}' 不存在于 {root_dir}")

    classes_txt_path = os.path.join(root_dir, "label", "classes.txt")
    if os.path.exists(classes_txt_path):
        with open(classes_txt_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"未找到类别文件: {classes_txt_path}")

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        if subdir == "background":
            for bg_file in os.listdir(subdir_path):
                if bg_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    bg_img = cv2.imread(os.path.join(subdir_path, bg_file))
                    if bg_img is not None:
                        backgrounds.append(bg_img)
            if not backgrounds:
                print("警告: 没有找到背景图片")
        elif subdir == "label":
            continue
        else:
            for img_file in os.listdir(subdir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir_path, img_file)
                    label_path = os.path.join(root_dir, "label", os.path.splitext(img_file)[0] + ".txt")
                    if os.path.exists(label_path):
                        image_paths.append(img_path)
                        label_paths.append(label_path)
                    else:
                        print(f"警告: 图片 {img_file} 没有对应的标签文件")

    return backgrounds, image_paths, label_paths, class_names


def rotate_and_scale_image(image, angle, scale):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return rotated, M, (new_w, new_h)


def transform_bbox(bbox, img_size, transform_matrix, new_img_size):
    x_center, y_center, width, height = bbox
    orig_h, orig_w = img_size
    abs_x = x_center * orig_w
    abs_y = y_center * orig_h
    abs_w = width * orig_w
    abs_h = height * orig_h
    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2
    points = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1]
    ])
    transformed_points = np.dot(points, transform_matrix.T)
    new_x1 = np.min(transformed_points[:, 0])
    new_y1 = np.min(transformed_points[:, 1])
    new_x2 = np.max(transformed_points[:, 0])
    new_y2 = np.max(transformed_points[:, 1])
    new_w, new_h = new_img_size
    new_x_center = ((new_x1 + new_x2) / 2) / new_w
    new_y_center = ((new_y1 + new_y2) / 2) / new_h
    new_width = (new_x2 - new_x1) / new_w
    new_height = (new_y2 - new_y1) / new_h
    new_width = max(0.02, min(1.0, new_width))
    new_height = max(0.02, min(1.0, new_height))
    return [new_x_center, new_y_center, new_width, new_height]


def blend_with_alpha(background, foreground, x, y):
    if foreground.shape[2] == 3:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)
    if x1 >= x2 or y1 >= y2:
        return background
    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)
    fg_roi = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
    fg_alpha = fg_roi[:, :, 3] / 255.0
    fg_alpha = np.expand_dims(fg_alpha, axis=-1)
    bg_roi = background[y1:y2, x1:x2]
    blended = fg_roi[:, :, :3] * fg_alpha + bg_roi * (1 - fg_alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def process_one_image(args):
    i, backgrounds, images, labels, class_names, output_dir = args
    output_files = []
    for angle in range(360):
        bg = random.choice(backgrounds).copy()
        bg_h, bg_w = bg.shape[:2]

        num_blocks = random.randint(8, 20)
        for _ in range(num_blocks):
            block_w = random.randint(bg_w // 30, bg_w // 8)
            block_h = random.randint(bg_h // 30, bg_h // 8)
            x1 = random.randint(0, bg_w - block_w)
            y1 = random.randint(0, bg_h - block_h)
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(bg, (x1, y1), (x1 + block_w, y1 + block_h), color, thickness=-1)

        if random.random() < 0.5:
            noise = np.random.randint(0, 50, (bg_h, bg_w, 3), dtype=np.uint8)
            bg = cv2.add(bg, noise)

        num_objects = random.randint(1, min(5, len(images)))
        placed_objects = []
        output_labels = []
        for _ in range(num_objects):
            idx = random.randint(0, len(images) - 1)
            img_path = images[idx]
            label_path = labels[idx]
            class_name = os.path.basename(os.path.dirname(img_path))
            class_id = class_names.index(class_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_h, img_w = img.shape[:2]
            scale = random.uniform(0.3, 0.8)
            rotated_img, M, (new_w, new_h) = rotate_and_scale_image(img, angle, scale)
            try:
                with open(label_path, 'r') as f:
                    bbox_lines = f.readlines()
            except Exception:
                continue
            transformed_bboxes = []
            for line in bbox_lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        bbox = list(map(float, parts[1:]))
                        transformed_bbox = transform_bbox(bbox, (img_h, img_w), M, (new_w, new_h))
                        transformed_bboxes.append((int(parts[0]), transformed_bbox))
                    except Exception:
                        continue
            max_attempts = 50
            placed = False
            for _ in range(max_attempts):
                x = random.randint(0, max(1, bg_w - new_w))
                y = random.randint(0, max(1, bg_h - new_h))
                overlap = False
                new_rect = (x, y, x + new_w, y + new_h)
                for rect in placed_objects:
                    if not (new_rect[2] < rect[0] or new_rect[0] > rect[2] or
                            new_rect[3] < rect[1] or new_rect[1] > rect[3]):
                        overlap = True
                        break
                if not overlap:
                    bg = blend_with_alpha(bg, rotated_img, x, y)
                    placed_objects.append(new_rect)
                    for class_id, bbox in transformed_bboxes:
                        new_x_center = (bbox[0] * new_w + x) / bg_w
                        new_y_center = (bbox[1] * new_h + y) / bg_h
                        new_width = bbox[2] * new_w / bg_w
                        new_height = bbox[3] * new_h / bg_h
                        new_x_center = max(0.0, min(1.0, new_x_center))
                        new_y_center = max(0.0, min(1.0, new_y_center))
                        new_width = max(0.02, min(1.0, new_width))
                        new_height = max(0.02, min(1.0, new_height))
                        output_labels.append(
                            f"{class_id} {new_x_center:.6f} {new_y_center:.6f} "
                            f"{new_width:.6f} {new_height:.6f}\n"
                        )
                    placed = True
                    break
            if not placed:
                continue
        output_filename = f"aug_{i:04d}_{angle:03d}"
        is_train = random.random() < 0.8
        split_dir = "train" if is_train else "val"
        output_img_path = os.path.join(output_dir, "images", split_dir, output_filename + ".jpg")
        output_label_path = os.path.join(output_dir, "labels", split_dir, output_filename + ".txt")
        cv2.imwrite(output_img_path, bg)
        with open(output_label_path, 'w') as f:
            f.writelines(output_labels)
        output_files.append(output_img_path)
    return output_files


def place_images_on_background(backgrounds, images, labels, class_names, output_dir,
                               num_output_images=1000):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    all_output_files = []
    args_list = [
        (i, backgrounds, images, labels, class_names, output_dir)
        for i in range(num_output_images)
    ]
    cpu_count = min(multiprocessing.cpu_count(), 24)
    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, args_list),
                           total=num_output_images, desc="多进程增强数据"):
            if result:
                all_output_files.extend(result)
    return all_output_files


def create_yaml_file(output_dir, class_names):
    yaml_content = {
        'train': os.path.join('.', 'images', 'train'),
        'val': os.path.join('.', 'images', 'val'),
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)


def create_class_file(output_dir, class_names):
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    class_file_path = os.path.join(label_dir, "classes.txt")
    with open(class_file_path, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}\n")
    print(f"已创建类别文件: {class_file_path}")


def generate_dataset():
    """功能一：生成增强数据集"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_data")
    output_dir = os.path.join(current_dir, "dataset")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    try:
        backgrounds, image_paths, label_paths, class_names = load_images_and_labels(input_dir)
        if not backgrounds:
            print("错误：没有找到背景图片")
            return
        if not image_paths:
            print("错误：没有找到类别图片")
            return
        if not class_names:
            print("错误：没有找到类别文件夹")
            return
        print(f"找到 {len(backgrounds)} 张背景图片")
        print(f"找到 {len(image_paths)} 张类别图片")
        print(f"找到 {len(class_names)} 个类别: {', '.join(class_names)}")
        print("开始生成增强数据集...")
        all_output_files = place_images_on_background(
            backgrounds, image_paths, label_paths, class_names, output_dir,
            num_output_images=150
        )
        create_yaml_file(output_dir, class_names)
        create_class_file(output_dir, class_names)
        print(f"\n数据集生成完成，共生成 {len(all_output_files)} 张图片")
        print(f"数据集已保存到: {output_dir}")
        print(f"YAML配置文件已生成: {os.path.join(output_dir, 'dataset.yaml')}")
        print(f"类别文件已生成: {os.path.join(output_dir, 'labels', 'classes.txt')}")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print("请检查:")
        print(f"1. 输入目录 {input_dir} 是否存在")
        print("2. 输入目录结构是否正确:")
        print("   input_data/")
        print("   ├── background/       # 存放背景图片")
        print("   ├── label/            # 存放YOLO格式标签(.txt)")
        print("   ├── class1/           # 类别1图片")
        print("   ├── class2/           # 类别2图片")
        print("   └── ...               # 其他类别")
        print("3. 确保所有图片和标签文件都是有效的")


# =====================================================================
#  第二部分：三模块自适应早停回调（Ultimate8.py 训练 + 新判定逻辑）
# =====================================================================

class AdaptiveEarlyStopper:
    """
    三模块自适应早停管理器

    Stop(T) = Overfit(T) ∨ (FullyLearned(T) ∧ Plateau(T))

    模块 A — Overfit Kill Switch:
        最近 k 轮 val_loss 均 > best_val_loss × (1+γ) → 熔断并回滚最佳权重

    模块 B — Fully Learned Guard:
        LR_T ≤ α·LR_initial 且 T ≥ T_min → 允许进入停止判定

    模块 C — Plateau Detector:
        耐心窗口内 fitness 无显著超越 best + ε，
        且 train_loss 下降率 ≤ δ → 收益递减，平稳停止
    """

    def __init__(self, k=5, gamma=0.02, alpha=0.1, T_min=100,
                 P=30, epsilon=0.001, delta=0.01):
        # 模块 A 参数
        self.k = k
        self.gamma = gamma
        # 模块 B 参数
        self.alpha = alpha
        self.T_min = T_min
        # 模块 C 参数
        self.P = P
        self.epsilon = epsilon
        self.delta = delta

        # 历史记录
        self.val_losses = []
        self.train_losses = []
        self.fitnesses = []
        self.lrs = []

        # 最佳记录
        self.best_val_loss = float('inf')
        self.best_val_epoch = -1
        self.best_fitness = -float('inf')
        self.lr_initial = None

        self.stop_flag = False
        self.stop_reason = ""

    def on_train_start(self, trainer):
        """训练开始时记录初始学习率"""
        lr_groups = trainer.optimizer.param_groups
        self.lr_initial = max(pg['lr'] for pg in lr_groups)
        print(f"\n[AdaptiveStopper] 初始学习率 LR_initial = {self.lr_initial:.6f}")
        print(f"[AdaptiveStopper] 参数: k={self.k}, γ={self.gamma}, α={self.alpha}, "
              f"T_min={self.T_min}, P={self.P}, ε={self.epsilon}, δ={self.delta}")

    def _collect_metrics(self, trainer):
        """从 trainer 对象收集当前 epoch 的各项指标"""
        epoch = trainer.epoch

        # 学习率：取所有参数组的均值
        lr_groups = trainer.optimizer.param_groups
        lr_t = sum(pg['lr'] for pg in lr_groups) / len(lr_groups)

        # 验证集损失：从 validator 的 loss 缓冲区取总和
        val_loss = float('inf')
        try:
            if hasattr(trainer, 'validator') and trainer.validator is not None:
                if hasattr(trainer.validator, 'loss') and trainer.validator.loss is not None:
                    val_loss = float(trainer.validator.loss.detach().sum().item())
        except Exception:
            pass

        # 训练集损失：box_loss + cls_loss + dfl_loss
        train_loss = float('inf')
        try:
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                items = trainer.loss_items
                if hasattr(items, 'detach'):
                    items = items.detach()
                if hasattr(items, 'cpu'):
                    items = items.cpu()
                if hasattr(items, 'numpy'):
                    arr = items.numpy()
                else:
                    arr = np.array(items)
                arr = np.atleast_1d(arr).flatten()
                if arr.size >= 3:
                    train_loss = float(arr[0] + arr[1] + arr[2])
                elif arr.size > 0:
                    train_loss = float(arr.sum())
        except Exception:
            pass

        # 适应度 fitness = (P * 0.1 + R * 0.9) 或直接用 metrics
        fitness = 0.0
        try:
            if hasattr(trainer, 'metrics') and trainer.metrics:
                p = float(trainer.metrics.get('metrics/precision(B)', 0.0))
                r = float(trainer.metrics.get('metrics/recall(B)', 0.0))
                fitness = p * 0.1 + r * 0.9
        except Exception:
            pass

        return epoch, lr_t, val_loss, train_loss, fitness

    def _check_overfit(self):
        """
        模块 A：Overfit Kill Switch

        Overfit(T) = ∀t ∈ [T-k, T]: L_val(t) > min_{i<t}(L_val(i)) × (1+γ)

        如果最近 k 个 Epoch 的验证集损失均持续高于历史最低值的 (1+γ) 倍，
        则判定过拟合，触发熔断。
        """
        if len(self.val_losses) < self.k + 1:
            return False

        recent = self.val_losses[-self.k:]
        threshold = self.best_val_loss * (1.0 + self.gamma)
        return all(v > threshold for v in recent)

    def _check_fully_learned(self):
        """
        模块 B：Fully Learned Guard

        FullyLearned(T) = (LR_T ≤ α · LR_initial) ∧ (T ≥ T_min)

        只有当学习率衰减到初始值的 α 比例以下，且训练轮数达到 T_min，
        才允许模型进入停止候选状态。
        """
        if self.lr_initial is None or len(self.lrs) == 0:
            return False
        current_lr = self.lrs[-1]
        epoch = len(self.lrs)
        return (current_lr <= self.alpha * self.lr_initial) and (epoch >= self.T_min)

    def _check_plateau(self):
        """
        模块 C：Plateau Detector

        Plateau(T) = (max_{t∈[T-P,T]} Fitness_t ≤ Fitness_best + ε)
                   ∧ ((L_train(T-P) - L_train(T)) / L_train(T-P) ≤ δ)

        在耐心窗口内，适应度未显著超越历史最优，
        且训练损失下降率已经微乎其微。
        """
        if len(self.fitnesses) < self.P + 1 or len(self.train_losses) < self.P + 1:
            return False

        recent_fitness = self.fitnesses[-self.P:]
        if max(recent_fitness) > self.best_fitness + self.epsilon:
            return False

        train_loss_old = self.train_losses[-(self.P + 1)]
        train_loss_new = self.train_losses[-1]
        if train_loss_old <= 0 or train_loss_old == float('inf'):
            return False
        drop_rate = (train_loss_old - train_loss_new) / train_loss_old
        return drop_rate <= self.delta

    def on_train_epoch_end(self, trainer):
        """每个 epoch 结束时执行判定逻辑"""
        epoch, lr_t, val_loss, train_loss, fitness = self._collect_metrics(trainer)

        self.lrs.append(lr_t)
        self.val_losses.append(val_loss)
        self.train_losses.append(train_loss)
        self.fitnesses.append(fitness)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch
        if fitness > self.best_fitness:
            self.best_fitness = fitness

        # === 三模块判定 ===
        overfit = self._check_overfit()
        fully_learned = self._check_fully_learned()
        plateau = self._check_plateau()

        # 日志输出
        status_parts = [
            f"[AdaptiveStopper] Epoch {epoch}",
            f"LR={lr_t:.6f}",
            f"L_val={val_loss:.4f} (best={self.best_val_loss:.4f}@{self.best_val_epoch})",
            f"L_train={train_loss:.4f}",
            f"Fit={fitness:.4f} (best={self.best_fitness:.4f})",
        ]
        print(" | ".join(status_parts))

        flags = []
        if overfit:
            flags.append("OVERFIT")
        if fully_learned:
            flags.append("FULLY_LEARNED")
        if plateau:
            flags.append("PLATEAU")
        if flags:
            print(f"  → 标志: {', '.join(flags)}")

        # Stop(T) = Overfit(T) ∨ (FullyLearned(T) ∧ Plateau(T))
        if overfit:
            self.stop_flag = True
            self.stop_reason = (
                f"过拟合熔断 (Module A): 最近 {self.k} 轮 L_val 持续高于 "
                f"best×(1+{self.gamma})={self.best_val_loss * (1 + self.gamma):.4f}"
            )
        elif fully_learned and plateau:
            self.stop_flag = True
            self.stop_reason = (
                f"充分学习 + 收益递减 (Module B+C): "
                f"LR={lr_t:.6f} ≤ α·LR_init={self.alpha * (self.lr_initial or 0):.6f}, "
                f"Epoch={epoch} ≥ T_min={self.T_min}, "
                f"fitness 窗口无提升且 train_loss 下降率 ≤ {self.delta}"
            )

        if self.stop_flag:
            print(f"\n{'='*60}")
            print(f"[AdaptiveStopper] 训练终止: {self.stop_reason}")
            print(f"{'='*60}\n")
            trainer.epochs = epoch
            trainer.stopper.best_fitness = trainer.best_fitness
            trainer.stop = True


# =====================================================================
#  训练入口
# =====================================================================

def resolve_project_dir():
    """解析项目工作目录（兼容 workspace 子目录布局）"""
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "dataset" / "dataset.yaml").exists():
        return script_dir
    workspace_dir = script_dir / "workspace"
    if (workspace_dir / "dataset" / "dataset.yaml").exists():
        return workspace_dir
    return script_dir


def resolve_model(project_dir):
    """自动检测可用的模型文件"""
    for candidate in ["yolo12n.pt", "yolo12n", "yolo11n.pt"]:
        path = project_dir / candidate
        if path.exists():
            return str(path)
        if candidate == "yolo12n":
            return candidate
    return "yolo12n"


def train():
    """功能二：YOLO 训练 + 三模块自适应早停"""
    from ultralytics import YOLO, callbacks

    project_dir = resolve_project_dir()
    os.chdir(project_dir)

    model = YOLO(resolve_model(project_dir))

    stopper = AdaptiveEarlyStopper(
        k=5,         # 过拟合观察窗口
        gamma=0.02,  # 过拟合容忍比例 (2%)
        alpha=0.1,   # 学习率衰减阈值 (10%)
        T_min=100,   # 最低训练轮数
        P=30,        # 收益递减耐心窗口
        epsilon=0.001,  # 适应度容忍波动
        delta=0.01,  # 训练损失下降率阈值 (1%)
    )

    callbacks.on_train_start.append(stopper.on_train_start)
    callbacks.on_train_epoch_end.append(stopper.on_train_epoch_end)

    model.train(
        data="./dataset/dataset.yaml",
        epochs=300,
        patience=0,       # 禁用内置 patience，完全由 AdaptiveStopper 接管
        batch=108,
        imgsz=640,
        save=True,
        save_period=10,
        cache=False,
        device="0",
        workers=8,
        name="escherichia_train",
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split="val",
        save_json=False,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        vid_stride=1,
        stream_buffer=False,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
        show=False,
        save_frames=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        format="torchscript",
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=True,
        nms=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
        tracker="botsort.yaml",
        save_dir="runs/detect/train",
    )

    if stopper.stop_flag:
        print(f"\n训练已由 AdaptiveStopper 终止:")
        print(f"  原因: {stopper.stop_reason}")
    else:
        print("\n训练已正常结束（未触发自适应早停）")


# =====================================================================
#  主入口：CLI 双功能调度
# =====================================================================

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower().strip()
    else:
        mode = "all"

    if mode == "train":
        train()
    elif mode in ("generate", "gen"):
        generate_dataset()
    elif mode == "all":
        generate_dataset()
        print("\n" + "=" * 60)
        print("数据集生成完毕，开始训练...")
        print("=" * 60 + "\n")
        train()
    else:
        print(f"未知模式: {mode}")
        print("用法: python Ultimate11.py [train|generate|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
