"""
Ultimate12.py — 配对约束下的数据增强生成（仅生成，不负责训练）

输入目录约定（相对项目根，自动解析 script_dir 或 workspace/）:

    input_data/
    ├── background/          # 背景池 1：仅与 A、B 结合
    ├── ground/              # 背景池 2：仅与 A_down、B_down 结合
    ├── label/
    │   ├── classes.txt      # 每行一个类别名，建议顺序: A B A_down B_down
    │   └── <stem>.txt       # YOLO 标签，与前景图同名
    ├── A/                   # 前景，只贴到 background
    ├── B/                   # 前景，只贴到 background
    ├── A_down/              # 前景，只贴到 ground
    └── B_down/              # 前景，只贴到 ground

配对规则（硬约束）:
    A, B        × background
    A_down,B_down × ground
    同一张合成图内不会混用两个背景池。

增强手法与 Ultimate9 对齐:
    源图隔离 train/val、背景池隔离、640 画布、仿射旋转缩放、
    alpha 合成、train-only 色块/高斯噪声、固定种子可复现。

用法:
    python Ultimate12.py
"""

from __future__ import annotations

import multiprocessing
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# =====================================================================
#  配置
# =====================================================================

SEED = 42
TRAIN_RATIO = 0.8
TRAIN_IMAGES = 12000
VAL_IMAGES = 3000
TARGET_SIZE = (640, 640)

MIN_SCALE = 0.25
MAX_SCALE = 0.75
MAX_OBJECTS_PER_IMAGE = 4
MAX_PLACEMENT_ATTEMPTS = 50

TRAIN_COLOR_BLOCK_PROB = 0.35
TRAIN_NOISE_PROB = 0.35
MIN_BOX_SIZE = 0.02

CLEAR_OUTPUT = True

# 前景类别 → 允许使用的背景池
CLASS_BG_MAP = {
    "A": "background",
    "B": "background",
    "A_down": "ground",
    "B_down": "ground",
}

# 保留目录名，不当作前景类别
RESERVED_DIRS = {"background", "ground", "label", "random", "dataset", "runs"}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


# =====================================================================
#  基础工具
# =====================================================================

@dataclass(frozen=True)
class ForegroundRecord:
    image_path: str
    label_path: str
    class_name: str
    class_id: int
    bg_kind: str  # "background" | "ground"


def resolve_project_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "input_data").exists():
        return script_dir
    workspace_dir = script_dir / "workspace"
    if (workspace_dir / "input_data").exists():
        return workspace_dir
    return script_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def load_class_names(label_dir: Path) -> list[str]:
    classes_path = label_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class file: {classes_path}")
    with classes_path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    unknown = [n for n in names if n not in CLASS_BG_MAP]
    if unknown:
        raise ValueError(
            f"classes.txt contains names without background binding: {unknown}. "
            f"Allowed: {sorted(CLASS_BG_MAP)}"
        )
    missing = [n for n in CLASS_BG_MAP if n not in names]
    if missing:
        print(f"[warn] classes.txt missing optional classes: {missing}")
    return names


def load_background_pools(input_dir: Path) -> dict[str, list[np.ndarray]]:
    pools: dict[str, list[np.ndarray]] = {}
    for kind in sorted(set(CLASS_BG_MAP.values())):
        bg_dir = input_dir / kind
        if not bg_dir.exists():
            raise FileNotFoundError(f"Missing background directory: {bg_dir}")

        images: list[np.ndarray] = []
        for path in sorted(bg_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            images.append(image)

        if not images:
            raise RuntimeError(f"No usable background images found: {bg_dir}")
        pools[kind] = images
        print(f"Background pool '{kind}': {len(images)} images")

    return pools


def load_foregrounds(input_dir: Path, class_names: list[str]) -> list[ForegroundRecord]:
    label_dir = input_dir / "label"
    records: list[ForegroundRecord] = []

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in RESERVED_DIRS:
            continue
        if class_dir.name not in class_names or class_dir.name not in CLASS_BG_MAP:
            continue

        folder_class_id = class_names.index(class_dir.name)
        bg_kind = CLASS_BG_MAP[class_dir.name]

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f"[warn] skip unlabeled image: {image_path.name}")
                continue
            records.append(
                ForegroundRecord(
                    image_path=str(image_path),
                    label_path=str(label_path),
                    class_name=class_dir.name,
                    class_id=folder_class_id,
                    bg_kind=bg_kind,
                )
            )

    if not records:
        raise RuntimeError(f"No labeled foreground images found in {input_dir}")

    by_class: dict[str, int] = {}
    by_bg: dict[str, int] = {}
    for r in records:
        by_class[r.class_name] = by_class.get(r.class_name, 0) + 1
        by_bg[r.bg_kind] = by_bg.get(r.bg_kind, 0) + 1
    print(f"Foregrounds by class: {by_class}")
    print(f"Foregrounds by bg pool: {by_bg}")
    return records


def split_records_by_source(
    records: list[ForegroundRecord], train_ratio: float
) -> tuple[list[ForegroundRecord], list[ForegroundRecord]]:
    """按 class_id 分层做源图隔离划分，避免同源同时进入 train/val。"""
    by_class: dict[int, list[ForegroundRecord]] = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)

    train_records: list[ForegroundRecord] = []
    val_records: list[ForegroundRecord] = []
    rng = random.Random(SEED)

    for class_id, class_records in sorted(by_class.items()):
        shuffled = list(class_records)
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            # 单源类无法隔离：只进 train，避免泄漏
            train_records.extend(shuffled)
            continue

        val_count = max(1, int(round(len(shuffled) * (1.0 - train_ratio))))
        val_count = min(val_count, len(shuffled) - 1)
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])

    if not train_records:
        raise RuntimeError("Train split is empty; check source image counts.")
    if not val_records:
        print("[warn] Val source split is empty; val generation will be skipped.")
    return train_records, val_records


def split_background_pool(
    backgrounds: list[np.ndarray], train_ratio: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(backgrounds) < 2:
        return backgrounds, backgrounds

    indices = list(range(len(backgrounds)))
    random.Random(SEED).shuffle(indices)
    val_count = max(1, int(round(len(indices) * (1.0 - train_ratio))))
    val_count = min(val_count, len(indices) - 1)
    val_indices = set(indices[:val_count])

    train_bgs = [bg for i, bg in enumerate(backgrounds) if i not in val_indices]
    val_bgs = [bg for i, bg in enumerate(backgrounds) if i in val_indices]
    return train_bgs or backgrounds, val_bgs or backgrounds


def split_background_pools(
    pools: dict[str, list[np.ndarray]], train_ratio: float
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    train_pools: dict[str, list[np.ndarray]] = {}
    val_pools: dict[str, list[np.ndarray]] = {}
    for kind, images in pools.items():
        train_pools[kind], val_pools[kind] = split_background_pool(images, train_ratio)
        print(
            f"Split '{kind}': train_bg={len(train_pools[kind])}, val_bg={len(val_pools[kind])}"
        )
    return train_pools, val_pools


def prepare_output_dirs(output_dir: Path) -> None:
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        path = output_dir / rel
        if CLEAR_OUTPUT and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


# =====================================================================
#  标签 / 几何 / 合成
# =====================================================================

def read_yolo_labels(label_path: str, expected_class_id: int) -> list[tuple[int, list[float]]]:
    labels: list[tuple[int, list[float]]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(float(parts[0]))
                bbox = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            # 目录名权威：强制对齐类别，防止标签文件错写
            labels.append((expected_class_id, bbox))
    return labels


def ensure_bgra(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def rotate_and_scale_image(
    image: np.ndarray, angle: float, scale: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int((h * sin) + (w * cos)))
    new_h = max(1, int((h * cos) + (w * sin)))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        ensure_bgra(image),
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated, matrix, (new_w, new_h)


def transform_bbox(
    bbox: list[float],
    img_size: tuple[int, int],
    transform_matrix: np.ndarray,
    new_img_size: tuple[int, int],
) -> list[float]:
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

    points = np.array(
        [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]],
        dtype=np.float32,
    )
    transformed = np.dot(points, transform_matrix.T)

    new_x1 = np.min(transformed[:, 0])
    new_y1 = np.min(transformed[:, 1])
    new_x2 = np.max(transformed[:, 0])
    new_y2 = np.max(transformed[:, 1])

    new_w, new_h = new_img_size
    new_x_center = ((new_x1 + new_x2) / 2) / new_w
    new_y_center = ((new_y1 + new_y2) / 2) / new_h
    new_width = (new_x2 - new_x1) / new_w
    new_height = (new_y2 - new_y1) / new_h

    return [
        max(0.0, min(1.0, new_x_center)),
        max(0.0, min(1.0, new_y_center)),
        max(MIN_BOX_SIZE, min(1.0, new_width)),
        max(MIN_BOX_SIZE, min(1.0, new_height)),
    ]


def fit_foreground_to_background(
    foreground: np.ndarray,
    bbox_list: list[tuple[int, list[float]]],
    bg_w: int,
    bg_h: int,
) -> tuple[np.ndarray, list[tuple[int, list[float]]]]:
    fg_h, fg_w = foreground.shape[:2]
    max_w = int(bg_w * 0.9)
    max_h = int(bg_h * 0.9)

    if fg_w <= max_w and fg_h <= max_h:
        return foreground, bbox_list

    factor = min(max_w / max(1, fg_w), max_h / max(1, fg_h))
    new_w = max(1, int(fg_w * factor))
    new_h = max(1, int(fg_h * factor))
    resized = cv2.resize(foreground, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 等比缩放整幅画布，归一化框保持不变
    return resized, bbox_list


def blend_with_alpha(
    background: np.ndarray, foreground: np.ndarray, x: int, y: int
) -> np.ndarray:
    foreground = ensure_bgra(foreground)
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
    fg_alpha = np.expand_dims(fg_roi[:, :, 3] / 255.0, axis=-1)
    bg_roi = background[y1:y2, x1:x2]
    blended = fg_roi[:, :, :3] * fg_alpha + bg_roi * (1 - fg_alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def add_train_only_background_noise(background: np.ndarray) -> np.ndarray:
    bg_h, bg_w = background.shape[:2]

    if random.random() < TRAIN_COLOR_BLOCK_PROB:
        for _ in range(random.randint(2, 8)):
            block_w = random.randint(max(4, bg_w // 40), max(5, bg_w // 12))
            block_h = random.randint(max(4, bg_h // 40), max(5, bg_h // 12))
            x1 = random.randint(0, max(0, bg_w - block_w))
            y1 = random.randint(0, max(0, bg_h - block_h))
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(background, (x1, y1), (x1 + block_w, y1 + block_h), color, -1)

    if random.random() < TRAIN_NOISE_PROB:
        noise = np.random.normal(0, 10, (bg_h, bg_w, 3)).astype(np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return background


def overlaps_any(new_rect: tuple[int, int, int, int], placed_rects: list[tuple[int, int, int, int]]) -> bool:
    for rect in placed_rects:
        if not (
            new_rect[2] < rect[0]
            or new_rect[0] > rect[2]
            or new_rect[3] < rect[1]
            or new_rect[1] > rect[3]
        ):
            return True
    return False


# =====================================================================
#  单样本生成
# =====================================================================

def make_one_sample(
    split: str,
    index: int,
    bg_pools: dict[str, list[np.ndarray]],
    records: list[ForegroundRecord],
    output_dir: Path,
) -> str | None:
    seed_everything(SEED + index + (0 if split == "train" else 1_000_000))

    # 前景按背景池分组；本张图只从一个池取背景、只贴该池兼容的前景
    scene_records: dict[str, list[ForegroundRecord]] = {}
    for record in records:
        scene_records.setdefault(record.bg_kind, []).append(record)

    available_kinds = [
        kind
        for kind, recs in scene_records.items()
        if recs and kind in bg_pools and bg_pools[kind]
    ]
    if not available_kinds:
        return None

    # 两池等概率出图，保证 background/ground 场景均衡
    bg_kind = random.choice(available_kinds)
    pool_records = scene_records[bg_kind]
    bg = random.choice(bg_pools[bg_kind]).copy()

    if split == "train":
        bg = add_train_only_background_noise(bg)

    bg_h, bg_w = bg.shape[:2]
    placed_rects: list[tuple[int, int, int, int]] = []
    output_labels: list[str] = []
    num_objects = random.randint(1, min(MAX_OBJECTS_PER_IMAGE, len(pool_records)))

    for _ in range(num_objects):
        record = random.choice(pool_records)
        image = cv2.imread(record.image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        source_labels = read_yolo_labels(record.label_path, record.class_id)
        if not source_labels:
            continue

        img_h, img_w = image.shape[:2]
        angle = random.uniform(0.0, 360.0)
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
        rotated, matrix, new_size = rotate_and_scale_image(image, angle, scale)

        transformed_labels = [
            (class_id, transform_bbox(bbox, (img_h, img_w), matrix, new_size))
            for class_id, bbox in source_labels
        ]
        rotated, transformed_labels = fit_foreground_to_background(
            rotated, transformed_labels, bg_w, bg_h
        )
        fg_h, fg_w = rotated.shape[:2]

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            x = random.randint(0, max(0, bg_w - fg_w))
            y = random.randint(0, max(0, bg_h - fg_h))
            new_rect = (x, y, x + fg_w, y + fg_h)
            if overlaps_any(new_rect, placed_rects):
                continue

            bg = blend_with_alpha(bg, rotated, x, y)
            placed_rects.append(new_rect)

            for class_id, bbox in transformed_labels:
                new_x_center = (bbox[0] * fg_w + x) / bg_w
                new_y_center = (bbox[1] * fg_h + y) / bg_h
                new_width = bbox[2] * fg_w / bg_w
                new_height = bbox[3] * fg_h / bg_h

                output_labels.append(
                    f"{class_id} "
                    f"{max(0.0, min(1.0, new_x_center)):.6f} "
                    f"{max(0.0, min(1.0, new_y_center)):.6f} "
                    f"{max(MIN_BOX_SIZE, min(1.0, new_width)):.6f} "
                    f"{max(MIN_BOX_SIZE, min(1.0, new_height)):.6f}\n"
                )
            break

    if not output_labels:
        return None

    stem = f"u12_{split}_{index:06d}"
    image_path = output_dir / "images" / split / f"{stem}.jpg"
    label_path = output_dir / "labels" / split / f"{stem}.txt"

    cv2.imwrite(str(image_path), bg)
    with label_path.open("w", encoding="utf-8") as f:
        f.writelines(output_labels)

    return str(image_path)


def worker(args):
    return make_one_sample(*args)


def generate_split(
    split: str,
    count: int,
    bg_pools: dict[str, list[np.ndarray]],
    records: list[ForegroundRecord],
    output_dir: Path,
) -> list[str]:
    if count <= 0:
        return []

    args = [(split, i, bg_pools, records, output_dir) for i in range(count)]
    cpu_count = min(multiprocessing.cpu_count(), 16)
    written: list[str] = []

    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(
            pool.imap_unordered(worker, args),
            total=count,
            desc=f"Generating {split}",
        ):
            if result:
                written.append(result)

    return written


def write_dataset_files(output_dir: Path, class_names: list[str]) -> None:
    yaml_content = {
        "train": "./images/train",
        "val": "./images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False, allow_unicode=True)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with (labels_dir / "classes.txt").open("w", encoding="utf-8") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")


# =====================================================================
#  主流程
# =====================================================================

def generate_dataset() -> None:
    seed_everything(SEED)
    project_dir = resolve_project_dir()
    input_dir = project_dir / "input_data"
    output_dir = project_dir / "dataset"

    print(f"Project: {project_dir}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Pairing: {CLASS_BG_MAP}")

    class_names = load_class_names(input_dir / "label")
    bg_pools = load_background_pools(input_dir)
    records = load_foregrounds(input_dir, class_names)

    train_records, val_records = split_records_by_source(records, TRAIN_RATIO)
    train_pools, val_pools = split_background_pools(bg_pools, TRAIN_RATIO)

    prepare_output_dirs(output_dir)

    train_written = generate_split(
        "train", TRAIN_IMAGES, train_pools, train_records, output_dir
    )
    val_written = generate_split(
        "val", VAL_IMAGES, val_pools, val_records, output_dir
    )
    write_dataset_files(output_dir, class_names)

    print("\n========== Generation summary ==========")
    print(f"Class names:     {class_names}")
    print(f"Train sources:   {len(train_records)}")
    print(f"Val sources:     {len(val_records)}")
    print(f"Train images:    {len(train_written)} / {TRAIN_IMAGES}")
    print(f"Val images:      {len(val_written)} / {VAL_IMAGES}")
    print(f"Dataset YAML:    {output_dir / 'dataset.yaml'}")
    print("Generation complete.")


def main() -> None:
    generate_dataset()


if __name__ == "__main__":
    main()
