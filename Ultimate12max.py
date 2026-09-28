"""
Ultimate12max.py
================

安全的一体化数据生成、质量检查和一次性训练入口。

设计目标：
1. 最大限度阻断 train/val 泄漏和合成数据捷径，降低过拟合风险。
2. 生成过程使用暂存目录，QA 通过后再原子替换 dataset，保留旧数据备份。
3. 训练前验证数据签名、hold-out、模型、软件版本、GPU 和磁盘空间。
4. 使用一次性训练锁，拒绝覆盖、递增运行目录或无意重复训练。

本脚本不能从数学上保证模型绝不过拟合；它保证的是可验证的工程约束。

目标环境：Python 3.8 / torch 2.1 + CUDA 12.1 / ultralytics 8.3.170 / RTX 4090。

典型用法：
    python Ultimate12max.py generate
    python Ultimate12max.py qa
    python Ultimate12max.py train \
        --confirm-train RUN_ONE_SHOT_TRAINING

输入结构：
    input_data/
    ├── background/
    ├── ground/
    ├── label/classes.txt       # 严格为 A, B, A_down, B_down
    ├── label/<stem>.txt
    ├── A/
    ├── B/
    ├── A_down/
    └── B_down/

前景必须是带有效 Alpha 通道的 PNG 抠图。训练前还要求：
    holdout/images/             # 20--50 张未参与生成的真实图片
    holdout/labels/             # 与图片同相对路径、同 stem 的 YOLO 标签
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ============================================================================
# 固定配置：修改这些值会改变数据签名和训练行为
# ============================================================================

PIPELINE_VERSION = "12max-1.0"
SEED = 42
EXPECTED_CLASSES: Tuple[str, ...] = ("A", "B", "A_down", "B_down")
CLASS_BG_MAP: Dict[str, str] = {
    "A": "background",
    "B": "background",
    "A_down": "ground",
    "B_down": "ground",
}

TRAIN_RATIO = 0.80
TRAIN_IMAGES = 12_000
VAL_IMAGES = 3_000
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

MIN_SCALE = 0.25
MAX_SCALE = 0.75
MAX_OBJECTS_PER_IMAGE = 4
MAX_PLACEMENT_ATTEMPTS = 50
SAMPLE_RETRIES = 5
MIN_BOX_SIZE = 0.02

TRAIN_COLOR_BLOCK_PROB = 0.35
TRAIN_NOISE_PROB = 0.35

# dHash 汉明距离不超过该值的同类图片视为近重复，必须留在同一 split。
NEAR_DUPLICATE_DISTANCE = 4
MIN_FOREGROUNDS_PER_CLASS = 2
MIN_BACKGROUNDS_PER_POOL = 2
MIN_HOLDOUT_IMAGES = 20
MAX_HOLDOUT_IMAGES = 50

IMAGE_SUFFIXES: Set[str] = {".png", ".jpg", ".jpeg"}
FOREGROUND_SUFFIXES: Set[str] = {".png"}
TRAIN_CONFIRMATION = "RUN_ONE_SHOT_TRAINING"

EXPECTED_PYTHON = (3, 8)
EXPECTED_TORCH_PREFIX = "2.1."
EXPECTED_CUDA_PREFIX = "12.1"
EXPECTED_ULTRALYTICS = "8.3.170"
EXPECTED_GPU_SUBSTRING = "4090"
MIN_GPU_TOTAL_GIB = 20.0
MIN_GPU_FREE_RATIO = 0.85
MIN_DISK_FREE_GIB = 15.0

RUN_NAME = "escherichia_train_u10"
RUN_PROJECT_REL = Path("runs") / "detect"
READY_FILE_NAME = ".u12max_ready.json"
QA_REPORT_NAME = "qa_report.json"
PROVENANCE_NAME = "provenance.jsonl"
QA_SHEET_NAME = "qa_samples.jpg"
TRAIN_LOCK_NAME = "ONE_SHOT_TRAINING.lock.json"
TRAIN_COMPLETE_NAME = "ONE_SHOT_TRAINING.completed.json"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass(frozen=True)
class LabelBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def values(self) -> Tuple[float, float, float, float]:
        return (self.cx, self.cy, self.width, self.height)


@dataclass(frozen=True)
class ForegroundRecord:
    image_path: str
    label_path: str
    class_name: str
    class_id: int
    bg_kind: str
    sha256: str
    dhash: int
    labels: Tuple[LabelBox, ...]


@dataclass(frozen=True)
class BackgroundRecord:
    image_path: str
    bg_kind: str
    sha256: str
    dhash: int
    width: int
    height: int


class PipelineError(RuntimeError):
    pass


class PlacementError(PipelineError):
    pass


# 每个 worker 只初始化一次；任务队列只传 index，避免重复序列化背景图数组。
_WORKER_CONTEXT: Dict[str, Any] = {}


# ============================================================================
# 通用工具
# ============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_project_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    if (script_dir / "input_data").exists():
        return script_dir
    workspace_dir = script_dir / "workspace"
    if (workspace_dir / "input_data").exists():
        return workspace_dir
    return script_dir


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temp_path), str(path))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PipelineError("JSON root must be an object: {}".format(path))
    return value


def image_dhash(image: np.ndarray) -> int:
    if image.ndim == 2:
        gray = image
    else:
        if image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            bgr = image[:, :, :3].astype(np.float32)
            composite = bgr * alpha + 127.0 * (1.0 - alpha)
            color = composite.astype(np.uint8)
        else:
            color = image[:, :, :3]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count() if hasattr(int, "bit_count") else bin(left ^ right).count("1")


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def ensure_disk_space(project_dir: Path, minimum_gib: float = MIN_DISK_FREE_GIB) -> None:
    free = shutil.disk_usage(str(project_dir)).free
    free_gib = free / (1024.0 ** 3)
    if free_gib < minimum_gib:
        raise PipelineError(
            "Insufficient free disk space: {:.2f} GiB available, {:.2f} GiB required.".format(
                free_gib, minimum_gib
            )
        )


# ============================================================================
# 输入 QA、标签与近重复分组
# ============================================================================

def load_class_names(label_dir: Path) -> Tuple[str, ...]:
    classes_path = label_dir / "classes.txt"
    if not classes_path.is_file():
        raise PipelineError("Missing classes file: {}".format(classes_path))
    with classes_path.open("r", encoding="utf-8-sig") as handle:
        names = tuple(line.strip() for line in handle if line.strip())
    if names != EXPECTED_CLASSES:
        raise PipelineError(
            "classes.txt must contain exactly these four lines in order: {}. Found: {}".format(
                list(EXPECTED_CLASSES), list(names)
            )
        )
    return names


def parse_yolo_label(
    label_path: Path,
    expected_class_id: Optional[int] = None,
) -> Tuple[LabelBox, ...]:
    if not label_path.is_file():
        raise PipelineError("Missing label: {}".format(label_path))

    boxes: List[LabelBox] = []
    with label_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise PipelineError(
                    "{}:{} must contain 5 fields, found {}.".format(
                        label_path, line_number, len(parts)
                    )
                )
            try:
                raw_class = float(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise PipelineError(
                    "{}:{} contains a non-numeric field.".format(label_path, line_number)
                ) from exc

            if not raw_class.is_integer():
                raise PipelineError(
                    "{}:{} class id must be an integer.".format(label_path, line_number)
                )
            class_id = int(raw_class)
            if class_id < 0 or class_id >= len(EXPECTED_CLASSES):
                raise PipelineError(
                    "{}:{} class id {} is outside [0, 3].".format(
                        label_path, line_number, class_id
                    )
                )
            if expected_class_id is not None and class_id != expected_class_id:
                raise PipelineError(
                    "{}:{} class id {} conflicts with folder class id {}.".format(
                        label_path, line_number, class_id, expected_class_id
                    )
                )

            if not all(math.isfinite(value) for value in values):
                raise PipelineError(
                    "{}:{} contains NaN or infinity.".format(label_path, line_number)
                )
            cx, cy, width, height = values
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise PipelineError(
                    "{}:{} center must be inside [0, 1].".format(label_path, line_number)
                )
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise PipelineError(
                    "{}:{} width and height must be inside (0, 1].".format(
                        label_path, line_number
                    )
                )
            epsilon = 1e-6
            if (
                cx - width / 2.0 < -epsilon
                or cy - height / 2.0 < -epsilon
                or cx + width / 2.0 > 1.0 + epsilon
                or cy + height / 2.0 > 1.0 + epsilon
            ):
                raise PipelineError(
                    "{}:{} bounding box extends outside the image.".format(
                        label_path, line_number
                    )
                )
            boxes.append(LabelBox(class_id, cx, cy, width, height))

    if not boxes:
        raise PipelineError("Label contains no valid boxes: {}".format(label_path))
    return tuple(boxes)


def validate_foreground_alpha(
    image_path: Path,
    image: np.ndarray,
    labels: Sequence[LabelBox],
) -> None:
    if image.ndim != 3 or image.shape[2] != 4:
        raise PipelineError(
            "Foreground must be a transparent four-channel PNG: {}".format(image_path)
        )

    alpha = image[:, :, 3]
    transparent_ratio = float(np.mean(alpha < 250))
    foreground_ratio = float(np.mean(alpha > 5))
    if transparent_ratio < 0.005 or foreground_ratio < 0.005:
        raise PipelineError(
            "Foreground Alpha is ineffective (transparent={:.4f}, foreground={:.4f}): {}".format(
                transparent_ratio, foreground_ratio, image_path
            )
        )

    ys, xs = np.where(alpha > 5)
    if xs.size == 0 or ys.size == 0:
        raise PipelineError("Foreground Alpha mask is empty: {}".format(image_path))

    image_h, image_w = alpha.shape
    alpha_x1 = float(xs.min()) / image_w
    alpha_y1 = float(ys.min()) / image_h
    alpha_x2 = float(xs.max() + 1) / image_w
    alpha_y2 = float(ys.max() + 1) / image_h

    for box in labels:
        box_x1 = box.cx - box.width / 2.0
        box_y1 = box.cy - box.height / 2.0
        box_x2 = box.cx + box.width / 2.0
        box_y2 = box.cy + box.height / 2.0
        intersection_w = max(0.0, min(box_x2, alpha_x2) - max(box_x1, alpha_x1))
        intersection_h = max(0.0, min(box_y2, alpha_y2) - max(box_y1, alpha_y1))
        overlap = (intersection_w * intersection_h) / max(box.width * box.height, 1e-12)
        if overlap < 0.50:
            raise PipelineError(
                "Label and Alpha foreground overlap by only {:.1%}: {}".format(
                    overlap, image_path
                )
            )


def collect_foregrounds(input_dir: Path) -> List[ForegroundRecord]:
    label_dir = input_dir / "label"
    load_class_names(label_dir)
    records: List[ForegroundRecord] = []
    seen_stems: Dict[str, Path] = {}
    sha_to_class: Dict[str, str] = {}

    for class_id, class_name in enumerate(EXPECTED_CLASSES):
        class_dir = input_dir / class_name
        if not class_dir.is_dir():
            raise PipelineError("Missing foreground directory: {}".format(class_dir))
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in FOREGROUND_SUFFIXES
        )
        if len(image_paths) < MIN_FOREGROUNDS_PER_CLASS:
            raise PipelineError(
                "Class {} needs at least {} transparent PNG foregrounds; found {}.".format(
                    class_name, MIN_FOREGROUNDS_PER_CLASS, len(image_paths)
                )
            )

        for image_path in image_paths:
            stem_key = image_path.stem.casefold()
            previous = seen_stems.get(stem_key)
            if previous is not None:
                raise PipelineError(
                    "Foreground stems must be globally unique because labels share one directory: {} and {}".format(
                        previous, image_path
                    )
                )
            seen_stems[stem_key] = image_path

            label_path = label_dir / (image_path.stem + ".txt")
            labels = parse_yolo_label(label_path, expected_class_id=class_id)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise PipelineError("Unreadable foreground image: {}".format(image_path))
            validate_foreground_alpha(image_path, image, labels)

            file_hash = sha256_file(image_path)
            previous_class = sha_to_class.get(file_hash)
            if previous_class is not None and previous_class != class_name:
                raise PipelineError(
                    "Identical foreground content appears in two classes: {} and {} ({})".format(
                        previous_class, class_name, image_path
                    )
                )
            sha_to_class[file_hash] = class_name
            records.append(
                ForegroundRecord(
                    image_path=str(image_path.resolve()),
                    label_path=str(label_path.resolve()),
                    class_name=class_name,
                    class_id=class_id,
                    bg_kind=CLASS_BG_MAP[class_name],
                    sha256=file_hash,
                    dhash=image_dhash(image),
                    labels=labels,
                )
            )

    return records


def collect_backgrounds(input_dir: Path) -> List[BackgroundRecord]:
    records: List[BackgroundRecord] = []
    seen_hashes: Dict[str, str] = {}
    seen_by_kind: Dict[str, List[BackgroundRecord]] = {}

    for bg_kind in ("background", "ground"):
        bg_dir = input_dir / bg_kind
        if not bg_dir.is_dir():
            raise PipelineError("Missing background directory: {}".format(bg_dir))
        image_paths = sorted(
            path
            for path in bg_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(image_paths) < MIN_BACKGROUNDS_PER_POOL:
            raise PipelineError(
                "Pool {} needs at least {} backgrounds; found {}.".format(
                    bg_kind, MIN_BACKGROUNDS_PER_POOL, len(image_paths)
                )
            )

        kind_records: List[BackgroundRecord] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise PipelineError("Unreadable background image: {}".format(image_path))
            height, width = image.shape[:2]
            if width < 64 or height < 64:
                raise PipelineError("Background is too small: {}".format(image_path))
            file_hash = sha256_file(image_path)
            previous_kind = seen_hashes.get(file_hash)
            if previous_kind is not None and previous_kind != bg_kind:
                raise PipelineError(
                    "Identical background content appears in both pools: {}".format(image_path)
                )
            seen_hashes[file_hash] = bg_kind
            record = BackgroundRecord(
                image_path=str(image_path.resolve()),
                bg_kind=bg_kind,
                sha256=file_hash,
                dhash=image_dhash(image),
                width=width,
                height=height,
            )
            kind_records.append(record)
            records.append(record)
        seen_by_kind[bg_kind] = kind_records

    # 跨池近重复会让场景语义和 train/val 隔离变得不可靠，直接拒绝。
    for left in seen_by_kind["background"]:
        for right in seen_by_kind["ground"]:
            if hamming_distance(left.dhash, right.dhash) <= NEAR_DUPLICATE_DISTANCE:
                raise PipelineError(
                    "Near-duplicate backgrounds appear in different pools: {} and {}".format(
                        left.image_path, right.image_path
                    )
                )
    return records


def near_duplicate_groups(records: Sequence[Any]) -> List[List[Any]]:
    count = len(records)
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(count):
        for right in range(left + 1, count):
            if (
                records[left].sha256 == records[right].sha256
                or hamming_distance(records[left].dhash, records[right].dhash)
                <= NEAR_DUPLICATE_DISTANCE
            ):
                union(left, right)

    grouped: Dict[int, List[Any]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def split_grouped_records(
    records: Sequence[Any],
    train_ratio: float,
    seed: int,
    description: str,
) -> Tuple[List[Any], List[Any]]:
    groups = near_duplicate_groups(records)
    if len(groups) < 2:
        raise PipelineError(
            "{} needs at least two visually distinct source groups for leakage-free train/val splitting.".format(
                description
            )
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    target_val_records = max(1, int(round(len(records) * (1.0 - train_ratio))))
    val_groups: List[List[Any]] = []
    val_count = 0
    for group in groups:
        if len(val_groups) >= len(groups) - 1:
            break
        if val_count < target_val_records:
            val_groups.append(group)
            val_count += len(group)

    if not val_groups:
        val_groups = [groups[0]]
    val_group_ids = {id(group) for group in val_groups}
    train = [item for group in groups if id(group) not in val_group_ids for item in group]
    val = [item for group in groups if id(group) in val_group_ids for item in group]
    if not train or not val:
        raise PipelineError("Leakage-free split is empty for {}.".format(description))

    print(
        "{}: sources={} groups={} train={} val={}".format(
            description, len(records), len(groups), len(train), len(val)
        )
    )
    return train, val


def split_foregrounds(
    records: Sequence[ForegroundRecord],
) -> Tuple[Dict[str, List[ForegroundRecord]], Dict[str, List[ForegroundRecord]]]:
    train: Dict[str, List[ForegroundRecord]] = {}
    val: Dict[str, List[ForegroundRecord]] = {}
    for class_id, class_name in enumerate(EXPECTED_CLASSES):
        class_records = [record for record in records if record.class_name == class_name]
        train[class_name], val[class_name] = split_grouped_records(
            class_records,
            TRAIN_RATIO,
            SEED + 100 + class_id,
            "foreground/{}".format(class_name),
        )
    return train, val


def split_backgrounds(
    records: Sequence[BackgroundRecord],
) -> Tuple[Dict[str, List[BackgroundRecord]], Dict[str, List[BackgroundRecord]]]:
    train: Dict[str, List[BackgroundRecord]] = {}
    val: Dict[str, List[BackgroundRecord]] = {}
    for offset, bg_kind in enumerate(("background", "ground")):
        pool_records = [record for record in records if record.bg_kind == bg_kind]
        train[bg_kind], val[bg_kind] = split_grouped_records(
            pool_records,
            TRAIN_RATIO,
            SEED + 200 + offset,
            "background/{}".format(bg_kind),
        )
    return train, val


# ============================================================================
# 合成与几何变换
# ============================================================================

def crop_to_alpha(
    image: np.ndarray,
    labels: Sequence[LabelBox],
) -> Tuple[np.ndarray, Tuple[LabelBox, ...]]:
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 5)
    if xs.size == 0 or ys.size == 0:
        raise PipelineError("Alpha mask became empty while generating a sample.")

    original_h, original_w = image.shape[:2]
    padding = 2
    # 裁剪范围同时覆盖 Alpha 和所有标签，避免为了去透明边而截短检测框。
    label_x1 = min((box.cx - box.width / 2.0) * original_w for box in labels)
    label_y1 = min((box.cy - box.height / 2.0) * original_h for box in labels)
    label_x2 = max((box.cx + box.width / 2.0) * original_w for box in labels)
    label_y2 = max((box.cy + box.height / 2.0) * original_h for box in labels)
    x1 = max(0, int(math.floor(min(float(xs.min()), label_x1))) - padding)
    y1 = max(0, int(math.floor(min(float(ys.min()), label_y1))) - padding)
    x2 = min(original_w, int(math.ceil(max(float(xs.max() + 1), label_x2))) + padding)
    y2 = min(original_h, int(math.ceil(max(float(ys.max() + 1), label_y2))) + padding)
    cropped = image[y1:y2, x1:x2]
    cropped_h, cropped_w = cropped.shape[:2]

    adjusted: List[LabelBox] = []
    for box in labels:
        old_x1 = (box.cx - box.width / 2.0) * original_w
        old_y1 = (box.cy - box.height / 2.0) * original_h
        old_x2 = (box.cx + box.width / 2.0) * original_w
        old_y2 = (box.cy + box.height / 2.0) * original_h
        new_x1 = max(0.0, min(float(cropped_w), old_x1 - x1))
        new_y1 = max(0.0, min(float(cropped_h), old_y1 - y1))
        new_x2 = max(0.0, min(float(cropped_w), old_x2 - x1))
        new_y2 = max(0.0, min(float(cropped_h), old_y2 - y1))
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            raise PipelineError("A label fell outside its Alpha crop.")
        adjusted.append(
            LabelBox(
                box.class_id,
                ((new_x1 + new_x2) / 2.0) / cropped_w,
                ((new_y1 + new_y2) / 2.0) / cropped_h,
                (new_x2 - new_x1) / cropped_w,
                (new_y2 - new_y1) / cropped_h,
            )
        )
    return cropped, tuple(adjusted)


def rotate_and_scale_image(
    image: np.ndarray,
    angle: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = max(1, int(math.ceil(height * sine + width * cosine)))
    new_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated, matrix, (new_width, new_height)


def transform_box(
    box: LabelBox,
    original_size: Tuple[int, int],
    matrix: np.ndarray,
    new_size: Tuple[int, int],
) -> LabelBox:
    original_h, original_w = original_size
    x1 = (box.cx - box.width / 2.0) * original_w
    y1 = (box.cy - box.height / 2.0) * original_h
    x2 = (box.cx + box.width / 2.0) * original_w
    y2 = (box.cy + box.height / 2.0) * original_h
    points = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    transformed = np.dot(points, matrix.T)
    new_width, new_height = new_size
    tx1 = max(0.0, min(float(new_width), float(np.min(transformed[:, 0]))))
    ty1 = max(0.0, min(float(new_height), float(np.min(transformed[:, 1]))))
    tx2 = max(0.0, min(float(new_width), float(np.max(transformed[:, 0]))))
    ty2 = max(0.0, min(float(new_height), float(np.max(transformed[:, 1]))))
    if tx2 <= tx1 or ty2 <= ty1:
        raise PlacementError("Transformed label became empty.")
    return LabelBox(
        box.class_id,
        ((tx1 + tx2) / 2.0) / new_width,
        ((ty1 + ty2) / 2.0) / new_height,
        (tx2 - tx1) / new_width,
        (ty2 - ty1) / new_height,
    )


def fit_foreground(
    image: np.ndarray,
    labels: Sequence[LabelBox],
    background_width: int,
    background_height: int,
) -> Tuple[np.ndarray, Tuple[LabelBox, ...]]:
    height, width = image.shape[:2]
    max_width = int(background_width * 0.90)
    max_height = int(background_height * 0.90)
    if width <= max_width and height <= max_height:
        return image, tuple(labels)
    factor = min(max_width / max(1, width), max_height / max(1, height))
    resized = cv2.resize(
        image,
        (max(1, int(width * factor)), max(1, int(height * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, tuple(labels)


def rectangles_overlap(
    candidate: Tuple[int, int, int, int],
    placed: Sequence[Tuple[int, int, int, int]],
) -> bool:
    for existing in placed:
        if not (
            candidate[2] <= existing[0]
            or candidate[0] >= existing[2]
            or candidate[3] <= existing[1]
            or candidate[1] >= existing[3]
        ):
            return True
    return False


def alpha_blend(background: np.ndarray, foreground: np.ndarray, x: int, y: int) -> None:
    height, width = foreground.shape[:2]
    region = background[y : y + height, x : x + width]
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
    blended = foreground[:, :, :3].astype(np.float32) * alpha + region.astype(np.float32) * (
        1.0 - alpha
    )
    background[y : y + height, x : x + width] = blended.astype(np.uint8)


def add_train_only_noise(
    background: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    height, width = background.shape[:2]
    if rng.random() < TRAIN_COLOR_BLOCK_PROB:
        for _ in range(rng.randint(2, 8)):
            block_width = rng.randint(max(4, width // 40), max(5, width // 12))
            block_height = rng.randint(max(4, height // 40), max(5, height // 12))
            x1 = rng.randint(0, max(0, width - block_width))
            y1 = rng.randint(0, max(0, height - block_height))
            color = tuple(rng.randint(0, 255) for _ in range(3))
            cv2.rectangle(
                background,
                (x1, y1),
                (x1 + block_width, y1 + block_height),
                color,
                thickness=-1,
            )
    if rng.random() < TRAIN_NOISE_PROB:
        noise = np_rng.normal(0.0, 10.0, (height, width, 3)).astype(np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return background


def init_worker(context: Dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    cv2.setNumThreads(1)


def sample_class_sequence(primary_class: str, count: int) -> List[str]:
    bg_kind = CLASS_BG_MAP[primary_class]
    compatible = [name for name in EXPECTED_CLASSES if CLASS_BG_MAP[name] == bg_kind]
    primary_index = compatible.index(primary_class)
    return [compatible[(primary_index + offset) % len(compatible)] for offset in range(count)]


def build_one_sample(index: int, attempt: int = 0) -> Dict[str, Any]:
    context = _WORKER_CONTEXT
    split = str(context["split"])
    output_dir = Path(str(context["output_dir"]))
    records_by_class: Dict[str, List[ForegroundRecord]] = context["records_by_class"]
    backgrounds_by_kind: Dict[str, List[BackgroundRecord]] = context["backgrounds_by_kind"]

    split_offset = 0 if split == "train" else 1_000_000_000
    seed = SEED + split_offset + index + attempt * 10_000_000
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 每四张至少各有一张以对应类别为主目标，避免类别数量随源文件数偏斜。
    primary_class = EXPECTED_CLASSES[index % len(EXPECTED_CLASSES)]
    bg_kind = CLASS_BG_MAP[primary_class]
    background_record = rng.choice(backgrounds_by_kind[bg_kind])
    background = cv2.imread(background_record.image_path, cv2.IMREAD_COLOR)
    if background is None:
        raise PipelineError("Background became unreadable: {}".format(background_record.image_path))
    background = cv2.resize(
        background,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    if split == "train":
        background = add_train_only_noise(background, rng, np_rng)

    object_count = rng.randint(1, MAX_OBJECTS_PER_IMAGE)
    class_sequence = sample_class_sequence(primary_class, object_count)
    placed_rectangles: List[Tuple[int, int, int, int]] = []
    output_boxes: List[LabelBox] = []
    object_provenance: List[Dict[str, Any]] = []

    for object_index, class_name in enumerate(class_sequence):
        record = rng.choice(records_by_class[class_name])
        foreground = cv2.imread(record.image_path, cv2.IMREAD_UNCHANGED)
        if foreground is None:
            raise PipelineError("Foreground became unreadable: {}".format(record.image_path))
        foreground, cropped_labels = crop_to_alpha(foreground, record.labels)
        original_size = foreground.shape[:2]
        angle = rng.uniform(0.0, 360.0)
        scale = rng.uniform(MIN_SCALE, MAX_SCALE)
        rotated, matrix, rotated_size = rotate_and_scale_image(foreground, angle, scale)
        transformed_labels = tuple(
            transform_box(box, original_size, matrix, rotated_size) for box in cropped_labels
        )
        rotated, transformed_labels = fit_foreground(
            rotated,
            transformed_labels,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        foreground_height, foreground_width = rotated.shape[:2]

        # 小框不能靠人为放大标签“通过”；实际目标过小就重试整张样本。
        if any(
            box.width * foreground_width / TARGET_WIDTH < MIN_BOX_SIZE
            or box.height * foreground_height / TARGET_HEIGHT < MIN_BOX_SIZE
            for box in transformed_labels
        ):
            if object_index == 0:
                raise PlacementError("The primary object became smaller than MIN_BOX_SIZE.")
            continue

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            x = rng.randint(0, TARGET_WIDTH - foreground_width)
            y = rng.randint(0, TARGET_HEIGHT - foreground_height)
            rectangle = (x, y, x + foreground_width, y + foreground_height)
            if rectangles_overlap(rectangle, placed_rectangles):
                continue
            alpha_blend(background, rotated, x, y)
            placed_rectangles.append(rectangle)
            for box in transformed_labels:
                cx = (box.cx * foreground_width + x) / TARGET_WIDTH
                cy = (box.cy * foreground_height + y) / TARGET_HEIGHT
                width = box.width * foreground_width / TARGET_WIDTH
                height = box.height * foreground_height / TARGET_HEIGHT
                output_boxes.append(
                    LabelBox(
                        record.class_id,
                        max(0.0, min(1.0, cx)),
                        max(0.0, min(1.0, cy)),
                        min(1.0, width),
                        min(1.0, height),
                    )
                )
            object_provenance.append(
                {
                    "class_name": record.class_name,
                    "class_id": record.class_id,
                    "source": record.image_path,
                    "source_sha256": record.sha256,
                    "angle": round(angle, 6),
                    "scale": round(scale, 6),
                }
            )
            placed = True
            break

        if not placed and object_index == 0:
            raise PlacementError("Could not place the primary object.")

    if not output_boxes or not object_provenance:
        raise PlacementError("Generated sample contains no objects.")

    stem = "u12max_{}_{:06d}".format(split, index)
    image_path = output_dir / "images" / split / (stem + ".jpg")
    label_path = output_dir / "labels" / split / (stem + ".txt")
    image_temp = image_path.with_name(stem + ".tmp.jpg")
    label_temp = label_path.with_name(stem + ".tmp.txt")

    if not cv2.imwrite(str(image_temp), background, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise PipelineError("Failed to write generated image: {}".format(image_temp))
    with label_temp.open("w", encoding="utf-8", newline="\n") as handle:
        for box in output_boxes:
            handle.write(
                "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    box.class_id, box.cx, box.cy, box.width, box.height
                )
            )
    os.replace(str(image_temp), str(image_path))
    os.replace(str(label_temp), str(label_path))

    return {
        "split": split,
        "index": index,
        "image": image_path.relative_to(output_dir).as_posix(),
        "label": label_path.relative_to(output_dir).as_posix(),
        "primary_class": primary_class,
        "background_kind": bg_kind,
        "background_source": background_record.image_path,
        "background_sha256": background_record.sha256,
        "objects": object_provenance,
    }


def worker_task(index: int) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    for attempt in range(SAMPLE_RETRIES):
        try:
            return build_one_sample(index, attempt)
        except PlacementError as exc:
            last_error = exc
    raise PlacementError(
        "Sample {} failed after {} retries: {}".format(index, SAMPLE_RETRIES, last_error)
    )


def prepare_staging_dirs(output_dir: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / relative).mkdir(parents=True, exist_ok=False)


def generate_split(
    split: str,
    count: int,
    records_by_class: Dict[str, List[ForegroundRecord]],
    backgrounds_by_kind: Dict[str, List[BackgroundRecord]],
    output_dir: Path,
    workers: int,
) -> List[Dict[str, Any]]:
    context = {
        "split": split,
        "output_dir": str(output_dir),
        "records_by_class": records_by_class,
        "backgrounds_by_kind": backgrounds_by_kind,
    }
    results: List[Dict[str, Any]] = []
    if workers <= 1:
        init_worker(context)
        for index in tqdm(range(count), total=count, desc="Generating {}".format(split)):
            results.append(worker_task(index))
        return results

    with multiprocessing.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(context,),
        maxtasksperchild=500,
    ) as pool:
        iterator = pool.imap_unordered(worker_task, range(count), chunksize=8)
        for result in tqdm(iterator, total=count, desc="Generating {}".format(split)):
            results.append(result)
    if len(results) != count:
        raise PipelineError(
            "Generated {} {} images, expected {}.".format(len(results), split, count)
        )
    return results


# ============================================================================
# 数据集 QA、签名与抽检图
# ============================================================================

def write_dataset_metadata(output_dir: Path) -> None:
    yaml_payload = {
        "train": "./images/train",
        "val": "./images/val",
        "nc": len(EXPECTED_CLASSES),
        "names": list(EXPECTED_CLASSES),
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(yaml_payload, handle, sort_keys=False, allow_unicode=True)
    classes_path = output_dir / "labels" / "classes.txt"
    with classes_path.open("w", encoding="utf-8", newline="\n") as handle:
        for class_name in EXPECTED_CLASSES:
            handle.write(class_name + "\n")


def write_provenance(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    provenance_path = output_dir / PROVENANCE_NAME
    with provenance_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: (item["split"], item["index"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_provenance(dataset_dir: Path) -> List[Dict[str, Any]]:
    path = dataset_dir / PROVENANCE_NAME
    if not path.is_file():
        raise PipelineError("Missing provenance file: {}".format(path))
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(
                    "Invalid provenance JSON at {}:{}".format(path, line_number)
                ) from exc
            if not isinstance(value, dict):
                raise PipelineError("Invalid provenance row at {}:{}".format(path, line_number))
            rows.append(value)
    return rows


def validate_dataset_yaml(dataset_dir: Path) -> None:
    path = dataset_dir / "dataset.yaml"
    if not path.is_file():
        raise PipelineError("Missing dataset.yaml: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PipelineError("dataset.yaml root must be a mapping.")
    if payload.get("nc") != 4 or tuple(payload.get("names", [])) != EXPECTED_CLASSES:
        raise PipelineError("dataset.yaml class definition is not the required four-class order.")
    if payload.get("train") != "./images/train" or payload.get("val") != "./images/val":
        raise PipelineError("dataset.yaml train/val paths were changed unexpectedly.")


def dataset_digest_paths(dataset_dir: Path) -> List[Path]:
    paths = [
        dataset_dir / "dataset.yaml",
        dataset_dir / "labels" / "classes.txt",
        dataset_dir / PROVENANCE_NAME,
    ]
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        directory = dataset_dir / relative
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return paths


def compute_dataset_digest(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in dataset_digest_paths(dataset_dir):
        if not path.is_file():
            raise PipelineError("Dataset integrity file is missing: {}".format(path))
        relative = path.relative_to(dataset_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def validate_composite_visibility(
    dataset_dir: Path,
    provenance_rows: Sequence[Dict[str, Any]],
    sample_count: int = 200,
) -> Dict[str, Any]:
    """Headless QA: verify that val boxes contain visible changes from their source background."""
    val_rows = sorted(
        (row for row in provenance_rows if row.get("split") == "val"),
        key=lambda row: int(row.get("index", -1)),
    )
    if not val_rows:
        raise PipelineError("No val provenance rows are available for composite visibility QA.")
    chosen_indices = sorted(
        set(int(value) for value in np.linspace(0, len(val_rows) - 1, min(sample_count, len(val_rows))))
    )
    margins: List[float] = []
    changed_fractions: List[float] = []
    failures: List[str] = []

    for row_index in chosen_indices:
        row = val_rows[row_index]
        bg_kind = str(row.get("background_kind"))
        objects = row.get("objects")
        if bg_kind not in ("background", "ground") or not isinstance(objects, list):
            raise PipelineError("Invalid pairing metadata in provenance.")
        for item in objects:
            class_name = str(item.get("class_name"))
            if class_name not in CLASS_BG_MAP or CLASS_BG_MAP[class_name] != bg_kind:
                raise PipelineError(
                    "Class/background pairing violation in provenance: class={}, pool={}.".format(
                        class_name, bg_kind
                    )
                )

        generated_path = dataset_dir / str(row.get("image"))
        label_path = dataset_dir / str(row.get("label"))
        background_path = Path(str(row.get("background_source")))
        generated = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if generated is None or background is None:
            raise PipelineError(
                "Composite visibility QA could not read {} or {}.".format(
                    generated_path, background_path
                )
            )
        background = cv2.resize(
            background,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        absolute_difference = np.mean(
            np.abs(generated.astype(np.int16) - background.astype(np.int16)),
            axis=2,
        )
        box_mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        boxes = parse_yolo_label(label_path)
        for box in boxes:
            x1 = max(0, int(math.floor((box.cx - box.width / 2.0) * TARGET_WIDTH)))
            y1 = max(0, int(math.floor((box.cy - box.height / 2.0) * TARGET_HEIGHT)))
            x2 = min(TARGET_WIDTH, int(math.ceil((box.cx + box.width / 2.0) * TARGET_WIDTH)))
            y2 = min(TARGET_HEIGHT, int(math.ceil((box.cy + box.height / 2.0) * TARGET_HEIGHT)))
            box_mask[y1:y2, x1:x2] = 1
        inside = absolute_difference[box_mask == 1]
        outside = absolute_difference[box_mask == 0]
        if inside.size == 0 or outside.size == 0:
            raise PipelineError("Composite visibility QA produced an empty mask: {}".format(generated_path))

        inside_mean = float(np.mean(inside))
        outside_mean = float(np.mean(outside))
        margin = inside_mean - outside_mean
        change_threshold = max(6.0, float(np.percentile(outside, 95)) + 2.0)
        changed_fraction = float(np.mean(inside > change_threshold))
        margins.append(margin)
        changed_fractions.append(changed_fraction)
        if margin < 1.0 or changed_fraction < 0.02:
            failures.append(generated_path.name)

    allowed_failures = max(1, int(math.floor(len(chosen_indices) * 0.01)))
    if len(failures) > allowed_failures:
        raise PipelineError(
            "Headless composite visibility QA failed for {}/{} samples; examples: {}".format(
                len(failures), len(chosen_indices), failures[:10]
            )
        )
    return {
        "samples": len(chosen_indices),
        "failures": len(failures),
        "minimum_box_background_difference": min(margins),
        "mean_box_background_difference": float(np.mean(margins)),
        "minimum_changed_pixel_fraction": min(changed_fractions),
        "mean_changed_pixel_fraction": float(np.mean(changed_fractions)),
    }


def validate_generated_dataset(
    dataset_dir: Path,
    expected_train: int = TRAIN_IMAGES,
    expected_val: int = VAL_IMAGES,
) -> Dict[str, Any]:
    validate_dataset_yaml(dataset_dir)
    class_counts = [0 for _ in EXPECTED_CLASSES]
    image_hashes: Dict[str, Set[str]] = {"train": set(), "val": set()}
    split_counts: Dict[str, int] = {}

    for split, expected_count in (("train", expected_train), ("val", expected_val)):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")
        if len(images) != expected_count or len(labels) != expected_count:
            raise PipelineError(
                "{} count mismatch: images={}, labels={}, expected={}.".format(
                    split, len(images), len(labels), expected_count
                )
            )
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise PipelineError("Image/label stems do not match in {}.".format(split))

        for image_path in tqdm(images, desc="QA images/{}".format(split)):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                raise PipelineError("Invalid generated image: {}".format(image_path))
            image_hashes[split].add(sha256_file(image_path))
            boxes = parse_yolo_label(label_dir / (image_path.stem + ".txt"))
            for box in boxes:
                class_counts[box.class_id] += 1
        if len(image_hashes[split]) != expected_count:
            raise PipelineError("Duplicate generated images exist inside {}.".format(split))
        split_counts[split] = len(images)

    duplicate_outputs = image_hashes["train"] & image_hashes["val"]
    if duplicate_outputs:
        raise PipelineError("Identical generated images appear in train and val.")

    rows = load_provenance(dataset_dir)
    if len(rows) != expected_train + expected_val:
        raise PipelineError("Provenance row count does not match generated image count.")
    foreground_hashes: Dict[str, Set[str]] = {"train": set(), "val": set()}
    background_hashes: Dict[str, Set[str]] = {"train": set(), "val": set()}
    for row in rows:
        split = row.get("split")
        if split not in ("train", "val"):
            raise PipelineError("Invalid split in provenance.")
        bg_kind = str(row.get("background_kind"))
        background_hash = row.get("background_sha256")
        if bg_kind not in ("background", "ground") or not isinstance(background_hash, str):
            raise PipelineError("Invalid background metadata in provenance.")
        background_hashes[split].add(background_hash)
        objects = row.get("objects")
        if not isinstance(objects, list) or not objects:
            raise PipelineError("A provenance row has no object sources.")
        for item in objects:
            class_name = str(item.get("class_name"))
            source_hash = item.get("source_sha256")
            if (
                class_name not in CLASS_BG_MAP
                or CLASS_BG_MAP[class_name] != bg_kind
                or not isinstance(source_hash, str)
            ):
                raise PipelineError("Class/background pairing violation in provenance.")
            foreground_hashes[split].add(source_hash)

    if foreground_hashes["train"] & foreground_hashes["val"]:
        raise PipelineError("Foreground source leakage detected between train and val.")
    if background_hashes["train"] & background_hashes["val"]:
        raise PipelineError("Background source leakage detected between train and val.")
    if min(class_counts) <= 0:
        raise PipelineError("At least one class is missing from generated labels.")
    balance_ratio = max(class_counts) / float(min(class_counts))
    if balance_ratio > 1.25:
        raise PipelineError(
            "Generated class imbalance is too high: counts={}, ratio={:.3f}.".format(
                class_counts, balance_ratio
            )
        )

    visibility_report = validate_composite_visibility(dataset_dir, rows)

    return {
        "passed": True,
        "checked_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "class_box_counts": {
            EXPECTED_CLASSES[index]: class_counts[index] for index in range(len(EXPECTED_CLASSES))
        },
        "class_balance_ratio": balance_ratio,
        "train_foreground_sources": len(foreground_hashes["train"]),
        "val_foreground_sources": len(foreground_hashes["val"]),
        "train_background_sources": len(background_hashes["train"]),
        "val_background_sources": len(background_hashes["val"]),
        "cross_split_output_duplicates": 0,
        "cross_split_foreground_leaks": 0,
        "cross_split_background_leaks": 0,
        "headless_composite_visibility": visibility_report,
    }


def create_qa_sheet(dataset_dir: Path, sample_count: int = 20) -> Path:
    image_dir = dataset_dir / "images" / "val"
    label_dir = dataset_dir / "labels" / "val"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < sample_count:
        raise PipelineError("Not enough val images for the visual QA sheet.")
    indices = np.linspace(0, len(images) - 1, sample_count, dtype=int)
    tiles: List[np.ndarray] = []
    colors = [(60, 220, 60), (60, 180, 255), (255, 120, 60), (220, 80, 220)]
    for index in indices:
        image_path = images[int(index)]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineError("Unreadable QA sample: {}".format(image_path))
        boxes = parse_yolo_label(label_dir / (image_path.stem + ".txt"))
        for box in boxes:
            x1 = int((box.cx - box.width / 2.0) * image.shape[1])
            y1 = int((box.cy - box.height / 2.0) * image.shape[0])
            x2 = int((box.cx + box.width / 2.0) * image.shape[1])
            y2 = int((box.cy + box.height / 2.0) * image.shape[0])
            color = colors[box.class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                EXPECTED_CLASSES[box.class_id],
                (max(0, x1), max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        tiles.append(cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA))

    rows: List[np.ndarray] = []
    for offset in range(0, sample_count, 5):
        rows.append(np.hstack(tiles[offset : offset + 5]))
    sheet = np.vstack(rows)
    output_path = dataset_dir / QA_SHEET_NAME
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise PipelineError("Failed to write QA sheet: {}".format(output_path))
    return output_path


def write_ready_marker(dataset_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    digest = compute_dataset_digest(dataset_dir)
    ready = {
        "ready": True,
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "dataset_sha256": digest,
        "train_images": TRAIN_IMAGES,
        "val_images": VAL_IMAGES,
        "classes": list(EXPECTED_CLASSES),
        "qa_report_sha256": sha256_file(dataset_dir / QA_REPORT_NAME),
    }
    write_json_atomic(dataset_dir / READY_FILE_NAME, ready)
    return ready


def run_dataset_qa(dataset_dir: Path) -> Dict[str, Any]:
    report = validate_generated_dataset(dataset_dir)
    write_json_atomic(dataset_dir / QA_REPORT_NAME, report)
    sheet_path = create_qa_sheet(dataset_dir)
    ready = write_ready_marker(dataset_dir, report)
    print("QA passed: {}".format(dataset_dir / QA_REPORT_NAME))
    print("Optional QA contact sheet (not a training gate on headless SSH): {}".format(sheet_path))
    print("Dataset digest: {}".format(ready["dataset_sha256"]))
    return ready


def promote_dataset(staging_dir: Path, dataset_dir: Path) -> Optional[Path]:
    backup_dir: Optional[Path] = None
    if dataset_dir.exists():
        backup_dir = dataset_dir.with_name("dataset_backup_{}".format(safe_timestamp()))
        if backup_dir.exists():
            raise PipelineError("Backup destination already exists: {}".format(backup_dir))
        os.replace(str(dataset_dir), str(backup_dir))
    try:
        os.replace(str(staging_dir), str(dataset_dir))
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not dataset_dir.exists():
            os.replace(str(backup_dir), str(dataset_dir))
        raise
    return backup_dir


def generate_dataset(project_dir: Path, workers: int) -> None:
    ensure_disk_space(project_dir)
    input_dir = project_dir / "input_data"
    if not input_dir.is_dir():
        raise PipelineError("Missing input_data directory: {}".format(input_dir))

    print("Validating source data...")
    foregrounds = collect_foregrounds(input_dir)
    backgrounds = collect_backgrounds(input_dir)
    train_foregrounds, val_foregrounds = split_foregrounds(foregrounds)
    train_backgrounds, val_backgrounds = split_backgrounds(backgrounds)

    staging_dir = project_dir / ".dataset_build_{}_{}".format(safe_timestamp(), os.getpid())
    if staging_dir.exists():
        raise PipelineError("Staging directory already exists: {}".format(staging_dir))
    prepare_staging_dirs(staging_dir)
    print("Generating into staging directory: {}".format(staging_dir))

    try:
        train_rows = generate_split(
            "train",
            TRAIN_IMAGES,
            train_foregrounds,
            train_backgrounds,
            staging_dir,
            workers,
        )
        val_rows = generate_split(
            "val",
            VAL_IMAGES,
            val_foregrounds,
            val_backgrounds,
            staging_dir,
            workers,
        )
        write_dataset_metadata(staging_dir)
        write_provenance(staging_dir, train_rows + val_rows)
        run_dataset_qa(staging_dir)
    except BaseException:
        print("Generation failed. Staging data was kept for diagnosis: {}".format(staging_dir))
        raise

    dataset_dir = project_dir / "dataset"
    backup_dir = promote_dataset(staging_dir, dataset_dir)
    print("Dataset promoted atomically: {}".format(dataset_dir))
    if backup_dir is not None:
        print("Previous dataset preserved at: {}".format(backup_dir))


# ============================================================================
# Hold-out 和训练前门禁
# ============================================================================

def validate_holdout(project_dir: Path, dataset_dir: Path) -> Dict[str, Any]:
    holdout_dir = project_dir / "holdout"
    images_dir = holdout_dir / "images"
    labels_dir = holdout_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise PipelineError(
            "holdout/images and holdout/labels are required before the one-shot training run."
        )

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not (MIN_HOLDOUT_IMAGES <= len(images) <= MAX_HOLDOUT_IMAGES):
        raise PipelineError(
            "Hold-out must contain {}--{} real images; found {}.".format(
                MIN_HOLDOUT_IMAGES, MAX_HOLDOUT_IMAGES, len(images)
            )
        )

    class_counts = [0 for _ in EXPECTED_CLASSES]
    hashes: Set[str] = set()
    used_source_hashes: Set[str] = set()
    for row in load_provenance(dataset_dir):
        background_hash = row.get("background_sha256")
        if isinstance(background_hash, str):
            used_source_hashes.add(background_hash)
        objects = row.get("objects")
        if isinstance(objects, list):
            for item in objects:
                source_hash = item.get("source_sha256")
                if isinstance(source_hash, str):
                    used_source_hashes.add(source_hash)

    for image_path in images:
        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineError("Unreadable hold-out image: {}".format(image_path))
        boxes = parse_yolo_label(label_path)
        for box in boxes:
            class_counts[box.class_id] += 1
        file_hash = sha256_file(image_path)
        if file_hash in hashes:
            raise PipelineError("Duplicate images exist inside hold-out: {}".format(image_path))
        if file_hash in used_source_hashes:
            raise PipelineError(
                "A hold-out image was used as a generation source or background: {}".format(
                    image_path
                )
            )
        hashes.add(file_hash)

    if min(class_counts) <= 0:
        raise PipelineError(
            "Hold-out must contain every class. Box counts: {}".format(class_counts)
        )
    return {
        "images": len(images),
        "class_box_counts": {
            EXPECTED_CLASSES[index]: class_counts[index] for index in range(len(EXPECTED_CLASSES))
        },
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
    }


def verify_ready_dataset(dataset_dir: Path) -> Dict[str, Any]:
    ready_path = dataset_dir / READY_FILE_NAME
    if not ready_path.is_file():
        raise PipelineError(
            "Dataset has no U12max ready marker. Run `python Ultimate12max.py qa` first."
        )
    ready = read_json(ready_path)
    if ready.get("pipeline_version") != PIPELINE_VERSION or ready.get("ready") is not True:
        raise PipelineError("Dataset ready marker belongs to a different pipeline version.")
    qa_report_path = dataset_dir / QA_REPORT_NAME
    if not qa_report_path.is_file():
        raise PipelineError("Dataset QA report is missing: {}".format(qa_report_path))
    if sha256_file(qa_report_path) != ready.get("qa_report_sha256"):
        raise PipelineError("Dataset QA report changed after the ready marker was created.")
    current_digest = compute_dataset_digest(dataset_dir)
    if current_digest != ready.get("dataset_sha256"):
        raise PipelineError(
            "Dataset changed after QA. Expected {}, found {}.".format(
                ready.get("dataset_sha256"), current_digest
            )
        )
    validate_dataset_yaml(dataset_dir)
    # 不盲目信任旧报告；在真正训练前重新扫描全部图像、标签、来源和配对关系。
    validate_generated_dataset(dataset_dir)
    return ready


def validate_training_environment(project_dir: Path) -> Dict[str, Any]:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise PipelineError(
            "Expected Python {}.{}, found {}.{}.".format(
                EXPECTED_PYTHON[0],
                EXPECTED_PYTHON[1],
                sys.version_info.major,
                sys.version_info.minor,
            )
        )

    import torch
    import ultralytics

    torch_version = str(torch.__version__)
    cuda_version = str(torch.version.cuda)
    ultralytics_version = str(ultralytics.__version__)
    if not torch_version.startswith(EXPECTED_TORCH_PREFIX):
        raise PipelineError(
            "Expected torch {}x, found {}.".format(EXPECTED_TORCH_PREFIX, torch_version)
        )
    if not cuda_version.startswith(EXPECTED_CUDA_PREFIX):
        raise PipelineError(
            "Expected torch CUDA {}, found {}.".format(EXPECTED_CUDA_PREFIX, cuda_version)
        )
    if ultralytics_version != EXPECTED_ULTRALYTICS:
        raise PipelineError(
            "Expected ultralytics {}, found {}.".format(
                EXPECTED_ULTRALYTICS, ultralytics_version
            )
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PipelineError("CUDA GPU is unavailable; training is blocked.")

    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    if EXPECTED_GPU_SUBSTRING not in gpu_name:
        raise PipelineError(
            "Training is restricted to the approved RTX 4090 target; found {}.".format(gpu_name)
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_gib = total_bytes / (1024.0 ** 3)
    free_ratio = free_bytes / float(total_bytes)
    if total_gib < MIN_GPU_TOTAL_GIB or free_ratio < MIN_GPU_FREE_RATIO:
        raise PipelineError(
            "GPU is not sufficiently free: name={}, total={:.2f} GiB, free={:.1%}.".format(
                gpu_name, total_gib, free_ratio
            )
        )
    ensure_disk_space(project_dir)
    return {
        "python": "{}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ),
        "torch": torch_version,
        "cuda": cuda_version,
        "ultralytics": ultralytics_version,
        "gpu": gpu_name,
        "gpu_total_gib": total_gib,
        "gpu_free_ratio": free_ratio,
    }


def acquire_training_lock(project_dir: Path, payload: Dict[str, Any]) -> Path:
    lock_path = project_dir / TRAIN_LOCK_NAME
    completed_path = project_dir / TRAIN_COMPLETE_NAME
    if completed_path.exists():
        raise PipelineError("One-shot training is already marked complete: {}".format(completed_path))
    if lock_path.exists():
        raise PipelineError(
            "A one-shot training lock already exists. Inspect it before any further action: {}".format(
                lock_path
            )
        )

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(str(lock_path), flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            lock_path.unlink()
        except OSError:
            pass
        raise
    return lock_path


def verify_checkpoint(best_path: Path) -> Dict[str, Any]:
    if not best_path.is_file() or best_path.stat().st_size < 1024 * 1024:
        raise PipelineError("best.pt is missing or implausibly small: {}".format(best_path))
    import torch

    checkpoint = torch.load(str(best_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or not any(key in checkpoint for key in ("model", "ema")):
        raise PipelineError("best.pt does not look like a valid Ultralytics checkpoint.")
    return {
        "path": str(best_path.resolve()),
        "bytes": best_path.stat().st_size,
        "sha256": sha256_file(best_path),
    }


def serializable_metrics(results_dict: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in results_dict.items():
        if isinstance(value, (np.floating, np.integer)):
            output[str(key)] = value.item()
        elif isinstance(value, (float, int, str, bool)) or value is None:
            output[str(key)] = value
        else:
            output[str(key)] = str(value)
    return output


def evaluate_holdout(best_path: Path, project_dir: Path) -> Dict[str, Any]:
    from ultralytics import YOLO

    holdout_dir = project_dir / "holdout"
    run_dir = project_dir / RUN_PROJECT_REL / RUN_NAME
    holdout_yaml = run_dir / "holdout_dataset.yaml"
    payload = {
        "path": str(holdout_dir.resolve()),
        "val": "images",
        "nc": len(EXPECTED_CLASSES),
        "names": list(EXPECTED_CLASSES),
    }
    with holdout_yaml.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    model = YOLO(str(best_path))
    metrics = model.val(
        data=str(holdout_yaml),
        split="val",
        imgsz=640,
        batch=32,
        device="0",
        workers=8,
        plots=True,
        project=str(run_dir),
        name="holdout_eval",
        exist_ok=False,
    )
    results = serializable_metrics(dict(metrics.results_dict))
    write_json_atomic(run_dir / "holdout_metrics.json", results)
    return results


def train_once(
    project_dir: Path,
    confirmation: str,
) -> None:
    if confirmation != TRAIN_CONFIRMATION:
        raise PipelineError(
            "Training confirmation is missing. Required token: {}".format(TRAIN_CONFIRMATION)
        )
    dataset_dir = project_dir / "dataset"
    ready = verify_ready_dataset(dataset_dir)
    holdout_report = validate_holdout(project_dir, dataset_dir)
    environment = validate_training_environment(project_dir)

    model_path = project_dir / "yolo12n.pt"
    if not model_path.is_file():
        raise PipelineError(
            "Local yolo12n.pt is required; implicit downloads and model fallbacks are disabled."
        )
    model_info = {
        "path": str(model_path.resolve()),
        "bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
    }
    if model_info["bytes"] < 1024 * 1024:
        raise PipelineError("yolo12n.pt is implausibly small.")

    run_project = project_dir / RUN_PROJECT_REL
    run_dir = run_project / RUN_NAME
    if run_dir.exists():
        raise PipelineError(
            "Exact run directory already exists; refusing overwrite or auto-increment: {}".format(
                run_dir
            )
        )

    lock_payload: Dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "dataset": ready,
        "holdout": holdout_report,
        "environment": environment,
        "model": model_info,
        "run_dir": str(run_dir.resolve()),
    }
    lock_path = acquire_training_lock(project_dir, lock_payload)

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path.resolve()))
        model.train(
            data=str((dataset_dir / "dataset.yaml").resolve()),
            epochs=300,
            patience=35,
            batch=64,
            imgsz=640,
            save=True,
            save_period=5,
            cache=False,
            device="0",
            workers=8,
            project=str(run_project.resolve()),
            name=RUN_NAME,
            exist_ok=False,
            pretrained=True,
            optimizer="AdamW",
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=30,
            resume=False,
            amp=True,
            fraction=1.0,
            multi_scale=False,
            dropout=0.1,
            val=True,
            split="val",
            plots=True,
            lr0=0.003,
            lrf=0.02,
            momentum=0.937,
            weight_decay=0.001,
            warmup_epochs=4.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.05,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            nbs=64,
            hsv_h=0.01,
            hsv_s=0.35,
            hsv_v=0.25,
            degrees=0.0,
            translate=0.08,
            scale=0.35,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            bgr=0.0,
            mosaic=0.6,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
        )

        best_path = run_dir / "weights" / "best.pt"
        checkpoint = verify_checkpoint(best_path)
        holdout_metrics = evaluate_holdout(best_path, project_dir)
        lock_payload.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "best_checkpoint": checkpoint,
                "holdout_metrics": holdout_metrics,
            }
        )
        write_json_atomic(lock_path, lock_payload)
        completed_path = project_dir / TRAIN_COMPLETE_NAME
        os.replace(str(lock_path), str(completed_path))
        print("One-shot training completed successfully.")
        print("Deliverable: {}".format(best_path))
        print("Hold-out metrics: {}".format(run_dir / "holdout_metrics.json"))
    except BaseException as exc:
        lock_payload.update(
            {
                "status": "failed_or_interrupted",
                "failed_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc()[-8000:],
            }
        )
        write_json_atomic(lock_path, lock_payload)
        raise


# ============================================================================
# CLI
# ============================================================================

def show_status(project_dir: Path) -> None:
    dataset_dir = project_dir / "dataset"
    ready_path = dataset_dir / READY_FILE_NAME
    lock_path = project_dir / TRAIN_LOCK_NAME
    complete_path = project_dir / TRAIN_COMPLETE_NAME
    print("Project: {}".format(project_dir))
    print("Dataset ready: {}".format(ready_path.is_file()))
    if ready_path.is_file():
        ready = read_json(ready_path)
        print("Dataset digest: {}".format(ready.get("dataset_sha256")))
    print("Training lock: {}".format(lock_path if lock_path.exists() else "absent"))
    print("Training complete: {}".format(complete_path if complete_path.exists() else "no"))
    print(
        "Expected best.pt: {}".format(
            project_dir / RUN_PROJECT_REL / RUN_NAME / "weights" / "best.pt"
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-resistant dataset generation and guarded one-shot YOLO training."
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory. Defaults to the script directory or its workspace/ child.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate, QA and atomically promote dataset.")
    generate_parser.add_argument(
        "--workers",
        type=int,
        default=min(multiprocessing.cpu_count(), 16),
        help="CPU worker count; worker state is initialized once.",
    )

    subparsers.add_parser("qa", help="Re-run strict QA and refresh the dataset integrity marker.")

    train_parser = subparsers.add_parser("train", help="Run the guarded one-shot training on the approved host.")
    train_parser.add_argument("--confirm-train", default="")

    subparsers.add_parser("status", help="Show dataset and one-shot training state.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_dir = resolve_project_dir(args.project_dir)
    if not project_dir.is_dir():
        raise PipelineError("Project directory does not exist: {}".format(project_dir))

    if args.command == "generate":
        if args.workers < 1 or args.workers > 32:
            raise PipelineError("--workers must be between 1 and 32.")
        generate_dataset(project_dir, args.workers)
    elif args.command == "qa":
        run_dataset_qa(project_dir / "dataset")
    elif args.command == "train":
        train_once(project_dir, args.confirm_train)
    elif args.command == "status":
        show_status(project_dir)
    else:
        parser.error("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except PipelineError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)

