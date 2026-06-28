import os
import random
import shutil
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


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


@dataclass(frozen=True)
class ForegroundRecord:
    image_path: str
    label_path: str
    class_name: str
    class_id: int


def resolve_project_dir():
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "input_data").exists():
        return script_dir
    workspace_dir = script_dir / "workspace"
    if (workspace_dir / "input_data").exists():
        return workspace_dir
    return script_dir


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def load_class_names(label_dir):
    classes_path = label_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class file: {classes_path}")
    with classes_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_backgrounds(background_dir):
    if not background_dir.exists():
        raise FileNotFoundError(f"Missing background directory: {background_dir}")

    backgrounds = []
    for path in sorted(background_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        backgrounds.append(image)

    if not backgrounds:
        raise RuntimeError(f"No usable background images found: {background_dir}")
    return backgrounds


def load_foregrounds(input_dir, class_names):
    label_dir = input_dir / "label"
    records = []

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in {"background", "label", "random"}:
            continue
        if class_dir.name not in class_names:
            continue

        folder_class_id = class_names.index(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            records.append(
                ForegroundRecord(
                    image_path=str(image_path),
                    label_path=str(label_path),
                    class_name=class_dir.name,
                    class_id=folder_class_id,
                )
            )

    if not records:
        raise RuntimeError(f"No labeled foreground images found in {input_dir}")
    return records


def split_records_by_source(records, train_ratio):
    by_class = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)

    train_records = []
    val_records = []
    rng = random.Random(SEED)

    for class_id, class_records in sorted(by_class.items()):
        shuffled = list(class_records)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_records.extend(shuffled)
            val_records.extend(shuffled)
            continue

        val_count = max(1, int(round(len(shuffled) * (1.0 - train_ratio))))
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])

    if not train_records or not val_records:
        raise RuntimeError("Train/validation split failed; check source image counts.")
    return train_records, val_records


def split_backgrounds(backgrounds, train_ratio):
    if len(backgrounds) < 2:
        return backgrounds, backgrounds

    indices = list(range(len(backgrounds)))
    random.Random(SEED).shuffle(indices)
    val_count = max(1, int(round(len(indices) * (1.0 - train_ratio))))
    val_indices = set(indices[:val_count])

    train_backgrounds = [bg for i, bg in enumerate(backgrounds) if i not in val_indices]
    val_backgrounds = [bg for i, bg in enumerate(backgrounds) if i in val_indices]
    return train_backgrounds or backgrounds, val_backgrounds or backgrounds


def prepare_output_dirs(output_dir):
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        path = output_dir / rel
        if CLEAR_OUTPUT and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def read_yolo_labels(label_path, expected_class_id):
    labels = []
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
            if class_id != expected_class_id:
                class_id = expected_class_id
            labels.append((class_id, bbox))
    return labels


def ensure_bgra(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image


def rotate_and_scale_image(image, angle, scale):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

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


def fit_foreground_to_background(foreground, bbox_list, bg_w, bg_h):
    fg_h, fg_w = foreground.shape[:2]
    max_w = int(bg_w * 0.9)
    max_h = int(bg_h * 0.9)

    if fg_w <= max_w and fg_h <= max_h:
        return foreground, bbox_list

    factor = min(max_w / max(1, fg_w), max_h / max(1, fg_h))
    new_w = max(1, int(fg_w * factor))
    new_h = max(1, int(fg_h * factor))
    resized = cv2.resize(foreground, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, bbox_list


def blend_with_alpha(background, foreground, x, y):
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


def add_train_only_background_noise(background):
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


def overlaps_any(new_rect, placed_rects):
    for rect in placed_rects:
        if not (
            new_rect[2] < rect[0]
            or new_rect[0] > rect[2]
            or new_rect[3] < rect[1]
            or new_rect[1] > rect[3]
        ):
            return True
    return False


def make_one_sample(split, index, backgrounds, records, output_dir):
    seed_everything(SEED + index + (0 if split == "train" else 1_000_000))
    bg = random.choice(backgrounds).copy()
    if split == "train":
        bg = add_train_only_background_noise(bg)

    bg_h, bg_w = bg.shape[:2]
    placed_rects = []
    output_labels = []
    num_objects = random.randint(1, min(MAX_OBJECTS_PER_IMAGE, len(records)))

    for _ in range(num_objects):
        record = random.choice(records)
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

    stem = f"u9_{split}_{index:06d}"
    image_path = output_dir / "images" / split / f"{stem}.jpg"
    label_path = output_dir / "labels" / split / f"{stem}.txt"

    cv2.imwrite(str(image_path), bg)
    with label_path.open("w", encoding="utf-8") as f:
        f.writelines(output_labels)

    return str(image_path)


def worker(args):
    return make_one_sample(*args)


def generate_split(split, count, backgrounds, records, output_dir):
    args = [(split, i, backgrounds, records, output_dir) for i in range(count)]
    cpu_count = min(multiprocessing.cpu_count(), 16)
    written = []

    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(
            pool.imap_unordered(worker, args),
            total=count,
            desc=f"Generating {split}",
        ):
            if result:
                written.append(result)

    return written


def write_dataset_files(output_dir, class_names):
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


def main():
    seed_everything(SEED)
    project_dir = resolve_project_dir()
    input_dir = project_dir / "input_data"
    output_dir = project_dir / "dataset"

    print(f"Project: {project_dir}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    class_names = load_class_names(input_dir / "label")
    backgrounds = load_backgrounds(input_dir / "background")
    records = load_foregrounds(input_dir, class_names)

    train_records, val_records = split_records_by_source(records, TRAIN_RATIO)
    train_backgrounds, val_backgrounds = split_backgrounds(backgrounds, TRAIN_RATIO)

    prepare_output_dirs(output_dir)
    train_written = generate_split(
        "train", TRAIN_IMAGES, train_backgrounds, train_records, output_dir
    )
    val_written = generate_split("val", VAL_IMAGES, val_backgrounds, val_records, output_dir)
    write_dataset_files(output_dir, class_names)

    print(f"Train sources: {len(train_records)}")
    print(f"Val sources: {len(val_records)}")
    print(f"Train images: {len(train_written)}")
    print(f"Val images: {len(val_written)}")
    print("Generation complete.")


if __name__ == "__main__":
    main()
