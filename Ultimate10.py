import os
from pathlib import Path

from ultralytics import YOLO


def resolve_project_dir():
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "dataset" / "dataset.yaml").exists():
        return script_dir
    workspace_dir = script_dir / "workspace"
    if (workspace_dir / "dataset" / "dataset.yaml").exists():
        return workspace_dir
    return script_dir


def resolve_model(project_dir):
    for candidate in ["yolo12n.pt", "yolo12n", "yolo11n.pt"]:
        path = project_dir / candidate
        if path.exists():
            return str(path)
        if candidate == "yolo12n":
            return candidate
    return "yolo12n"


def main():
    project_dir = resolve_project_dir()
    os.chdir(project_dir)

    model = YOLO(resolve_model(project_dir))
    model.train(
        data="./dataset/dataset.yaml",
        epochs=220,
        patience=35,
        batch=64,
        imgsz=640,
        save=True,
        save_period=10,
        cache=False,
        device="0",
        workers=8,
        project="runs/detect",
        name="escherichia_train_u10",
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=25,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.05,
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
        pose=12.0,
        kobj=1.0,
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
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.2,
        tracker="botsort.yaml",
    )


if __name__ == "__main__":
    main()
