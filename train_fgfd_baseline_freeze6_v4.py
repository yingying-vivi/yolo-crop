from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"


def train_baseline_freeze6_v4():
    print(f"\n{'='*60}")
    print(f"[FGFD] Baseline freeze=6 (freeze shallow 6 layers, standard C3k2)")
    print(f"  Control experiment for star-deep freeze=6 comparison")
    print(f"{'='*60}\n")

    model = YOLO(PRETRAINED)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_baseline_freeze6_v4",
        exist_ok=True,
        workers=8,
        seed=42,
        freeze=6,
        mosaic=0.5,
        close_mosaic=20,
        mixup=0.0,
        cutmix=0.0,
        auto_augment=None,
        erasing=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        warmup_bias_lr=0.01,
        patience=200,
        dropout=0.2,
        overlap_mask=True,
        mask_ratio=4,
        weight_decay=0.001,
    )
    return results


if __name__ == "__main__":
    train_baseline_freeze6_v4()