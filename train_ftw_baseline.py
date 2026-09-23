from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/ftw_stitched/data.yaml"
EPOCHS = 200
IMGSZ = 512
BATCH = 32
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"


def train_ftw_baseline():
    model = YOLO(PRETRAINED)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="ftw_baseline_v3",
        exist_ok=True,
        workers=8,
        seed=42,
        freeze=0,
        mosaic=0.0,
        close_mosaic=0,
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
    train_ftw_baseline()
