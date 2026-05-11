from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 512
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
STAR_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-star-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"

AUG_ARGS = dict(
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    auto_augment=None,
    erasing=0.0,
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.2,
    degrees=0.0,
    translate=0.05,
    scale=0.3,
    fliplr=0.5,
    flipud=0.5,
    close_mosaic=0,
)

TRAIN_ARGS = dict(
    lr0=0.005,
    lrf=0.01,
    warmup_epochs=5,
    patience=50,
    dropout=0.1,
    overlap_mask=True,
    mask_ratio=4,
)


def train_star_v2():
    print(f"\n{'='*60}")
    print(f"[FGFD] Training YOLO11-StarBlock v3 (no internal shortcut)")
    print(f"{'='*60}\n")
    model = YOLO(STAR_YAML)
    model.load(PRETRAINED)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_star_v3",
        exist_ok=True,
        workers=8,
        seed=42,
        **AUG_ARGS,
        **TRAIN_ARGS,
    )
    return results


if __name__ == "__main__":
    train_star_v2()