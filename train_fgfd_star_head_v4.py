from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
STAR_HEAD_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-star-head-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"

AUG_ARGS = dict(
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
)

TRAIN_ARGS = dict(
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


def train_star_head():
    print(f"\n{'='*60}")
    print(f"[FGFD] Training YOLO11-StarHead (backbone standard, head StarBlock)")
    print(f"{'='*60}\n")
    model = YOLO(STAR_HEAD_YAML)
    model.load(PRETRAINED)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_star_head_v4",
        exist_ok=True,
        workers=8,
        seed=42,
        freeze=10,
        **AUG_ARGS,
        **TRAIN_ARGS,
    )
    return results


if __name__ == "__main__":
    train_star_head()