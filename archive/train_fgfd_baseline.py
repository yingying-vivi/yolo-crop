from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd/data.yaml"
EPOCHS = 100
IMGSZ = 512
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"


def train_baseline():
    print(f"\n{'='*60}")
    print(f"[FGFD] Training YOLO11 baseline (pretrained)")
    print(f"{'='*60}\n")
    model = YOLO(PRETRAINED)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment",
        name="fgfd_baseline",
        exist_ok=True,
        workers=8,
        seed=42,
    )
    return results


if __name__ == "__main__":
    train_baseline()