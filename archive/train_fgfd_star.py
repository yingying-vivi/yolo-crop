from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd/data.yaml"
EPOCHS = 100
IMGSZ = 512
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
MODEL_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-star-seg.yaml"


def train_star():
    print(f"\n{'='*60}")
    print(f"[FGFD] Training YOLO11-StarBlock (pretrained)")
    print(f"{'='*60}\n")
    model = YOLO(MODEL_YAML)
    model.load(PRETRAINED)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment",
        name="fgfd_star",
        exist_ok=True,
        workers=8,
        seed=42,
    )
    return results


if __name__ == "__main__":
    train_star()