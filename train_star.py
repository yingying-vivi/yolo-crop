from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/paddy_field/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0

MODEL_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-star-seg.yaml"


def train_star_model():
    print(f"\n{'='*60}")
    print(f"Training YOLO11-StarBlock (backbone replaced)")
    print(f"{'='*60}\n")
    model = YOLO(MODEL_YAML)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment",
        name="yolov11_star",
        exist_ok=True,
        workers=8,
        seed=42,
    )
    print(f"\nYOLO11-StarBlock training complete!")
    return results


if __name__ == "__main__":
    train_star_model()