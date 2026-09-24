from ultralytics import YOLO

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/paddy_field/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0

MODELS = {
    "yolov8": "yolov8n-seg.pt",
    # "yolov11": "yolo11n-seg.pt",
    # "yolov26": "yolo26n-seg.pt",
}


def train_model(name, model_path):
    print(f"\n{'=' * 60}")
    print(f"Training {name} ({model_path})")
    print(f"{'=' * 60}\n")
    model = YOLO(model_path)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment",
        name=name,
        exist_ok=True,
        workers=8,
        seed=42,
    )
    print(f"\n{name} training complete!")
    print(f"Results saved to: {name}/")
    return results


if __name__ == "__main__":
    for name, model_path in MODELS.items():
        train_model(name, model_path)
