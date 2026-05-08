from ultralytics import YOLO
import torch
from ultralytics.utils.torch_utils import intersect_dicts

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/paddy_field/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
MODEL_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-eca-seg.yaml"

SHIFT_MAP = {
    17: 18,
    18: 19,
    19: 20,
    20: 22,
    21: 23,
    22: 24,
    23: 26,
}


def load_with_shift(model, pretrained_path, shift_map, verbose=True):
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    pretrained_sd = ckpt["model"].float().state_dict()
    model_sd = model.state_dict()

    matched = intersect_dicts(pretrained_sd, model_sd)
    model.load_state_dict(matched, strict=False)
    n_matched = len(matched)

    remapped = {}
    for old_idx, new_idx in shift_map.items():
        old_prefix = f"model.{old_idx}."
        new_prefix = f"model.{new_idx}."
        for key, value in pretrained_sd.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in model_sd and model_sd[new_key].shape == value.shape:
                    remapped[new_key] = value

    if remapped:
        model.load_state_dict(remapped, strict=False)
        n_matched += len(remapped)

    if verbose:
        print(f"Transferred {n_matched}/{len(model_sd)} items from pretrained weights")
        print(f"  Direct match: {len(matched)}, Shift-remapped: {len(remapped)}")
    return model


def train_eca_model():
    print(f"\n{'='*60}")
    print(f"Training YOLO11-ECA with pretrained weights (shift-remap)")
    print(f"{'='*60}\n")
    model = YOLO(MODEL_YAML)
    load_with_shift(model.model, PRETRAINED, SHIFT_MAP)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment",
        name="yolov11_eca_pretrained",
        exist_ok=True,
        workers=8,
        seed=42,
    )
    print(f"\nYOLO11-ECA (pretrained) training complete!")
    return results


if __name__ == "__main__":
    train_eca_model()