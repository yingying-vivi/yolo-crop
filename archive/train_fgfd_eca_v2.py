from ultralytics import YOLO
import torch
from ultralytics.utils.torch_utils import intersect_dicts

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 512
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
ECA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-eca-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"

SHIFT_MAP = {17: 18, 18: 19, 19: 20, 20: 22, 21: 23, 22: 24, 23: 26}

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
    return model


def train_eca_v2():
    print(f"\n{'='*60}")
    print(f"[FGFD] Training YOLO11-ECA v2")
    print(f"{'='*60}\n")
    model = YOLO(ECA_YAML)
    load_with_shift(model.model, PRETRAINED, SHIFT_MAP)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_eca_v2",
        exist_ok=True,
        workers=8,
        seed=42,
        **AUG_ARGS,
        **TRAIN_ARGS,
    )
    return results


if __name__ == "__main__":
    train_eca_v2()