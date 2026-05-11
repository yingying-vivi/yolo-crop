from ultralytics import YOLO
import torch
from ultralytics.utils.torch_utils import intersect_dicts

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-star-deep-eca-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"

SHIFT_MAP = {
    19: 20,
    22: 24,
    23: 26,
}


def load_with_shift(model, pretrained_path, shift_map):
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    pretrained_sd = ckpt["model"].float().state_dict()
    model_sd = model.state_dict()

    matched = intersect_dicts(pretrained_sd, model_sd)
    model.load_state_dict(matched, strict=False)
    n_direct = len(matched)

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
        n_matched = n_direct + len(remapped)
    else:
        n_matched = n_direct

    total = len(model_sd)
    print(f"Weight loading: {n_matched}/{total} ({n_matched/total*100:.1f}%)")
    print(f"  Direct: {n_direct}, Shift-remapped: {len(remapped)}")
    return model


def train_star_deep_eca_v4():
    print(f"\n{'='*60}")
    print(f"[FGFD] Star-Deep + ECA (freeze=0)")
    print(f"  Backbone: shallow C3k2 (pretrained), deep C3k2_Star (StarBlock)")
    print(f"  Head: C3k2 + ECA on P3/P4/P5")
    print(f"  freeze=0 so StarBlock and ECA can train")
    print(f"{'='*60}\n")

    model = YOLO(YAML)
    load_with_shift(model.model, PRETRAINED, SHIFT_MAP)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_star_deep_eca_v4",
        exist_ok=True,
        workers=8,
        seed=42,
        freeze=0,
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
    train_star_deep_eca_v4()