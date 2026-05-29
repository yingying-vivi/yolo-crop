import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo26n-seg.pt"
YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/26/yolo26-star-deep-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"


def load_with_partial(model, pretrained_path):
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    pretrained_sd = ckpt["model"].float().state_dict()
    model_sd = model.state_dict()

    matched = intersect_dicts(pretrained_sd, model_sd)
    model.load_state_dict(matched, strict=False)

    n_matched = len(matched)
    total = len(model_sd)
    print(f"Weight loading: {n_matched}/{total} ({n_matched/total*100:.1f}%)")
    return model


def train_fgfd_v26_star_deep():
    print(f"\n{'='*60}")
    print(f"[FGFD] YOLOv26 ablation: StarBlock deep backbone only")
    print(f"  Backbone layers 6,8: C3k2_Star (StarBlock, shortcut=False)")
    print(f"  Head: standard v26-seg (no ECA, no custom loss)")
    print(f"  end2end=True, reg_max=1")
    print(f"{'='*60}\n")

    model = YOLO(YAML)
    load_with_partial(model.model, PRETRAINED)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_v26_star_deep",
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
    train_fgfd_v26_star_deep()