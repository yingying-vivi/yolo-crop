from ultralytics import YOLO
import torch
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.nn.tasks import FieldSegmentationModel
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from copy import copy

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd/data.yaml"
EPOCHS = 100
IMGSZ = 512
BATCH = 16
DEVICE = 0
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"

PRETRAINED = "yolo11n-seg.pt"
STAR_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-star-seg.yaml"
ECA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-eca-seg.yaml"
BASELINE_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-seg.yaml"

SHIFT_MAP = {17: 18, 18: 19, 19: 20, 20: 22, 21: 23, 22: 24, 23: 26}


class FieldSegmentationTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = FieldSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "biou_loss", "bmask_loss"
        from ultralytics.models.yolo.segment import SegmentationValidator
        return SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
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


def run_experiment(name, model, **kwargs):
    print(f"\n{'='*60}")
    print(f"[FGFD] Starting experiment: {name}")
    print(f"{'='*60}\n")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=name,
        exist_ok=True,
        workers=8,
        seed=42,
        **kwargs,
    )
    print(f"\n[FGFD] Experiment '{name}' complete!")
    return results


EXPERIMENTS = {
    "fgfd_baseline": lambda: run_experiment("fgfd_baseline", YOLO(PRETRAINED)),
    "fgfd_star": lambda: run_experiment("fgfd_star", (m := YOLO(STAR_YAML), m.load(PRETRAINED), m)[2]),
    "fgfd_eca": lambda: run_experiment("fgfd_eca", (m := YOLO(ECA_YAML), load_with_shift(m.model, PRETRAINED, SHIFT_MAP), m)[2]),
    "fgfd_field_loss": lambda: run_experiment(
        "fgfd_field_loss",
        (m := YOLO(BASELINE_YAML), m.load(PRETRAINED), setattr(m, "trainer", FieldSegmentationTrainer), m)[3],
    ),
}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        if exp_name in EXPERIMENTS:
            EXPERIMENTS[exp_name]()
        else:
            print(f"Unknown experiment: {exp_name}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
    else:
        for name, func in EXPERIMENTS.items():
            func()