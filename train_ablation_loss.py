import torch
from ultralytics.nn.tasks import SizeAwareMaskModel
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import intersect_dicts
from copy import copy

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 640
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"


class LossTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SizeAwareMaskModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            if isinstance(weights, str):
                ckpt = torch.load(weights, map_location="cpu", weights_only=False)
                pretrained_sd = ckpt["model"].float().state_dict()
            else:
                pretrained_sd = weights.float().state_dict()
            model_sd = model.state_dict()
            matched = intersect_dicts(pretrained_sd, model_sd)
            model.load_state_dict(matched, strict=False)
            n_matched = len(matched)
            total = len(model_sd)
            print(f"Weight loading: {n_matched}/{total} ({n_matched/total*100:.1f}%)")
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        from ultralytics.models.yolo.segment import SegmentationValidator
        return SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


def train_ablation_loss():
    args = dict(
        model=YAML,
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="ablation_loss",
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
        pretrained=PRETRAINED,
    )

    trainer = LossTrainer(overrides=args)
    results = trainer.train()
    return results


if __name__ == "__main__":
    train_ablation_loss()