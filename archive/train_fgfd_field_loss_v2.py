from ultralytics import YOLO
from ultralytics.nn.tasks import FieldSegmentationModel
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from copy import copy

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls/data.yaml"
EPOCHS = 200
IMGSZ = 512
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
BASELINE_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-seg.yaml"
PROJECT = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment"

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


def train_field_loss_v2():
    print(f"\n{'='*60}")
    print(f"[FGFD] Training YOLO11-FieldLoss v2")
    print(f"{'='*60}\n")
    model = YOLO(BASELINE_YAML)
    model.load(PRETRAINED)
    model.trainer = FieldSegmentationTrainer
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name="fgfd_field_loss_v2",
        exist_ok=True,
        workers=8,
        seed=42,
        **AUG_ARGS,
        **TRAIN_ARGS,
    )
    return results


if __name__ == "__main__":
    train_field_loss_v2()