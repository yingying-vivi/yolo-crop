from ultralytics import YOLO
from ultralytics.nn.tasks import FieldSegmentationModel
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from copy import copy
from pathlib import Path

DATA_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd/data.yaml"
EPOCHS = 100
IMGSZ = 512
BATCH = 16
DEVICE = 0

PRETRAINED = "yolo11n-seg.pt"
MODEL_YAML = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/ultralytics/cfg/models/11/yolo11-seg.yaml"


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


if __name__ == "__main__":
    model = YOLO(MODEL_YAML)
    model.load(PRETRAINED)
    model.trainer = FieldSegmentationTrainer
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="/home/fumu/xyy/ultralytics-crop/ultralytics-crop/runs/segment",
        name="fgfd_field_loss",
        exist_ok=True,
        workers=8,
        seed=42,
    )