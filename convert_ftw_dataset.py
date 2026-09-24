from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

SRC_DIR = Path("/home/fumu/datadisk/split_ftw_4")
OUT_DIR = Path("/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/ftw")

MIN_INSTANCE_AREA = 200
NC = 1
CLASS_NAME = "field"


def convert_image(tif_path, out_png_path):
    arr = np.array(Image.open(tif_path))
    rgb = arr[:, :, [3, 1, 0]].astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax > vmin:
            rgb[:, :, c] = (ch - vmin) / (vmax - vmin) * 255.0
        else:
            rgb[:, :, c] = 0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_png_path)


def convert_label(mask_path, out_txt_path, img_h, img_w):
    mask = np.array(Image.open(mask_path))
    labeled, num_instances = ndimage.label(mask)
    lines = []
    kept = 0
    for inst_id in range(1, num_instances + 1):
        inst_mask = (labeled == inst_id).astype(np.uint8)
        area = inst_mask.sum()
        if area < MIN_INSTANCE_AREA:
            continue
        kept += 1
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = []
        for pt in approx:
            px = pt[0][0] / img_w
            py = pt[0][1] / img_h
            points.append(f"{px:.6f}")
            points.append(f"{py:.6f}")
        line = f"{0} " + " ".join(points)
        lines.append(line)
    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines))
    return kept


def convert_split(split):
    src_img_dir = SRC_DIR / split / "img"
    src_lbl_dir = SRC_DIR / split / "label"
    out_img_dir = OUT_DIR / "images" / split
    out_lbl_dir = OUT_DIR / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(src_img_dir.glob("*.tif"))
    total_instances = 0
    for tif_path in tif_files:
        stem = tif_path.stem
        png_path = out_img_dir / f"{stem}.png"
        txt_path = out_lbl_dir / f"{stem}.txt"
        mask_path = src_lbl_dir / f"{stem}.png"

        convert_image(tif_path, png_path)

        mask_pil = Image.open(mask_path)
        img_w, img_h = mask_pil.size
        n = convert_label(mask_path, txt_path, img_h, img_w)
        total_instances += n

    print(f"[{split}] {len(tif_files)} images, {total_instances} instances")


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)

    data_yaml = f"""path: {OUT_DIR}
train: images/train
val: images/val
test: images/test
nc: {NC}
names: ['{CLASS_NAME}']
"""
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml)
    print(f"\ndata.yaml saved to {yaml_path}")


if __name__ == "__main__":
    main()
