import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

FGFD_ROOT = Path("/home/fumu/conda_disk/baiduwangpan/Fine-Grained Farmland Dataset ")
OUTPUT_ROOT = Path("/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/fgfd_1cls")

MIN_AREA = 100
APPROX_EPSILON_RATIO = 0.003
MAX_POINTS = 50


def mask_to_polygons(mask, epsilon_ratio=APPROX_EPSILON_RATIO):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue
        if len(points) > MAX_POINTS:
            step = max(1, len(points) // MAX_POINTS)
            points = points[::step]
        polygons.append(points)
    return polygons


def convert_label(label_path, img_h, img_w):
    label_img = cv2.imread(str(label_path))
    if label_img is None:
        return []

    farmland_mask = ((label_img[:, :, 1] == 128) | (label_img[:, :, 2] == 128)).astype(np.uint8) * 255

    lines = []
    polygons = mask_to_polygons(farmland_mask)
    for poly in polygons:
        coords = []
        for x, y in poly:
            coords.extend([x / img_w, y / img_h])
        lines.append("0 " + " ".join(f"{c:.6f}" for c in coords))

    return lines


def convert_split(split_name):
    src_img_dir = FGFD_ROOT / split_name / "img"
    src_lbl_dir = FGFD_ROOT / split_name / "label"

    dst_img_dir = OUTPUT_ROOT / "images" / split_name
    dst_lbl_dir = OUTPUT_ROOT / "labels" / split_name
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(os.listdir(src_img_dir))
    total = len(img_files)
    converted = 0
    total_instances = 0

    for i, img_file in enumerate(img_files):
        img_path = src_img_dir / img_file
        lbl_file = img_file.replace("_image_", "_label_").replace(".tif", ".png")
        lbl_path = src_lbl_dir / lbl_file

        if not lbl_path.exists():
            continue

        pil_img = Image.open(str(img_path))
        img_w, img_h = pil_img.size

        dst_img_file = img_file.replace(".tif", ".jpg")
        pil_img.save(str(dst_img_dir / dst_img_file), "JPEG", quality=95)

        lines = convert_label(str(lbl_path), img_h, img_w)

        dst_lbl_file = dst_img_file.replace(".jpg", ".txt")
        with open(dst_lbl_dir / dst_lbl_file, "w") as f:
            f.write("\n".join(lines))

        total_instances += len(lines)
        converted += 1

        if (i + 1) % 500 == 0:
            print(f"  [{split_name}] {i+1}/{total} processed...")

    print(f"  [{split_name}] Done: {converted} converted, {total_instances} instances")
    return converted, total_instances


def main():
    print("Converting FGFD to YOLO format (nc=1, farmland vs background)...")

    total_converted = 0
    total_instances = 0
    for split in ["train", "val", "test"]:
        n, ni = convert_split(split)
        total_converted += n
        total_instances += ni

    data_yaml = OUTPUT_ROOT / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUTPUT_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 1\n")
        f.write("names: ['farmland']\n")

    print(f"\nTotal: {total_converted} images, {total_instances} instances")
    print(f"data.yaml written to {data_yaml}")


if __name__ == "__main__":
    main()