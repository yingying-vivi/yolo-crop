import os
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage
import cv2

SRC_DIR = Path("/home/fumu/datadisk/split_ftw_4")
OUT_DIR = Path("/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/ftw_stitched")

MIN_INSTANCE_AREA = 200
NC = 1
CLASS_NAME = "field"
GRID_N = 2
TILE_SIZE = 256
STITCHED_SIZE = GRID_N * TILE_SIZE


def parse_filename(stem):
    parts = stem.split("_")
    grid_id = parts[0]
    parent_row = int(parts[1])
    col_idx = int(parts[2])
    return grid_id, parent_row, col_idx


def build_tile_index(split):
    img_dir = SRC_DIR / split / "img"
    lbl_dir = SRC_DIR / split / "label"
    img_set = {f.replace(".tif", "") for f in os.listdir(img_dir) if f.endswith(".tif")}
    lbl_set = {f.replace(".png", "") for f in os.listdir(lbl_dir) if f.endswith(".png")}
    valid_stems = img_set & lbl_set
    index = {}
    for stem in valid_stems:
        grid_id, parent_row, col_idx = parse_filename(stem)
        index[(grid_id, parent_row, col_idx)] = stem
    return index


def convert_image_rgba_to_rgb(arr_rgba):
    rgb = arr_rgba[:, :, [3, 1, 0]].astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax > vmin:
            rgb[:, :, c] = (ch - vmin) / (vmax - vmin) * 255.0
        else:
            rgb[:, :, c] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def convert_label(mask, img_h, img_w, out_txt_path):
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


def process_stitched_group(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir,
                           stems, parent_row_tl, col, grid_id):
    imgs_rgba = []
    labels = []
    for stem in stems:
        tif_path = src_img_dir / f"{stem}.tif"
        lbl_path = src_lbl_dir / f"{stem}.png"
        imgs_rgba.append(np.array(Image.open(tif_path)))
        labels.append(np.array(Image.open(lbl_path)))

    stitched_rgba = np.zeros((STITCHED_SIZE, STITCHED_SIZE, 4), dtype=np.uint8)
    stitched_rgba[0:TILE_SIZE, 0:TILE_SIZE, :] = imgs_rgba[0]
    stitched_rgba[0:TILE_SIZE, TILE_SIZE:STITCHED_SIZE, :] = imgs_rgba[1]
    stitched_rgba[TILE_SIZE:STITCHED_SIZE, 0:TILE_SIZE, :] = imgs_rgba[2]
    stitched_rgba[TILE_SIZE:STITCHED_SIZE, TILE_SIZE:STITCHED_SIZE, :] = imgs_rgba[3]

    out_name = f"{grid_id}_r{parent_row_tl}_c{col}_2x2"
    out_png = out_img_dir / f"{out_name}.png"
    out_txt = out_lbl_dir / f"{out_name}.txt"

    stitched_rgb = convert_image_rgba_to_rgb(stitched_rgba)
    Image.fromarray(stitched_rgb).save(out_png)

    stitched_label = np.zeros((STITCHED_SIZE, STITCHED_SIZE), dtype=np.uint8)
    stitched_label[0:TILE_SIZE, 0:TILE_SIZE] = labels[0]
    stitched_label[0:TILE_SIZE, TILE_SIZE:STITCHED_SIZE] = labels[1]
    stitched_label[TILE_SIZE:STITCHED_SIZE, 0:TILE_SIZE] = labels[2]
    stitched_label[TILE_SIZE:STITCHED_SIZE, TILE_SIZE:STITCHED_SIZE] = labels[3]

    n = convert_label(stitched_label, STITCHED_SIZE, STITCHED_SIZE, out_txt)
    return n


def process_single_tile(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, stem):
    tif_path = src_img_dir / f"{stem}.tif"
    lbl_path = src_lbl_dir / f"{stem}.png"

    img_arr = np.array(Image.open(tif_path))
    lbl_arr = np.array(Image.open(lbl_path))

    out_name = stem
    out_png = out_img_dir / f"{out_name}.png"
    out_txt = out_lbl_dir / f"{out_name}.txt"

    rgb = convert_image_rgba_to_rgb(img_arr)
    Image.fromarray(rgb).save(out_png)

    img_w, img_h = Image.open(lbl_path).size
    n = convert_label(lbl_arr, img_h, img_w, out_txt)
    return n


def stitch_split(split):
    src_img_dir = SRC_DIR / split / "img"
    src_lbl_dir = SRC_DIR / split / "label"
    out_img_dir = OUT_DIR / "images" / split
    out_lbl_dir = OUT_DIR / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    index = build_tile_index(split)
    grid_ids = sorted(set(k[0] for k in index))

    used_tiles = set()
    total_stitched = 0
    total_single = 0
    total_instances = 0

    for grid_id in grid_ids:
        rows_by_grid = sorted(set(k[1] for k in index if k[0] == grid_id))

        for r_idx in range(len(rows_by_grid) - 1):
            parent_row_tl = rows_by_grid[r_idx]
            parent_row_bl = rows_by_grid[r_idx + 1]
            if parent_row_bl != parent_row_tl + 1:
                continue
            for col in range(19 - 1):
                positions = [
                    (grid_id, parent_row_tl, col),
                    (grid_id, parent_row_tl, col + 1),
                    (grid_id, parent_row_bl, col),
                    (grid_id, parent_row_bl, col + 1),
                ]
                stems = [index.get(p) for p in positions]
                if not all(stems):
                    continue

                for p in positions:
                    used_tiles.add(p)

                n = process_stitched_group(
                    src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir,
                    stems, parent_row_tl, col, grid_id,
                )
                total_stitched += 1
                total_instances += n

    for key, stem in index.items():
        if key not in used_tiles:
            n = process_single_tile(
                src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, stem,
            )
            total_single += 1
            total_instances += n

    print(f"[{split}] {total_stitched} stitched 2x2 + {total_single} single = "
          f"{total_stitched + total_single} total, {total_instances} instances")
    return total_stitched, total_single


def main():
    counts = {}
    for split in ["train", "val", "test"]:
        counts[split] = stitch_split(split)

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
    print(f"\nResults per split:")
    for split, (stitched, single) in counts.items():
        print(f"  {split}: {stitched} stitched 2x2 + {single} single = {stitched + single} total")


if __name__ == "__main__":
    main()