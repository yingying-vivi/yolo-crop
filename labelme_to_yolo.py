import glob
import json
import os
import random
import shutil

LABELME_DIR = (
    "/home/fumu/conda_disk/baiduwangpan/无人机视角空田水田稻田识别分割数据集labelme格式528张3类别/labelme_data"
)
OUTPUT_DIR = "/home/fumu/xyy/ultralytics-crop/ultralytics-crop/datasets/paddy_field"

CLASS_MAP = {"empty_field": 0, "flooded_field": 1, "rice_field": 2}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)


def convert_labelme_to_yolo(json_path, output_label_path):
    data = json.load(open(json_path))
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[label]
        if shape["shape_type"] == "polygon":
            points = shape["points"]
            normalized_points = []
            for px, py in points:
                normalized_points.extend([px / img_w, py / img_h])
            line = f"{cls_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
            lines.append(line)
    with open(output_label_path, "w") as f:
        f.write("\n".join(lines))


def main():
    json_files = sorted(glob.glob(os.path.join(LABELME_DIR, "*.json")))
    print(f"Found {len(json_files)} JSON files")

    names = [os.path.splitext(os.path.basename(f))[0] for f in json_files]

    random.shuffle(names)
    n = len(names)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n - n_train - n_val

    train_names = names[:n_train]
    val_names = names[n_train : n_train + n_val]
    test_names = names[n_train + n_val :]

    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    splits = {"train": train_names, "val": val_names, "test": test_names}

    for split, split_names in splits.items():
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for name in split_names:
            src_img = os.path.join(LABELME_DIR, name + ".jpg")
            src_json = os.path.join(LABELME_DIR, name + ".json")
            dst_img = os.path.join(img_dir, name + ".jpg")
            dst_lbl = os.path.join(lbl_dir, name + ".txt")

            if not os.path.exists(src_img):
                print(f"Warning: image {src_img} not found, skipping")
                continue

            shutil.copy2(src_img, dst_img)
            convert_labelme_to_yolo(src_json, dst_lbl)

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(CLASS_MAP)}\n")
        f.write(f"names: {list(CLASS_MAP.keys())}\n")

    print(f"Dataset YAML saved to {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
