#!/usr/bin/env python3
"""
YOLO format dataset → COCO JSON format converter.

Source : /home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml
Output : /home/hsjeong/workspace/VIT/RepVIT/detection/data/qnsfl/annotations/
           ├── instances_train.json
           ├── instances_val.json
           └── instances_test.json

Image paths are stored as absolute paths inside JSON (direct-path mode).
img_prefix should be set to '' in the mmdet config.

YOLO label format : <class_id> <cx> <cy> <w> <h>  (normalised 0-1)
COCO bbox format  : [x_min, y_min, width, height]   (absolute pixels)
Category IDs      : YOLO 0-indexed → COCO 1-indexed
"""

import os
import json
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
YAML_PATH  = "/home/hsjeong/workspace/Yolo26/DATA-YAML/data_qnsfl_new.yaml"
OUTPUT_DIR = "/home/hsjeong/workspace/VIT/RepVIT/detection/data/qnsfl/annotations"

# ── Classes (must match data_qnsfl_new.yaml nc / names) ──────────────────────
CLASSES = ["person", "car", "motorcycle", "plate_number"]

# ── Supported image extensions ────────────────────────────────────────────────
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_yaml_path_pairs(yaml_path: str) -> dict:
    """
    Parse the yaml and return per-split list of (img_dir, lbl_dir) pairs.

    The yaml lists paths alternating image/label directories:
        train:
          - /ds1/images   ← index 0 (even)
          - /ds1/labels   ← index 1 (odd)
          - /ds2/images
          - /ds2/labels
          ...
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    result = {}
    for split in ("train", "val", "test"):
        raw = data.get(split, [])
        pairs = []
        for i in range(0, len(raw), 2):
            img_dir = raw[i].strip()
            lbl_dir = raw[i + 1].strip() if i + 1 < len(raw) else None
            pairs.append((img_dir, lbl_dir))
        result[split] = pairs
    return result


def get_image_wh(img_path: Path):
    """Return (width, height) by reading only the image header."""
    with Image.open(img_path) as img:
        return img.width, img.height


def convert_split(pairs: list, split_name: str) -> dict:
    """
    Convert one split (train / val / test) into a COCO-format dict.

    Args:
        pairs      : list of (img_dir, lbl_dir) strings
        split_name : 'train', 'val', or 'test'  (used only for progress labels)

    Returns:
        COCO dict with keys 'images', 'annotations', 'categories'
    """
    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(CLASSES)
    ]
    images      = []
    annotations = []
    image_id    = 1
    ann_id      = 1
    skipped_img = 0
    skipped_ann = 0

    for img_dir_str, lbl_dir_str in pairs:
        img_dir = Path(img_dir_str)
        lbl_dir = Path(lbl_dir_str) if lbl_dir_str else None

        if not img_dir.exists():
            print(f"  [SKIP] Directory not found: {img_dir}")
            continue

        img_files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        ds_tag = img_dir.parts[-3] if len(img_dir.parts) >= 3 else img_dir.name
        print(f"  {ds_tag}/{split_name}: {len(img_files)} images")

        for img_path in tqdm(img_files, desc=f"    {ds_tag}", leave=False):
            # ── Image record ──────────────────────────────────────────────
            try:
                w, h = get_image_wh(img_path)
            except Exception as e:
                print(f"  [WARN] Cannot read {img_path.name}: {e}")
                skipped_img += 1
                continue

            images.append({
                "id":        image_id,
                "file_name": str(img_path),  # absolute path (direct-path mode)
                "width":     w,
                "height":    h,
            })

            # ── Annotation records ────────────────────────────────────────
            if lbl_dir is not None:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    with open(lbl_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) != 5:
                                continue

                            cls_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:])

                            # Normalised → absolute pixel coords
                            x_min = (cx - bw / 2) * w
                            y_min = (cy - bh / 2) * h
                            box_w = bw * w
                            box_h = bh * h

                            # Clamp to image boundaries
                            x_min = max(0.0, x_min)
                            y_min = max(0.0, y_min)
                            box_w = min(box_w, w - x_min)
                            box_h = min(box_h, h - y_min)

                            area = box_w * box_h
                            if area <= 0:
                                skipped_ann += 1
                                continue

                            annotations.append({
                                "id":          ann_id,
                                "image_id":    image_id,
                                "category_id": cls_id + 1,  # 0-indexed → 1-indexed
                                "bbox":  [round(x_min, 2), round(y_min, 2),
                                          round(box_w, 2), round(box_h, 2)],
                                "area":  round(area, 2),
                                "iscrowd": 0,
                            })
                            ann_id += 1

            image_id += 1

    if skipped_img:
        print(f"  [INFO] Skipped {skipped_img} unreadable images")
    if skipped_ann:
        print(f"  [INFO] Skipped {skipped_ann} zero-area boxes")

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs_by_split = parse_yaml_path_pairs(YAML_PATH)

    stats = {}
    for split in ("train", "val", "test"):
        print(f"\n{'='*60}")
        print(f"  Converting split: {split}")
        print(f"{'='*60}")

        coco = convert_split(pairs_by_split[split], split)
        out_path = os.path.join(OUTPUT_DIR, f"instances_{split}.json")

        with open(out_path, "w") as f:
            json.dump(coco, f)

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        stats[split] = (n_img, n_ann)
        print(f"  Saved  → {out_path}")
        print(f"  images : {n_img:>7,}")
        print(f"  boxes  : {n_ann:>7,}")

    print(f"\n{'='*60}")
    print("  Conversion complete — summary")
    print(f"{'='*60}")
    for split, (n_img, n_ann) in stats.items():
        print(f"  {split:<6} │ images {n_img:>7,} │ boxes {n_ann:>8,}")
    print(f"\n  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
