#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse
from pathlib import Path
from PIL import Image
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_names(yaml_path: Path):
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if "names" not in y:
        raise ValueError(f"'names' not found in {yaml_path}")
    names = list(y["names"])
    return names

def iter_images(images_dir: Path):
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def yolo_txt_for(img_path: Path, images_root: Path, labels_root: Path):
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")

def yolo_to_xyxy(line, W, H):
    # YOLO: cls cx cy w h (normalized)
    parts = line.strip().split()
    if len(parts) < 5: return None
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    x = (cx - w/2.0) * W
    y = (cy - h/2.0) * H
    bw = w * W
    bh = h * H
    x1 = max(0.0, x); y1 = max(0.0, y)
    x2 = min(W, x + bw); y2 = min(H, y + bh)
    return cls, x1, y1, x2, y2

def convert_split(images_dir: Path, labels_dir: Path, names, out_json: Path):
    images = []
    annotations = []
    categories = [{"id": i+1, "name": n} for i, n in enumerate(names)]
    ann_id = 1
    img_id = 1
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()

    for img_path in iter_images(images_dir):
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({
            "id": img_id,
            "file_name": str(img_path),
            "width": W, "height": H
        })
        txt = yolo_txt_for(img_path, images_dir, labels_dir)
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip(): continue
                parsed = yolo_to_xyxy(line, W, H)
                if parsed is None: continue
                cls, x1, y1, x2, y2 = parsed
                w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0: continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # COCO类id从1开始
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "info": {"description": "YOLO->COCO converted", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }), encoding="utf-8")
    print(f"[OK] COCO json -> {out_json} (images={len(images)}, anns={len(annotations)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="e.g., datasets/bdd_yolo/all/images/train")
    ap.add_argument("--labels_dir", required=True, help="e.g., datasets/bdd_yolo/all/labels/train")
    ap.add_argument("--names_yaml", required=True, help="e.g., yamls/bdd_all.yaml (must contain 'names')")
    ap.add_argument("--out_json", required=True, help="output COCO json file")
    args = ap.parse_args()
    names = load_names(Path(args.names_yaml))
    convert_split(Path(args.images_dir), Path(args.labels_dir), names, Path(args.out_json))

if __name__ == "__main__":
    main()
