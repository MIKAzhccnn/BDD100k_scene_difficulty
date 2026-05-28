#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, shutil, ijson
from pathlib import Path

# BDD100K 10 classes (official detection classes)
BDD_CLASSES = [
    "person","rider","car","truck","bus","train",
    "motorcycle","bicycle","traffic light","traffic sign"
]
CLS2ID = {c: i for i, c in enumerate(BDD_CLASSES)}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, symlink=True):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    # Convert to float for compatibility with decimal.Decimal
    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
    w = float(w); h = float(h)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    return cx / w, cy / h, bw / w, bh / h

def write_yolo_label(txt_path: Path, labels, img_w, img_h):
    lines = []
    for lab in labels:
        if "box2d" not in lab or "category" not in lab:
            continue
        cat = lab["category"]
        if cat not in CLS2ID:
            continue
        b = lab["box2d"]
        x1 = float(b["x1"]); y1 = float(b["y1"])
        x2 = float(b["x2"]); y2 = float(b["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
        cx = max(0, min(1, cx)); cy = max(0, min(1, cy))
        bw = max(0, min(1, bw)); bh = max(0, min(1, bh))
        cls_id = CLS2ID[cat]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if lines:
        ensure_dir(txt_path.parent)
        txt_path.write_text("\n".join(lines), encoding="utf-8")

def build_image_index(images_root: Path, split: str, images_size: str = "100k"):
    """
    Recursively traverse images_root/<images_size>/<split> and
    build an index {filename: absolute_path}, supporting jpg/JPG/jpeg.
    """
    base = images_root / images_size / split
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    idx = {}
    for pat in exts:
        for p in base.rglob(pat):
            idx[p.name] = p
    return idx

def pick_items_by_weather(json_path, images_root, weather, max_items, images_size="100k"):
    """
    Extract a list of sample names for the specified weather from the JSON (train/val),
    keeping only files that actually exist on disk.
    """
    chosen = []
    split = "val" if Path(json_path).name.endswith("_val.json") else "train"
    idx = build_image_index(Path(images_root), split, images_size=images_size)

    with open(json_path, "rb") as f:
        for item in ijson.items(f, "item"):
            a = item.get("attributes", {})
            if a.get("weather", "undefined") != weather:
                continue
            name = item["name"]
            if name not in idx:
                continue
            chosen.append((name, 1280, 720))
            if len(chosen) >= max_items:
                break
    return chosen

def export_split(json_path, images_root, out_root, weather, names, split, symlink=True, images_size="100k"):
        """
        Export the specified split (train/val):
            - images/{split}/  symlink or copy images
            - labels/{split}/ write YOLO labels
        """
    split_dir_img = Path(out_root) / weather / "images" / split
    split_dir_lab = Path(out_root) / weather / "labels" / split
    ensure_dir(split_dir_img); ensure_dir(split_dir_lab)

    json_split = "val" if Path(json_path).name.endswith("_val.json") else "train"
    idx = build_image_index(Path(images_root), json_split, images_size=images_size)

    name_index = {n for n, _w, _h in names}

    with open(json_path, "rb") as f:
        for item in ijson.items(f, "item"):
            name = item["name"]
            if name not in name_index:
                continue
            if name not in idx:
                continue
            # link/copy image
            src = idx[name]
            dst = split_dir_img / name
            link_or_copy(src, dst, symlink=symlink)
            # write label
            labels = item.get("labels", [])
            w, h = 1280, 720
            txt = split_dir_lab / (Path(name).stem + ".txt")
            write_yolo_label(txt, labels, w, h)

def write_yaml(yaml_path: Path, dataset_dir: Path):
    names_yaml = "\n".join([f"  - {n}" for n in BDD_CLASSES])
    content = f"""# Auto-generated
path: {dataset_dir}
train: images/train
val: images/val

names:
{names_yaml}
"""
    ensure_dir(yaml_path.parent)
    yaml_path.write_text(content, encoding="utf-8")
    print("wrote", yaml_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_json",  required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--images_root", required=True, help=".../bdd100k/images")
    ap.add_argument("--images_size", default="100k", choices=["100k","10k"])
    ap.add_argument("--out_root", default="~/Projects/moca/datasets/bdd_yolo")
    ap.add_argument("--weather", default="clear,rainy,snowy,overcast,partly cloudy,foggy")
    ap.add_argument("--train_count", type=int, default=200)
    ap.add_argument("--val_count", type=int, default=100)
    ap.add_argument("--symlink", action="store_true")
    args = ap.parse_args()

    val_json = Path(os.path.expanduser(args.val_json))
    train_json = Path(os.path.expanduser(args.train_json))
    images_root = Path(os.path.expanduser(args.images_root))
    out_root = Path(os.path.expanduser(args.out_root)).expanduser()

    weathers = [w.strip() for w in args.weather.split(",") if w.strip()]
    print("weathers:", weathers)

    # Sample for each weather
    picks = {}
    for w in weathers:
        tr = pick_items_by_weather(train_json, images_root, w, args.train_count, images_size=args.images_size)
        va = pick_items_by_weather(val_json,   images_root, w, args.val_count,   images_size=args.images_size)
        print(f"[{w}] train={len(tr)} val={len(va)}")
        picks[w] = (tr, va)

    # Export + write YAML for each weather
    for w,(tr,va) in picks.items():
        export_split(train_json, images_root, out_root, w, tr, split="train", symlink=args.symlink, images_size=args.images_size)
        export_split(val_json,   images_root, out_root, w, va, split="val",   symlink=args.symlink, images_size=args.images_size)
        safe_w = w.replace(" ", "_")
        yaml_path = out_root.parent.parent / "yamls" / f"bdd_{safe_w}.yaml"
        write_yaml(yaml_path, Path(out_root)/w)

    # Generate 'all' (mixed baseline)
    all_train = sum((tr for tr,_ in picks.values()), [])
    all_val   = sum((va for _,va in picks.values()), [])
    export_split(train_json, images_root, out_root, "all", all_train, split="train", symlink=args.symlink, images_size=args.images_size)
    export_split(val_json,   images_root, out_root, "all", all_val,   split="val",   symlink=args.symlink, images_size=args.images_size)
    yaml_path = out_root.parent.parent / "yamls" / "bdd_all.yaml"
    write_yaml(yaml_path, Path(out_root)/"all")

    print("✅ Done. You can start training YOLO with the generated yamls.")

if __name__ == "__main__":
    main()
