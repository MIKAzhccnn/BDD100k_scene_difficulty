#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RT-DETR (Ultralytics) on bdd_yolo/all and summarize metrics/latency.

Outputs:
- Training run: scripts/runs/detect/rtdetr_all (weights/best.pt, results.csv, etc.)
- Summary CSV:  results/rtdetr_all/results.csv  (mAP50, mAP50_95, p50_ms, p95_ms)
"""

import argparse, time, csv
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO

# -------- helpers --------
def resolve_val_dirs(yaml_path: Path):
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    base = Path(y.get("path", "."))
    val = y.get("val")
    if val is None:
        raise KeyError(f"No 'val' in {yaml_path}")
    val_dir = (base / val) if not Path(val).is_absolute() else Path(val)
    if not val_dir.exists():
        # Support: relative path relative to the yaml
        val_dir = (yaml_path.parent / val_dir).resolve()
    if not val_dir.exists():
        raise FileNotFoundError(f"val images dir not found: {val_dir}")

    # Infer labels directory (replace 'images' with 'labels'; ignore if not applicable)
    s = str(val_dir)
    if "/images/" in s:
        labels_dir = Path(s.replace("/images/", "/labels/"))
    else:
        labels_dir = val_dir.parent.parent / "labels" / val_dir.name
    return val_dir, labels_dir

@torch.no_grad()
def measure_latency_ultra(model: YOLO, img_dir: Path, imgsz: int = 640, runs: int = 200, warmup: int = 10, device: str = None):
    # Collect several images
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    paths = []
    for e in exts:
        paths += list(img_dir.rglob(e))
    if not paths:
        raise FileNotFoundError(f"No images under {img_dir}")
    paths = paths[:runs] if runs > 0 else paths

    # Warmup
    for p in paths[:warmup]:
        _ = model.predict(source=str(p), imgsz=imgsz, conf=0.001, verbose=False, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timing
    times = []
    for p in paths:
        t0 = time.time()
        _ = model.predict(source=str(p), imgsz=imgsz, conf=0.001, verbose=False, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)

    t = np.array(times, dtype=np.float64)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}

def read_final_maps_from_results_csv(results_csv: Path):
    """
    Read the last row from an Ultralytics run's results.csv
    and return mAP50 and mAP50-95 (handles different column names).
    """
    if not results_csv.exists():
        return None
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    # Handle common column name variants
    cand_50 = ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"]
    cand_5095 = ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50_95", "mAP50_95"]
    def pick(d, keys, default=None):
        for k in keys:
            if k in d and d[k] != "":
                try:
                    return float(d[k])
                except Exception:
                    pass
        return default
    m50 = pick(last, cand_50, None)
    m5095 = pick(last, cand_5095, None)
    if m50 is None or m5095 is None:
        return None
    return {"mAP50": m50, "mAP50_95": m5095}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", default="yamls/bdd_all.yaml")
    ap.add_argument("--model", default="rtdetr-l.pt", help="ultralytics model name or path")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)   # For 8GB GPUs, 6~8 batch is usually okay
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None, help="e.g., 0 or cpu; None=auto")
    ap.add_argument("--project", default="scripts/runs/detect")
    ap.add_argument("--name", default="rtdetr_all")
    ap.add_argument("--exist_ok", action="store_true")
    ap.add_argument("--outdir", default="results/rtdetr_all")
    ap.add_argument("--val_conf", type=float, default=0.001)
    ap.add_argument("--latency_runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Train
    model = YOLO(args.model)
    train_res = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True
    )

    # run directory
    run_dir = Path(train_res.save_dir) if hasattr(train_res, "save_dir") else Path(args.project) / args.name
    results_csv = run_dir / "results.csv"
    best_w = run_dir / "weights" / "best.pt"

    # Evaluation (Ultralytics built-in)
    val_res = model.val(data=args.data_yaml, imgsz=args.imgsz, batch=args.batch,
                    conf=args.val_conf, device=args.device, verbose=False)
    # Prefer reading final map from run's results.csv; fallback to returned object if failed
    maps = read_final_maps_from_results_csv(results_csv)
    if maps is None:
        # Fallback: different versions may have different fields
        try:
            m50 = float(getattr(val_res.box, "map50"))
            m5095 = float(getattr(val_res.box, "map"))
            maps = {"mAP50": m50, "mAP50_95": m5095}
        except Exception:
            # Second fallback: generic results_dict
            d = getattr(val_res, "results_dict", {}) or {}
            m50 = float(d.get("metrics/mAP50", d.get("metrics/mAP50(B)", 0.0)))
            m5095 = float(d.get("metrics/mAP50-95", d.get("metrics/mAP50-95(B)", 0.0)))
            maps = {"mAP50": m50, "mAP50_95": m5095}

    # Latency (per-image p50/p95)
    val_imgs, _ = resolve_val_dirs(Path(args.data_yaml))
    lat = measure_latency_ultra(model, val_imgs, imgsz=args.imgsz,
                                runs=args.latency_runs, warmup=args.warmup, device=args.device)

    # Output summary
    out_csv = Path(args.outdir) / "results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(
        "mAP50,mAP50_95,p50_ms,p95_ms\n{:.6f},{:.6f},{:.3f},{:.3f}\n".format(
            maps["mAP50"], maps["mAP50_95"], lat["p50_ms"], lat["p95_ms"]
        ),
        encoding="utf-8"
    )

    print(f"[OK] Run dir: {run_dir}")
    print(f"[OK] Best weights: {best_w}")
    print(f"[OK] Summary CSV -> {out_csv}")
    print(f"[Final] mAP50_95={maps['mAP50_95']:.4f}  mAP50={maps['mAP50']:.4f}  "
          f"p50={lat['p50_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms")

if __name__ == "__main__":
    # Small optimization
    torch.backends.cudnn.benchmark = True
    main()
