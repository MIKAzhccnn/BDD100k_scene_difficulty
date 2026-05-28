#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time, csv
from pathlib import Path
import yaml, numpy as np
from ultralytics import YOLO
import torch

def resolve_val_imgs(yaml_path: Path) -> Path:
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    base = Path(y.get("path", "."))
    val = y.get("val")
    if val is None: raise KeyError(f"No 'val' in {yaml_path}")
    val_dir = (base/val) if not Path(val).is_absolute() else Path(val)
    if not val_dir.exists(): val_dir = (yaml_path.parent/val_dir).resolve()
    if not val_dir.exists(): raise FileNotFoundError(f"val images not found: {val_dir}")
    return val_dir

@torch.no_grad()
def measure_latency_ultra(model: YOLO, img_dir: Path, imgsz=640, runs=200, warmup=10, device=None):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    paths = []
    for e in exts: paths += list(img_dir.rglob(e))
    if not paths: raise FileNotFoundError(f"No images under {img_dir}")
    paths = paths[:runs] if runs>0 else paths
    # warmup
    for p in paths[:warmup]:
        _ = model.predict(source=str(p), imgsz=imgsz, conf=0.001, verbose=False, device=device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    # timing
    ts=[]
    for p in paths:
        t0=time.time()
        _ = model.predict(source=str(p), imgsz=imgsz, conf=0.001, verbose=False, device=device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ts.append((time.time()-t0)*1000.0)
    a=np.array(ts, dtype=np.float64)
    return {"p50_ms": float(np.percentile(a,50)), "p95_ms": float(np.percentile(a,95))}

def read_maps_from_run(run_dir: Path):
    csv_path = run_dir/"results.csv"
    if not csv_path.exists(): return None
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows: return None
    last = rows[-1]
    def pick(d, keys):
        for k in keys:
            if k in d and d[k]!="":
                try: return float(d[k])
                except: pass
        return None
    m50   = pick(last, ["metrics/mAP50(B)","metrics/mAP50","mAP50"])
    m5095 = pick(last, ["metrics/mAP50-95(B)","metrics/mAP50-95","metrics/mAP50_95","mAP50_95"])
    if m50 is None or m5095 is None: return None
    return {"mAP50":m50, "mAP50_95":m5095}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="scripts/runs/detect/rtdetr_all/weights/best.pt")
    ap.add_argument("--yaml_dir", default="yamls")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--latency_runs", type=int, default=200)
    ap.add_argument("--out_csv", default="results/rtdetr_by_weather.csv")
    args = ap.parse_args()

    model = YOLO(args.weights)
    weathers = ["clear","partly_cloudy","overcast","rainy","snowy","foggy"]
    rows=[]
    for w in weathers:
        ypath = Path(args.yaml_dir)/f"bdd_{w}.yaml"
        if not ypath.exists():
            print(f"[Skip] no yaml: {ypath}"); continue
        # val metrics
        val_res = model.val(data=str(ypath), imgsz=args.imgsz, batch=args.batch,
                            device=args.device, split='val', verbose=False)
        # Handle different field names
        try:
            m50   = float(getattr(val_res.box, "map50"))
            m5095 = float(getattr(val_res.box, "map"))
        except Exception:
            d = getattr(val_res, "results_dict", {}) or {}
            m50   = float(d.get("metrics/mAP50", d.get("metrics/mAP50(B)", 0.0)))
            m5095 = float(d.get("metrics/mAP50-95", d.get("metrics/mAP50-95(B)", 0.0)))
        # latency
        val_imgs = resolve_val_imgs(ypath)
        lat = measure_latency_ultra(model, val_imgs, imgsz=args.imgsz,
                                    runs=args.latency_runs, warmup=10, device=args.device)
        print(f"[{w}] mAP50_95={m5095:.4f} mAP50={m50:.4f} p50={lat['p50_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms")
        rows.append({"weather": w, "model": "rtdetr_all",
                     "mAP50": m50, "mAP50_95": m5095,
                     "p50_ms": lat["p50_ms"], "p95_ms": lat["p95_ms"]})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[OK] per-weather metrics -> {args.out_csv}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
