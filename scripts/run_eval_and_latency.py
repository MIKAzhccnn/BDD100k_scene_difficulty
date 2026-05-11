#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run YOLO val on multiple weather splits + measure inference latency.
Saves results locally to JSON and CSV for later plotting.
"""

import argparse, json, time, statistics, csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO


WEATHERS_DEFAULT = ["clear", "rainy", "snowy", "overcast", "partly_cloudy", "foggy"]


def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def measure_latency(model: YOLO, imgsz: int, device: str, runs: int = 200, warmup: int = 20):
    """
    Measure end-to-end prediction latency using ultralytics .predict()
    on a dummy image (imgsz x imgsz, uint8). Returns ms stats.
    """
    # prepare one dummy RGB frame
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # warmup
    for _ in range(warmup):
        _ = model.predict(source=img, imgsz=imgsz, device=device, verbose=False)

    # timed runs
    lat = []
    for _ in range(runs):
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)  # ms

    lat.sort()
    mean = float(statistics.mean(lat))
    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    minv = float(lat[0])
    maxv = float(lat[-1])
    return dict(mean_ms=mean, p50_ms=p50, p95_ms=p95, min_ms=minv, max_ms=maxv, runs=runs)


def val_one(model: YOLO, data_yaml: Path, imgsz: int, device: str, batch: int = 1):
    """
    Run validation and return key metrics.
    """
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        batch=batch,
        plots=False,
        verbose=False,
    )
    # metrics.box.map  (mAP@[.5:.95])
    # metrics.box.map50 (mAP@.5)
    # metrics.box.map75, metrics.box.maps (per-class)
    out = dict(
        images=int(getattr(metrics, "images", 0) or 0),
        instances=int(getattr(metrics, "boxes", 0) or 0),
        map_50_95=float(metrics.box.map),
        map_50=float(metrics.box.map50),
        precision=float(getattr(metrics.box, "mp", 0.0) or 0.0),  # mean precision
        recall=float(getattr(metrics.box, "mr", 0.0) or 0.0),     # mean recall
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="~/Projects/moca", help="Project root (contains yamls/)")
    ap.add_argument("--model", required=True, help="Path to YOLO best.pt, e.g. runs/detect/yv8n_all_e50/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="'0' or 'cpu'")
    ap.add_argument("--runs", type=int, default=200, help="Latency runs")
    ap.add_argument("--weathers", default=",".join(WEATHERS_DEFAULT), help="Comma list: clear,rainy,...")
    ap.add_argument("--batch", type=int, default=1, help="val batch size")
    ap.add_argument("--outdir", default="results", help="Dir (under project_root) to save exp_log.json/csv")
    args = ap.parse_args()

    root = Path(args.project_root).expanduser()
    yamls_dir = root / "yamls"
    outdir = (root / args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser()
    assert model_path.exists(), f"model not found: {model_path}"

    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]
    print(f"[{now_iso()}] Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Measure latency once for the model (shared across weathers)
    print(f"[{now_iso()}] Measuring latency: imgsz={args.imgsz}, device={args.device}, runs={args.runs}")
    lat = measure_latency(model, imgsz=args.imgsz, device=args.device, runs=args.runs)

    rows = []
    for w in weathers:
        yaml_path = yamls_dir / f"bdd_{w}.yaml"
        if not yaml_path.exists():
            print(f"[WARN] YAML not found for weather '{w}': {yaml_path} (skip)")
            continue

        print(f"[{now_iso()}] VAL on weather={w}  data={yaml_path}")
        m = val_one(model, yaml_path, imgsz=args.imgsz, device=args.device, batch=args.batch)

        row = {
            "timestamp": now_iso(),
            "weather": w,
            "imgsz": args.imgsz,
            "device": args.device,
            "model": str(model_path),
            "images": m["images"],
            "instances": m["instances"],
            "precision": m["precision"],
            "recall": m["recall"],
            "mAP50": m["map_50"],
            "mAP50_95": m["map_50_95"],
            "lat_mean_ms": lat["mean_ms"],
            "lat_p50_ms": lat["p50_ms"],
            "lat_p95_ms": lat["p95_ms"],
            "lat_min_ms": lat["min_ms"],
            "lat_max_ms": lat["max_ms"],
            "lat_runs": lat["runs"],
        }
        rows.append(row)

    # Save JSON
    json_path = outdir / "exp_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[{now_iso()}] Saved JSON: {json_path}")

    # Save CSV
    csv_path = outdir / "exp_log.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[{now_iso()}] Saved CSV:  {csv_path}")

    # quick preview
    for r in rows:
        print(f" - {r['weather']:>13} | mAP50={r['mAP50']:.3f} | mAP50-95={r['mAP50_95']:.3f} | p50={r['lat_p50_ms']:.2f} ms | p95={r['lat_p95_ms']:.2f} ms")


if __name__ == "__main__":
    main()
