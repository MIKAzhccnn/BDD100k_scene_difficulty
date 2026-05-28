#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click evaluation:
- Iterate over all submodels under --runs_dir
- Latency measurement:
    * If --glob_template is provided, use it to build the image path
    * Otherwise parse val paths from bdd_{weather}.yaml under --yaml_dir
- Save latency_eval.csv
- Summarize results and plot
"""

import argparse, time, glob, math, json, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

WEATHERS = ["clear", "partly_cloudy", "overcast", "rainy", "snowy", "foggy"]
ALIASES = {"partlycloudy": "partly_cloudy", "partly-cloudy": "partly_cloudy"}

def infer_weather(name: str):
    s = name.lower()
    for w in WEATHERS:
        if w in s:
            return w
    for a, c in ALIASES.items():
        if a in s:
            return c
    return "unknown"

def group_from_name(name: str):
    s = name.lower()
    if ("ft" in s or "fine" in s) and infer_weather(s) != "unknown":
        return "Per-weather FT"
    return "All-weather"

def measure_latency(weights_path, images_glob, warmup=10, runs=50, imgsz=640, device=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights_path))
    model.to(device)

    imgs = sorted(glob.glob(images_glob))
    if not imgs:
        raise RuntimeError(f"No images matched: {images_glob}")

    for i in range(min(warmup, len(imgs))):
        _ = model.predict(imgs[i], imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for i in range(min(runs, len(imgs))):
        t0 = time.time()
        _ = model.predict(imgs[i], imgsz=imgsz, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return p50, p95, len(times)

def val_glob_from_yaml(yaml_dir: Path, weather: str):
    """
    Read scripts/yamls/bdd_{weather}.yaml:
    - If it contains 'path', treat 'val' as relative to that path
    - If 'val' is a directory, append /*.{jpg,png,jpeg}
    - If 'val' is a glob, use it directly
    - If 'val' is a list, take the first item (you may change to test all)
    Returns: a glob string (must match at least one image, otherwise raise)
    """
    import yaml, os, glob

    yaml_path = yaml_dir / f"bdd_{weather}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No yaml for {weather}: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        yobj = yaml.safe_load(f) or {}

    base = yobj.get("path")  # May be absolute or relative
    val = yobj.get("val")
    if val is None:
        raise KeyError(f"No 'val' field in {yaml_path}")

    # Handle list / string
    if isinstance(val, list):
        val0 = val[0]
    else:
        val0 = val

    # If it's a relative path and base is present, join with base
    val_path = Path(val0)
    if not val_path.is_absolute() and base is not None:
        val_path = Path(base) / val_path

    # If still a relative path with no base, resolve relative to the yaml file's directory
    if not val_path.is_absolute():
        val_path = (yaml_path.parent / val_path).resolve()

    # If it's a directory, try common extensions
    if val_path.exists() and val_path.is_dir():
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        # Prefer jpg first
        for pat in exts:
            g = str(val_path / pat)
            if glob.glob(g):
                return g
        # If none found, return jpg pattern (later code may raise)
        return str(val_path / "*.jpg")

    # Looks like a glob or specific file
    g = str(val_path)
    if glob.glob(g):
        return g

    # If val itself is a glob and making it absolute failed, try the original val0
    if isinstance(val0, str) and any(ch in val0 for ch in "*?[]"):
        if glob.glob(val0):
            return val0

    raise RuntimeError(f"No images matched after resolving yaml: {yaml_path} -> {g}")


def grouped_bar(ax, x_labels, series_dict, title, ylabel, fmt="{:.3f}"):
    n = len(x_labels); k = len(series_dict)
    idx = np.arange(n); width = 0.8 / max(k, 1)
    for i, (lab, vals) in enumerate(series_dict.items()):
        off = (i - (k-1)/2) * width
        bars = ax.bar(idx + off, vals, width, label=lab)
        for b, v in zip(bars, vals):
            if np.isnan(v): continue
            ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(idx); ax.set_xticklabels(x_labels, rotation=20)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.legend(frameon=False); ax.grid(axis="y", linestyle="--", alpha=0.4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="scripts/runs/detect")
    ap.add_argument("--glob_template", default=None,
                    help="Image path template, e.g., datasets/bdd_yolo/{weather}/val/*.jpg")
    ap.add_argument("--yaml_dir", default=None,
                    help="Directory containing bdd_{weather}.yaml files")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--outdir", default="results/figs")
    ap.add_argument("--combined_csv", default="results/combined_weather_results.csv")
    args = ap.parse_args()

    if not args.glob_template and not args.yaml_dir:
        ap.error("You must provide either --glob_template or --yaml_dir")

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path(args.combined_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir(): continue
        weather = infer_weather(run.name)
        group = group_from_name(run.name)
        weights = run / "weights" / "best.pt"
        if not weights.exists():
            print(f"[Skip] {run.name} no weights")
            continue

        lat_csv = run / "latency_eval.csv"
        if not lat_csv.exists():
            if args.glob_template:
                img_glob = args.glob_template.format(weather=weather)
            else:
                img_glob = val_glob_from_yaml(Path(args.yaml_dir), weather)
            try:
                p50, p95, n = measure_latency(weights, img_glob,
                                              warmup=args.warmup, runs=args.runs, imgsz=args.imgsz)
                pd.DataFrame([{"p50_ms": p50, "p95_ms": p95, "N": n, "imgsz": args.imgsz}]
                             ).to_csv(lat_csv, index=False)
                print(f"[OK] {run.name}: p50={p50:.2f}ms p95={p95:.2f}ms")
            except Exception as e:
                print(f"[ERR] {run.name}: {e}")

        # Results
        mAP50, mAP5095, p50v, p95v = np.nan, np.nan, np.nan, np.nan
        res_csv = run / "results.csv"
        if res_csv.exists():
            df = pd.read_csv(res_csv)
            last = df.iloc[-1]
            mAP50 = last.get("metrics/mAP50(B)", last.get("mAP50", np.nan))
            mAP5095 = last.get("metrics/mAP50-95(B)", last.get("mAP50-95", np.nan))
        if lat_csv.exists():
            lat = pd.read_csv(lat_csv).iloc[0]
            p50v, p95v = lat["p50_ms"], lat["p95_ms"]

        rows.append({"model": run.name, "group": group, "weather": weather,
                     "mAP50": mAP50, "mAP50_95": mAP5095,
                     "p50_ms": p50v, "p95_ms": p95v})

    df = pd.DataFrame(rows)
    df.to_csv(args.combined_csv, index=False)
    print(f"[OK] Combined CSV -> {args.combined_csv}")

    if df.empty: return
    weathers = [w for w in WEATHERS if w in set(df["weather"])]

    def pick(metric):
        out = {"All-weather": [], "Per-weather FT": []}
        for w in weathers:
            sub = df[df["weather"] == w]
            for g in out:
                row = sub[sub["group"] == g]
                out[g].append(float(row[metric].iloc[0]) if not row.empty else np.nan)
        return out

    # # mAP50
    # fig, ax = plt.subplots(figsize=(10,5))
    # grouped_bar(ax, weathers, pick("mAP50"), "YOLO V8N mAP50 by Weather", "mAP50")
    # fig.savefig(outdir/"map50_by_weather.png", dpi=200)

    # # mAP50-95
    # fig, ax = plt.subplots(figsize=(10,5))
    # grouped_bar(ax, weathers, pick("mAP50_95"), "YOLO V8N mAP50-95 by Weather", "mAP50-95")
    # fig.savefig(outdir/"map5095_by_weather.png", dpi=200)

    # === Combined Accuracy (mAP50 & mAP50-95 in one figure) ===
    # weathers and pick() already defined above
    fig, ax = plt.subplots(figsize=(12, 6))

    series_acc = {
        "All: mAP50":      pick("mAP50")["All-weather"],
        "All: mAP50-95":   pick("mAP50_95")["All-weather"],
        "FT: mAP50":       pick("mAP50")["Per-weather FT"],
        "FT: mAP50-95":    pick("mAP50_95")["Per-weather FT"],
    }
    grouped_bar(ax, weathers, series_acc,
                title="Accuracy (mAP) by Weather",
                ylabel="mAP", fmt="{:.3f}")
    fig.tight_layout()
    fig.savefig(outdir / "accuracy_combined_by_weather.png", dpi=200)

    # latency
    fig, ax = plt.subplots(figsize=(10,5))
    dd = []
    for w in weathers:
        sub = df[df["weather"] == w]
        row = sub[sub["group"] == "Per-weather FT"]
        if row.empty: row = sub[sub["group"] == "All-weather"]
        if not row.empty: dd.append(row.iloc[0])
    if dd:
        dd = pd.DataFrame(dd)
        latency = {"p50 (ms)": dd["p50_ms"].tolist(), "p95 (ms)": dd["p95_ms"].tolist()}
        grouped_bar(ax, dd["weather"].tolist(), latency, "YOLO V8N Latency by Weather", "Latency (ms)", fmt="{:.2f}")
        fig.savefig(outdir/"latency_by_weather.png", dpi=200)

    print(f"[Done] Plots saved to {outdir}")

if __name__ == "__main__":
    main()
