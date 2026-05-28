#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect YOLO per-weather run results from runs/detect/* and plot comparisons.

It will:
  - scan runs/detect/*/results.csv (+ optional *latency*.csv if present)
  - infer weather from run folder name
  - normalize common column names
  - output combined CSV and figures

Usage:
  python collect_and_plot_weather.py \
      --runs_dir runs/detect \
      --outdir results/figs \
      [--combined_csv results/combined_weather_results.csv]
"""

import argparse
from pathlib import Path
import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- config ----------
WEATHERS = ["clear", "partly_cloudy", "overcast", "rainy", "snowy", "foggy"]
WEATHER_ALIASES = {
    "partlycloudy": "partly_cloudy",
    "partly-cloudy": "partly_cloudy",
}

# columns we try to detect (case-insensitive)
CANON_COLS = {
    "map50":   ["map50", "mAP50", "mAP_50", "metrics/mAP50(B)", "metrics/mAP50"],
    "map5095": ["map50_95", "mAP50_95", "mAP50-95", "mAP_50:95", "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP"],
    "p50":     ["p50_ms", "latency_p50_ms", "lat_p50", "p50", "median_ms"],
    "p95":     ["p95_ms", "latency_p95_ms", "lat_p95", "p95"],
}

def _find_col(df, aliases):
    for want in aliases:
        for col in df.columns:
            if col.lower().strip() == want.lower().strip():
                return col
    return None

def norm_weather_from_name(name: str):
    s = name.lower()
    for w in WEATHERS:
        if w in s:
            return w
    # Support aliases
    for alias, canon in WEATHER_ALIASES.items():
        if alias in s:
            return canon
    return None

def group_from_run_name(name: str):
    s = name.lower()
    # If the name contains ft/fine-tune and includes a weather, treat as per-weather fine-tune
    if ("ft" in s or "fine" in s) and norm_weather_from_name(s):
        return "Per-weather FT"
    # Otherwise -> treat as all-weather base model
    return "All-weather"

def read_results_csv(csv_path: Path):
    """
    Ultralytics results.csv: multiple rows (one per epoch); take the last row.
    Also supports single-row files.
    """
    df = pd.read_csv(csv_path)
    last = df.iloc[-1:].copy()
    return last

def maybe_read_latency(run_dir: Path):
    """
    Optional: read latency statistics file (if you have a separate script that produced a latency CSV).
    Filenames matching *latency*.csv; column names should be compatible with CANON_COLS.
    Returns None if not found.
    """
    cand = list(run_dir.glob("*latency*.csv"))
    if not cand:
        return None
    df = pd.read_csv(cand[0])
    # Either take the first or the last row; here we take the first row
    return df.iloc[:1].copy()

def canonicalize_row(df_like: pd.DataFrame):
    """
    Extract map50 / map50_95 / p50_ms / p95_ms from a single-row DataFrame.
    Missing columns return NaN.
    """
    row = df_like.iloc[0]
    def pick(aliases):
        col = _find_col(df_like, aliases)
        return pd.to_numeric(row[col], errors="coerce") if col else np.nan

    return {
        "mAP50":    float(pick(CANON_COLS["map50"])),
        "mAP50_95": float(pick(CANON_COLS["map5095"])),
        "p50_ms":   float(pick(CANON_COLS["p50"])),
        "p95_ms":   float(pick(CANON_COLS["p95"])),
    }

def collect_runs(runs_dir: Path):
    rows = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        res_csv = run / "results.csv"
        if not res_csv.exists():
            continue

        # training/validation results
        last = read_results_csv(res_csv)
        canon = canonicalize_row(last)

        # Optional latency file (if present, override p50/p95)
        lat = maybe_read_latency(run)
        if lat is not None:
            lat_vals = canonicalize_row(lat)
            if not np.isnan(lat_vals["p50_ms"]): canon["p50_ms"] = lat_vals["p50_ms"]
            if not np.isnan(lat_vals["p95_ms"]): canon["p95_ms"] = lat_vals["p95_ms"]

        weather = norm_weather_from_name(run.name) or "unknown"
        group = group_from_run_name(run.name)

        rows.append({
            "run_dir": str(run),
            "model": run.name,
            "group": group,
            "weather": weather,
            **canon,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs found with results.csv.")
    return df

def order_weathers(values):
    # Order by preset sequence; put unknown weathers last
    seen = list(dict.fromkeys(values))  # preserve order and deduplicate
    ordered = [w for w in WEATHERS if w in seen] + [w for w in seen if w not in WEATHERS]
    return ordered

def grouped_bar(ax, x_labels, series_dict, title, ylabel, ylim=None, fmt="{:.3f}"):
    n = len(x_labels)
    k = len(series_dict)
    idx = np.arange(n)
    width = 0.8 / max(k, 1)

    for i, (lab, vals) in enumerate(series_dict.items()):
        offset = (i - (k-1)/2) * width
        bars = ax.bar(idx + offset, vals, width, label=lab)
        for b, v in zip(bars, vals):
            if v is None or (isinstance(v, float) and (math.isnan(v) or not np.isfinite(v))):
                continue
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(idx)
    ax.set_xticklabels(x_labels, rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

def plot_all(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    # Deduplicate: keep the entry with highest mAP50_95 per (weather, group)
    use = df.copy()
    use["weather_norm"] = use["weather"].astype(str)
    use.sort_values(["weather_norm", "group", "mAP50_95"], ascending=[True, True, False], inplace=True)
    use = use.drop_duplicates(subset=["weather_norm", "group"], keep="first")

    weathers = order_weathers(list(use["weather_norm"].unique()))
    def pick(metric):
        got = {"All-weather": [], "Per-weather FT": []}
        for w in weathers:
            sub = use[use["weather_norm"] == w]
            for g in got:
                row = sub[sub["group"] == g]
                got[g].append(float(row[metric].iloc[0]) if not row.empty else np.nan)
        return got

    outdir.mkdir(parents=True, exist_ok=True)

    # mAP50
    fig, ax = plt.subplots(figsize=(10,5))
    grouped_bar(ax, weathers, pick("mAP50"), "mAP50 by Weather", "mAP50", fmt="{:.3f}")
    fig.tight_layout(); fig.savefig(outdir / "map50_by_weather.png", dpi=200)

    # mAP50-95
    fig, ax = plt.subplots(figsize=(10,5))
    grouped_bar(ax, weathers, pick("mAP50_95"), "mAP50-95 by Weather", "mAP50-95", fmt="{:.3f}")
    fig.tight_layout(); fig.savefig(outdir / "map5095_by_weather.png", dpi=200)

    # Latency (plot p50 & p95 on the same figure; prefer Per-weather FT, otherwise All-weather)
    chosen = []
    for w in weathers:
        sub = use[use["weather_norm"] == w]
        row = sub[sub["group"] == "Per-weather FT"]
        if row.empty:
            row = sub[sub["group"] == "All-weather"]
        if not row.empty:
            chosen.append(row.iloc[0])
    if chosen:
        dd = pd.DataFrame(chosen)
        fig, ax = plt.subplots(figsize=(10,5))
        latency_series = {
            "p50 (ms)": [float(x) if np.isfinite(x) else np.nan for x in dd["p50_ms"].values],
            "p95 (ms)": [float(x) if np.isfinite(x) else np.nan for x in dd["p95_ms"].values],
        }
        grouped_bar(ax, list(dd["weather_norm"].astype(str)), latency_series,
                    "Latency by Weather (picked FT if available)", "Latency (ms)", fmt="{:.2f}")
        fig.tight_layout(); fig.savefig(outdir / "latency_by_weather.png", dpi=200)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs/detect", help="Directory containing per-run folders")
    ap.add_argument("--outdir", default="results/figs", help="Where to save figures")
    ap.add_argument("--combined_csv", default="results/combined_weather_results.csv", help="Where to save combined CSV")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    outdir = Path(args.outdir)
    combined_csv = Path(args.combined_csv)
    combined_csv.parent.mkdir(parents=True, exist_ok=True)

    df = collect_runs(runs_dir)
    if not df.empty:
        df.to_csv(combined_csv, index=False)
        print(f"[OK] Combined CSV saved to: {combined_csv.resolve()}")
    else:
        print("[WARN] No results collected; plots will be skipped.")

    plot_all(df, outdir)
    print(f"[Done] Figures saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
