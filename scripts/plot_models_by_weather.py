#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge YOLO-all (exp_log.csv), Faster R-CNN-all (frcnn_by_weather.csv),
RT-DETR-all (rtdetr_by_weather.csv), and plot:

- ACCURACY: two subplots (mAP50, mAP50-95), grouped by weather, bars per model
  (legend outside on top)
- LATENCY: p50 (ms), grouped bars per weather, optional p95 error bars
  (legend outside on the right)

Outputs:
  <outdir>/combined_all_models_by_weather.csv
  <outdir>/acc_by_weather_all_models.png
  <outdir>/latency_by_weather_all_models.png
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WEATHER_ORDER = ["clear", "partly_cloudy", "overcast", "rainy", "snowy", "foggy"]
MODEL_ORDER = ["YOLO_v8n", "YOLO (all)", "FRCNN", "RT-DETR"]
MODEL_COLORS = {
    "YOLO_v8n": "#79A6D2",
    "YOLO (all)": "#79A6D2",
    "FRCNN": "#E9A06C",
    "RT-DETR": "#7EAE8B",
}

# 显示名直映（在启发式之前尝试）
MODEL_NAME_MAP = {
    "yolo": "YOLO (all)",
    "yolo_all": "YOLO (all)",
    "frcnn_all": "FRCNN",
    "frcnn": "FRCNN",
    "faster-rcnn": "FRCNN",
    "faster_rcnn": "FRCNN",
    "rtdetr_all": "RT-DETR",
    "rtdetr": "RT-DETR",
}

def normalize_model_display(s: str) -> str:
    """
    将各种模型名称/路径归一化为用于图例的展示名。
    例：'runs/.../yv8n_all.../best.pt' -> 'YOLO (all)'
        'rtdetr_all' -> 'RT-DETR (all)'
        'faster_rcnn' -> 'FRCNN (all)'
    """
    t = str(s).strip()
    low = t.lower()

    # 先尝试直映
    if t in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[t]
    if low in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[low]

    # 启发式归一化
    if any(k in low for k in ["rtdetr", "rt-detr", "rt_detr"]):
        return "RT-DETR"
    if any(k in low for k in ["frcnn", "fasterrcnn", "faster-rcnn", "faster_rcnn"]):
        return "FRCNN"
    if any(k in low for k in [
        "yolo", "yolov", "yolov5", "yolov7", "yolov8", "yolov9", "yolov10", "yolo11",
        "yv5", "yv7", "yv8", "yv9", "yv10", "runs/detect"
    ]):
        return "YOLO_v8n"

    # 兜底：返回原始（很少发生）
    return t

def pick_col(d: pd.DataFrame, candidates, default=None):
    for k in candidates:
        if k in d.columns:
            return d[k]
    return default

def load_and_normalize(csv_path: Path, default_model_name=None) -> pd.DataFrame:
    """
    标准化列名为：
    ['weather','model','mAP50','mAP50_95','p50_ms','p95_ms']
    """
    df = pd.read_csv(csv_path)

    # weather
    weather = pick_col(df, ["weather", "cond", "condition", "Condition", "label"])
    if weather is None:
        raise ValueError(f"{csv_path} 缺少 weather/cond 列")

    # model
    model_series = pick_col(df, ["model", "Model"])
    if model_series is None:
        mname = default_model_name or "yolo_all"
        model_series = pd.Series([mname] * len(df))

    # metrics
    mAP50 = pick_col(df, ["mAP50", "map50", "metrics/mAP50", "metrics/mAP50(B)"])
    mAP5095 = pick_col(df, ["mAP50_95", "mAP50-95", "map5095", "metrics/mAP50-95", "metrics/mAP50-95(B)"])
    p50 = pick_col(df, ["p50_ms", "latency_p50_ms", "p50", "median_ms"])
    p95 = pick_col(df, ["p95_ms", "p95", "tail_ms"])

    out = pd.DataFrame({
        "weather": weather.astype(str).str.strip().str.lower(),
        "model": model_series.astype(str).str.strip(),
        "mAP50": mAP50 if mAP50 is not None else np.nan,
        "mAP50_95": mAP5095 if mAP5095 is not None else np.nan,
        "p50_ms": p50 if p50 is not None else np.nan,
        "p95_ms": p95 if p95 is not None else np.nan,
    })

    # 归一化模型显示名（解决 YOLO 路径长的问题）
    out["model"] = out["model"].map(normalize_model_display)
    return out

def merge_dfs(dfs):
    df = pd.concat(dfs, ignore_index=True)
    # 只保留指定天气并排序
    df = df[df["weather"].isin(WEATHER_ORDER)].copy()
    df["weather"] = pd.Categorical(df["weather"], categories=WEATHER_ORDER, ordered=True)
    # 同一天气同一模型如果有重复，取均值
    df = df.groupby(["weather", "model"], as_index=False, observed=False).mean(numeric_only=True)
    return df

def plot_grouped_bars(ax, data_pivot, title, ylabel, ylim=None, annotate=False, add_legend=False):
    """
    data_pivot: index=weather, columns=model, values=metric
    """
    models = [m for m in MODEL_ORDER if m in data_pivot.columns]
    models += [m for m in data_pivot.columns if m not in models]
    x = np.arange(len(data_pivot.index))
    n = len(models)
    width = min(0.8 / max(1, n), 0.22)  # 每根柱宽度
    for i, m in enumerate(models):
        vals = data_pivot[m].values
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width=width, label=m,
               color=MODEL_COLORS.get(m))
        if annotate:
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi + (i - (n - 1) / 2) * width, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", " ") for w in data_pivot.index], rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if add_legend:
        ax.legend(frameon=False, ncol=min(3, len(models)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_csv", default="results/exp_log.csv", help="YOLO(all) by-weather CSV, e.g. exp_log.csv")
    ap.add_argument("--frcnn_csv", default="results/frcnn_by_weather.csv")
    ap.add_argument("--rtdetr_csv", default="results/rtdetr_by_weather.csv")
    ap.add_argument("--outdir", default="results/figs")
    ap.add_argument("--annotate", action="store_true", help="Annotate bars with values")
    ap.add_argument("--acc_legend_top", action="store_true",
                    help="(可选)也给 ACC 子图各自保留图例；默认使用上方全局图例")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    dfs.append(load_and_normalize(Path(args.yolo_csv), default_model_name="yolo_all"))
    dfs.append(load_and_normalize(Path(args.frcnn_csv)))
    dfs.append(load_and_normalize(Path(args.rtdetr_csv)))

    all_df = merge_dfs(dfs)
    out_combined = outdir / "combined_all_models_by_weather.csv"
    all_df.to_csv(out_combined, index=False)

    # ---------- ACCURACY：mAP50 / mAP50_95 ----------
    acc_df = all_df[["weather", "model", "mAP50", "mAP50_95"]].copy()

    # mAP50
    m50_pivot = acc_df.pivot(index="weather", columns="model", values="mAP50").sort_index()
    # mAP50_95
    m5095_pivot = acc_df.pivot(index="weather", columns="model", values="mAP50_95").sort_index()

    if m50_pivot.notna().any().any() or m5095_pivot.notna().any().any():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        plot_grouped_bars(axes[0], m50_pivot, "mAP@0.50 (by weather)", "mAP50", ylim=(0, 1.0),
                          annotate=args.annotate, add_legend=args.acc_legend_top)
        plot_grouped_bars(axes[1], m5095_pivot, "mAP@[0.50:0.95] (by weather)", "mAP50–95", ylim=(0, 1.0),
                          annotate=args.annotate, add_legend=args.acc_legend_top)

        # 移除子图内图例，统一放到图外上方
        if not args.acc_legend_top:
            for ax in axes:
                leg = ax.get_legend()
                if leg:
                    leg.remove()
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                           ncol=min(4, len(labels)), frameon=False)

        acc_path = outdir / "acc_by_weather_all_models.png"
        fig.savefig(acc_path, dpi=200, bbox_inches="tight")
        print(f"[OK] ACC figure -> {acc_path}")
        plt.close(fig)
    else:
        print("[WARN] 没有可用的 mAP 列，跳过 ACC 图。")

    # ---------- LATENCY：p50 ----------
    lat_df = all_df[["weather", "model", "p50_ms", "p95_ms"]].copy()
    if lat_df["p50_ms"].notna().any():
        p50_pivot = lat_df.pivot(index="weather", columns="model", values="p50_ms").sort_index()
        # 误差线用 p95 - p50（如果存在）
        err = None
        if lat_df["p95_ms"].notna().any():
            p95_pivot = lat_df.pivot(index="weather", columns="model", values="p95_ms").sort_index()
            err = (p95_pivot - p50_pivot).clip(lower=0)

        # 绘图（图例放右侧图外）
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        models = [m for m in MODEL_ORDER if m in p50_pivot.columns]
        models += [m for m in p50_pivot.columns if m not in models]
        x = np.arange(len(p50_pivot.index))
        n = len(models)
        width = min(0.8 / max(1, n), 0.25)
        for i, m in enumerate(models):
            vals = p50_pivot[m].values
            yerr = err[m].values if err is not None and m in err.columns else None
            ax.bar(x + (i - (n - 1) / 2) * width, vals, width=width, label=m,
                   yerr=yerr, capsize=3 if yerr is not None else 0,
                   color=MODEL_COLORS.get(m))
            if args.annotate:
                for xi, v in zip(x, vals):
                    if np.isfinite(v):
                        ax.text(xi + (i - (n - 1) / 2) * width, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace("_", " ") for w in p50_pivot.index], rotation=20)
        ax.set_title("Latency p50 (ms) by weather")
        ax.set_ylabel("Latency (ms, p50)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # 图外右侧图例
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        lat_path = outdir / "latency_by_weather_all_models.png"
        fig.savefig(lat_path, dpi=200, bbox_inches="tight")
        print(f"[OK] LAT figure -> {lat_path}")
        plt.close(fig)
    else:
        print("[WARN] 没有 p50_ms 列，跳过 LAT 图。")

    print(f"[OK] 合并表 -> {outdir/'combined_all_models_by_weather.csv'}")

if __name__ == "__main__":
    main()
