#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Paths ----------------
root = Path('~/Projects/moca/results').expanduser()       # 与原脚本一致（All 模型的结果根目录）
all_json = root / 'exp_log.json'                          # All 的结果（json 列里应含 weather/mAP50/mAP50_95/lat_p50_ms）
ft_csv   = root / 'combined_weather_results.csv'          # 我们一键脚本生成的 FT 汇总

# ---------------- Load: All (from exp_log.json) ----------------
j = json.loads(all_json.read_text(encoding='utf-8'))
df_all = pd.DataFrame(j)

order = ["clear","partly_cloudy","overcast","rainy","snowy","foggy"]
df_all['weather'] = pd.Categorical(df_all['weather'], categories=order, ordered=True)
df_all = df_all.sort_values('weather').reset_index(drop=True)

# 统一列名（稳妥起见）
if 'mAP50_95' not in df_all.columns and 'mAP50-95' in df_all.columns:
    df_all = df_all.rename(columns={'mAP50-95':'mAP50_95'})
if 'lat_p50_ms' not in df_all.columns and 'p50_ms' in df_all.columns:
    df_all = df_all.rename(columns={'p50_ms':'lat_p50_ms'})

# ---------------- Load: FT (from combined_weather_results.csv) ----------------
if not ft_csv.exists():
    raise FileNotFoundError(f"FT 汇总不存在：{ft_csv}\n先运行 weather_eval_all.py 生成该文件。")

df_ft_raw = pd.read_csv(ft_csv)
# 只取 Per-weather FT；若重复，按 mAP50_95 最高的保留
df_ft_raw = df_ft_raw[df_ft_raw['group'] == 'Per-weather FT'].copy()
df_ft_raw['weather'] = pd.Categorical(df_ft_raw['weather'], categories=order, ordered=True)
df_ft_raw.sort_values(['weather','mAP50_95'], ascending=[True, False], inplace=True)
df_ft = df_ft_raw.drop_duplicates(subset=['weather'], keep='first').reset_index(drop=True)

# ---------------- Align by weather ----------------
weathers = [w for w in order if w in set(df_all['weather'].astype(str)) or w in set(df_ft['weather'].astype(str))]

def arr_from(df, col):
    s = []
    for w in weathers:
        row = df[df['weather'].astype(str) == w]
        if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
            s.append(np.nan)
        else:
            s.append(float(row.iloc[0][col]))
    return s

# All
a_m50    = arr_from(df_all, 'mAP50')
a_m5095  = arr_from(df_all, 'mAP50_95')
a_lat50  = arr_from(df_all, 'lat_p50_ms')

# FT
f_m50    = arr_from(df_ft, 'mAP50')
f_m5095  = arr_from(df_ft, 'mAP50_95')
f_lat50  = arr_from(df_ft, 'p50_ms')   # FT 的 p50 列名

# ---------------- Plot helpers ----------------
def annotate_bars(ax, bars, fmt):
    for b in bars:
        v = b.get_height()
        if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
            continue
        ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v),
                ha='center', va='bottom', fontsize=8)

# ---------------- Figure 1: Accuracy (4 bars per weather) ----------------
fig1, ax1 = plt.subplots(figsize=(11,6))
x = np.arange(len(weathers))
k = 4
width = 0.8 / k

bars = []
bars.append(ax1.bar(x + (-1.5)*width, a_m50,   width, label='All: mAP@0.5'))
bars.append(ax1.bar(x + (-0.5)*width, a_m5095, width, label='All: mAP@0.5:0.95'))
bars.append(ax1.bar(x + ( 0.5)*width, f_m50,   width, label='FT:  mAP@0.5'))
bars.append(ax1.bar(x + ( 1.5)*width, f_m5095, width, label='FT:  mAP@0.5:0.95'))

ax1.set_xticks(x); ax1.set_xticklabels(weathers, rotation=20, ha='right')
ax1.set_ylabel('mAP'); ax1.set_title('YOLOv8n Accuracy by Weather (All vs Per-weather FT)')
ax1.grid(axis='y', linestyle='--', alpha=0.35); ax1.legend(frameon=False, ncols=2)

for bb,fmt in zip(bars, ["{:.3f}","{:.3f}","{:.3f}","{:.3f}"]):
    annotate_bars(ax1, bb, fmt)

fig1.tight_layout()
(fig1).savefig(root/'acc_all_vs_ft_by_weather.png', dpi=180)

# ---------------- Figure 2: Latency p50 (2 bars per weather) ----------------
fig2, ax2 = plt.subplots(figsize=(10,5))
k = 2
width = 0.8 / k

b_all = ax2.bar(x - width/2, a_lat50, width, label='All: p50 (ms)')
b_ft  = ax2.bar(x + width/2, f_lat50, width, label='FT:  p50 (ms)')

ax2.set_xticks(x); ax2.set_xticklabels(weathers, rotation=20, ha='right')
ax2.set_ylabel('Latency (ms)'); ax2.set_title('Latency p50 by Weather (All vs Per-weather FT)')
ax2.grid(axis='y', linestyle='--', alpha=0.35); ax2.legend(frameon=False)

for bb in (b_all, b_ft):
    annotate_bars(ax2, bb, "{:.2f}")

fig2.tight_layout()
(fig2).savefig(root/'latency_p50_all_vs_ft_by_weather.png', dpi=180)

print('Saved to:',
      root/'acc_all_vs_ft_by_weather.png',
      'and',
      root/'latency_p50_all_vs_ft_by_weather.png')
