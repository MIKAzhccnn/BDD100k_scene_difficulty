#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-weather YOLOv8 fine-tuning launcher.

- For each weather in the list, runs:
    yolo detect train ... name=yv8n_ft_<weather>
    yolo val   ... (on the same weather split)

- Saves raw stdout/stderr to logs/<weather>_train.log and _val.log
- You can tweak hyper-params via command line flags.

Example:
  python train_ft_by_weather.py \
    --project_root ~/Projects/moca \
    --weathers clear,rainy,snowy,overcast,partly_cloudy \
    --model yolov8n.pt --epochs 30 --imgsz 640 --batch 16 --device 0
"""

import argparse
import subprocess
import shlex
from pathlib import Path
from datetime import datetime

def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"==== CMD @ {datetime.now().isoformat(sep=' ', timespec='seconds')} ====\n")
        f.write(cmd + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="")
        ret = proc.wait()
    return ret

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="~/Projects/moca", help="root that contains yamls/")
    ap.add_argument("--weathers", default="clear,rainy,snowy,overcast,partly_cloudy,foggy",
                    help="comma list (use partly_cloudy with underscore)")
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained model to start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--close_mosaic", type=int, default=5)
    ap.add_argument("--cos_lr", type=str, default="True")  # keep string for CLI
    ap.add_argument("--resume", action="store_true", help="resume if there is an existing run")
    args = ap.parse_args()

    root = Path(args.project_root).expanduser()
    yamls_dir = root / "yamls"
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]
    print(f"Weathers: {weathers}")
    print(f"YAMLs dir: {yamls_dir}")

    for w in weathers:
        yaml_path = yamls_dir / f"bdd_{w}.yaml"
        if not yaml_path.exists():
            print(f"[WARN] YAML not found: {yaml_path} (skip)")
            continue

        run_name = f"yv8n_ft_{w}"
        # --- train ---
        if args.resume:
            train_cmd = (
                f"yolo detect train resume=True "
                f"data={yaml_path} imgsz={args.imgsz} batch={args.batch} device={args.device} workers={args.workers} "
                f"epochs={args.epochs} lr0={args.lr0} freeze={args.freeze} "
                f"patience={args.patience} close_mosaic={args.close_mosaic} cos_lr={args.cos_lr} "
                f"name={run_name}"
            )
        else:
            train_cmd = (
                f"yolo detect train "
                f"model={args.model} data={yaml_path} imgsz={args.imgsz} batch={args.batch} "
                f"device={args.device} workers={args.workers} "
                f"epochs={args.epochs} lr0={args.lr0} freeze={args.freeze} "
                f"patience={args.patience} close_mosaic={args.close_mosaic} cos_lr={args.cos_lr} "
                f"name={run_name}"
            )

        print(f"\n===== TRAIN [{w}] =====")
        code = run(train_cmd, logs_dir / f"{w}_train.log")
        if code != 0:
            print(f"[ERROR] train failed on {w} (exit {code}), skipping val")
            continue

        # --- val on the same weather split ---
        best_path = root / "runs" / "detect" / run_name / "weights" / "best.pt"
        val_cmd = (
            f"yolo val model={best_path} data={yaml_path} "
            f"imgsz={args.imgsz} batch=1 device={args.device}"
        )
        print(f"\n===== VAL [{w}] =====")
        code = run(val_cmd, logs_dir / f"{w}_val.log")
        if code != 0:
            print(f"[ERROR] val failed on {w} (exit {code})")

    print("\nAll done.")

if __name__ == "__main__":
    main()
