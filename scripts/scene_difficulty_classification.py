#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene difficulty classification for the BDD100K proposal experiment.

The proposal defines "difficult" driving scenes using signals such as low
visibility, high object density, and many small objects. This script turns the
BDD100K image labels into structured tabular features, derives an easy/hard
target from those signals, and evaluates several classifiers.

Outputs are written to results/scene_difficulty by default:
  - train_features.csv / val_features.csv
  - metrics.csv
  - predictions_<best_model>.csv
  - classification_report_<best_model>.txt
  - confusion_matrix_<best_model>.png
  - metrics_bar.png
  - feature_importance_<best_model>.png
  - weather_difficulty_rate.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import ijson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BDD_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

ADVERSE_WEATHER = {"rainy", "snowy", "foggy"}
LOW_LIGHT_TIMES = {"night", "dawn/dusk"}
IMG_W = 1280.0
IMG_H = 720.0
IMG_AREA = IMG_W * IMG_H
SMALL_OBJECT_AREA = 32.0 * 32.0


def iter_bdd_items(json_path: Path) -> Iterable[dict]:
    with json_path.open("rb") as f:
        yield from ijson.items(f, "item")


def safe_attr(attrs: dict, key: str) -> str:
    value = attrs.get(key, "undefined")
    if value is None:
        return "undefined"
    return str(value).strip().lower().replace(" ", "_")


def image_brightness(images_root: Path, split: str, name: str, images_size: str) -> float:
    img_path = images_root / images_size / split / name
    if not img_path.exists():
        return math.nan
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L").resize((32, 18))
            return float(np.asarray(gray, dtype=np.float32).mean() / 255.0)
    except Exception:
        return math.nan


def extract_features(
    json_path: Path,
    images_root: Path,
    split: str,
    images_size: str,
    max_items: int,
    use_brightness: bool,
) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(iter_bdd_items(json_path)):
        if max_items > 0 and len(rows) >= max_items:
            break

        attrs = item.get("attributes", {}) or {}
        labels = item.get("labels", []) or []
        weather = safe_attr(attrs, "weather")
        timeofday = safe_attr(attrs, "timeofday")
        scene = safe_attr(attrs, "scene")

        obj_count = 0
        areas = []
        small_count = 0
        occluded_count = 0
        truncated_count = 0
        class_counts = defaultdict(int)

        for lab in labels:
            box = lab.get("box2d")
            cat = lab.get("category")
            if not box or cat not in BDD_CLASSES:
                continue

            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            if area <= 0:
                continue

            obj_count += 1
            areas.append(area / IMG_AREA)
            small_count += int(area < SMALL_OBJECT_AREA)
            class_counts[cat] += 1

            lab_attrs = lab.get("attributes", {}) or {}
            occluded_count += int(bool(lab_attrs.get("occluded", False)))
            truncated_count += int(bool(lab_attrs.get("truncated", False)))

        area_arr = np.asarray(areas, dtype=float)
        brightness = (
            image_brightness(images_root, split, item["name"], images_size)
            if use_brightness
            else math.nan
        )

        row = {
            "split": split,
            "name": item.get("name", ""),
            "weather": weather,
            "timeofday": timeofday,
            "scene": scene,
            "object_count": obj_count,
            "avg_box_area": float(area_arr.mean()) if len(area_arr) else 0.0,
            "median_box_area": float(np.median(area_arr)) if len(area_arr) else 0.0,
            "max_box_area": float(area_arr.max()) if len(area_arr) else 0.0,
            "small_object_ratio": small_count / obj_count if obj_count else 0.0,
            "occluded_ratio": occluded_count / obj_count if obj_count else 0.0,
            "truncated_ratio": truncated_count / obj_count if obj_count else 0.0,
            "brightness": brightness,
            "is_adverse_weather": int(weather in ADVERSE_WEATHER),
            "is_low_light": int(timeofday in LOW_LIGHT_TIMES),
        }
        for cls in BDD_CLASSES:
            row[f"count_{cls.replace(' ', '_')}"] = class_counts[cls]
        rows.append(row)

        if (idx + 1) % 5000 == 0:
            print(f"[feature] scanned {idx + 1:,} items, kept {len(rows):,}")

    return pd.DataFrame(rows)


def add_difficulty_target(train: pd.DataFrame, val: pd.DataFrame, hard_quantile: float):
    density_scale = max(float(train["object_count"].quantile(0.90)), 1.0)
    bright_q25 = float(train["brightness"].dropna().quantile(0.25)) if train["brightness"].notna().any() else 0.35
    bright_q75 = float(train["brightness"].dropna().quantile(0.75)) if train["brightness"].notna().any() else 0.65
    bright_range = max(bright_q75 - bright_q25, 1e-6)

    def score(df: pd.DataFrame) -> pd.Series:
        brightness = df["brightness"].fillna(train["brightness"].median())
        low_visibility = ((bright_q75 - brightness) / bright_range).clip(0.0, 1.0)
        density = (df["object_count"] / density_scale).clip(0.0, 1.0)
        small = df["small_object_ratio"].clip(0.0, 1.0)
        occlusion = df[["occluded_ratio", "truncated_ratio"]].max(axis=1).clip(0.0, 1.0)
        context = 0.15 * df["is_adverse_weather"] + 0.15 * df["is_low_light"]
        return 0.35 * density + 0.25 * small + 0.20 * low_visibility + 0.20 * occlusion + context

    train = train.copy()
    val = val.copy()
    train["difficulty_score"] = score(train)
    val["difficulty_score"] = score(val)
    threshold = float(train["difficulty_score"].quantile(hard_quantile))
    train["hard_scene"] = (train["difficulty_score"] >= threshold).astype(int)
    val["hard_scene"] = (val["difficulty_score"] >= threshold).astype(int)
    return train, val, threshold


def build_models(random_state: int):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def plot_metrics(metrics: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_df = metrics.set_index("model")[["accuracy", "precision", "recall", "f1"]]
    plot_df.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Scene Difficulty Classification Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_weather_rates(train: pd.DataFrame, val: pd.DataFrame, out_path: Path):
    both = pd.concat([train, val], ignore_index=True)
    rates = both.groupby(["split", "weather"])["hard_scene"].mean().unstack(0).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rates.plot(kind="bar", ax=ax, width=0.72)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("hard-scene rate")
    ax.set_title("Hard Scene Rate by Weather")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(pipeline: Pipeline, numeric_cols, categorical_cols, out_path: Path):
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocess"]
    feature_names = list(pre.get_feature_names_out())

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        return

    top = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(18)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#4c78a8")
    ax.set_title("Top Features for Best Classifier")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="~/Downloads/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json")
    ap.add_argument("--val_json", default="~/Downloads/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json")
    ap.add_argument("--images_root", default="~/Downloads/BDD100K/bdd100k/bdd100k/images")
    ap.add_argument("--images_size", default="100k", choices=["100k", "10k"])
    ap.add_argument("--outdir", default="results/scene_difficulty")
    ap.add_argument("--max_train", type=int, default=12000, help="0 means use all train labels")
    ap.add_argument("--max_val", type=int, default=3000, help="0 means use all val labels")
    ap.add_argument("--hard_quantile", type=float, default=0.65)
    ap.add_argument("--no_brightness", action="store_true")
    ap.add_argument("--reuse_features", action="store_true")
    ap.add_argument("--random_state", type=int, default=683)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_csv = outdir / "train_features.csv"
    val_csv = outdir / "val_features.csv"

    if args.reuse_features and train_csv.exists() and val_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    else:
        train_df = extract_features(
            Path(args.train_json).expanduser(),
            Path(args.images_root).expanduser(),
            "train",
            args.images_size,
            args.max_train,
            not args.no_brightness,
        )
        val_df = extract_features(
            Path(args.val_json).expanduser(),
            Path(args.images_root).expanduser(),
            "val",
            args.images_size,
            args.max_val,
            not args.no_brightness,
        )

    train_df, val_df, threshold = add_difficulty_target(train_df, val_df, args.hard_quantile)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    categorical_cols = ["weather", "timeofday", "scene"]
    numeric_cols = [
        c
        for c in train_df.columns
        if c
        not in {
            "split",
            "name",
            "weather",
            "timeofday",
            "scene",
            "difficulty_score",
            "hard_scene",
        }
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ]
    )

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df["hard_scene"]
    X_val = val_df[numeric_cols + categorical_cols]
    y_val = val_df["hard_scene"]

    rows = []
    fitted = {}
    for name, clf in build_models(args.random_state).items():
        pipe = Pipeline([("preprocess", preprocess), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_val, pred),
                "precision": precision_score(y_val, pred, zero_division=0),
                "recall": recall_score(y_val, pred, zero_division=0),
                "f1": f1_score(y_val, pred, zero_division=0),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "hard_threshold": threshold,
                "train_hard_rate": float(y_train.mean()),
                "val_hard_rate": float(y_val.mean()),
            }
        )
        fitted[name] = (pipe, pred)
        print(f"[model] {name}: F1={rows[-1]['f1']:.3f}, Acc={rows[-1]['accuracy']:.3f}")

    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    best_name = str(metrics.iloc[0]["model"])
    best_pipe, best_pred = fitted[best_name]

    pd.DataFrame(
        {
            "name": val_df["name"],
            "weather": val_df["weather"],
            "timeofday": val_df["timeofday"],
            "scene": val_df["scene"],
            "difficulty_score": val_df["difficulty_score"],
            "hard_scene": y_val,
            "pred_hard_scene": best_pred,
        }
    ).to_csv(outdir / f"predictions_{best_name}.csv", index=False)

    report = classification_report(y_val, best_pred, target_names=["easy", "hard"], digits=4)
    (outdir / f"classification_report_{best_name}.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_val, best_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["easy", "hard"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_matrix_{best_name}.png", dpi=200)
    plt.close(fig)

    plot_metrics(metrics, outdir / "metrics_bar.png")
    plot_weather_rates(train_df, val_df, outdir / "weather_difficulty_rate.png")
    plot_feature_importance(best_pipe, numeric_cols, categorical_cols, outdir / f"feature_importance_{best_name}.png")

    summary = {
        "best_model": best_name,
        "hard_threshold": threshold,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "train_hard_rate": float(y_train.mean()),
        "val_hard_rate": float(y_val.mean()),
        "metrics": metrics.to_dict(orient="records"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Best model: {best_name}")
    print(f"[done] Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
