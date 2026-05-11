# BDD100K Weather and Scene Difficulty Project

This repository contains the code and generated test results for the CPSC 683 final project:
**Analyzing the Impact of Environmental Conditions on Driving Scene Difficulty Using BDD100K**.

The work is split into two experiment tracks:

1. **Proposal-aligned scene difficulty classification**: extracts structured scene features from BDD100K metadata and object annotations, defines easy/hard scenes, and evaluates classical ML classifiers.
2. **Object detection robustness by weather**: compares YOLOv8n, Faster R-CNN, and RT-DETR across weather-specific BDD100K splits using mAP and latency.

## Environment

Use the project conda environment:

```bash
conda activate moca
```

Packages installed for the proposal-aligned experiment:

```bash
python -m pip install numpy pandas scikit-learn pillow matplotlib ijson
```

## Data

Original BDD100K data is expected at:

```text
~/Downloads/BDD100K
```

Weather-split YOLO-format data is already prepared at:

```text
datasets/bdd_yolo
```

The YOLO dataset can be regenerated from the BDD100K image-label JSON files:

```bash
python scripts/bdd2yolo_weather.py \
  --train_json ~/Downloads/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --val_json ~/Downloads/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --images_root ~/Downloads/BDD100K/bdd100k/bdd100k/images \
  --out_root datasets/bdd_yolo \
  --train_count 200 \
  --val_count 100 \
  --symlink
```

## Scene Difficulty Classification

Run the proposal-aligned experiment:

```bash
python scripts/scene_difficulty_classification.py
```

Default run size is 12,000 training images and 3,000 validation images for quick iteration. The current saved results were generated on all available BDD100K image labels:

```bash
python scripts/scene_difficulty_classification.py --max_train 0 --max_val 0
```

Outputs:

```text
results/scene_difficulty/
  train_features.csv
  val_features.csv
  metrics.csv
  summary.json
  predictions_gradient_boosting.csv
  classification_report_gradient_boosting.txt
  metrics_bar.png
  confusion_matrix_gradient_boosting.png
  feature_importance_gradient_boosting.png
  weather_difficulty_rate.png
```

Current validation results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.857 | 0.769 | 0.830 | 0.799 |
| Logistic Regression | 0.848 | 0.741 | 0.854 | 0.794 |
| Decision Tree | 0.787 | 0.651 | 0.812 | 0.723 |
| Random Forest | 0.786 | 0.674 | 0.726 | 0.699 |

The best model is **Gradient Boosting**. The hard-scene label uses the 65th percentile of the training difficulty score as the threshold, producing a 35.0% hard-scene rate in train and 34.3% in validation.

## Detection Robustness by Weather

Existing model outputs are stored under:

```text
scripts/runs/detect/
results/
```

Regenerate the all-model weather comparison tables and plots:

```bash
python scripts/plot_models_by_weather.py --annotate
```

Outputs:

```text
results/figs/combined_all_models_by_weather.csv
results/figs/acc_by_weather_all_models.png
results/figs/latency_by_weather_all_models.png
```

Current detector summary:

| Weather | Best mAP50 | Best mAP50-95 | Fastest p50 latency |
|---|---:|---:|---:|
| clear | Faster R-CNN, 0.599 | Faster R-CNN, 0.329 | YOLOv8n, 5.79 ms |
| partly cloudy | Faster R-CNN, 0.530 | RT-DETR, 0.282 | YOLOv8n, 5.79 ms |
| overcast | Faster R-CNN, 0.507 | Faster R-CNN, 0.285 | YOLOv8n, 5.79 ms |
| rainy | Faster R-CNN, 0.444 | Faster R-CNN, 0.237 | YOLOv8n, 5.79 ms |
| snowy | RT-DETR, 0.579 | RT-DETR, 0.331 | YOLOv8n, 5.79 ms |
| foggy | Faster R-CNN, 0.507 | Faster R-CNN, 0.253 | YOLOv8n, 5.79 ms |

## Main Scripts

| Script | Purpose |
|---|---|
| `scripts/scene_difficulty_classification.py` | Proposal-aligned feature extraction, easy/hard labeling, classifier training, metrics, and plots. |
| `scripts/bdd2yolo_weather.py` | Converts BDD100K image labels into YOLO-format weather-specific datasets. |
| `scripts/run_eval_and_latency.py` | Evaluates a YOLO model on each weather split and measures latency. |
| `scripts/train_ft_by_weather.py` | Fine-tunes YOLOv8n separately per weather split. |
| `scripts/collect_and_plot_weather.py` | Collects YOLO per-weather fine-tuning runs and plots comparisons. |
| `scripts/plot_models_by_weather.py` | Merges YOLO, Faster R-CNN, and RT-DETR CSVs and generates final comparison plots. |
| `scripts/frcnn_train_all.py` / `scripts/frcnn_eval_by_weather.py` | Faster R-CNN training and weather-wise evaluation. |
| `scripts/rtdetr_train_all.py` / `scripts/rtdetr_eval_by_weather.py` | RT-DETR training and weather-wise evaluation. |
