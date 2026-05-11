# Upload Manifest

This directory contains the files intended for the final project code repository.

Included:

- `README.md`: project overview, setup, data locations, commands, and summarized results.
- `requirements.txt`: Python packages needed for the included scripts.
- `scripts/`: preprocessing, training, evaluation, plotting, and scene-difficulty classification code.
- `yamls/`: YOLO dataset configuration files for each BDD100K weather split.
- `results/`: compact result CSVs, summary JSON/TXT files, and final figures.

Excluded:

- Raw BDD100K data.
- `datasets/` symlinks/copies.
- `runs/` and `scripts/runs/` training directories.
- Model checkpoint files such as `.pt` and `.pth`.

The excluded files are large and can be regenerated or referenced from the paths documented in `README.md`.

