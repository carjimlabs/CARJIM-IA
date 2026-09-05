# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Blood-smear cell analyzer, **two-stage**:
1. **Detector** (`models/carjim_best.pt`) — YOLOv8, 3 classes: RBC/Hemácia, WBC/Leucócito,
   Platelets/Plaqueta. Works well on wide-field teacher photos.
2. **Subtype classifier** (`models/wbc_classifier.pt`) — YOLOv8-cls; takes each detected WBC
   crop and labels it Neutrophil/Lymphocyte/Monocyte/Eosinophil/Basophil.

Didactic tool: a teacher drops an image into `Imagens a serem analisadas/` (or picks one in
the GUI) and gets it back annotated with the 7 displayed classes (Hemácia / Neutrófilo /
Linfócito / Monócito / Eosinófilo / Basófilo / Plaqueta).

**Why two stages** (read before trying to "simplify" back to one model): a single 7-class
detector was tried repeatedly and kept collapsing. Raabin-WBC is the only source with
leukocyte subtypes and it only provides zoomed single-cell crops — training a detector on it
makes the model expect large, centered WBCs and fail on wide-field photos, and every attempt
to bridge the scale gap (pseudo-labels, copy-paste aug) traded one regression for another.
Detecting first, then classifying the crop, keeps RBC/platelet detection identical to the
proven 3-class model. Dead-end artifacts from that effort still exist:
`models/carjim_v3_7class_singledetector.pt`, `models/carjim_7class_colapsado.pt`,
`scripts/09`–`11`, and the 7-class state of `data/bccd/` (restore from
`data/bccd/labels_backup_pre_migracao/` if you ever retrain the 3-class detector).

## Setup and commands

```
pip install -r requirements.txt
```

There is no test suite, linter, or build tool configured in this repo — don't invent
`pytest`/lint commands. Everything is run directly with the Python interpreter.

**Dataset + training pipeline** (numbered scripts in `scripts/`, run in order — each has a
docstring explaining what it does and why):

**Stage-1 detector** (3-class YOLOv8 — `models/carjim_best.pt`):
```
python scripts/01_download_dataset.py       # BCCD dataset -> data/bccd/raw/
python scripts/02_prepare_dataset.py         # Pascal VOC -> YOLO format, train/val split, dataset.yaml
python scripts/06_merge_txlpbc.py            # optional: merges TXL-PBC into train
python scripts/07_add_platelet_crops.py      # optional: merges platelet.zip crops (auto-boxed by color segmentation)
python scripts/08_merge_roboflow_allidb.py   # optional: merges a Roboflow ALL_IDB export (WBC+platelet, no RBC — see caveat)
python scripts/03_train.py                   # fine-tunes YOLOv8 (imgsz=1280), writes models/carjim_best.pt
```
The current `models/carjim_best.pt` is the working 3-class checkpoint
(`models/carjim_best_3class_pre_diferencial.pt` is an identical backup). `data/bccd/` was
later migrated to a dead-end 7-class layout — before re-running `03_train.py`, restore the
3-class labels/yaml from `data/bccd/labels_backup_pre_migracao/` (+ `dataset.yaml.bak_pre_migracao`).

**Stage-2 WBC subtype classifier** (YOLOv8-cls — `models/wbc_classifier.pt`):
```
python scripts/12_build_wbc_classifier_data.py  # Raabin zips -> data/wbc_cls/<split>/<Class>/*.jpg (ImageFolder)
python scripts/13_train_wbc_classifier.py       # trains YOLOv8-cls (imgsz=128), balances rare classes by duplication, writes models/wbc_classifier.pt
```

**Abandoned single-detector experiment** — `scripts/09_migrate_wbc_taxonomy.py`,
`10_merge_raabin_wbc.py`, `11_paste_wbc_wide_field.py` built a 7-class detector dataset; it
never generalized to wide field. Kept for reference only; not part of the live pipeline.

**End-user tools** (need `models/carjim_best.pt`; classifier optional — without it WBCs show as
generic "Leucócito"):

```
python app.py                        # interactive GUI: pick classes, pick an image file, see result
python scripts/04_watch_and_infer.py # watches Imagens a serem analisadas/, auto-processes new files
```

`03_train.py` always fine-tunes from `models/carjim_best.pt` if it exists (falls back to the
pretrained `yolov8s.pt` at repo root only the very first time). Training is slow and GPU-bound —
see the background-task caveat below before launching it from an agent session.

**One-off reinforcement**: `python scripts/05_add_wide_field_sample.py "path/to/photo.jpg"` generates
pseudo-labels for a single real photo (via tiling + the current model) and adds it to the training
set — used when a photo's zoom level doesn't match the training data. Always inspect the preview in
`data/bccd/pseudo_label_previews/` before trusting it, then re-run `03_train.py`.

## Architecture

**Class taxonomy** — `scripts/detection_core.py` is the single source of truth:
- The **detector** emits 3 classes (`RBC`/`WBC`/`Platelets`).
- `CLASSES` = `["RBC", "Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil",
  "Platelets"]` is the **display** taxonomy (tuple class-ids: RBC=0, subtypes 1–5,
  Platelets=6). Id 7 = `"WBC"` = fallback when the classifier is absent.
- `primary_detections()` returns tuples already in this display taxonomy: RBC/Platelets pass
  straight through, each WBC box is cropped and run through the classifier to get its subtype id.
- `CLASS_LABELS_PT` / `CLASS_COLORS` — PT-BR names and colors. Never redeclare these elsewhere.
- The classifier's own dataset (`12`) is built straight from the Raabin zips; the abandoned
  `09`/`10` still import `CLASSES` but are not run.

**Shared detection logic lives in `scripts/detection_core.py`**, imported by both
`scripts/04_watch_and_infer.py` (folder-watch CLI) and `app.py` (Tkinter GUI). Do not
duplicate detection/drawing logic in either consumer — add it to `detection_core.py` instead.
`load_model()` returns a `(Detector, device)` pair — `Detector` wraps the 3-class YOLO
detector + the YOLOv8-cls classifier and exposes `.names` (the display taxonomy dict) so the
consumers need no change. Key pieces: `primary_detections()` (full-image detect + per-WBC
classify), `platelet_tile_scan()` + `merge_platelet_detections()` (a second,
slower reinforcement pass in small tiles specifically for platelets — they're much smaller
than RBC/WBC and disappear in a single full-image pass on wide-field photos; expected platelet
size is computed as a ratio of the RBC size detected in the *same* image, so it self-adjusts to
photo zoom), `draw_detections()`.

`detection_core._resolve()` (used for both `MODEL_PATH` and `CLASSIFIER_PATH`, and exported as
`resolve_model_path()`) resolves `models/*` relative to `sys.executable` when frozen
(PyInstaller) or relative to `__file__` otherwise — this must stay that way for the app to find
the models both as a script and once packaged as a `.exe`.

**Dataset merge scripts (`06`/`07`/`08`) are additive and append-only**: each copies/generates
images+labels into `data/bccd/images/train/` and `data/bccd/labels/train/` with a distinct
filename prefix (`txlpbc_`, `pltcrop_`, `allidb_`) and leaves `data/bccd/images/val/` (the
original BCCD split) untouched, so validation metrics stay comparable across retrains. When
writing a new merge script, follow this same pattern (prefix + train-only + preserve val).

**One-time exception to "preserve val": `10_merge_raabin_wbc.py`** also writes a fraction of
its images into `images/val/` + `labels/val/` (prefix `raabin_`). The original BCCD val split
has zero leukocyte-subtype boxes, so the 5 new classes could not be measured at all otherwise.
This is a deliberate, documented break of the rule above — justified only because the taxonomy
itself changed. Future merges under the 7-class taxonomy go back to train-only.

**Why `10` emits crops, not full fields**: Raabin's full-field images are NOT exhaustively
annotated (they boxed a sample of cells per image), so a dense field has visible leukocytes
with no box → training-as-background. `10` keeps only *isolated* annotated cells (no other
annotated cell within `MIN_CELL_SEPARATION` px) and emits a `CROP_SIZE`-px window around each
(half-window ≤ separation, so no other annotated cell fits) — one leukocyte per image, boxed,
with RBC context at a scale close to BCCD/TXL. Read the module docstring before changing the
orientation handling (First microscope = no rotation, Second = 90° CW; verified against the
data, the official demo script's rotation is wrong for First).

**Known caveat baked into `08_merge_roboflow_allidb.py`**: that Roboflow export only has
WBC/Platelet ground truth (no RBC boxes), and pseudo-labeling RBC for it via the current model
was tried and abandoned — the export's own baked-in brightness/contrast augmentation pushes
some images out of the model's known color domain and it hallucinates hundreds of false RBC
boxes. That script therefore does a plain merge of the real WBC/Platelet boxes only. Read its
module docstring before changing this. (Under the 7-class taxonomy its generic WBC boxes are
also dropped — see the taxonomy note above — so in practice it now merges platelet boxes only.)

**GUI threading model (`app.py`)**: Tkinter is single-threaded and not thread-safe. Model
loading and inference run on background `threading.Thread`s that post results through a
`queue.Queue`, polled on the main thread via `root.after(100, ...)`. Never touch Tk widgets
from a worker thread. Checkbox toggles re-filter/redraw from the cached last-inference result
without re-running the model — except re-enabling "Plaqueta" after it was processed unchecked,
which triggers the (slow) platelet tile-scan on demand since that pass was skipped originally.

**Windows/background-task caveat**: in this environment, launching long GPU jobs (training) via
the Bash tool's `run_in_background` has been unreliable — the task can be killed on its own well
before completion, and `TaskStop` on an unrelated `Monitor` can kill a sibling background Bash
task. For `03_train.py` or anything else expected to run more than ~10-30 minutes, launch it as
a fully detached OS process instead (PowerShell `Start-Process ... -RedirectStandardOutput ...
-PassThru`) and poll its log file with one-off (non-backgrounded) reads.
