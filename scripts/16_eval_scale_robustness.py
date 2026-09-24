"""
Mede o quanto o detector aguenta fotos em distancias diferentes.

A validacao original (data/bccd/images/val, BCCD) so tem UM zoom. Este script
gera versoes sinteticas dela:

  - zoom 1.5x: recorte de 1/1.5 da imagem ampliado de volta (mais perto)
  - zoom 1x:   a imagem original
  - zoom 1/k:  mosaico k x k de imagens de val em resolucao cheia (a foto fica
               k vezes maior com as celulas k vezes menores em relacao ao
               quadro -- como uma foto de celular de alta resolucao tirada de
               mais longe)

e roda, para cada modelo passado, os modos de deteccao de detection_core
(base / TTA / normalizacao de escala / ambos), medindo precisao, recall e F1
por classe com IoU >= 0.5.

Ressalva: o BCCD nao anota todas as hemacias de cada imagem, entao a precisao
de RBC sai subestimada em todos os modos. Use o script para COMPARAR modos e
modelos, nao como numero absoluto.

Uso:
  python scripts/16_eval_scale_robustness.py                       # models/carjim_best.pt
  python scripts/16_eval_scale_robustness.py models/a.pt models/b.pt
Relatorio: runs/eval_scale/<modelo>.md (+ .csv)
"""
import csv
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detection_core as dc  # noqa: E402

VAL_IMAGES = PROJECT_ROOT / "data" / "bccd" / "images" / "val"
VAL_LABELS = PROJECT_ROOT / "data" / "bccd" / "labels" / "val"
OUT_DIR = PROJECT_ROOT / "runs" / "eval_scale"
DET_CLASSES = ["RBC", "WBC", "Platelets"]   # ids do dataset = ids do detector
ZOOMS = [1.5, 1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 6]
MOSAICS_PER_ZOOM = 40
IOU_MATCH = 0.5
RANDOM_SEED = 0
MODES = {
    "base": dict(scale_norm=False, tta=False),
    "tta": dict(scale_norm=False, tta=True),
    "escala": dict(scale_norm=True, tta=False),
    "escala+tta": dict(scale_norm=True, tta=True),
}


def load_val():
    items = []
    for img_path in sorted(VAL_IMAGES.iterdir()):
        if img_path.suffix.lower() not in dc.VALID_EXTENSIONS:
            continue
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        boxes = []
        label_path = VAL_LABELS / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().split("\n"):
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *(float(v) for v in p[1:])
                boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        items.append((image, boxes))
    return items


def zoom_in(image, boxes, factor, rng):
    W, H = image.size
    cw, ch = W / factor, H / factor
    x0, y0 = rng.uniform(0, W - cw), rng.uniform(0, H - ch)
    crop = image.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((W, H), Image.BICUBIC)
    out = []
    for c, x1, y1, x2, y2 in boxes:
        nx1, ny1 = max(x1, x0), max(y1, y0)
        nx2, ny2 = min(x2, x0 + cw), min(y2, y0 + ch)
        # mantem so celulas com pelo menos metade visivel
        if nx2 <= nx1 or ny2 <= ny1 or (nx2 - nx1) * (ny2 - ny1) < 0.5 * (x2 - x1) * (y2 - y1):
            continue
        out.append((c, (nx1 - x0) * factor, (ny1 - y0) * factor, (nx2 - x0) * factor, (ny2 - y0) * factor))
    return crop, out


def mosaic(items, k, rng):
    W, H = items[0][0].size
    canvas = Image.new("RGB", (W * k, H * k))
    out = []
    picks = rng.sample(items, k * k) if k * k <= len(items) else [rng.choice(items) for _ in range(k * k)]
    for idx, (image, boxes) in enumerate(picks):
        ox, oy = (idx % k) * W, (idx // k) * H
        canvas.paste(image.resize((W, H)), (ox, oy))
        sx, sy = W / image.size[0], H / image.size[1]
        out += [(c, x1 * sx + ox, y1 * sy + oy, x2 * sx + ox, y2 * sy + oy) for c, x1, y1, x2, y2 in boxes]
    return canvas, out


def build_zoom_sets(items):
    rng = random.Random(RANDOM_SEED)
    sets = {}
    for z in ZOOMS:
        if z > 1:
            sets[z] = [zoom_in(im, bx, z, rng) for im, bx in items]
        elif z == 1:
            sets[z] = list(items)
        else:
            k = round(1 / z)
            sets[z] = [mosaic(items, k, rng) for _ in range(MOSAICS_PER_ZOOM)]
    return sets


def match_counts(preds, gts, n_classes):
    """(tp, fp, fn) por classe, casamento guloso por confianca com IoU >= IOU_MATCH."""
    counts = [[0, 0, 0] for _ in range(n_classes)]
    for c in range(n_classes):
        p = sorted([d for d in preds if d[0] == c], key=lambda d: -d[5])
        g = [b[1:5] for b in gts if b[0] == c]
        if not p:
            counts[c][2] += len(g)
            continue
        if not g:
            counts[c][1] += len(p)
            continue
        ious = box_iou(torch.tensor([d[1:5] for d in p]), torch.tensor(g))
        used = torch.zeros(len(g), dtype=torch.bool)
        for i in range(len(p)):
            row = ious[i].clone()
            row[used] = -1
            j = int(row.argmax())
            if row[j] >= IOU_MATCH:
                used[j] = True
                counts[c][0] += 1
            else:
                counts[c][1] += 1
        counts[c][2] += int((~used).sum())
    return counts


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def zoom_label(z):
    return f"{z:g}x" if z >= 1 else f"1/{round(1 / z)}x"


def evaluate(model_path: Path, zoom_sets, device):
    det = YOLO(str(model_path))
    det.to(device)
    detector = dc.Detector(det, None, device)
    rows = []
    for mode, kwargs in MODES.items():
        for z, samples in zoom_sets.items():
            totals = [[0, 0, 0] for _ in DET_CLASSES]
            t0 = time.time()
            for image, gts in samples:
                preds = dc.detect_cells(detector, image, **kwargs)
                for c, (tp, fp, fn) in enumerate(match_counts(preds, gts, len(DET_CLASSES))):
                    totals[c][0] += tp
                    totals[c][1] += fp
                    totals[c][2] += fn
            sec = (time.time() - t0) / len(samples)
            row = {"modelo": model_path.name, "modo": mode, "zoom": zoom_label(z), "s_por_img": round(sec, 2)}
            f1s = []
            for name, (tp, fp, fn) in zip(DET_CLASSES, totals):
                p, r, f = f1(tp, fp, fn)
                row[f"{name}_P"], row[f"{name}_R"], row[f"{name}_F1"] = round(p, 3), round(r, 3), round(f, 3)
                f1s.append(f)
            row["F1_medio"] = round(sum(f1s) / len(f1s), 3)
            rows.append(row)
            print(f"{model_path.name} | {mode:>10} | {row['zoom']:>5} | F1 medio {row['F1_medio']:.3f} "
                  f"| RBC {row['RBC_F1']:.3f} WBC {row['WBC_F1']:.3f} PLT {row['Platelets_F1']:.3f} | {sec:.2f}s/img",
                  flush=True)
    return rows


def write_report(model_path: Path, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem
    with open(OUT_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    zooms = [zoom_label(z) for z in ZOOMS]
    lines = [f"# Robustez a escala -- {model_path.name}", "",
             "F1 medio (RBC/WBC/Plaqueta, IoU>=0.5) por zoom sintetico:", "",
             "| modo | " + " | ".join(zooms) + " | media | s/img (1x) |",
             "|---" * (len(zooms) + 3) + "|"]
    for mode in MODES:
        mr = {r["zoom"]: r for r in rows if r["modo"] == mode}
        vals = [mr[z]["F1_medio"] for z in zooms]
        lines.append(f"| {mode} | " + " | ".join(f"{v:.3f}" for v in vals)
                     + f" | {sum(vals) / len(vals):.3f} | {mr['1x']['s_por_img']} |")
    for cls in DET_CLASSES:
        lines += ["", f"F1 {cls}:", "", "| modo | " + " | ".join(zooms) + " |", "|---" * (len(zooms) + 1) + "|"]
        for mode in MODES:
            mr = {r["zoom"]: r for r in rows if r["modo"] == mode}
            lines.append(f"| {mode} | " + " | ".join(f"{mr[z][f'{cls}_F1']:.3f}" for z in zooms) + " |")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Relatorio: {OUT_DIR / f'{stem}.md'}")


def main():
    model_paths = [Path(a) for a in sys.argv[1:]] or [dc.MODEL_PATH]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoom_sets = build_zoom_sets(load_val())
    for mp in model_paths:
        write_report(mp, evaluate(mp, zoom_sets, device))


if __name__ == "__main__":
    main()
