"""
Compara detectores em FOTOS REAIS (frames dos videos de varredura em Videos/)
simulando distancias diferentes.

Os frames nao tem rotulo manual, entao a medida e de CONSISTENCIA: a
referencia de cada frame e a UNIAO das caixas de todos os modelos
comparados no frame original (passada normal e com normalizacao de escala),
com duplicatas fundidas. Ressalva: falso positivo de qualquer modelo vira
"verdade" -- confira os previews salvos. Depois o mesmo campo e mostrado:

  - mais perto: recorte central de 1/z do frame ampliado de volta (z = 1.5, 2)
  - mais longe: mosaico k x k de frames reduzido ao tamanho de um frame
                (celulas k vezes menores, mesma resolucao da camera)

e mede-se P/R/F1 contra a referencia transformada, por modelo e modo
(base / normalizacao de escala).

Frames: a +1.5 s do grid de 3 s que o 14_add_wide_field_video_samples.py
usou no treino (posicoes da lamina entre as treinadas; a varredura e lenta,
entao ha semelhanca -- mas igual para todos os modelos, que treinaram nos
mesmos frames). Descarta frames escuros e com poucas hemacias.

Uso:
  python scripts/17_eval_real_video_frames.py models/a.pt models/b.pt
Relatorio: runs/eval_real/relatorio.md (+ previews em runs/eval_real/)
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detection_core as dc  # noqa: E402
import importlib  # noqa: E402

ev = importlib.import_module("16_eval_scale_robustness")

VIDEOS_DIR = PROJECT_ROOT / "Videos"
OUT_DIR = PROJECT_ROOT / "runs" / "eval_real"
FRAMES_PER_VIDEO = 6
OFFSET_SEC = 1.5
GRID_SEC = 3.0
MIN_BRIGHTNESS = 80.0
MIN_REF_RBC = 20
ZOOMS = [2.0, 1.5, 1.0, 1 / 2, 1 / 3]
MOSAICS_PER_ZOOM = 12
RANDOM_SEED = 0
DET_CLASSES = ev.DET_CLASSES


def sample_frames():
    rng = random.Random(RANDOM_SEED)
    frames = []
    for video in sorted(VIDEOS_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        slots = list(range(int(OFFSET_SEC * fps), n, int(GRID_SEC * fps)))
        rng.shuffle(slots)
        got = 0
        for idx in slots:
            if got >= FRAMES_PER_VIDEO:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame.mean() < MIN_BRIGHTNESS:
                continue
            frames.append((f"{video.stem}_f{idx:06d}", Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            got += 1
        cap.release()
        print(f"{video.name}: {got} frames", flush=True)
    return frames


def union_reference(box_sets):
    """Uniao das caixas de todos os modelos (duplicatas fundidas por NMS).
    Consenso (intersecao) escondia celulas que so um modelo achava -- com a
    uniao, uma celula perdida por um modelo conta contra ele."""
    allb = [b for s in box_sets for b in s]
    if not allb:
        return []
    boxes = torch.tensor([b[1:5] for b in allb])
    keep = batched_nms(boxes, torch.tensor([b[5] for b in allb]), torch.tensor([b[0] for b in allb]), ev.IOU_MATCH)
    return [(allb[i][0], *allb[i][1:5]) for i in keep.tolist()]


def main():
    model_paths = [Path(a) for a in sys.argv[1:]]
    if len(model_paths) < 2:
        raise SystemExit("passe ao menos 2 modelos")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detectors = {}
    for mp in model_paths:
        det = YOLO(str(mp))
        det.to(device)
        detectors[mp.stem] = dc.Detector(det, None, device)

    items = []
    for name, image in sample_frames():
        per_model = [dc.detect_cells(d, image, scale_norm=sn) for d in detectors.values() for sn in (False, True)]
        ref = union_reference(per_model)
        n_rbc = sum(1 for b in ref if b[0] == 0)
        if n_rbc >= MIN_REF_RBC:
            items.append((image, [tuple(b) for b in ref]))
    print(f"{len(items)} frames com referencia valida", flush=True)

    rng = random.Random(RANDOM_SEED)
    zoom_sets = {}
    for z in ZOOMS:
        if z > 1:
            zoom_sets[z] = [ev.zoom_in(im, bx, z, rng) for im, bx in items]
        elif z == 1:
            zoom_sets[z] = list(items)
        else:
            k = round(1 / z)
            sets = []
            for _ in range(MOSAICS_PER_ZOOM):
                big, boxes = ev.mosaic(items, k, rng)
                W, H = items[0][0].size
                sets.append((big.resize((W, H), Image.BILINEAR),
                             [(c, x1 / k, y1 / k, x2 / k, y2 / k) for c, x1, y1, x2, y2 in boxes]))
            zoom_sets[z] = sets

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zl = [ev.zoom_label(z) for z in ZOOMS]
    lines = ["# Consistencia em frames reais (Videos/)", "",
             f"{len(items)} frames; referencia = uniao das deteccoes dos modelos no frame original.", "",
             "F1 medio (RBC/WBC/Plaqueta) por distancia simulada:", "",
             "| modelo | modo | " + " | ".join(zl) + " | media |", "|---" * (len(zl) + 3) + "|"]
    per_class_lines = []
    for mname, detector in detectors.items():
        for mode, kw in (("base", dict(scale_norm=False)), ("escala", dict(scale_norm=True))):
            row, cls_rows = [], {c: [] for c in DET_CLASSES}
            for z in ZOOMS:
                totals = [[0, 0, 0] for _ in DET_CLASSES]
                for i, (image, gts) in enumerate(zoom_sets[z]):
                    preds = dc.detect_cells(detector, image, **kw)
                    for c, t in enumerate(ev.match_counts(preds, gts, len(DET_CLASSES))):
                        for j in range(3):
                            totals[c][j] += t[j]
                    if i == 0 and mode == "escala":
                        det_to_display = {0: 0, 1: dc.WBC_FALLBACK_ID, 2: dc.CLASSES.index("Platelets")}
                        disp = [(det_to_display[p[0]], *p[1:]) for p in preds]
                        dc.draw_detections(image.copy(), disp, dc.DISPLAY_NAMES, dc.load_font(28)) \
                            .save(OUT_DIR / f"{mname}_{ev.zoom_label(z).replace('/', 'de')}.jpg", quality=85)
                f1s = [ev.f1(*t)[2] for t in totals]
                for c, f in zip(DET_CLASSES, f1s):
                    cls_rows[c].append(f)
                row.append(sum(f1s) / len(f1s))
                print(f"{mname} | {mode} | {ev.zoom_label(z)} | F1 {row[-1]:.3f} "
                      + " ".join(f"{c} {f:.3f}" for c, f in zip(DET_CLASSES, f1s)), flush=True)
            lines.append(f"| {mname} | {mode} | " + " | ".join(f"{v:.3f}" for v in row)
                         + f" | {sum(row) / len(row):.3f} |")
            for c in DET_CLASSES:
                per_class_lines.append(f"| {mname} | {mode} | {c} | "
                                       + " | ".join(f"{v:.3f}" for v in cls_rows[c]) + " |")
    lines += ["", "Por classe:", "", "| modelo | modo | classe | " + " | ".join(zl) + " |",
              "|---" * (len(zl) + 3) + "|"] + per_class_lines
    (OUT_DIR / "relatorio.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Relatorio: {OUT_DIR / 'relatorio.md'}")


if __name__ == "__main__":
    main()
