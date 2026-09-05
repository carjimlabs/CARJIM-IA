"""
Monta o dataset de CLASSIFICACAO de subtipo de leucocito (etapa 2 do pipeline
de 2 estagios: o detector de 3 classes acha as caixas de WBC, este
classificador diz qual dos 5 subtipos e).

Fonte: os zips do Raabin-WBC ja extraidos em data/raabin_wbc/raw/ (First e
Second microscope). Para cada celula onde os DOIS especialistas concordam na
classe mapeada, recorta a celula (caixa do JSON + margem), corrigindo a
orientacao (Second microscope = girar 90 horario), e salva em

    data/wbc_cls/<split>/<ClasseEN>/<film>_<img>_<i>.jpg

(estrutura ImageFolder, que o YOLOv8-cls le direto). Split estratificado.

Nao ha filtro de "celula isolada" aqui -- para classificar so importa a
celula central, e a caixa do Raabin ja e centrada nela.

Uso: python scripts/12_build_wbc_classifier_data.py
"""
import io
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAABIN_RAW = PROJECT_ROOT / "data" / "raabin_wbc" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "wbc_cls"
VAL_FRACTION = 0.15
RANDOM_SEED = 42
BOX_PAD = 0.18          # margem em volta da caixa do Raabin (fracao do lado)
MIN_CROP_PX = 40
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

LABEL_MAP = {
    "Neutrophil": "Neutrophil",
    "Band": "Neutrophil",
    "Small Lymph": "Lymphocyte",
    "Large Lymph": "Lymphocyte",
    "Lymphocyte": "Lymphocyte",
    "Monocyte": "Monocyte",
    "Eosinophil": "Eosinophil",
    "Basophil": "Basophil",
}
CLASSES_EN = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
MICROSCOPE_ROTATION = {"First_microscope": None, "Second_microscope": "CW"}


def find_microscope_dirs():
    found = {}
    for key in MICROSCOPE_ROTATION:
        matches = [p for p in RAABIN_RAW.rglob("*") if p.is_dir() and p.name.endswith(key)]
        matches = [p for p in matches if list(p.glob("*.zip"))]
        if matches:
            found[key] = matches[0]
    if not found:
        raise SystemExit(f"Nenhuma pasta *_microscope com .zip em {RAABIN_RAW}.")
    return found


def agreed_cells(data):
    try:
        n = int(data.get("Cell Numbers", 0))
    except (TypeError, ValueError):
        n = 0
    out = []
    for i in range(n):
        c = data.get(f"Cell_{i}")
        if not isinstance(c, dict):
            continue
        m1 = LABEL_MAP.get((c.get("Label1") or "").strip())
        m2 = LABEL_MAP.get((c.get("Label2") or "").strip())
        if m1 is None or m1 != m2:
            continue
        try:
            x1, x2 = sorted((int(float(c["x1"])), int(float(c["x2"]))))
            y1, y2 = sorted((int(float(c["y1"])), int(float(c["y2"]))))
        except (KeyError, TypeError, ValueError):
            continue
        out.append((m1, x1, y1, x2, y2, i))
    return out


def main():
    rng = random.Random(RANDOM_SEED)
    micro = find_microscope_dirs()

    if OUT_ROOT.exists():
        for f in OUT_ROOT.rglob("*.jpg"):
            f.unlink()
    for split in ("train", "val"):
        for cls in CLASSES_EN:
            (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"train": Counter(), "val": Counter()}
    skipped = 0

    for key, mdir in micro.items():
        rotation = MICROSCOPE_ROTATION[key]
        for zpath in sorted(mdir.glob("*.zip")):
            film = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in zpath.stem)
            try:
                zf = zipfile.ZipFile(zpath)
            except zipfile.BadZipFile:
                continue
            names = zf.namelist()
            img_by_stem = {Path(n).stem: n for n in names if n.lower().endswith(IMAGE_EXTS)}
            for jn in (n for n in names if n.lower().endswith(".json")):
                stem = Path(jn).stem
                if stem not in img_by_stem:
                    continue
                try:
                    data = json.loads(zf.read(jn))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                cells = agreed_cells(data)
                if not cells:
                    continue
                try:
                    image = Image.open(io.BytesIO(zf.read(img_by_stem[stem]))).convert("RGB")
                except Exception:  # noqa: BLE001
                    skipped += 1
                    continue
                if rotation == "CW":
                    image = image.rotate(-90, expand=True)
                W, H = image.size
                for name, x1, y1, x2, y2, ci in cells:
                    if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                        skipped += 1
                        continue
                    pad_x = int((x2 - x1) * BOX_PAD)
                    pad_y = int((y2 - y1) * BOX_PAD)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
                    if cx2 - cx1 < MIN_CROP_PX or cy2 - cy1 < MIN_CROP_PX:
                        skipped += 1
                        continue
                    crop = image.crop((cx1, cy1, cx2, cy2))
                    split = "val" if rng.random() < VAL_FRACTION else "train"
                    crop.save(OUT_ROOT / split / name / f"{key[0].lower()}_{film}_{stem}_{ci}.jpg",
                              quality=92)
                    counts[split][name] += 1
            print(f"  {key}/{zpath.name} feito")

    print("\n=== Resumo ===")
    for split in ("train", "val"):
        print(f"  {split}: {dict(counts[split])}  total {sum(counts[split].values())}")
    print(f"  pulados: {skipped}")
    print(f"\nDataset em {OUT_ROOT}")


if __name__ == "__main__":
    main()
