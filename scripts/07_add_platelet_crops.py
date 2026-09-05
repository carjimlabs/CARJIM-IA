"""
Integra o `platelet.zip` (crops individuais de plaquetas, um por imagem, sem
anotacao) ao conjunto de treino. Como o dataset nao vem com caixas, este
script gera as caixas automaticamente por segmentacao de cor: a plaqueta
(roxo/violeta escuro, corada) se destaca claramente das hemacias (rosa claro)
e do fundo em HSV. Cada blob roxo encontrado vira uma caixa da classe
"Platelets" -- normalmente ha 1 plaqueta central por imagem, mas algumas
imagens tem fragmentos/aglomerados extras, que tambem sao capturados.

Como a segmentacao e automatica (nao supervisionada por olho humano), o
script salva uma previa com as caixas desenhadas para cada imagem em
data/bccd/platelet_crop_previews/ -- confira uma amostra antes de rodar o
treino, e apague o quadro inteiro do split de treino (arquivos com prefixo
"pltcrop_") se a qualidade nao parecer boa.

Uso: python scripts/07_add_platelet_crops.py
"""
import random
import zipfile
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES  # noqa: E402 -- fonte unica da taxonomia

ZIP_PATH = PROJECT_ROOT / "platelet.zip"
RAW_DIR = PROJECT_ROOT / "data" / "platelet_crops" / "raw"

TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "bccd" / "images" / "train"
TRAIN_LABELS_DIR = PROJECT_ROOT / "data" / "bccd" / "labels" / "train"
PREVIEW_DIR = PROJECT_ROOT / "data" / "bccd" / "platelet_crop_previews"

PLATELET_CLASS_ID = CLASSES.index("Platelets")
FILE_PREFIX = "pltcrop_"

# Faixa HSV (OpenCV: H 0-179) calibrada nas amostras do dataset: a plaqueta e
# roxo/violeta escuro e saturado, bem diferente do rosa claro/dessaturado das
# hemacias ao redor.
HSV_LOWER = np.array([100, 70, 0])
HSV_UPPER = np.array([175, 255, 210])
MIN_BOX_AREA = 18
MIN_BOX_DIM = 4
BOX_PAD = 2
MAX_BOXES_PER_IMAGE = 10  # imagens com mais que isso costumam ser artefato de coloracao, nao plaquetas
CONTACT_SHEET_SAMPLE = 48


def extract_zip() -> Path:
    if not ZIP_PATH.exists():
        raise RuntimeError(f"Arquivo nao encontrado: {ZIP_PATH}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = [
            n for n in zf.namelist()
            if n.startswith("platelet/PLATELET_") and n.lower().endswith(".jpg")
        ]
        for name in names:
            dest = RAW_DIR / Path(name).name
            if dest.exists():
                continue
            with zf.open(name) as src, open(dest, "wb") as out:
                out.write(src.read())

    print(f"{len(names)} imagens extraidas/verificadas em {RAW_DIR}")
    return RAW_DIR


def find_boxes(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_BOX_AREA or bw < MIN_BOX_DIM or bh < MIN_BOX_DIM:
            continue
        x0 = max(0, x - BOX_PAD)
        y0 = max(0, y - BOX_PAD)
        x1 = min(w, x + bw + BOX_PAD)
        y1 = min(h, y + bh + BOX_PAD)
        boxes.append((x0, y0, x1, y1, area))

    if len(boxes) > MAX_BOXES_PER_IMAGE:
        # provavelmente artefato de coloracao sendo super-segmentado em varios
        # pedacos, nao plaquetas de verdade -- mantem so as maiores caixas.
        boxes.sort(key=lambda b: b[4], reverse=True)
        boxes = boxes[:MAX_BOXES_PER_IMAGE]

    boxes = [(x0, y0, x1, y1) for x0, y0, x1, y1, _area in boxes]
    return boxes, w, h


def boxes_to_yolo(boxes, width, height):
    lines = []
    for x0, y0, x1, y1 in boxes:
        xc = ((x0 + x1) / 2) / width
        yc = ((y0 + y1) / 2) / height
        bw = (x1 - x0) / width
        bh = (y1 - y0) / height
        lines.append(f"{PLATELET_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def save_preview(img, boxes, dest: Path):
    disp = img.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 0, 255), 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), disp)


def build_contact_sheet(sample_previews):
    if not sample_previews:
        return
    thumb_size = 150
    thumbs = [cv2.resize(cv2.imread(str(p)), (thumb_size, thumb_size)) for p in sample_previews]
    cols = 8
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    rows = [r for r in rows if len(r) == cols]
    if not rows:
        return
    sheet = np.vstack([np.hstack(r) for r in rows])
    dest = PREVIEW_DIR / "_contact_sheet.jpg"
    cv2.imwrite(str(dest), sheet)
    print(f"Contact sheet (amostra de {len(rows) * cols} imagens) salvo em: {dest}")


def main():
    raw_dir = extract_zip()

    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(raw_dir.glob("*.jpg"))
    added = 0
    skipped_no_box = 0
    total_boxes = 0
    preview_paths = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        boxes, width, height = find_boxes(img)
        preview_path = PREVIEW_DIR / f"{img_path.stem}_preview.jpg"
        save_preview(img, boxes, preview_path)
        preview_paths.append(preview_path)

        if not boxes:
            skipped_no_box += 1
            continue

        dest_img = TRAIN_IMAGES_DIR / f"{FILE_PREFIX}{img_path.name}"
        dest_label = TRAIN_LABELS_DIR / f"{FILE_PREFIX}{img_path.stem}.txt"

        cv2.imwrite(str(dest_img), img)
        dest_label.write_text(boxes_to_yolo(boxes, width, height), encoding="utf-8")

        added += 1
        total_boxes += len(boxes)

    random.Random(42).shuffle(preview_paths)
    build_contact_sheet(preview_paths[:CONTACT_SHEET_SAMPLE])

    print(
        f"Concluido. {added} imagens adicionadas ao treino "
        f"({total_boxes} caixas de plaqueta no total, "
        f"{total_boxes / added:.2f} por imagem em media). "
        f"{skipped_no_box} imagens sem plaqueta detectada (ignoradas)."
    )
    print(f"Previas de TODAS as imagens em: {PREVIEW_DIR}")
    print(
        "Confira a previa (ou o contact sheet) antes de treinar -- se a "
        "segmentacao tiver saido ruim, apague os arquivos com prefixo "
        f"'{FILE_PREFIX}' de images/train e labels/train."
    )


if __name__ == "__main__":
    main()
