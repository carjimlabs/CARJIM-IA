"""
Mescla o Raabin-WBC (raabindata.com) -- a fonte que traz os 5 subtipos de
leucocito do diferencial. E o merge que "preenche" os ids 1..5 esvaziados
pela migracao de taxonomia (09_migrate_wbc_taxonomy.py). RODE 09 ANTES.

Fonte esperada (extraia o WBCData.rar aqui, sem achatar):
    data/raabin_wbc/raw/**/First_microscope/*.zip
    data/raabin_wbc/raw/**/Second_microscope/*.zip
Cada .zip = um filme de sangue, com pares images/<x>.jpg + jsons/<x>.json.
O JSON tem, por celula: Label1 e Label2 (dois especialistas) + caixa
absoluta x1,x2,y1,y2. Metadados no nivel do arquivo: Zoom, Film ID, etc.

Decisoes embutidas (todas verificadas na propria base):

- ORIENTACAO. First_microscope: as coordenadas batem com a imagem como
  esta (5312x2988), sem rotacao -- o script-demo oficial que gira 90 CCW
  esta errado para estas imagens (testado: 1550/1550 caixas dentro dos
  limites sem girar). Second_microscope: as coordenadas estao no
  referencial girado 90 no sentido HORARIO (4160x3120 -> 3120x4160);
  giramos a imagem e usamos as caixas como vem.

- RECORTE POR CELULA ISOLADA. As imagens de campo completo do Raabin NAO
  sao anotadas exaustivamente -- eles marcaram so uma amostra de celulas
  por imagem. So consideramos celulas ISOLADAS (nenhuma outra celula
  anotada a menos de MIN_CELL_SEPARATION px), e recortamos uma janela de
  CROP_SIZE px (metade <= separacao, entao nenhuma outra celula anotada
  cabe no recorte). 1 leucocito por recorte, com contexto de hemacias.

- COTA POR CLASSE. Neutrofilo e linfocito sao muito mais comuns; sem cota
  eles dominariam o treino e o modelo colaria no dominio "recorte Raabin"
  (celula ocupando ~metade do quadro), perdendo competencia nas fotos de
  campo largo do BCCD/uso real. MAX_PER_CLASS limita os dois; os raros
  (monocito/eosinofilo/basofilo) entram todos.

- PSEUDO-ROTULO DE HEMACIA/PLAQUETA. O recorte do Raabin so tem a caixa do
  leucocito. Sem hemacia rotulada, o treino aprende "hemacia aqui = fundo"
  e o modelo para de detectar hemacia em campo largo. Entao rodamos o
  modelo ANTIGO de 3 classes (carjim_best_3class...) em cada recorte e
  adicionamos as caixas de RBC/plaqueta que ele achar (o recorte e foto
  limpa, o modelo antigo detecta bem -- diferente do export aumentado que
  quebrou o pseudo-rotulo no script 08). Caixas do modelo antigo que se
  sobrepoem ao leucocito sao descartadas (IoU).

- ROTULO DO LEUCOCITO. So entra quando os DOIS especialistas concordam na
  classe mapeada. "Small Lymph" vs "Large Lymph" conta como concordancia.
  Fora do mapa (Artifact/Burst/Unknn/Not centered/Meta/NRBC/'') e
  descartado. "Band" -> Neutrofilo.

- VAL. Ao contrario de 06/07/08, este script TAMBEM escreve uma fracao das
  imagens em images/val + labels/val (prefixo raabin_) -- o val original
  do BCCD nao tem nenhuma caixa de subtipo de leucocito. Split por imagem
  de origem (recortes do mesmo campo vao juntos para train ou val).

Uso: python scripts/10_merge_raabin_wbc.py
"""
import io
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES, CLASS_COLORS, CLASS_LABELS_PT  # noqa: E402

RAABIN_RAW = PROJECT_ROOT / "data" / "raabin_wbc" / "raw"
BCCD_ROOT = PROJECT_ROOT / "data" / "bccd"
DATASET_YAML = BCCD_ROOT / "dataset.yaml"
IMAGES_DIR = BCCD_ROOT / "images"
LABELS_DIR = BCCD_ROOT / "labels"
PREVIEW_DIR = BCCD_ROOT / "raabin_wbc_previews"

# Modelo antigo de 3 classes, usado so para pseudo-rotular RBC/plaqueta.
OLD_MODEL_PATH = PROJECT_ROOT / "models" / "carjim_best_3class_pre_diferencial.pt"

FILE_PREFIX = "raabin_"
VAL_FRACTION = 0.12
RANDOM_SEED = 42

MIN_CELL_SEPARATION = 600
CROP_SIZE = 1200
MAX_SIDE = 1200
JPEG_QUALITY = 88
MIN_BOX_PX = 6
PREVIEW_SAMPLE = 60

# Cota de recortes por classe de leucocito (aplicada ao TOTAL train+val antes
# do split). Neutrofilo/linfocito tem 10k+/3k+ disponiveis; os outros ~700-900
# (monocito/eosinofilo) e ~280 (basofilo) -- esses entram todos.
MAX_PER_CLASS = {
    "Neutrophil": 2500,
    "Lymphocyte": 1800,
}

# Pseudo-rotulo (modelo antigo de 3 classes) -- so aproveitamos RBC e plaqueta.
PSEUDO_IMGSZ = 1280
PSEUDO_RBC_CONF = 0.35
PSEUDO_PLATELET_CONF = 0.45
PSEUDO_DROP_IOU = 0.35   # caixa pseudo que sobrepoe o leucocito -> descartada
# O modelo antigo acha ~40 hemacias por recorte. Isso e sinal demais -- deixaria
# a hemacia ~25x mais frequente que os leucocitos e o treino mais lento. Bastam
# ~15 (aleatorias) para o modelo aprender que "aquilo e hemacia, nao fundo".
MAX_PSEUDO_RBC_PER_CROP = 15
MAX_PSEUDO_PLATELET_PER_CROP = 6

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

MICROSCOPE_ROTATION = {
    "First_microscope": None,
    "Second_microscope": "CW",
}


def find_microscope_dirs() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not RAABIN_RAW.exists():
        raise SystemExit(
            f"{RAABIN_RAW} nao existe. Extraia o WBCData.rar la dentro "
            "(https://dl.raabindata.com/WBCData.rar)."
        )
    for key in MICROSCOPE_ROTATION:
        matches = [p for p in RAABIN_RAW.rglob("*") if p.is_dir() and p.name.endswith(key)]
        matches = [p for p in matches if list(p.glob("*.zip"))]
        if matches:
            found[key] = matches[0]
    if not found:
        raise SystemExit(
            f"Nao encontrei nenhuma pasta *_microscope com .zip dentro de {RAABIN_RAW}."
        )
    return found


def check_taxonomy_migrated() -> None:
    if not DATASET_YAML.exists():
        raise SystemExit(f"{DATASET_YAML} nao existe. Rode 02_prepare_dataset.py e 09_migrate_wbc_taxonomy.py antes.")
    names = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8")).get("names", {})
    values = set(names.values() if isinstance(names, dict) else names)
    if "Neutrophil" not in values:
        raise SystemExit(
            "dataset.yaml ainda esta na taxonomia antiga. "
            "Rode `python scripts/09_migrate_wbc_taxonomy.py` primeiro."
        )


def load_old_model():
    """Carrega o modelo antigo de 3 classes para pseudo-rotular RBC/plaqueta.
    Devolve (model, rbc_id, platelet_id) ou (None, None, None) se ausente."""
    if not OLD_MODEL_PATH.exists():
        print(f"AVISO: {OLD_MODEL_PATH.name} nao encontrado -- recortes ficarao "
              "SEM pseudo-rotulo de hemacia/plaqueta (o treino pode aprender "
              "hemacia como fundo).")
        return None, None, None
    from ultralytics import YOLO
    m = YOLO(str(OLD_MODEL_PATH))
    inv = {v: k for k, v in m.names.items()}
    return m, inv.get("RBC"), inv.get("Platelets")


def rotate_cw(image: Image.Image) -> Image.Image:
    return image.rotate(-90, expand=True)


def cells_from_json(data: dict):
    try:
        n = int(data.get("Cell Numbers", 0))
    except (TypeError, ValueError):
        n = 0
    out = []
    for i in range(n):
        cell = data.get(f"Cell_{i}")
        if not isinstance(cell, dict):
            continue
        m1 = LABEL_MAP.get((cell.get("Label1") or "").strip())
        m2 = LABEL_MAP.get((cell.get("Label2") or "").strip())
        if m1 is None or m1 != m2:
            continue
        try:
            x1, x2 = sorted((int(float(cell["x1"])), int(float(cell["x2"]))))
            y1, y2 = sorted((int(float(cell["y1"])), int(float(cell["y2"]))))
        except (KeyError, TypeError, ValueError):
            continue
        out.append((m1, x1, y1, x2, y2))
    return out


def crop_window(w: int, h: int, cx: float, cy: float):
    size = min(CROP_SIZE, w, h)
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))
    return x0, y0, x0 + size, y0 + size


def boxes_in_crop(boxes, crop):
    cx0, cy0, cx1, cy1 = crop
    out = []
    for name, x1, y1, x2, y2 in boxes:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if not (cx0 <= mx < cx1 and cy0 <= my < cy1):
            continue
        nx1 = max(x1, cx0) - cx0
        ny1 = max(y1, cy0) - cy0
        nx2 = min(x2, cx1) - cx0
        ny2 = min(y2, cy1) - cy0
        if nx2 - nx1 < MIN_BOX_PX or ny2 - ny1 < MIN_BOX_PX:
            continue
        out.append((name, nx1, ny1, nx2, ny2))
    return out


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def pseudo_rbc_platelet(old_model, rbc_id, platelet_id, crop_img, wbc_boxes, rng):
    """Roda o modelo antigo no recorte e devolve caixas (name, x1,y1,x2,y2)
    de RBC/plaqueta que NAO se sobrepoem a um leucocito anotado. Limita a
    quantidade por classe (as mais confiantes primeiro, depois amostra)."""
    if old_model is None:
        return []
    res = old_model.predict(source=crop_img, imgsz=PSEUDO_IMGSZ, conf=min(PSEUDO_RBC_CONF, PSEUDO_PLATELET_CONF),
                            verbose=False)[0]
    wbc_xyxy = [(b[1], b[2], b[3], b[4]) for b in wbc_boxes]
    rbc, plt = [], []
    for box in res.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        if cid == rbc_id and conf >= PSEUDO_RBC_CONF:
            bucket = rbc
        elif cid == platelet_id and conf >= PSEUDO_PLATELET_CONF:
            bucket = plt
        else:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        if any(_iou((x1, y1, x2, y2), wb) > PSEUDO_DROP_IOU for wb in wbc_xyxy):
            continue
        bucket.append((conf, x1, y1, x2, y2))

    def limit(items, cap, name):
        if len(items) > cap:
            items = rng.sample(items, cap)
        return [(name, x1, y1, x2, y2) for _c, x1, y1, x2, y2 in items]

    return limit(rbc, MAX_PSEUDO_RBC_PER_CROP, "RBC") + limit(plt, MAX_PSEUDO_PLATELET_PER_CROP, "Platelets")


def to_yolo_lines(boxes, w, h):
    lines = []
    for name, x1, y1, x2, y2 in boxes:
        cls_id = CLASSES.index(name)
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def maybe_downscale(image: Image.Image, boxes):
    w, h = image.size
    if max(w, h) <= MAX_SIDE:
        return image, boxes, w, h
    scale = MAX_SIDE / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    image = image.resize((nw, nh), Image.LANCZOS)
    boxes = [(n, a * scale, b * scale, c * scale, d * scale) for n, a, b, c, d in boxes]
    return image, boxes, nw, nh


def clear_previous() -> int:
    removed = 0
    for split in ("train", "val"):
        for d in (IMAGES_DIR / split, LABELS_DIR / split):
            for f in d.glob(f"{FILE_PREFIX}*"):
                f.unlink()
                removed += 1
    for c in BCCD_ROOT.glob("*.cache"):
        c.unlink()
    if PREVIEW_DIR.exists():
        for f in PREVIEW_DIR.glob("*.jpg"):
            f.unlink()
    return removed


def _encode_jpg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def build_previews(written: list[tuple[str, str]], rng: random.Random):
    if not written:
        return
    sample = written if len(written) <= PREVIEW_SAMPLE else rng.sample(written, PREVIEW_SAMPLE)
    thumb, cols = 300, 6
    tiles = []
    for split, base in sample:
        img = Image.open(IMAGES_DIR / split / f"{base}.jpg").convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)
        for ln in (LABELS_DIR / split / f"{base}.txt").read_text().splitlines():
            if not ln.strip():
                continue
            cid, xc, yc, bw, bh = ln.split()
            cid = int(cid)
            xc, yc, bw, bh = (float(v) for v in (xc, yc, bw, bh))
            x1, y1 = (xc - bw / 2) * W, (yc - bh / 2) * H
            x2, y2 = (xc + bw / 2) * W, (yc + bh / 2) * H
            color = CLASS_COLORS.get(CLASSES[cid], (255, 200, 0))
            wdt = 4 if CLASSES[cid] not in ("RBC", "Platelets") else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=wdt)
        img.save(PREVIEW_DIR / f"{base}_preview.jpg", quality=75)
        tiles.append(img.resize((thumb, thumb)))
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    rows = [r for r in rows if len(r) == cols]
    if rows:
        sheet = Image.new("RGB", (thumb * cols, thumb * len(rows)), (20, 20, 20))
        for ry, row in enumerate(rows):
            for cx, t in enumerate(row):
                sheet.paste(t, (cx * thumb, ry * thumb))
        sheet.save(PREVIEW_DIR / "_contact_sheet.jpg", quality=82)
        print(f"Contact sheet ({len(rows) * cols} amostras) em: {PREVIEW_DIR / '_contact_sheet.jpg'}")


def collect_candidates(micro_dirs: dict[str, Path]):
    """Pass 1 (so JSON): lista de dicts com o que precisamos para emitir cada
    recorte de celula isolada, sem abrir imagem ainda."""
    cands = []
    for key, mdir in micro_dirs.items():
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
                cells = cells_from_json(data)
                if not cells:
                    continue
                centers = [((x1 + x2) / 2, (y1 + y2) / 2) for _n, x1, y1, x2, y2 in cells]
                for ci in range(len(cells)):
                    if any(
                        cj != ci
                        and abs(centers[ci][0] - centers[cj][0]) < MIN_CELL_SEPARATION
                        and abs(centers[ci][1] - centers[cj][1]) < MIN_CELL_SEPARATION
                        for cj in range(len(cells))
                    ):
                        continue
                    cands.append({
                        "key": key, "zip": str(zpath), "film": film, "img": img_by_stem[stem],
                        "stem": stem, "cell_idx": ci, "cls": cells[ci][0], "cells": cells,
                    })
    return cands


def apply_class_caps(cands, rng: random.Random):
    by_cls: dict[str, list] = {}
    for c in cands:
        by_cls.setdefault(c["cls"], []).append(c)
    kept = []
    for cls, lst in by_cls.items():
        cap = MAX_PER_CLASS.get(cls)
        if cap and len(lst) > cap:
            rng.shuffle(lst)
            lst = lst[:cap]
        kept.extend(lst)
        print(f"  {cls}: {len(lst)} recortes"
              + (f" (cota {cap}, de {len(by_cls[cls])})" if cap and len(by_cls[cls]) > cap else ""))
    rng.shuffle(kept)
    return kept


def main() -> None:
    check_taxonomy_migrated()
    micro_dirs = find_microscope_dirs()
    print("Pastas encontradas:")
    for k, v in micro_dirs.items():
        print(f"  {k}: {v}  ({len(list(v.glob('*.zip')))} zips)")

    for split in ("train", "val"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    removed = clear_previous()
    if removed:
        print(f"Removidos {removed} arquivos raabin_* de uma execucao anterior (+ caches).")

    rng = random.Random(RANDOM_SEED)

    print("\nPass 1: varrendo JSONs...")
    cands = collect_candidates(micro_dirs)
    print(f"  {len(cands)} celulas isoladas encontradas")
    print("Aplicando cotas por classe:")
    kept = apply_class_caps(cands, rng)
    print(f"  -> {len(kept)} recortes a emitir")

    old_model, rbc_id, platelet_id = load_old_model()
    if old_model is not None:
        print(f"Pseudo-rotulo RBC/plaqueta: modelo antigo carregado (RBC id={rbc_id}, plaqueta id={platelet_id})")

    # split por imagem-fonte (recortes do mesmo campo vao juntos)
    src_split: dict[tuple, str] = {}

    per_class = {"train": Counter(), "val": Counter()}
    n_crops = {"train": 0, "val": 0}
    skipped_bad = 0
    written: list[tuple[str, str]] = []

    # agrupa por zip para abrir cada zip uma vez
    by_zip: dict[str, list] = {}
    for c in kept:
        by_zip.setdefault(c["zip"], []).append(c)

    print("\nPass 2: emitindo recortes...")
    for zpath, items in by_zip.items():
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            continue
        rotation = MICROSCOPE_ROTATION[items[0]["key"]]
        # ordenado por imagem-fonte -> cache de 1 entrada e suficiente (a
        # maioria das imagens tem 1 celula isolada; guardar todas as imagens
        # 5312x2988 de um zip estourava a RAM).
        items.sort(key=lambda c: c["img"])
        cached_name, cached_img = None, None
        for c in items:
            if c["img"] != cached_name:
                try:
                    im = Image.open(io.BytesIO(zf.read(c["img"]))).convert("RGB")
                except Exception:  # noqa: BLE001
                    skipped_bad += 1
                    cached_name, cached_img = None, None
                    continue
                if rotation == "CW":
                    im = rotate_cw(im)
                cached_name, cached_img = c["img"], im
            image = cached_img
            if image is None:
                continue
            w, h = image.size

            target = c["cells"][c["cell_idx"]]
            _n, tx1, ty1, tx2, ty2 = target
            if not (0 <= tx1 < tx2 <= w and 0 <= ty1 < ty2 <= h):
                skipped_bad += 1  # orientacao/limites inesperados
                continue
            mx, my = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            crop = crop_window(w, h, mx, my)
            # so as celulas anotadas que cabem na imagem e caem no recorte
            valid_cells = [
                cc for cc in c["cells"]
                if 0 <= cc[1] < cc[3] <= w and 0 <= cc[2] < cc[4] <= h
            ]
            wbc_boxes = boxes_in_crop(valid_cells, crop)
            if not wbc_boxes:
                skipped_bad += 1
                continue

            sub = image.crop(crop)
            pseudo = pseudo_rbc_platelet(old_model, rbc_id, platelet_id, sub, wbc_boxes, rng)
            all_boxes = wbc_boxes + pseudo
            sub, all_boxes, cw, ch = maybe_downscale(sub, all_boxes)
            lines = to_yolo_lines(all_boxes, cw, ch)

            skey = (zpath, c["stem"])
            if skey not in src_split:
                src_split[skey] = "val" if rng.random() < VAL_FRACTION else "train"
            split = src_split[skey]

            base = f"{FILE_PREFIX}{c['key'][0].lower()}_{c['film']}_{c['stem']}_c{c['cell_idx']}"
            (IMAGES_DIR / split / f"{base}.jpg").write_bytes(_encode_jpg(sub))
            (LABELS_DIR / split / f"{base}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_crops[split] += 1
            for bn, *_ in all_boxes:
                per_class[split][bn] += 1
            written.append((split, base))
        cached_img = None

    build_previews(written, rng)

    print("\n=== Resumo ===")
    for split in ("train", "val"):
        total = sum(per_class[split].values())
        by = ", ".join(f"{CLASS_LABELS_PT[c]}={per_class[split][c]}" for c in CLASSES if per_class[split][c])
        print(f"  {split}: {n_crops[split]} recortes, {total} caixas  [{by}]")
    print(f"  recortes pulados (imagem ilegivel / caixa fora): {skipped_bad}")
    print(f"\nConfira o contact sheet em {PREVIEW_DIR} antes de treinar.")


if __name__ == "__main__":
    main()
