"""
Copy-paste augmentation: cola leucocitos (recortados dos crops do Raabin) DENTRO
das imagens de campo largo (BCCD / TXL-PBC / ALL_IDB) em tamanho pequeno e
realista, com o rotulo do subtipo.

Por que: o Raabin so fornece o leucocito num recorte onde a celula ocupa ~metade
do quadro. Treinando so nisso, o modelo aprende a reconhecer leucocito grande e
desenha caixas do tamanho errado -- em foto de campo largo (o uso real: professor
fotografa a lamina no microscopio, ~15-25 hemacias por campo, leucocito ~1.5-2x
uma hemacia) ele nao acha nada. Este script gera o sinal que falta.

Como:
  1. Fonte do leucocito: cada crop `data/bccd/images/train/raabin_*.jpg` tem 1
     caixa de subtipo. Segmentamos a celula (nucleo roxo saturado + citoplasma
     ao redor, via HSV + dilatacao + maior contorno), com borda suavizada.
  2. Alvo: imagens de campo largo do proprio dataset (prefixo BloodImage /
     txlpbc_ / allidb_). Estimamos o tamanho tipico de hemacia pelas caixas de
     RBC do alvo (ou por heuristica se nao houver).
  3. Colamos 1-3 leucocitos por alvo, escalados para ~1.4-2.4x a hemacia, em
     posicao aleatoria (evitando cair em cima de um leucocito ja rotulado),
     com leve rotacao/jitter. Adicionamos a caixa YOLO do subtipo.
  4. Salvamos como `wbcpaste_*.jpg` (+ label = caixas originais do alvo +
     caixas coladas). Uma fracao vai para val, para dar como medir leucocito
     em campo largo (a val atual so tem crops Raabin + 73 imagens BCCD).

Classes raras (basofilo/eosinofilo/monocito) sao super-amostradas na hora de
escolher qual leucocito colar.

Rode DEPOIS de 10_merge_raabin_wbc.py. Uso: python scripts/11_paste_wbc_wide_field.py
"""
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES, CLASS_LABELS_PT  # noqa: E402

# modelo antigo de 3 classes: usado para achar os leucocitos REAIS ja presentes
# na imagem-alvo (que ficaram sem rotulo apos a migracao de taxonomia) e
# cobri-los com um leucocito sintetico rotulado -> imagem sintetica 100%
# consistente (nenhum leucocito visivel sem caixa).
OLD_MODEL_PATH = PROJECT_ROOT / "models" / "carjim_best_3class_pre_diferencial.pt"
OLD_WBC_CONF = 0.30

BCCD_ROOT = PROJECT_ROOT / "data" / "bccd"
IMAGES_DIR = BCCD_ROOT / "images"
LABELS_DIR = BCCD_ROOT / "labels"
PREVIEW_DIR = BCCD_ROOT / "wbcpaste_previews"

FILE_PREFIX = "wbcpaste_"
RANDOM_SEED = 42
VAL_FRACTION = 0.10
JPEG_QUALITY = 90
PREVIEW_SAMPLE = 40

# alvos: so campos largos de verdade (BloodImage 640x480, allidb ~2000x1500).
# txlpbc foi testado e descartado -- muitas dessas imagens ja sao recortes
# apertados (575x575 / 360x360), colar leucocito grande em cima fica irreal.
TARGET_PREFIXES = ("BloodImage", "allidb_")
# quantas imagens sinteticas gerar (alvos reutilizados se preciso)
N_SYNTHETIC = 3000
PASTES_PER_IMAGE = (1, 3)          # min, max (inclusive) leucocitos colados
WBC_TO_RBC_RATIO = (1.4, 2.5)      # largura do leucocito / largura da hemacia
FALLBACK_RBC_FRAC = 1 / 22         # se o alvo nao tem caixa de RBC: ~22 hemacias na largura
MAX_WBC_FRAC_OF_IMG = 0.16         # leucocito colado nao passa disso da largura da imagem
MAX_OVERLAP_WITH_WBC = 0.15        # nao colar em cima de um leucocito existente (IoU)
RBC_CLASS_ID = CLASSES.index("RBC")
LEUKOCYTE_IDS = [CLASSES.index(n) for n in
                 ("Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil")]

# peso na amostragem de qual leucocito colar (super-amostra os raros)
CLASS_SAMPLING_WEIGHT = {
    "Neutrophil": 1.0,
    "Lymphocyte": 1.0,
    "Monocyte": 2.2,
    "Eosinophil": 2.2,
    "Basophil": 3.0,
}


def read_label(path: Path):
    boxes = []
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            c, xc, yc, w, h = ln.split()
            boxes.append((int(c), float(xc), float(yc), float(w), float(h)))
    return boxes


def yolo_to_xyxy(b, W, H):
    _c, xc, yc, w, h = b
    return (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cell_extent(region_bgr):
    """Acha a bounding box da celula corada dentro da regiao (nucleo roxo +
    granulos + citoplasma corado). Devolve (x0,y0,x1,y1) ou None."""
    h, w = region_bgr.shape[:2]
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    stained = (((S >= 70) & (V <= 235)) & (
        ((Hh >= 110) & (Hh <= 165)) |   # roxo (nucleo)
        (Hh <= 22) | (Hh >= 150)        # rosa/laranja (granulos eosinofilo)
    )).astype(np.uint8) * 255
    ker = max(5, int(0.10 * min(h, w)) | 1)
    stained = cv2.morphologyEx(stained, cv2.MORPH_CLOSE, np.ones((ker, ker), np.uint8))
    stained = cv2.dilate(stained, np.ones((ker, ker), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(stained, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.03 * h * w:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    # citoplasma costuma ir um pouco alem da parte mais corada
    pad = int(0.12 * max(bw, bh))
    return (max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad))


def make_cutout(region_bgr, extent):
    """Recorta a celula na `extent` e monta uma mascara eliptica suavizada
    (celula e redonda; a borda da elipse dissolve o fundo/hemacia da fonte)."""
    x0, y0, x1, y1 = extent
    cell = region_bgr[y0:y1, x0:x1]
    ch, cw = cell.shape[:2]
    if ch < 12 or cw < 12:
        return None
    mask = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(mask, (cw // 2, ch // 2), (int(cw * 0.46), int(ch * 0.46)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, 0.10 * min(ch, cw)))
    return cell, mask


def load_wbc_pool(rng):
    """Segmenta os leucocitos dos crops raabin_*. Devolve dict
    subtipo -> lista de (cell_bgr, mask)."""
    pool: dict[str, list] = {n: [] for n in ("Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil")}
    crops = sorted((IMAGES_DIR / "train").glob("raabin_*.jpg"))
    rng.shuffle(crops)
    for img_path in crops:
        boxes = read_label(LABELS_DIR / "train" / f"{img_path.stem}.txt")
        wbc = [b for b in boxes if b[0] in LEUKOCYTE_IDS]
        if len(wbc) != 1:
            continue
        name = CLASSES[wbc[0][0]]
        # ja temos o bastante desse subtipo?
        if len(pool[name]) >= 900:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        Hh, Ww = bgr.shape[:2]
        x1, y1, x2, y2 = yolo_to_xyxy(wbc[0], Ww, Hh)
        # encolhe a caixa do Raabin (folgada) e da uma margem pequena
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * 0.62, (y2 - y1) * 0.62
        rx1 = int(max(0, cx - bw / 2 * 1.25)); ry1 = int(max(0, cy - bh / 2 * 1.25))
        rx2 = int(min(Ww, cx + bw / 2 * 1.25)); ry2 = int(min(Hh, cy + bh / 2 * 1.25))
        region = bgr[ry1:ry2, rx1:rx2]
        if region.size == 0:
            continue
        ext = cell_extent(region)
        if ext is None:
            # fallback: usa o miolo da caixa Raabin encolhida
            rh, rw = region.shape[:2]
            ext = (int(rw * 0.12), int(rh * 0.12), int(rw * 0.88), int(rh * 0.88))
        cut = make_cutout(region, ext)
        if cut is None:
            continue
        pool[name].append(cut)
        if all(len(v) >= 900 for v in pool.values()):
            break
    return pool


def median_rbc_width(boxes, W):
    ws = sorted((b[3] * W) for b in boxes if b[0] == RBC_CLASS_ID)
    return ws[len(ws) // 2] if ws else None


def load_old_model():
    if not OLD_MODEL_PATH.exists():
        print(f"AVISO: {OLD_MODEL_PATH.name} ausente -- leucocitos reais da imagem-alvo "
              "nao serao cobertos (podem ficar visiveis sem caixa).")
        return None, None
    from ultralytics import YOLO
    m = YOLO(str(OLD_MODEL_PATH))
    wbc_id = {v: k for k, v in m.names.items()}.get("WBC")
    return m, wbc_id


def real_wbc_boxes(old_model, wbc_id, img_path):
    """Caixas (x1,y1,x2,y2) dos leucocitos reais que o modelo antigo acha na
    imagem-alvo -- para cobrir cada um com um sintetico rotulado."""
    if old_model is None or wbc_id is None:
        return []
    r = old_model.predict(source=str(img_path), imgsz=1280, conf=OLD_WBC_CONF,
                          max_det=100, verbose=False)[0]
    return [tuple(float(v) for v in b.xyxy[0]) for b in r.boxes if int(b.cls[0]) == wbc_id]


def paste_one(canvas, existing_xyxy, cell, mask, target_w, rng, angle_jitter=12, at=None):
    """Cola uma celula no canvas. Se `at`=(x1,y1,x2,y2) for dado, centra ali
    (cobrindo um leucocito real); senao procura posicao aleatoria valida.
    Devolve a caixa (x1,y1,x2,y2) colada ou None."""
    ch, cw = cell.shape[:2]
    if at is not None:
        target_w = max(at[2] - at[0], at[3] - at[1]) * 1.15
    target_w = min(target_w, MAX_WBC_FRAC_OF_IMG * canvas.shape[1])
    scale = target_w / cw
    nw, nh = max(6, int(cw * scale)), max(6, int(ch * scale))
    cell_r = cv2.resize(cell, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)

    ang = rng.uniform(-angle_jitter, angle_jitter)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), ang, 1.0)
    cell_r = cv2.warpAffine(cell_r, M, (nw, nh), borderMode=cv2.BORDER_REFLECT)
    mask_r = cv2.warpAffine(mask_r, M, (nw, nh))

    # jitter de cor leve
    if rng.random() < 0.7:
        hsv = cv2.cvtColor(cell_r, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-18, 18), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-18, 18), 0, 255)
        cell_r = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    soft = cv2.GaussianBlur(mask_r, (0, 0), sigmaX=max(1.5, nw * 0.04))
    alpha = (soft.astype(np.float32) / 255.0)[..., None]

    Himg, Wimg = canvas.shape[:2]
    if nw >= Wimg or nh >= Himg:
        return None

    if at is not None:
        acx, acy = (at[0] + at[2]) / 2, (at[1] + at[3]) / 2
        candidates = [(int(np.clip(acx - nw / 2, 0, Wimg - nw)),
                       int(np.clip(acy - nh / 2, 0, Himg - nh)))]
    else:
        candidates = [(rng.randint(0, Wimg - nw), rng.randint(0, Himg - nh)) for _ in range(30)]

    for px, py in candidates:
        box = (px, py, px + nw, py + nh)
        if at is None and any(iou_xyxy(box, e) > MAX_OVERLAP_WITH_WBC for e in existing_xyxy):
            continue
        roi = canvas[py:py + nh, px:px + nw].astype(np.float32)
        canvas[py:py + nh, px:px + nw] = (alpha * cell_r + (1 - alpha) * roi).astype(np.uint8)
        ys, xs = np.where(soft > 40)
        if len(xs) < 10:
            return box
        return (px + int(xs.min()), py + int(ys.min()), px + int(xs.max()), py + int(ys.max()))
    return None


def main():
    rng = random.Random(RANDOM_SEED)

    for split in ("train", "val"):
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    removed = 0
    for split in ("train", "val"):
        for d in (IMAGES_DIR / split, LABELS_DIR / split):
            for f in d.glob(f"{FILE_PREFIX}*"):
                f.unlink(); removed += 1
    for c in BCCD_ROOT.glob("*.cache"):
        c.unlink()
    if removed:
        print(f"Removidos {removed} arquivos {FILE_PREFIX}* anteriores (+ caches).")

    print("Segmentando leucocitos dos crops Raabin...")
    pool = load_wbc_pool(rng)
    for k, v in pool.items():
        print(f"  {k}: {len(v)} recortes segmentados")
    if sum(len(v) for v in pool.values()) < 100:
        raise SystemExit("Poucos leucocitos segmentados -- rode 10_merge_raabin_wbc.py antes.")

    names = list(pool)
    weights = [CLASS_SAMPLING_WEIGHT[n] * (1 if pool[n] else 0) for n in names]

    old_model, old_wbc_id = load_old_model()

    targets = []
    for pref in TARGET_PREFIXES:
        targets += sorted((IMAGES_DIR / "train").glob(f"{pref}*"))
    targets = [t for t in targets if not t.name.startswith(FILE_PREFIX)]
    print(f"{len(targets)} imagens-alvo de campo largo")
    if not targets:
        raise SystemExit("Nenhuma imagem-alvo encontrada.")

    # leucocitos reais em cada alvo (uma inferencia por alvo, cacheada)
    print("Detectando leucocitos reais nos alvos (modelo antigo)...")
    real_wbc_cache = {t: real_wbc_boxes(old_model, old_wbc_id, t) for t in set(targets)}
    n_real = sum(len(v) for v in real_wbc_cache.values())
    print(f"  {n_real} leucocitos reais a cobrir em {len(real_wbc_cache)} alvos")

    per_class = {"train": Counter(), "val": Counter()}
    n_out = {"train": 0, "val": 0}
    previews = []

    def add_box(boxes, cid, x1, y1, x2, y2, W, H):
        boxes.append((cid, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))

    for i in range(N_SYNTHETIC):
        tgt = targets[rng.randrange(len(targets))]
        canvas = cv2.imread(str(tgt))
        if canvas is None:
            continue
        Himg, Wimg = canvas.shape[:2]
        boxes = read_label(LABELS_DIR / "train" / f"{tgt.stem}.txt")
        wbc_xyxy = [yolo_to_xyxy(b, Wimg, Himg) for b in boxes if b[0] in LEUKOCYTE_IDS]

        rbc_w = median_rbc_width(boxes, Wimg) or (Wimg * FALLBACK_RBC_FRAC)
        new_boxes = []

        # 1. cobre cada leucocito real com um sintetico rotulado
        for rb in real_wbc_cache.get(tgt, []):
            name = rng.choices(names, weights=weights, k=1)[0]
            cell, mask = pool[name][rng.randrange(len(pool[name]))]
            placed = paste_one(canvas, [], cell, mask, 0, rng, at=rb)
            if placed is None:
                continue
            new_boxes.append(placed)
            add_box(boxes, CLASSES.index(name), *placed, Wimg, Himg)

        # 2. alguns leucocitos extras em posicao aleatoria
        for _ in range(rng.randint(*PASTES_PER_IMAGE)):
            name = rng.choices(names, weights=weights, k=1)[0]
            cell, mask = pool[name][rng.randrange(len(pool[name]))]
            target_w = rbc_w * rng.uniform(*WBC_TO_RBC_RATIO)
            placed = paste_one(canvas, wbc_xyxy + new_boxes, cell, mask, target_w, rng)
            if placed is None:
                continue
            new_boxes.append(placed)
            add_box(boxes, CLASSES.index(name), *placed, Wimg, Himg)

        if not new_boxes:
            continue

        split = "val" if rng.random() < VAL_FRACTION else "train"
        base = f"{FILE_PREFIX}{i:05d}_{tgt.stem}"
        cv2.imwrite(str(IMAGES_DIR / split / f"{base}.jpg"),
                    canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        (LABELS_DIR / split / f"{base}.txt").write_text(
            "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes),
            encoding="utf-8")
        n_out[split] += 1
        for c, *_ in boxes:
            if c in LEUKOCYTE_IDS:
                per_class[split][CLASSES[c]] += 1
        if len(previews) < PREVIEW_SAMPLE:
            previews.append((split, base))

    # contact sheet
    tiles = []
    for split, base in previews:
        im = cv2.imread(str(IMAGES_DIR / split / f"{base}.jpg"))
        Hh, Ww = im.shape[:2]
        for ln in (LABELS_DIR / split / f"{base}.txt").read_text().splitlines():
            c, xc, yc, w, h = ln.split()
            c = int(c)
            if c not in LEUKOCYTE_IDS:
                continue
            x1, y1, x2, y2 = yolo_to_xyxy((c, float(xc), float(yc), float(w), float(h)), Ww, Hh)
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        tiles.append(cv2.resize(im, (320, 320)))
    if tiles:
        cols = 5
        rows = [tiles[k:k + cols] for k in range(0, len(tiles), cols)]
        rows = [r for r in rows if len(r) == cols]
        if rows:
            sheet = np.vstack([np.hstack(r) for r in rows])
            cv2.imwrite(str(PREVIEW_DIR / "_contact_sheet.jpg"), sheet)
            print(f"Contact sheet: {PREVIEW_DIR / '_contact_sheet.jpg'}")

    print("\n=== Resumo ===")
    for split in ("train", "val"):
        by = ", ".join(f"{CLASS_LABELS_PT[c]}={per_class[split][c]}"
                       for c in CLASSES if per_class[split][c])
        print(f"  {split}: {n_out[split]} imagens sinteticas | leucocitos colados: [{by}]")


if __name__ == "__main__":
    main()
