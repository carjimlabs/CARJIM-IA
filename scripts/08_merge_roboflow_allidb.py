"""
Mescla o dataset "custom + ALL_IDB" (Roboflow Universe, CC BY 4.0, 1598
imagens) no treino. Esse dataset so tem WBC e plaqueta anotados -- nao tem
hemacia (RBC) nenhuma rotulada, embora as hemacias apareçam nas fotos.

Tentativa descartada: pseudo-rotular RBC com o modelo atual (como
05_add_wide_field_sample.py faz). Nao funcionou -- esse export do Roboflow
inclui copias com augmentation pesada (brilho/contraste alterado, ruido) que
tiram a imagem do dominio de cor que o modelo conhece, e ele passa a
"alucinar" centenas de falsos RBC em cima do fundo/ruido nessas copias (nao e
so ruido de pixel isolado: ate suavizar com blur de mediana piorou, de ~145
para ~470 falsos positivos numa imagem de teste -- entao nao da pra filtrar
so por contagem de caixas ou por deteccao de ruido). Como isso afeta uma
fracao grande do dataset de forma dificil de detectar com seguranca,
preferimos nao arriscar contaminar o RBC com pseudo-rotulo ruim.

Por isso o merge e simples: copiamos so as caixas originais de WBC/plaqueta
(rotulo humano, confiavel) remapeadas para nossas classes, sem nenhum RBC.
Igual ao risco ja aceito em 06_merge_txlpbc.py quando uma fonte nao cobre
todas as classes -- as hemacias visiveis-mas-nao-rotuladas nessas imagens
podem ensinar o treino a tratar RBC ali como fundo, mas isso e mais seguro
que adicionar milhares de caixas de RBC erradas.

Uso: python scripts/08_merge_roboflow_allidb.py
"""
import random
import zipfile
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES as OUR_CLASSES  # noqa: E402 -- fonte unica da taxonomia

ZIP_PATH = PROJECT_ROOT / "custom - ALL_IDB.v1i.yolov8.zip"
RAW_DIR = PROJECT_ROOT / "data" / "roboflow_allidb" / "raw"

TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "bccd" / "images" / "train"
TRAIN_LABELS_DIR = PROJECT_ROOT / "data" / "bccd" / "labels" / "train"
PREVIEW_DIR = PROJECT_ROOT / "data" / "bccd" / "roboflow_allidb_previews"

FILE_PREFIX = "allidb_"
SPLITS = ("train", "valid", "test")
CONTACT_SHEET_SAMPLE = 48


def extract_zip() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.glob("train/images/*")):
        print(f"Ja extraido em: {RAW_DIR}")
        return
    if not ZIP_PATH.exists():
        raise RuntimeError(f"Arquivo nao encontrado: {ZIP_PATH}")

    print(f"Extraindo {ZIP_PATH.name} (pode demorar, o zip e grande)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    print("Extracao concluida.")


def build_class_remap() -> dict:
    with open(RAW_DIR / "data.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    src_classes = config["names"]

    normalized = {name.strip().lower(): i for i, name in enumerate(OUR_CLASSES)}
    aliases = {"platelet": "platelets", "rbcs": "rbc", "wbcs": "wbc"}
    # "wbc" generico nao existe mais na taxonomia (sem subtipo anotado nessa
    # fonte) -> mapeado para None e descartado, igual a migracao 09. Na pratica
    # esse export so tem WBC+plaqueta, entao restam so as caixas de plaqueta.
    droppable = {"wbc", "leukocyte", "leucocito"}
    remap = {}
    for src_id, name in enumerate(src_classes):
        key = name.strip().lower()
        key = aliases.get(key, key)
        if key in normalized:
            remap[src_id] = normalized[key]
        elif key in droppable:
            remap[src_id] = None
        else:
            raise RuntimeError(f"Classe '{name}' do dataset nao existe em OUR_CLASSES={OUR_CLASSES}")

    print(f"Classes do dataset: {src_classes} -> remapeadas para {OUR_CLASSES}: {remap}")
    if any(v is None for v in remap.values()):
        print("  Aviso: caixas de WBC generico serao descartadas (taxonomia atual exige subtipo).")
    return remap


def load_ground_truth_boxes(label_path: Path, remap: dict):
    """Le o rotulo original (ja normalizado) e devolve linhas YOLO com a classe remapeada."""
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        src_class = int(parts[0])
        new_class = remap[src_class]
        if new_class is None:  # classe sem equivalente na taxonomia atual
            continue
        boxes.append((new_class, *parts[1:5]))
    return boxes


def boxes_to_yolo(boxes):
    return "\n".join(f"{class_id} {xc} {yc} {w} {h}" for class_id, xc, yc, w, h in boxes)


def save_preview(image: Image.Image, boxes, dest: Path):
    width, height = image.size
    disp = image.copy()
    draw = ImageDraw.Draw(disp)
    for _class_id, xc, yc, w, h in boxes:
        xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
        draw.rectangle([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], outline=(0, 200, 0), width=3)
    dest.parent.mkdir(parents=True, exist_ok=True)
    disp.save(dest, quality=70)


def build_contact_sheet(sample_previews):
    if not sample_previews:
        return
    thumb_size = 150
    cols = 8
    thumbs = [Image.open(p).resize((thumb_size, thumb_size)) for p in sample_previews]
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    rows = [r for r in rows if len(r) == cols]
    if not rows:
        return
    sheet = Image.new("RGB", (thumb_size * cols, thumb_size * len(rows)))
    for ry, row in enumerate(rows):
        for cx, thumb in enumerate(row):
            sheet.paste(thumb, (cx * thumb_size, ry * thumb_size))
    dest = PREVIEW_DIR / "_contact_sheet.jpg"
    sheet.save(dest, quality=80)
    print(f"Contact sheet (amostra de {len(rows) * cols} imagens) salvo em: {dest}")


def main():
    extract_zip()
    remap = build_class_remap()

    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    gt_counts = {}
    images_done = 0
    preview_paths = []

    for split in SPLITS:
        images_dir = RAW_DIR / split / "images"
        labels_dir = RAW_DIR / split / "labels"
        if not images_dir.exists():
            continue

        img_paths = sorted(images_dir.glob("*.jpg"))
        print(f"[{split}] {len(img_paths)} imagens...")

        for img_path in img_paths:
            boxes = load_ground_truth_boxes(labels_dir / f"{img_path.stem}.txt", remap)
            for class_id, *_ in boxes:
                name = OUR_CLASSES[class_id]
                gt_counts[name] = gt_counts.get(name, 0) + 1

            dest_img = TRAIN_IMAGES_DIR / f"{FILE_PREFIX}{img_path.name}"
            dest_label = TRAIN_LABELS_DIR / f"{FILE_PREFIX}{img_path.stem}.txt"
            dest_img.write_bytes(img_path.read_bytes())
            dest_label.write_text(boxes_to_yolo(boxes), encoding="utf-8")

            preview_path = PREVIEW_DIR / f"{FILE_PREFIX}{img_path.stem}_preview.jpg"
            image = Image.open(img_path).convert("RGB")
            save_preview(image, boxes, preview_path)
            preview_paths.append(preview_path)

            images_done += 1

    random.Random(42).shuffle(preview_paths)
    build_contact_sheet(preview_paths[:CONTACT_SHEET_SAMPLE])

    print(f"Concluido. {images_done} imagens adicionadas ao treino. Rotulos originais: {gt_counts}.")
    print(f"Previas (verde = rotulo humano) em: {PREVIEW_DIR}")
    print(
        "Nenhum RBC foi adicionado para essas imagens -- veja o comentario no "
        "topo do script para o porque."
    )


if __name__ == "__main__":
    main()
