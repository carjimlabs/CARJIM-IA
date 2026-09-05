"""
Converte as anotacoes Pascal VOC XML do BCCD para o formato YOLO,
faz o split treino/validacao e gera o dataset.yaml.
"""
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES  # noqa: E402 -- fonte unica da taxonomia

BCCD_ROOT = PROJECT_ROOT / "data" / "bccd" / "raw" / "BCCD_Dataset-master" / "BCCD"
IMAGES_SRC = BCCD_ROOT / "JPEGImages"
ANNOTATIONS_SRC = BCCD_ROOT / "Annotations"

OUT_ROOT = PROJECT_ROOT / "data" / "bccd"
# BCCD so tem "RBC"/"WBC"/"Platelets"; com a taxonomia de 7 classes o "WBC"
# generico nao existe mais (nao ha subtipo anotado no BCCD) e cai fora no
# filtro `name not in CLASSES` abaixo -- mesmo efeito da migracao 09.
VAL_SPLIT = 0.2
RANDOM_SEED = 42


def parse_voc_annotation(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue
        class_id = CLASSES.index(name)

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def yolo_line(box) -> str:
    class_id, xc, yc, w, h = box
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def build_dataset():
    xml_files = sorted(ANNOTATIONS_SRC.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(
            f"Nenhuma anotacao encontrada em {ANNOTATIONS_SRC}. "
            "Rode primeiro 01_download_dataset.py."
        )

    stems = [f.stem for f in xml_files]
    random.Random(RANDOM_SEED).shuffle(stems)

    split_idx = int(len(stems) * (1 - VAL_SPLIT))
    train_stems = set(stems[:split_idx])
    val_stems = set(stems[split_idx:])

    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped_no_boxes": 0}

    for stem in stems:
        split = "train" if stem in train_stems else "val"

        xml_path = ANNOTATIONS_SRC / f"{stem}.xml"
        img_path = IMAGES_SRC / f"{stem}.jpg"

        if not img_path.exists():
            print(f"Aviso: imagem nao encontrada para {stem}, pulando.")
            continue

        boxes = parse_voc_annotation(xml_path)
        if not boxes:
            counts["skipped_no_boxes"] += 1
            continue

        dst_img = OUT_ROOT / "images" / split / img_path.name
        dst_label = OUT_ROOT / "labels" / split / f"{stem}.txt"

        shutil.copyfile(img_path, dst_img)
        dst_label.write_text("\n".join(yolo_line(b) for b in boxes), encoding="utf-8")

        counts[split] += 1

    return counts


def write_dataset_yaml():
    data = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    yaml_path = OUT_ROOT / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"dataset.yaml gerado em: {yaml_path}")


if __name__ == "__main__":
    counts = build_dataset()
    write_dataset_yaml()
    print(
        f"Concluido. Treino: {counts['train']} imagens | "
        f"Validacao: {counts['val']} imagens | "
        f"Sem caixas (ignoradas): {counts['skipped_no_boxes']}"
    )
