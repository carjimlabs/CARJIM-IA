"""
Gera rotulos automaticamente (via tiling + pseudo-labeling com o modelo atual)
para uma imagem de campo largo (muitas celulas pequenas, fora da escala do
BCCD) e adiciona essa imagem + rotulos ao conjunto de treino, para o proximo
fine-tuning aprender tambem essa escala.

Uso:
    python scripts/05_add_wide_field_sample.py "caminho/para/imagem.jpg"
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "carjim_best.pt"
TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "bccd" / "images" / "train"
TRAIN_LABELS_DIR = PROJECT_ROOT / "data" / "bccd" / "labels" / "train"
PREVIEW_DIR = PROJECT_ROOT / "data" / "bccd" / "pseudo_label_previews"

TILE_SIZE = 300
OVERLAP = 60
PREDICT_IMGSZ = 640
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5


def make_tiles(width: int, height: int):
    stride = TILE_SIZE - OVERLAP
    xs = list(range(0, max(width - TILE_SIZE, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - TILE_SIZE, 0) + 1, stride)) or [0]
    if xs[-1] + TILE_SIZE < width:
        xs.append(width - TILE_SIZE)
    if ys[-1] + TILE_SIZE < height:
        ys.append(height - TILE_SIZE)

    tiles = []
    for y in ys:
        for x in xs:
            x0 = max(0, min(x, width - TILE_SIZE)) if width > TILE_SIZE else 0
            y0 = max(0, min(y, height - TILE_SIZE)) if height > TILE_SIZE else 0
            x1 = min(x0 + TILE_SIZE, width)
            y1 = min(y0 + TILE_SIZE, height)
            tiles.append((x0, y0, x1, y1))
    return list(set(tiles))


def pseudo_label(image_path: Path, model: YOLO):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    all_boxes = []  # (x1, y1, x2, y2)
    all_scores = []
    all_classes = []

    for (x0, y0, x1, y1) in make_tiles(width, height):
        tile = image.crop((x0, y0, x1, y1))
        results = model.predict(source=tile, imgsz=PREDICT_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        for box in r.boxes:
            bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0]]
            all_boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
            all_scores.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    if not all_boxes:
        return [], width, height

    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    classes_t = torch.tensor(all_classes, dtype=torch.int64)

    final = []
    for class_id in classes_t.unique():
        mask = classes_t == class_id
        keep = nms(boxes_t[mask], scores_t[mask], NMS_IOU_THRESHOLD)
        for idx in keep:
            x1, y1, x2, y2 = boxes_t[mask][idx].tolist()
            final.append((int(class_id), x1, y1, x2, y2, float(scores_t[mask][idx])))

    return final, width, height


def save_preview(image_path: Path, detections, model_names, dest: Path):
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for class_id, x1, y1, x2, y2, conf in detections:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), f"{model_names[class_id]} {conf:.0%}", fill=(255, 0, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)


def save_yolo_labels(detections, width, height, dest: Path):
    lines = []
    for class_id, x1, y1, x2, y2, _conf in detections:
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    dest.write_text("\n".join(lines), encoding="utf-8")


def main():
    if len(sys.argv) != 2:
        print("Uso: python scripts/05_add_wide_field_sample.py <caminho da imagem>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise RuntimeError(f"Imagem nao encontrada: {image_path}")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo nao encontrado em {MODEL_PATH}. Rode antes 03_train.py.")

    model = YOLO(str(MODEL_PATH))
    detections, width, height = pseudo_label(image_path, model)

    print(f"{len(detections)} pseudo-rotulos gerados para {image_path.name} ({width}x{height}).")
    counts = {}
    for class_id, *_ in detections:
        name = model.names[class_id]
        counts[name] = counts.get(name, 0) + 1
    print("Contagem por classe:", counts)

    preview_path = PREVIEW_DIR / f"{image_path.stem}_preview.jpg"
    save_preview(image_path, detections, model.names, preview_path)
    print(f"Preview salvo em: {preview_path}")

    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    dest_image = TRAIN_IMAGES_DIR / image_path.name
    dest_label = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

    Image.open(image_path).convert("RGB").save(dest_image)
    save_yolo_labels(detections, width, height, dest_label)

    print(f"Imagem adicionada ao treino: {dest_image}")
    print(f"Rotulos adicionados ao treino: {dest_label}")


if __name__ == "__main__":
    main()
