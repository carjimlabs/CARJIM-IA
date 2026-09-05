"""
Logica de deteccao reutilizavel, em DOIS ESTAGIOS:

  1. Detector (models/carjim_best.pt): YOLO de 3 classes -- RBC / WBC / Platelets.
     E o modelo que funciona bem em foto de campo largo (o uso real).
  2. Classificador (models/wbc_classifier.pt): YOLOv8-cls que recebe o recorte
     de cada caixa "WBC" e diz qual dos 5 subtipos e (neutrofilo, linfocito,
     monocito, eosinofilo, basofilo).

Tentativas anteriores de treinar UM detector de 7 classes falharam: os dados
do Raabin (unica fonte com subtipo) so vem como recortes com zoom, e o modelo
colapsava em campo largo. Detectar + classificar separadamente resolve --
a deteccao de RBC/plaqueta continua igual a do modelo de 3 classes.

Usado por 04_watch_and_infer.py (monitora pasta) e app.py (GUI) -- nenhum dos
dois deve duplicar essa logica.
"""
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou, nms
from ultralytics import YOLO


def _resolve(name: str) -> Path:
    """Caminho de models/<name>, relativo a onde o codigo roda -- funciona via
    `python scripts/...py` (Path(__file__)) e empacotado como .exe
    (Path(sys.executable))."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    return base / "models" / name


def resolve_model_path() -> Path:
    return _resolve("carjim_best.pt")


MODEL_PATH = resolve_model_path()               # detector 3 classes
CLASSIFIER_PATH = _resolve("wbc_classifier.pt")  # classificador de subtipo

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONFIDENCE_THRESHOLD = 0.25
PREDICT_IMGSZ = 1280
# Fotos de campo largo tem facilmente 300+ hemacias. O default do YOLO
# (max_det=300) cortaria leucocitos/plaquetas menos confiantes.
PREDICT_MAX_DET = 1500
CLASSIFIER_IMGSZ = 128
# margem em volta da caixa do detector ao recortar para o classificador
# (mesma proporcao usada em 12_build_wbc_classifier_data.py)
CLASSIFIER_BOX_PAD = 0.18

PLATELET_CLASS_NAME = "Platelets"
RBC_CLASS_NAME = "RBC"
WBC_CLASS_NAME = "WBC"
TILE_SIZE = 300
TILE_STEP = 200
TILE_PREDICT_IMGSZ = 640
TILE_CONF_THRESHOLD = 0.15
PLATELET_MIN_CONF = 0.55
PLATELET_MIN_SIZE_RATIO = 0.10
PLATELET_MAX_SIZE_RATIO = 0.55
PLATELET_MERGE_IOU = 0.3

# Taxonomia exibida (ordem = class-id usado nas tuplas de deteccao). Os ids
# 1..5 sao atribuidos pelo classificador; RBC=0 e Platelets=6 vem direto do
# detector. O id 7 ("WBC") e o fallback quando o classificador nao esta
# disponivel.
CLASSES = [
    "RBC",
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil",
    "Platelets",
]
WBC_FALLBACK_ID = 7
DISPLAY_NAMES = {i: n for i, n in enumerate(CLASSES)}
DISPLAY_NAMES[WBC_FALLBACK_ID] = WBC_CLASS_NAME

CLASS_LABELS_PT = {
    "RBC": "Hemacia",
    "Neutrophil": "Neutrofilo",
    "Lymphocyte": "Linfocito",
    "Monocyte": "Monocito",
    "Eosinophil": "Eosinofilo",
    "Basophil": "Basofilo",
    "Platelets": "Plaqueta",
    "WBC": "Leucocito",
}

CLASS_COLORS = {
    "RBC": (220, 60, 60),
    "Neutrophil": (60, 120, 220),
    "Lymphocyte": (170, 90, 200),
    "Monocyte": (235, 145, 40),
    "Eosinophil": (230, 90, 150),
    "Basophil": (45, 175, 175),
    "Platelets": (60, 180, 90),
    "WBC": (90, 110, 200),
}


class Detector:
    """Empacota o detector de 3 classes + o classificador de subtipo."""

    def __init__(self, det_model: YOLO, cls_model: YOLO | None, device: str):
        self.det = det_model
        self.cls = cls_model
        self.device = device
        self.names = dict(DISPLAY_NAMES)   # {id: nome} para os consumidores
        dn = {v: k for k, v in det_model.names.items()}
        self.det_rbc_id = dn.get(RBC_CLASS_NAME)
        self.det_wbc_id = dn.get(WBC_CLASS_NAME)
        self.det_platelet_id = dn.get(PLATELET_CLASS_NAME)
        # nome de saida do classificador -> id na taxonomia de exibicao
        self._cls_name_to_id = {}
        if cls_model is not None:
            for idx, cname in cls_model.names.items():
                if cname in CLASSES:
                    self._cls_name_to_id[idx] = CLASSES.index(cname)

    def classify_crops(self, crops: list[Image.Image]) -> list[int]:
        """Recebe uma lista de recortes de leucocito, devolve a lista de ids
        de subtipo (na taxonomia de exibicao). Se nao ha classificador,
        devolve WBC_FALLBACK_ID para todos."""
        if self.cls is None or not crops:
            return [WBC_FALLBACK_ID] * len(crops)
        results = self.cls.predict(source=crops, imgsz=CLASSIFIER_IMGSZ, verbose=False)
        out = []
        for r in results:
            top = int(r.probs.top1)
            out.append(self._cls_name_to_id.get(top, WBC_FALLBACK_ID))
        return out


def load_model(device: str | None = None):
    """Carrega detector + classificador. Devolve (Detector, device).
    Se models/wbc_classifier.pt nao existir, o app roda so com o detector e
    os leucocitos aparecem como 'Leucocito' generico."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(str(MODEL_PATH))
    det.to(device)
    cls = None
    if CLASSIFIER_PATH.exists():
        cls = YOLO(str(CLASSIFIER_PATH))
        cls.to(device)
    return Detector(det, cls, device), device


def load_font(size: int = 16):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections, names, font) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for class_id, x1, y1, x2, y2, conf in detections:
        class_name = names[class_id]
        label_pt = CLASS_LABELS_PT.get(class_name, class_name)
        color = CLASS_COLORS.get(class_name, (255, 200, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label_pt} {conf:.0%}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        text_y = max(0, y1 - th - 4)
        draw.rectangle([x1, text_y, x1 + tw + 4, text_y + th + 4], fill=color)
        draw.text((x1 + 2, text_y + 1), text, fill=(255, 255, 255), font=font)
    return image


def _crop_with_pad(image: Image.Image, x1, y1, x2, y2, pad_frac):
    W, H = image.size
    px = (x2 - x1) * pad_frac
    py = (y2 - y1) * pad_frac
    return image.crop((max(0, x1 - px), max(0, y1 - py),
                       min(W, x2 + px), min(H, y2 + py)))


def primary_detections(detector: Detector, image: Image.Image):
    """Detecta RBC/WBC/Platelets na imagem inteira e classifica cada WBC no
    subtipo. Devolve lista de (class_id, x1, y1, x2, y2, conf) na taxonomia
    de exibicao (RBC=0, subtipos 1..5, Platelets=6, WBC generico=7)."""
    result = detector.det.predict(
        source=image, imgsz=PREDICT_IMGSZ, conf=CONFIDENCE_THRESHOLD,
        max_det=PREDICT_MAX_DET, verbose=False,
    )[0]

    detections = []
    wbc_slots = []   # (indice na lista detections, crop)
    for box in result.boxes:
        det_id = int(box.cls[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        conf = float(box.conf[0])
        if det_id == detector.det_rbc_id:
            detections.append((0, x1, y1, x2, y2, conf))
        elif det_id == detector.det_platelet_id:
            detections.append((CLASSES.index("Platelets"), x1, y1, x2, y2, conf))
        elif det_id == detector.det_wbc_id:
            detections.append((WBC_FALLBACK_ID, x1, y1, x2, y2, conf))
            wbc_slots.append((len(detections) - 1, _crop_with_pad(image, x1, y1, x2, y2, CLASSIFIER_BOX_PAD)))

    if wbc_slots:
        subtype_ids = detector.classify_crops([c for _, c in wbc_slots])
        for (slot, _), sid in zip(wbc_slots, subtype_ids):
            d = detections[slot]
            detections[slot] = (sid, d[1], d[2], d[3], d[4], d[5])

    return detections


def estimate_rbc_size(detections, rbc_class_id) -> float | None:
    sizes = [
        ((x2 - x1) + (y2 - y1)) / 2
        for cls, x1, y1, x2, y2, _conf in detections
        if cls == rbc_class_id
    ]
    if not sizes:
        return None
    sizes.sort()
    return sizes[len(sizes) // 2]


def platelet_tile_scan(detector: Detector, image: Image.Image, platelet_class_id: int, rbc_size_px: float):
    """Passada extra em recortes pequenos, so para reforcar deteccao de
    plaquetas (muito menores que as demais celulas). Tamanho esperado da
    plaqueta relativo ao tamanho das hemacias na mesma imagem."""
    min_size_px = rbc_size_px * PLATELET_MIN_SIZE_RATIO
    max_size_px = rbc_size_px * PLATELET_MAX_SIZE_RATIO
    width, height = image.size
    boxes, scores = [], []

    for y in range(0, height, TILE_STEP):
        for x in range(0, width, TILE_STEP):
            x0 = min(x, max(width - TILE_SIZE, 0))
            y0 = min(y, max(height - TILE_SIZE, 0))
            x1 = min(x0 + TILE_SIZE, width)
            y1 = min(y0 + TILE_SIZE, height)
            tile = image.crop((x0, y0, x1, y1))
            result = detector.det.predict(
                source=tile, imgsz=TILE_PREDICT_IMGSZ, conf=TILE_CONF_THRESHOLD, verbose=False
            )[0]
            for box in result.boxes:
                if int(box.cls[0]) != detector.det_platelet_id:
                    continue
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0])
                bw, bh = bx2 - bx1, by2 - by1
                if conf < PLATELET_MIN_CONF:
                    continue
                if not (min_size_px <= bw <= max_size_px) or not (min_size_px <= bh <= max_size_px):
                    continue
                boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                scores.append(conf)

    if not boxes:
        return []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, PLATELET_MERGE_IOU)
    return [(platelet_class_id, *boxes_t[i].tolist(), float(scores_t[i])) for i in keep]


def merge_platelet_detections(base_detections, tile_platelets, platelet_class_id):
    """Adiciona plaquetas do reforco em recortes, descartando as que se
    sobrepoem a uma deteccao ja existente."""
    existing = [d for d in base_detections if d[0] == platelet_class_id]
    if not existing or not tile_platelets:
        return base_detections + tile_platelets
    existing_boxes = torch.tensor([d[1:5] for d in existing], dtype=torch.float32)
    merged = list(base_detections)
    for det in tile_platelets:
        candidate_box = torch.tensor([det[1:5]], dtype=torch.float32)
        if box_iou(candidate_box, existing_boxes).max().item() < PLATELET_MERGE_IOU:
            merged.append(det)
    return merged
