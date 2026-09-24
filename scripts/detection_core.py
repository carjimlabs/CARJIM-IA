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
import hashlib
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import batched_nms, box_iou, nms
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
# banco de exemplos (embeddings) para o kNN do classificador -- gerado por
# scripts/15_build_wbc_memory.py; opcional
WBC_MEMORY_PATH = _resolve("wbc_memory.pt")

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

# --- Normalizacao de escala (fotos em distancias diferentes) ---
# A hemacia tem tamanho fisico quase constante (~7-8 um), entao o tamanho dela
# em pixels mede o zoom da foto. O detector foi treinado com hemacias em duas
# faixas de tamanho (px na entrada da rede, lado maior = PREDICT_IMGSZ):
# fotos de campo largo reais (realslide_, ~25-55 px) e BCCD / TXL-PBC
# (~175-300 px). Medido com 16_eval_scale_robustness.py e varredura manual:
#   - imagem tipo BCCD com hemacia <= ~55 px: o detector acha ZERO hemacias;
#     ampliada para ~100-150 px volta a F1 ~0.7 (o melhor alvo foi ~100-110).
#   - fotos realslide_ (~38 px): a passada normal e a melhor; ampliar piora.
# Por isso a faixa pequena so e mantida quando a 1a passada de fato achou
# hemacias; se achou poucas, testa ampliado/reduzido (ver detect_cells).
SCALE_NORMALIZATION = True
RBC_SMALL_BAND = (20.0, 60.0)     # faixa das fotos reais: mantida se achou hemacias
RBC_SMALL_TARGET = 38.0           # alvo quando a hemacia e menor que a faixa pequena
RBC_GAP_TARGET = 110.0            # alvo entre as faixas (nenhum dado de treino ali)
RBC_LARGE_BAND = (175.0, 300.0)
RBC_LARGE_TARGET = 250.0          # alvo quando a hemacia e maior que a faixa grande
# com menos hemacias que isso na 1a passada o tamanho medido nao e confiavel:
# testa tambem 0.5x e SCALE_PROBE_ZOOM x antes de decidir
SCALE_PROBE_MIN_RBC = 5
SCALE_PROBE_ZOOM = 3.0
MAX_SCALE_TILES = 36
SCALE_TILE_BATCH = 4
MIN_PREDICT_IMGSZ = 320
TILE_EDGE_MARGIN_PX = 3
SCALE_MERGE_IOU = 0.5
# Test-time augmentation do Ultralytics (flip + 3 escalas por passada):
# ~2-3x mais lento.
USE_TTA = False

# --- kNN sobre exemplos (classificador de subtipo) ---
# Probabilidade final = (1 - KNN_BLEND) * classificador + KNN_BLEND * votos
# dos KNN_K exemplos mais parecidos do banco models/wbc_memory.pt.
KNN_BLEND = 0.3
KNN_K = 10
KNN_TEMPERATURE = 0.05

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

    def __init__(self, det_model: YOLO, cls_model: YOLO | None, device: str, memory: dict | None = None):
        self.det = det_model
        self.cls = cls_model
        self.device = device
        # banco do kNN: embeddings normalizados (N, D) + indice de classe do
        # classificador (N,)
        self.memory_emb = None
        self.memory_labels = None
        if cls_model is not None and memory is not None:
            self.memory_emb = memory["emb"].float().to(device)
            self.memory_labels = memory["labels"].long().to(device)
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
        probs, emb = self.classify_probs(crops)
        if self.memory_emb is not None and KNN_BLEND > 0:
            probs = (1 - KNN_BLEND) * probs + KNN_BLEND * self.knn_probs(emb)
        return [self._cls_name_to_id.get(int(i), WBC_FALLBACK_ID) for i in probs.argmax(1).tolist()]

    def classify_probs(self, crops: list[Image.Image]):
        """Roda o classificador. Devolve (probs (N, C), embeddings (N, D)
        normalizados) -- o embedding e a entrada da camada linear final,
        capturada por hook na mesma passada."""
        captured = []
        head_linear = self.cls.model.model[-1].linear
        hook = head_linear.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].detach()))
        try:
            results = self.cls.predict(source=crops, imgsz=CLASSIFIER_IMGSZ, verbose=False)
        finally:
            hook.remove()
        probs = torch.stack([r.probs.data.float() for r in results]).to(self.device)
        # na 1a chamada o Ultralytics faz um warmup que tambem passa pelo hook
        emb = torch.cat(captured)[-len(crops):].float().to(self.device)
        return probs, torch.nn.functional.normalize(emb, dim=1)

    def knn_probs(self, emb: torch.Tensor) -> torch.Tensor:
        """Distribuicao de classe pelos KNN_K exemplos mais parecidos do banco
        (similaridade de cosseno, votos ponderados por softmax)."""
        sims = emb @ self.memory_emb.T
        k = min(KNN_K, sims.shape[1])
        top_sims, top_idx = sims.topk(k, dim=1)
        weights = torch.softmax(top_sims / KNN_TEMPERATURE, dim=1)
        out = torch.zeros(emb.shape[0], len(self.cls.names), device=emb.device)
        out.scatter_add_(1, self.memory_labels[top_idx], weights)
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
    memory = None
    if CLASSIFIER_PATH.exists():
        cls = YOLO(str(CLASSIFIER_PATH))
        cls.to(device)
        memory = load_wbc_memory()
    return Detector(det, cls, device, memory), device


def load_wbc_memory(path: Path = WBC_MEMORY_PATH) -> dict | None:
    """Banco do kNN, se existir e tiver sido gerado com o classificador atual
    (embeddings de outro checkpoint nao sao comparaveis)."""
    if not path.exists():
        return None
    memory = torch.load(path, map_location="cpu")
    if memory.get("classifier_sha1") != classifier_sha1():
        print(f"Aviso: {path.name} foi gerado com outro wbc_classifier.pt -- "
              f"ignorado. Rode scripts/15_build_wbc_memory.py de novo.")
        return None
    return memory


def classifier_sha1() -> str:
    return hashlib.sha1(CLASSIFIER_PATH.read_bytes()).hexdigest()


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


def _predict_boxes(detector: Detector, images, imgsz: int, tta: bool):
    """predict do detector em lote. Devolve, por imagem, lista de
    (det_id, x1, y1, x2, y2, conf) nas coordenadas daquela imagem."""
    out = []
    for i in range(0, len(images), SCALE_TILE_BATCH):
        results = detector.det.predict(
            source=images[i:i + SCALE_TILE_BATCH], imgsz=imgsz, conf=CONFIDENCE_THRESHOLD,
            max_det=PREDICT_MAX_DET, augment=tta, verbose=False,
        )
        for r in results:
            b = r.boxes
            out.append([(int(c), *xyxy, float(s))
                        for c, xyxy, s in zip(b.cls.tolist(), b.xyxy.tolist(), b.conf.tolist())])
    return out


def _median_rbc_size(boxes, rbc_id):
    sizes = sorted(((x2 - x1) + (y2 - y1)) / 2 for c, x1, y1, x2, y2, _ in boxes if c == rbc_id)
    return (sizes[len(sizes) // 2], len(sizes)) if sizes else (None, 0)


def _tiled_predict(detector: Detector, image: Image.Image, zoom: float, cell_px: float, tta: bool):
    """Deteccao com a imagem ampliada `zoom` vezes (relativo a passada normal
    em PREDICT_IMGSZ), feita em blocos que a rede ve em PREDICT_IMGSZ.
    Sobreposicao >= 2.5 celulas: caixas encostadas numa borda interna do bloco
    (celula cortada) sao descartadas porque a celula aparece inteira no
    vizinho. `cell_px` = tamanho da hemacia na imagem original."""
    W, H = image.size
    tile = max(W, H) / zoom
    while True:
        tw, th = min(tile, W), min(tile, H)
        overlap = min(max(0.2 * tile, 2.5 * cell_px), 0.5 * tile)
        step = tile - overlap
        nx = 1 if tw >= W else math.ceil((W - tw) / step) + 1
        ny = 1 if th >= H else math.ceil((H - th) / step) + 1
        if nx * ny <= MAX_SCALE_TILES:
            break
        tile *= 1.15   # menos zoom para caber no limite de blocos
    tw, th = int(round(tw)), int(round(th))
    xs = [min(round(i * step), W - tw) for i in range(nx)]
    ys = [min(round(j * step), H - th) for j in range(ny)]
    origins = [(x0, y0) for y0 in ys for x0 in xs]
    tiles = [image.crop((x0, y0, x0 + tw, y0 + th)) for x0, y0 in origins]

    boxes, scores, classes = [], [], []
    m = TILE_EDGE_MARGIN_PX
    for (x0, y0), tile_boxes in zip(origins, _predict_boxes(detector, tiles, PREDICT_IMGSZ, tta)):
        for c, bx1, by1, bx2, by2, conf in tile_boxes:
            if (bx1 <= m and x0 > 0) or (by1 <= m and y0 > 0) \
                    or (bx2 >= tw - m and x0 + tw < W) or (by2 >= th - m and y0 + th < H):
                continue
            boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
            scores.append(conf)
            classes.append(c)
    if not boxes:
        return []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    classes_t = torch.tensor(classes)
    keep = batched_nms(boxes_t, scores_t, classes_t, SCALE_MERGE_IOU)
    return [(classes[i], *boxes[i], scores[i]) for i in keep.tolist()]


def _target_rbc_net_size(net_size: float, small_band_ok: bool) -> float | None:
    """Tamanho-alvo da hemacia (px na rede) ou None se ja esta numa faixa em
    que o detector funciona. `small_band_ok` = a passada normal achou
    hemacias suficientes, ou seja, a imagem se parece com as fotos de campo
    largo do treino e a faixa pequena serve."""
    if net_size > RBC_LARGE_BAND[1]:
        return RBC_LARGE_TARGET
    if net_size >= RBC_LARGE_BAND[0]:
        return None
    if small_band_ok:
        if net_size < RBC_SMALL_BAND[0]:
            return RBC_SMALL_TARGET
        if net_size <= RBC_SMALL_BAND[1]:
            return None
    return max(RBC_GAP_TARGET, net_size)


def detect_cells(detector: Detector, image: Image.Image, scale_norm: bool | None = None,
                 tta: bool | None = None):
    """Deteccao de 3 classes (ids do detector) com normalizacao de escala
    opcional -- ver RBC_NET_BANDS."""
    scale_norm = SCALE_NORMALIZATION if scale_norm is None else scale_norm
    tta = USE_TTA if tta is None else tta
    long_side = max(image.size)
    base = _predict_boxes(detector, [image], PREDICT_IMGSZ, tta)[0]
    if not scale_norm or detector.det_rbc_id is None:
        return base

    rbc_px, n_rbc = _median_rbc_size(base, detector.det_rbc_id)
    small_band_ok = n_rbc >= SCALE_PROBE_MIN_RBC
    chosen, chosen_zoom = base, 1.0
    if not small_band_ok:
        # quase nenhuma hemacia: a foto pode estar tao longe/perto que o
        # detector nem enxerga. Testa reduzida e ampliada e fica com a que
        # achar mais hemacias (so para medir o tamanho delas).
        probes = [
            (0.5, _predict_boxes(detector, [image], _round_imgsz(PREDICT_IMGSZ * 0.5), tta)[0]),
            (SCALE_PROBE_ZOOM, _tiled_predict(detector, image, SCALE_PROBE_ZOOM, long_side / 60, tta)),
        ]
        for zoom, probe in probes:
            px, n = _median_rbc_size(probe, detector.det_rbc_id)
            if n > n_rbc:
                chosen, chosen_zoom, rbc_px, n_rbc = probe, zoom, px, n
        if n_rbc < SCALE_PROBE_MIN_RBC:
            return base

    net_size = rbc_px * PREDICT_IMGSZ / long_side
    target = _target_rbc_net_size(net_size, small_band_ok)
    zoom = 1.0 if target is None else target / net_size
    if abs(math.log(zoom / chosen_zoom)) < 0.1:   # ja rodou nessa escala
        return chosen
    if zoom < 1:
        return _predict_boxes(detector, [image], _round_imgsz(PREDICT_IMGSZ * zoom), tta)[0]
    return _tiled_predict(detector, image, zoom, rbc_px, tta)


def _round_imgsz(size: float) -> int:
    return max(MIN_PREDICT_IMGSZ, int(round(size / 32)) * 32)


def primary_detections(detector: Detector, image: Image.Image, scale_norm: bool | None = None,
                       tta: bool | None = None):
    """Detecta RBC/WBC/Platelets (com normalizacao de escala -- ver
    detect_cells) e classifica cada WBC no subtipo. Devolve lista de
    (class_id, x1, y1, x2, y2, conf) na taxonomia de exibicao (RBC=0,
    subtipos 1..5, Platelets=6, WBC generico=7)."""
    detections = []
    wbc_slots = []   # (indice na lista detections, crop)
    for det_id, x1, y1, x2, y2, conf in detect_cells(detector, image, scale_norm, tta):
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
