"""
Extrai frames de videos de varredura de lamina (Videos/*.mp4) e fotos soltas
(Pictures/*.jpg) do mesmo banco de laminas do usuario, e aplica a cada uma o
mesmo pseudo-labeling por tiling de 05_add_wide_field_sample.py, adicionando
ao treino do detector 3-classes com o prefixo "realslide_".

Frames de video sao amostrados a cada SAMPLE_INTERVAL_SEC segundos (poda a
redundancia de uma varredura lenta) e dois filtros evitam lixo no treino:
  - dedup: pula um frame quase identico ao ultimo frame MANTIDO (diff medio
    de pixel numa miniatura), o que acontece quando a varredura pausa;
  - qualidade: so adiciona ao treino um frame cujo pseudo-labeling encontrou
    pelo menos MIN_DETECTIONS caixas no total -- um campo real dessa lamina
    tem sempre dezenas de hemacias, entao poucas/nenhuma deteccao sinaliza
    frame borrado por movimento, fora de foco, ou fora da lamina;
  - brilho: descarta ANTES de rodar o modelo um frame escuro demais (luz do
    microscopio cortada/transicao entre campos) -- confirmado em producao que
    esses frames (media de brilho ~35-40, contra >125 dos frames validos)
    fazem o modelo alucinar uma caixa de WBC em quase todo tile da grade de
    tiling, o que o filtro de MIN_DETECTIONS sozinho NAO pega (o total fica
    alto, so que todo alucinado).

Cada imagem aceita gera preview em data/bccd/pseudo_label_previews/ (mesma
pasta do script 05) para conferencia manual antes do proximo 03_train.py.

Uso:
    python scripts/14_add_wide_field_video_samples.py
"""
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICTURES_DIR = PROJECT_ROOT / "Pictures"
VIDEOS_DIR = PROJECT_ROOT / "Videos"
STAGING_DIR = PROJECT_ROOT / "data" / "bccd" / "wide_field_video_frames"
PREFIX = "realslide_"

SAMPLE_INTERVAL_SEC = 3.0
DEDUP_DIFF_THRESHOLD = 4.0
MAX_CANDIDATES_PER_VIDEO = 80
MIN_DETECTIONS = 20
MIN_BRIGHTNESS_MEAN = 80.0

_spec = importlib.util.spec_from_file_location(
    "add_wide_field_sample", PROJECT_ROOT / "scripts" / "05_add_wide_field_sample.py"
)
wfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfs)


def thumb(frame):
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def extract_candidate_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * SAMPLE_INTERVAL_SEC)), 1)

    kept = []
    last_thumb = None
    frame_idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = thumb(frame)
                if last_thumb is None or float(np.mean(cv2.absdiff(t, last_thumb))) >= DEDUP_DIFF_THRESHOLD:
                    kept.append((frame_idx, frame))
                    last_thumb = t
        frame_idx += 1
    cap.release()

    if len(kept) > MAX_CANDIDATES_PER_VIDEO:
        ratio = len(kept) / MAX_CANDIDATES_PER_VIDEO
        kept = [kept[int(i * ratio)] for i in range(MAX_CANDIDATES_PER_VIDEO)]

    return kept


def stage_video_frames():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    for video in sorted(VIDEOS_DIR.glob("*.mp4")):
        print(f"Extraindo candidatos de {video.name}...")
        frames = extract_candidate_frames(video)
        print(f"  {len(frames)} frames candidatos (amostra a cada {SAMPLE_INTERVAL_SEC}s, dedup aplicado)")
        for frame_idx, frame in frames:
            out_path = STAGING_DIR / f"{video.stem}_f{frame_idx:06d}.jpg"
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            staged.append(out_path)
    return staged


def is_too_dark(image_path: Path) -> bool:
    gray = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
    return bool(gray.mean() < MIN_BRIGHTNESS_MEAN)


def main():
    candidates = sorted({p.resolve() for p in PICTURES_DIR.glob("*.[jJ][pP][gG]")})
    candidates += stage_video_frames()
    print(f"\nTotal de candidatos para pseudo-labeling: {len(candidates)}")

    model = wfs.YOLO(str(wfs.MODEL_PATH))

    added, skipped, too_dark = 0, 0, 0
    total_counts = {}
    t_start = time.time()

    for i, image_path in enumerate(candidates, 1):
        if is_too_dark(image_path):
            too_dark += 1
            print(f"[{i}/{len(candidates)}] {image_path.name}: SKIP (frame escuro demais, provavel transicao/sem luz)")
            continue

        detections, width, height = wfs.pseudo_label(image_path, model)

        if len(detections) < MIN_DETECTIONS:
            skipped += 1
            print(f"[{i}/{len(candidates)}] {image_path.name}: SKIP ({len(detections)} deteccoes, abaixo de {MIN_DETECTIONS})")
            continue

        counts = {}
        for class_id, *_ in detections:
            name = model.names[class_id]
            counts[name] = counts.get(name, 0) + 1
            total_counts[name] = total_counts.get(name, 0) + 1

        preview_path = wfs.PREVIEW_DIR / f"{PREFIX}{image_path.stem}_preview.jpg"
        wfs.save_preview(image_path, detections, model.names, preview_path)

        dest_image = wfs.TRAIN_IMAGES_DIR / f"{PREFIX}{image_path.stem}.jpg"
        dest_label = wfs.TRAIN_LABELS_DIR / f"{PREFIX}{image_path.stem}.txt"
        wfs.Image.open(image_path).convert("RGB").save(dest_image)
        wfs.save_yolo_labels(detections, width, height, dest_label)

        added += 1
        print(f"[{i}/{len(candidates)}] {image_path.name}: {counts}")

    elapsed = time.time() - t_start
    print(f"\nConcluido em {elapsed:.0f}s. Adicionadas: {added}. Puladas (poucas deteccoes): {skipped}. Puladas (escuras): {too_dark}.")
    print("Total por classe (imagens novas adicionadas):", total_counts)
    print(f"Previews em: {wfs.PREVIEW_DIR}")
    print(f"Imagens adicionadas em: {wfs.TRAIN_IMAGES_DIR}")
    print(f"Labels adicionados em: {wfs.TRAIN_LABELS_DIR}")


if __name__ == "__main__":
    main()
