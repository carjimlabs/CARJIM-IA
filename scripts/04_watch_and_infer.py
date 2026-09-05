"""
Monitora a pasta 'Imagens a serem analisadas/'; a cada imagem nova,
roda a deteccao com o modelo treinado, desenha as molduras com o nome
da celula em portugues e salva o resultado em 'Imagens analisadas/'.

Uso: python scripts/04_watch_and_infer.py
Pressione Ctrl+C para parar.
"""
import time
from pathlib import Path

from detection_core import (
    MODEL_PATH,
    PLATELET_CLASS_NAME,
    RBC_CLASS_NAME,
    VALID_EXTENSIONS,
    draw_detections,
    estimate_rbc_size,
    load_font,
    load_model,
    merge_platelet_detections,
    platelet_tile_scan,
    primary_detections,
)
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "Imagens a serem analisadas"
OUTPUT_DIR = PROJECT_ROOT / "Imagens analisadas"
PROCESSED_DIR = INPUT_DIR / "_processadas"

POLL_INTERVAL_SECONDS = 3


def process_image(model, image_path: Path, font) -> None:
    print(f"Processando: {image_path.name}")
    image = Image.open(image_path).convert("RGB")

    detections = primary_detections(model, image)

    platelet_class_id = next((cid for cid, name in model.names.items() if name == PLATELET_CLASS_NAME), None)
    rbc_class_id = next((cid for cid, name in model.names.items() if name == RBC_CLASS_NAME), None)
    rbc_size_px = estimate_rbc_size(detections, rbc_class_id) if rbc_class_id is not None else None

    if platelet_class_id is not None and rbc_size_px:
        tile_platelets = platelet_tile_scan(model, image, platelet_class_id, rbc_size_px)
        detections = merge_platelet_detections(detections, tile_platelets, platelet_class_id)

    image = draw_detections(image, detections, model.names, font)

    output_path = OUTPUT_DIR / image_path.name
    image.save(output_path)
    print(f"  -> Salvo em: {output_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    image_path.replace(PROCESSED_DIR / image_path.name)


def watch_loop(model, font) -> None:
    print(f"Monitorando '{INPUT_DIR}'. Pressione Ctrl+C para parar.")
    while True:
        pending = sorted(
            p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        for image_path in pending:
            try:
                process_image(model, image_path, font)
            except Exception as exc:
                print(f"Erro ao processar {image_path.name}: {exc}")
        time.sleep(POLL_INTERVAL_SECONDS)


def main():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo nao encontrado em {MODEL_PATH}. Rode antes 03_train.py."
        )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, device = load_model()
    print(f"Usando dispositivo: {device}")
    font = load_font()

    try:
        watch_loop(model, font)
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario.")


if __name__ == "__main__":
    main()
