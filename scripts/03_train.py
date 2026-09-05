"""
Treina o YOLOv8 (fine-tuning) na taxonomia de data/bccd/dataset.yaml
(7 classes: hemacia + 5 subtipos de leucocito + plaqueta).
Detecta automaticamente GPU (cuda) e cai para CPU se nao houver.

Nota: ao continuar de models/carjim_best.pt (treinado com 3 classes), o
ultralytics reaproveita o backbone e reinicializa a cabeca de deteccao
porque o numero de classes mudou -- e esperado e correto.
"""
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "data" / "bccd" / "dataset.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

PRETRAINED_MODEL = "yolov8s.pt"
CARJIM_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "carjim_best.pt"
EPOCHS = 100
# 1280 (em vez de 640) preserva melhor celulas pequenas em fotos de campo largo
# (ex.: foto de celular no microscopio com muitas celulas por imagem), que
# ficam praticamente invisiveis apos redimensionar para 640. Mantenha o mesmo
# imgsz aqui e em 04_watch_and_infer.py / 05_add_wide_field_sample.py.
IMG_SIZE = 1280
# batch=8 numa RTX 3060 de 12 GB com imgsz=1280 usa ~7 GB e cabe com folga
# QUANDO a GPU nao esta disputando VRAM com navegador/launchers. Com o desktop
# cheio, o pico da validacao ja estourou os 12 GB com batch=8 -- nesse caso
# baixe para 4.
BATCH = 8
PATIENCE = 20
RUN_NAME = "carjim_train"
# workers=2 (nao o default 8): a maquina so tem 16 GB de RAM. No Windows cada
# worker do dataloader e um processo que recarrega torch/opencv (~1 GB), e a
# validacao sobe workers a mais -- com 8 o SO matou workers na validacao da
# epoca 1 ("DataLoader worker exited unexpectedly"). Com 2 cabe na RAM.
WORKERS = 2


def main():
    if not DATASET_YAML.exists():
        raise RuntimeError(
            f"{DATASET_YAML} nao encontrado. Rode antes 02_prepare_dataset.py."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("Nenhuma GPU CUDA detectada, treinando na CPU (mais lento).")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if CARJIM_CHECKPOINT.exists():
        print(f"Continuando fine-tuning a partir de: {CARJIM_CHECKPOINT}")
        start_model = str(CARJIM_CHECKPOINT)
    else:
        print(f"Nenhum checkpoint anterior encontrado, partindo de: {PRETRAINED_MODEL}")
        start_model = PRETRAINED_MODEL

    model = YOLO(start_model)
    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=device,
        patience=PATIENCE,
        project=str(RUNS_DIR / "detect"),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_weights = RUNS_DIR / "detect" / RUN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"Checkpoint final nao encontrado em {best_weights}")

    dest = MODELS_DIR / "carjim_best.pt"
    shutil.copyfile(best_weights, dest)
    print(f"Treinamento concluido. Melhor checkpoint copiado para: {dest}")


if __name__ == "__main__":
    main()
