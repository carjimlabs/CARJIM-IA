"""
Treina o classificador de subtipo de leucocito (etapa 2). Le o ImageFolder
gerado por 12_build_wbc_classifier_data.py e treina um YOLOv8-cls.

O dataset e bem desbalanceado (neutrofilo ~10k, basofilo ~260). Antes de
treinar, equilibramos o SPLIT DE TREINO duplicando os arquivos das classes
raras (a augmentation do YOLO -- flip, HSV, rotacao -- da variedade a cada
copia). O val fica intocado, para a metrica ser honesta.

Saida: models/wbc_classifier.pt (usado por detection_core.py).

Uso: python scripts/13_train_wbc_classifier.py
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_ROOT = PROJECT_ROOT / "data" / "wbc_cls"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

PRETRAINED = "yolov8s-cls.pt"
EPOCHS = 60
IMG_SIZE = 128
BATCH = 64
PATIENCE = 15
RUN_NAME = "wbc_classifier"
TARGET_PER_CLASS = 2500   # alvo de balanceamento do treino (duplicando os raros)


def balance_train():
    train = DATA_ROOT / "train"
    counts = {d.name: sorted(d.glob("*.jpg")) for d in train.iterdir() if d.is_dir()}
    print("Treino antes do balanceamento:", {k: len(v) for k, v in counts.items()})
    for cls, files in counts.items():
        n = len(files)
        if n == 0 or n >= TARGET_PER_CLASS:
            continue
        need = TARGET_PER_CLASS - n
        for i in range(need):
            src = files[i % n]
            dst = src.with_name(f"{src.stem}__dup{i}.jpg")
            if not dst.exists():
                shutil.copyfile(src, dst)
    after = {d.name: len(list(d.glob("*.jpg"))) for d in train.iterdir() if d.is_dir()}
    print("Treino apos balanceamento:", after)


def clean_dups():
    for d in (DATA_ROOT / "train").iterdir():
        if d.is_dir():
            for f in d.glob("*__dup*.jpg"):
                f.unlink()


def main():
    if not (DATA_ROOT / "train").exists():
        raise SystemExit(f"{DATA_ROOT} nao existe. Rode 12_build_wbc_classifier_data.py antes.")

    clean_dups()          # comeca de um estado limpo (execucoes anteriores)
    balance_train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = YOLO(PRETRAINED)
    model.train(
        data=str(DATA_ROOT),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        patience=PATIENCE,
        workers=2,
        project=str(RUNS_DIR / "classify"),
        name=RUN_NAME,
        exist_ok=True,
        # augmentation um pouco mais forte (celulas variam de orientacao/cor)
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.3,
    )

    best = RUNS_DIR / "classify" / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"best.pt nao encontrado em {best}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "wbc_classifier.pt"
    shutil.copyfile(best, dest)
    print(f"\nClassificador salvo em: {dest}")

    clean_dups()          # nao deixa as copias no disco


if __name__ == "__main__":
    main()
