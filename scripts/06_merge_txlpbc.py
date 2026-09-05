"""
Baixa o dataset TXL-PBC (https://github.com/lugan113/TXL-PBC_Dataset) -- uma
versao curada e maior, que integra BCCD + 3 outras fontes publicas (1260
imagens, 18143 caixas: ~16k RBC, ~1.3k WBC, ~0.5k Platelets) -- e mescla no
split de treino do projeto, remapeando os IDs de classe (que vem em ordem
diferente da nossa).

Uso: python scripts/06_merge_txlpbc.py
"""
import io
import shutil
import sys
import zipfile
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES as OUR_CLASSES  # noqa: E402 -- fonte unica da taxonomia

RAW_DIR = PROJECT_ROOT / "data" / "txl_pbc" / "raw"
ZIP_URL = "https://github.com/lugan113/TXL-PBC_Dataset/archive/refs/heads/master.zip"

TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "bccd" / "images" / "train"
TRAIN_LABELS_DIR = PROJECT_ROOT / "data" / "bccd" / "labels" / "train"

FILE_PREFIX = "txlpbc_"


def download_and_extract() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extracted_root = RAW_DIR / "TXL-PBC_Dataset-master" / "TXL-PBC"

    if extracted_root.exists():
        print(f"TXL-PBC ja baixado em: {extracted_root}")
        return extracted_root

    print(f"Baixando TXL-PBC de {ZIP_URL} ...")
    response = requests.get(ZIP_URL, timeout=180)
    response.raise_for_status()

    print("Extraindo arquivos...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(RAW_DIR)

    if not extracted_root.exists():
        raise RuntimeError(f"Pasta esperada nao encontrada: {extracted_root}")

    return extracted_root


def build_class_remap(txlpbc_root: Path) -> dict:
    data_yaml = txlpbc_root / "data.yaml"
    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    txl_classes = config["names"]
    remap = {}
    for txl_id, name in enumerate(txl_classes):
        # "WBC" generico do TXL-PBC nao tem subtipo anotado -> descartado
        # (None), igual ao que a migracao 09 faz com os labels antigos.
        remap[txl_id] = OUR_CLASSES.index(name) if name in OUR_CLASSES else None

    print(f"Classes TXL-PBC: {txl_classes} -> remapeadas para {OUR_CLASSES}")
    dropped = [txl_classes[i] for i, v in remap.items() if v is None]
    if dropped:
        print(f"  Classes sem equivalente na taxonomia atual (caixas descartadas): {dropped}")
    return remap


def remap_label_file(src: Path, dest: Path, remap: dict) -> int:
    """Reescreve o .txt com os ids remapeados. Linhas cuja classe nao tem
    equivalente (remap -> None) sao descartadas. Devolve quantas caixas
    foram descartadas."""
    lines_out = []
    dropped = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_class = int(parts[0])
        new_class = remap[old_class]
        if new_class is None:
            dropped += 1
            continue
        lines_out.append(" ".join([str(new_class)] + parts[1:]))
    dest.write_text("\n".join(lines_out), encoding="utf-8")
    return dropped


def merge_split(txlpbc_root: Path, split: str, remap: dict) -> tuple[int, int]:
    images_src = txlpbc_root / "images" / split
    labels_src = txlpbc_root / "labels" / split

    count = 0
    dropped_boxes = 0
    for img_path in images_src.iterdir():
        if not img_path.is_file():
            continue
        label_path = labels_src / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        dest_img = TRAIN_IMAGES_DIR / f"{FILE_PREFIX}{img_path.name}"
        dest_label = TRAIN_LABELS_DIR / f"{FILE_PREFIX}{img_path.stem}.txt"

        shutil.copyfile(img_path, dest_img)
        dropped_boxes += remap_label_file(label_path, dest_label, remap)
        count += 1

    return count, dropped_boxes


def main():
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    txlpbc_root = download_and_extract()
    remap = build_class_remap(txlpbc_root)

    # As 3 divisoes (train/val/test) do TXL-PBC entram todas no NOSSO split de
    # treino -- mantemos o split de validacao original do BCCD intacto, para
    # medir o progresso do modelo sempre na mesma referencia.
    total = 0
    total_dropped = 0
    for split in ("train", "val", "test"):
        n, dropped = merge_split(txlpbc_root, split, remap)
        print(f"  {split}: {n} imagens adicionadas ao treino")
        total += n
        total_dropped += dropped

    print(f"Total adicionado ao treino: {total} imagens")
    if total_dropped:
        print(f"Caixas descartadas (classes sem equivalente na taxonomia atual): {total_dropped}")


if __name__ == "__main__":
    main()
