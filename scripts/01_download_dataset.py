"""
Baixa e extrai o dataset publico BCCD (Blood Cell Count and Detection).
Fonte: https://github.com/Shenggan/BCCD_Dataset (licenca MIT)
"""
import io
import shutil
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "bccd" / "raw"
ZIP_URL = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"


def download_and_extract() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    extracted_root = RAW_DIR / "BCCD_Dataset-master"
    if (extracted_root / "BCCD" / "JPEGImages").exists():
        print(f"Dataset ja baixado em: {extracted_root}")
        return extracted_root

    print(f"Baixando dataset de {ZIP_URL} ...")
    response = requests.get(ZIP_URL, timeout=120)
    response.raise_for_status()

    print("Extraindo arquivos...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(RAW_DIR)

    if not extracted_root.exists():
        candidates = [p for p in RAW_DIR.iterdir() if p.is_dir()]
        raise RuntimeError(
            f"Pasta extraida esperada nao encontrada ({extracted_root}). "
            f"Pastas encontradas: {candidates}"
        )

    return extracted_root


def validate(extracted_root: Path) -> None:
    images_dir = extracted_root / "BCCD" / "JPEGImages"
    annotations_dir = extracted_root / "BCCD" / "Annotations"

    if not images_dir.exists() or not annotations_dir.exists():
        raise RuntimeError(
            "Estrutura do dataset inesperada. Esperava encontrar "
            f"'{images_dir}' e '{annotations_dir}'."
        )

    num_images = len(list(images_dir.glob("*.jpg")))
    num_annotations = len(list(annotations_dir.glob("*.xml")))
    print(f"OK: {num_images} imagens e {num_annotations} anotacoes encontradas.")

    if num_images == 0 or num_annotations == 0:
        raise RuntimeError("Dataset extraido, mas sem imagens/anotacoes.")


if __name__ == "__main__":
    root = download_and_extract()
    validate(root)
    print("Download concluido com sucesso.")
