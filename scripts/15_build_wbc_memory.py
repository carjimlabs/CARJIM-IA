"""
Monta o banco de exemplos do kNN do classificador de subtipo de leucocito
(models/wbc_memory.pt), usado por detection_core.Detector.classify_crops.

A ideia e um "RAG" para imagem: cada recorte de leucocito vira um embedding
(a entrada da camada linear final do proprio wbc_classifier.pt), e na hora de
classificar uma celula nova buscam-se os KNN_K exemplos mais parecidos do
banco. Os votos deles sao misturados com a probabilidade do classificador
(KNN_BLEND em detection_core).

O ganho pratico: exemplos NOVOS entram no banco sem retreinar nada. Coloque
recortes corrigidos pelo professor em

    Exemplos de leucocitos/<Classe>/*.jpg

(<Classe> = Neutrofilo, Linfocito, Monocito, Eosinofilo, Basofilo -- ou o nome
em ingles) e rode este script de novo. Eles entram todos (sem limite por
classe), alem de ate MAX_PER_CLASS recortes do Raabin (data/wbc_cls/train/).

No fim mede a acuracia em data/wbc_cls/val/ para varios valores de KNN_BLEND.
Atencao: o split do 12 e por celula, entao celulas do mesmo esfregaco podem
estar em train e val -- o kNN tende a parecer melhor ali do que e em fotos
realmente novas.

O banco guarda o sha1 do wbc_classifier.pt: se o classificador for retreinado
(13), rode este script de novo (detection_core ignora banco desatualizado).

Uso: python scripts/15_build_wbc_memory.py
"""
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detection_core as dc  # noqa: E402

TRAIN_DIR = PROJECT_ROOT / "data" / "wbc_cls" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "wbc_cls" / "val"
TEACHER_DIR = PROJECT_ROOT / "Exemplos de leucocitos"
MAX_PER_CLASS = 1500
RANDOM_SEED = 42
BATCH = 256
BLENDS_TO_REPORT = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)

PT_TO_EN = {v.lower(): k for k, v in dc.CLASS_LABELS_PT.items()}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in dc.VALID_EXTENSIONS)


def class_folders(root: Path):
    """{nome EN da classe: pasta} aceitando nomes de pasta em EN ou PT-BR."""
    out = {}
    if not root.exists():
        return out
    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = PT_TO_EN.get(d.name.lower(), d.name)
        if name in dc.CLASSES:
            out[name] = d
    return out


def embed_paths(detector, paths):
    probs, embs = [], []
    for i in range(0, len(paths), BATCH):
        crops = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        p, e = detector.classify_probs(crops)
        probs.append(p.cpu())
        embs.append(e.cpu())
        print(f"  {min(i + BATCH, len(paths))}/{len(paths)}", end="\r")
    print()
    return torch.cat(probs), torch.cat(embs)


def main():
    detector, device = dc.load_model()
    if detector.cls is None:
        raise RuntimeError(f"{dc.CLASSIFIER_PATH} nao encontrado. Rode antes 13_train_wbc_classifier.py.")
    cls_index = {name: idx for idx, name in detector.cls.names.items()}
    rng = random.Random(RANDOM_SEED)

    paths, labels = [], []
    for name, folder in sorted(class_folders(TRAIN_DIR).items()):
        imgs = list_images(folder)
        rng.shuffle(imgs)
        imgs = imgs[:MAX_PER_CLASS]
        paths += imgs
        labels += [cls_index[name]] * len(imgs)
    n_raabin = len(paths)
    for name, folder in sorted(class_folders(TEACHER_DIR).items()):
        imgs = list_images(folder)
        paths += imgs
        labels += [cls_index[name]] * len(imgs)
    print(f"Banco: {n_raabin} recortes do Raabin + {len(paths) - n_raabin} do professor")
    print("  por classe:", {detector.cls.names[k]: v for k, v in sorted(Counter(labels).items())})

    _, emb = embed_paths(detector, paths)
    memory = {
        "emb": emb.half(),
        "labels": torch.tensor(labels, dtype=torch.int16),
        "classes": [detector.cls.names[i] for i in range(len(detector.cls.names))],
        "classifier_sha1": dc.classifier_sha1(),
    }
    dc.WBC_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory, dc.WBC_MEMORY_PATH)
    print(f"Salvo em {dc.WBC_MEMORY_PATH} ({emb.shape[0]} x {emb.shape[1]})")

    # --- avaliacao no val ---
    detector.memory_emb = memory["emb"].float().to(device)
    detector.memory_labels = memory["labels"].long().to(device)
    val_paths, val_labels = [], []
    for name, folder in sorted(class_folders(VAL_DIR).items()):
        imgs = list_images(folder)
        val_paths += imgs
        val_labels += [cls_index[name]] * len(imgs)
    if not val_paths:
        return
    print(f"Avaliando em {len(val_paths)} recortes de {VAL_DIR}")
    probs, emb = embed_paths(detector, val_paths)
    knn = detector.knn_probs(emb.to(device)).cpu()
    y = torch.tensor(val_labels)
    print(f"{'KNN_BLEND':>9}  {'acuracia':>8}  " + "  ".join(f"{detector.cls.names[i][:5]:>6}" for i in range(len(detector.cls.names))))
    for blend in BLENDS_TO_REPORT:
        pred = ((1 - blend) * probs + blend * knn).argmax(1)
        per_class = [(pred[y == i] == i).float().mean().item() for i in range(len(detector.cls.names))]
        print(f"{blend:>9.1f}  {(pred == y).float().mean().item():>8.4f}  "
              + "  ".join(f"{a:>6.3f}" for a in per_class))
    print(f"(KNN_BLEND atual em detection_core: {dc.KNN_BLEND})")


if __name__ == "__main__":
    main()
