"""
Migracao unica da taxonomia: de 3 classes (RBC / WBC / Platelets) para 7,
separando o "WBC" generico nos 5 subtipos do diferencial de leucocitos.

    ANTES            ->  DEPOIS
    0 RBC                0 RBC
    1 WBC                (sem equivalente -- ver abaixo)
    2 Platelets          6 Platelets
                         1..5 = Neutrophil / Lymphocyte / Monocyte /
                                Eosinophil / Basophil  (vem do Raabin-WBC,
                                mesclado depois por 10_merge_raabin_wbc.py)

O que este script faz (rode UMA vez, depois que detection_core.CLASSES ja
tem as 7 classes):

1. Reescreve data/bccd/dataset.yaml com os 7 nomes/ids novos.
2. Percorre data/bccd/labels/{train,val}/*.txt e reescreve cada linha:
     classe 0 (RBC)       -> mantem id 0
     classe 1 (WBC)       -> REMOVE a linha
     classe 2 (Platelets) -> vira id 6
3. Apaga os *.cache do ultralytics (ficam invalidos com a taxonomia nova).
4. Imprime, por origem do arquivo (prefixo), quantas caixas de WBC generico
   foram descartadas.

Risco aceito e explicito (mesmo espirito do caveat em 06/08): depois desta
migracao, milhares de imagens de treino ficam com leucocitos VISIVEIS mas
SEM nenhuma caixa. YOLO trata area sem caixa como fundo -- isso pode
suprimir o recall das 5 classes novas nessas imagens. Mitigacao: o volume
grande do Raabin-WBC (Passo 3) deve dominar o sinal de leucocito. Se a
validacao mostrar recall ruim, o proximo passo seria excluir as imagens
mais densas em leucocito das fontes antigas em vez de so remover a caixa.

Uso: python scripts/09_migrate_wbc_taxonomy.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import CLASSES  # noqa: E402 -- fonte unica da taxonomia

BCCD_ROOT = PROJECT_ROOT / "data" / "bccd"
DATASET_YAML = BCCD_ROOT / "dataset.yaml"

# Mapa da taxonomia ANTIGA (id -> novo id, ou None para descartar).
OLD_TO_NEW = {
    0: CLASSES.index("RBC"),        # 0 -> 0
    1: None,                         # WBC generico: sem subtipo, descartado
    2: CLASSES.index("Platelets"),  # 2 -> 6
}

PREFIXES = ("txlpbc_", "pltcrop_", "allidb_", "raabin_")


def source_of(stem: str) -> str:
    for p in PREFIXES:
        if stem.startswith(p):
            return p.rstrip("_")
    return "bccd(original)"


def already_migrated() -> bool:
    if not DATASET_YAML.exists():
        return False
    config = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8")) or {}
    names = config.get("names", {})
    values = names.values() if isinstance(names, dict) else names
    return "Neutrophil" in set(values)


def migrate_label_file(path: Path) -> tuple[int, int]:
    """Reescreve um .txt para a taxonomia nova. Devolve
    (caixas_wbc_descartadas, caixas_mantidas)."""
    dropped = kept = 0
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id not in OLD_TO_NEW:
            raise SystemExit(
                f"{path}: classe {old_id} fora da taxonomia antiga (0/1/2). "
                "O arquivo ja parece migrado ou foi gerado com outra taxonomia -- "
                "abortando para nao corromper os rotulos."
            )
        new_id = OLD_TO_NEW[old_id]
        if new_id is None:
            dropped += 1
            continue
        out_lines.append(" ".join([str(new_id)] + parts[1:]))
        kept += 1
    path.write_text("\n".join(out_lines), encoding="utf-8")
    return dropped, kept


def rewrite_dataset_yaml() -> None:
    config = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8")) or {}
    config["names"] = {i: name for i, name in enumerate(CLASSES)}
    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"dataset.yaml reescrito com {len(CLASSES)} classes: {CLASSES}")


def clear_caches() -> None:
    removed = []
    for sub in ("labels", "images"):
        for cache in (BCCD_ROOT / sub).glob("*.cache"):
            cache.unlink()
            removed.append(cache.name)
    if removed:
        print(f"Caches invalidados e removidos: {removed}")


def main() -> None:
    if not DATASET_YAML.exists():
        raise SystemExit(f"Nao encontrei {DATASET_YAML}. Rode antes 02_prepare_dataset.py.")
    if already_migrated():
        raise SystemExit(
            "dataset.yaml ja contem a taxonomia nova (Neutrophil/...). "
            "A migracao ja foi feita -- nada a fazer."
        )
    if len(CLASSES) != 7 or "Neutrophil" not in CLASSES:
        raise SystemExit(
            f"detection_core.CLASSES={CLASSES} ainda nao esta na taxonomia de 7 classes. "
            "Atualize detection_core.py antes de migrar."
        )

    dropped_by_source: dict[str, int] = defaultdict(int)
    kept_by_source: dict[str, int] = defaultdict(int)
    files_by_source: dict[str, int] = defaultdict(int)

    for split in ("train", "val"):
        labels_dir = BCCD_ROOT / "labels" / split
        if not labels_dir.exists():
            continue
        for txt in sorted(labels_dir.glob("*.txt")):
            src = source_of(txt.stem)
            d, k = migrate_label_file(txt)
            dropped_by_source[src] += d
            kept_by_source[src] += k
            files_by_source[src] += 1

    rewrite_dataset_yaml()
    clear_caches()

    print("\nCaixas de WBC generico descartadas (por origem):")
    for src in sorted(files_by_source):
        print(
            f"  {src:<18} {files_by_source[src]:>5} arquivos | "
            f"WBC descartado: {dropped_by_source[src]:>6} | "
            f"caixas mantidas: {kept_by_source[src]:>7}"
        )
    total_dropped = sum(dropped_by_source.values())
    print(
        f"\nTotal de caixas de WBC removidas: {total_dropped}. "
        "Os ids 1..5 (subtipos de leucocito) ficam vazios ate rodar "
        "10_merge_raabin_wbc.py."
    )


if __name__ == "__main__":
    main()
