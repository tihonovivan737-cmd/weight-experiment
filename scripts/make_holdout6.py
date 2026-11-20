"""Фиксация НОВОГО независимого train-пула (волна 6) ДО выбора победителя.

Правила, которые здесь закрепляются:
  * пул строится только из строк train, которые НЕ входили в search(250) / dev(100) / smoke(24);
  * сохраняется список индексов и контрольные суммы ВОПРОСОВ (первые 16 hex sha256);
  * правильные ответы (gold) сюда НЕ записываются и не просматриваются;
  * пул нельзя использовать для отбора вариантов: он нужен только для финальной
    проверки заранее зарегистрированного победителя, оригинала и w8.

Детерминированно: numpy PCG64(seed=20261111). Индексы сохраняются в
runs/splits/holdout6.json вместе с indices_sha256 (sha256 от канонической
строки индексов), чтобы набор нельзя было подменить.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/mpiscil/weight-experiment")
TRAIN = ROOT / "datasets/gsm8k/main/train-00000-of-00001.parquet"
SPLITS = ROOT / "runs/splits/splits.json"
OUT = ROOT / "runs/splits/holdout6.json"

SEED = 20261111
N_HOLDOUT = 400


def qhash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def spec_of(indices: list[int]) -> str:
    """Сжатое представление списка индексов для --split custom --indices."""
    parts = []
    i = 0
    n = len(indices)
    while i < n:
        j = i
        while j + 1 < n and indices[j + 1] == indices[j] + 1:
            j += 1
        parts.append(str(indices[i]) if i == j else f"{indices[i]}-{indices[j]}")
        i = j + 1
    return ",".join(parts)


def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    used = set(splits["search_indices"]) | set(splits["dev_indices"]) | set(splits["smoke_indices"])

    df = pd.read_parquet(TRAIN)
    n = len(df)
    free = np.array(sorted(set(range(n)) - used), dtype=np.int64)

    rng = np.random.default_rng(SEED)
    picked = sorted(int(i) for i in rng.choice(free, size=N_HOLDOUT, replace=False))

    assert len(picked) == N_HOLDOUT
    assert not (set(picked) & used), "пересечение с уже использованными train-индексами"

    payload = {
        "name": "holdout6",
        "purpose": ("независимый контрольный train-пул волны 6; зафиксирован ДО отбора "
                    "победителя; ответы не просматривались; для отбора вариантов не используется"),
        "seed": SEED,
        "generator": (f"numpy.random.default_rng({SEED}).choice(sorted(set(range({n})) - used), "
                      f"{N_HOLDOUT}, replace=False)"),
        "train_rows": n,
        "used_train_indices_total": len(used),
        "free_train_rows": int(len(free)),
        "holdout_indices": picked,
        "indices_sha256": hashlib.sha256(",".join(str(i) for i in picked).encode("utf-8")).hexdigest(),
        "indices_spec": spec_of(picked),
        "counts": {"holdout": len(picked)},
        "overlap_with_used_train": sorted(set(picked) & used),
        "question_hashes": {str(i): qhash(str(df["question"][i])) for i in picked},
        "gold_answers_recorded": False,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {OUT}")
    print(f"holdout={len(picked)} из {len(free)} свободных train-строк "
          f"(всего train {n}, использовано ранее {len(used)})")
    print("indices_sha256:", payload["indices_sha256"])
    print("пересечение с search/dev/smoke:", payload["overlap_with_used_train"])
    print("holdout[:12]:", picked[:12])
    print("spec (для --split custom --indices):")
    print(payload["indices_spec"])


if __name__ == "__main__":
    main()
