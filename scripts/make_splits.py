"""Фиксация разрезов train: 250 примеров для поиска изменений + 100 для промежуточной оценки.

Детерминированно: numpy PCG64(seed). Индексы сохраняются в runs/splits/splits.json
вместе с контрольными суммами вопросов, чтобы нельзя было случайно подменить набор.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/mpiscil/weight-experiment")
TRAIN = ROOT / "datasets/gsm8k/main/train-00000-of-00001.parquet"
OUT_DIR = ROOT / "runs/splits"
SEED = 20261110
N_SEARCH = 250
N_DEV = 100
N_SMOKE = 24  # техническая проверка, не входит ни в один разрешённый набор


def qhash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def main():
    df = pd.read_parquet(TRAIN)
    n = len(df)
    rng = np.random.default_rng(SEED)
    picked = rng.choice(n, size=N_SEARCH + N_DEV + N_SMOKE, replace=False)
    search = sorted(int(i) for i in picked[:N_SEARCH])
    dev = sorted(int(i) for i in picked[N_SEARCH : N_SEARCH + N_DEV])
    smoke = sorted(int(i) for i in picked[N_SEARCH + N_DEV :])

    assert not (set(search) & set(dev)), "search и dev пересекаются"
    assert not (set(search) & set(smoke)), "search и smoke пересекаются"
    assert not (set(dev) & set(smoke)), "dev и smoke пересекаются"
    assert len(search) == N_SEARCH and len(dev) == N_DEV

    payload = {
        "seed": SEED,
        "generator": "numpy.random.default_rng(SEED).choice(7473, 374, replace=False)",
        "train_rows": n,
        "search_indices": search,
        "dev_indices": dev,
        "smoke_indices": smoke,
        "counts": {"search": len(search), "dev": len(dev), "smoke": len(smoke)},
        "overlap_search_dev": sorted(set(search) & set(dev)),
        "question_hashes": {
            "search": {str(i): qhash(str(df["question"][i])) for i in search},
            "dev": {str(i): qhash(str(df["question"][i])) for i in dev},
            "smoke": {str(i): qhash(str(df["question"][i])) for i in smoke},
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "splits.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {out}")
    print(f"search={len(search)} dev={len(dev)} smoke={len(smoke)}")
    print("пересечение search/dev:", payload["overlap_search_dev"])
    print("search[:12]:", search[:12])
    print("dev[:12]:", dev[:12])
    print("smoke[:12]:", smoke[:12])


if __name__ == "__main__":
    main()
