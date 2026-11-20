#!/usr/bin/env python3
"""Разбор попарных переключений (исправлено/сломано) на формат и арифметику.

Для каждой пары задач, где исходная модель была верна, а вариант нет (и наоборот),
определяет:
  - изменился ли статус извлечения ответа (answer_extracted);
  - изменилось ли извлечённое число;
  - есть ли обрыв генерации (truncated).
Категории:
  формат   — извлечение изменилось (число появилось/пропало), обрыва нет;
  обрыв    — извлечение пропало из-за truncation;
  арифметика — ответ извлечён в обеих моделях, но число другое.
Скрипт только читает готовые результаты прогонов.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")

PAIRS = [
    ("search250", "runs/baseline_search/results.jsonl", "search"),
    ("dev100", "runs/baseline_dev/results.jsonl", "dev"),
    ("test1319", "runs/baseline_test/results.jsonl", "test"),
]
VARIANTS = sys.argv[1:] or [
    "w1_final_token_amp", "w7_format_boost_mild", "w8_z_gate_mild_format",
    "w14_z93_final13", "w16_final_rows12_only", "w17_z93_only",
]


def load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[int(r["index"])] = r
    return out


def main() -> int:
    lines = []
    for name, base_path, split in PAIRS:
        base_file = ROOT / base_path
        for v in VARIANTS:
            vfile = ROOT / f"runs/variants/{v}/{split}/results.jsonl"
            if not vfile.exists():
                continue
            b, w = load(base_file), load(vfile)
            common = sorted(set(b) & set(w))
            fixed = [i for i in common if not b[i]["correct"] and w[i]["correct"]]
            broken = [i for i in common if b[i]["correct"] and not w[i]["correct"]]
            if not fixed and not broken:
                continue

            def classify(i: int, direction: str) -> str:
                rb, rw = b[i], w[i]
                if rb["answer_extracted"] != rw["answer_extracted"]:
                    if not rw["answer_extracted"] and rw["truncated"]:
                        return "обрыв"
                    return "формат"
                if (rb["predicted"] != rw["predicted"]):
                    return "арифметика"
                return "иное"

            for label, idxs in (("исправлено", fixed), ("сломано", broken)):
                cats: dict[str, int] = {}
                for i in idxs:
                    cats[classify(i, label)] = cats.get(classify(i, label), 0) + 1
                cat_str = ", ".join(f"{k}:{n}" for k, n in sorted(cats.items(), key=lambda x: -x[1]))
                lines.append(f"{name:<10} {v:<24} {label:<12} n={len(idxs):>3}  {cat_str}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
