#!/usr/bin/env python3
"""Контроль целостности удержанного набора из 400 задач и печать его спецификации.

Проверяет четыре вещи (любой провал — код 1, прогон не начинается):
  1) эталонные ответы в файл не записаны (`gold_answers_recorded` ложь);
  2) пересечения с уже использованными индексами `search`, `dev`, `smoke` нет;
  3) хеш канонической строки индексов совпадает с зафиксированным `indices_sha256`;
  4) `indices_spec` разворачивается ровно в тот же список индексов.

Вывод: на стандартный поток — строка спецификации (её передаёт прогонщик в `--indices`).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLD = ROOT / "runs/splits/holdout6.json"
SPLITS = ROOT / "runs/splits/splits.json"


def expand(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def fail(msg: str):
    print(f"ОШИБКА: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    d = json.loads(HOLD.read_text(encoding="utf-8"))
    if d.get("gold_answers_recorded") is not False:
        fail("в файле удержанного набора помечены записанными эталонные ответы")
    if d.get("overlap_with_used_train"):
        fail(f"набор пересекается с использованными индексами: {len(d['overlap_with_used_train'])} шт.")
    idx = [int(i) for i in d["holdout_indices"]]
    if len(idx) != 400 or len(set(idx)) != 400:
        fail(f"в наборе {len(idx)} индексов (уникальных {len(set(idx))}), ожидалось 400")
    canon = hashlib.sha256(",".join(str(i) for i in idx).encode("utf-8")).hexdigest()
    if canon != d.get("indices_sha256"):
        fail(f"хеш индексов {canon[:16]}… не совпал с зафиксированным {str(d.get('indices_sha256'))[:16]}…")
    spec = d.get("indices_spec") or ""
    if expand(spec) != sorted(idx):
        fail("спецификация indices_spec не совпала со списком holdout_indices")
    sp = json.loads(SPLITS.read_text(encoding="utf-8"))
    used = set()
    for key in ("search_indices", "dev_indices", "smoke_indices"):
        used |= {int(x) for x in sp.get(key, [])}
    inter = sorted(set(idx) & used)
    if inter:
        fail(f"набор пересекается с search/dev/smoke в {len(inter)} индексах")
    print(spec)


if __name__ == "__main__":
    main()
