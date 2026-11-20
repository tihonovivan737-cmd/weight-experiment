#!/usr/bin/env python3
"""Диагностический пересчёт: насколько потеря варианта объясняется форматом, а не арифметикой.

ВНИМАНИЕ. Официальные результаты — только из eval_gsm8k.py с единым фиксированным экстрактором
(`FINAL:` в начале строки). Этот скрипт не заменяет официальную метрику и не участвует в отборе:
он отвечает на один вопрос — «если бы строка финального ответа была оформлена чуть иначе, сколько
задач стало бы засчитано». Правила извлечения здесь одинаковы для baseline и всех вариантов.

Диагностическое правило: последнее вхождение FINAL/Final/final (в любой позиции строки, допускается
двоение вида «Final: FINAL: N») и следующее за ним число.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
LOOSE = re.compile(r"(?i)\bfinal(?:\s*:\s*\n?\s*final)?\s*:\s*\$?(-?[\d,]+(?:\.\d+)?)")


def tol_eq(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 1e-6 * abs(b))


def parse_num(s: str):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def loose_count(recs: list[dict]) -> tuple[int, int]:
    """Возвращает (засчитано диагностически, из них не засчитывались официально)."""
    c = 0
    extra = 0
    for r in recs:
        m = None
        for m in LOOSE.finditer(r["model_answer"]):
            pass
        val = parse_num(m.group(1)) if m else None
        exp = parse_num(str(r["expected"]))
        ok = val is not None and exp is not None and tol_eq(val, exp)
        if ok:
            c += 1
            if not r["correct"]:
                extra += 1
    return c, extra


def main() -> None:
    runs = [("baseline", ROOT / "runs/baseline_search/results.jsonl")]
    for v in sorted((ROOT / "runs/variants").iterdir()):
        p = v / "search/results.jsonl"
        if p.exists():
            runs.append((v.name, p))

    print("Диагностический пересчёт search250 (официальная метрика не меняется)")
    print(f"{'variant':24} {'офиц':>6} {'диагн':>6} {'диагн-офиц':>11} {'не извлеч. офиц':>16} {'из них с числом-ответом':>24}")
    print("-" * 92)
    for name, p in runs:
        recs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        off = sum(1 for r in recs if r["correct"])
        diag, extra = loose_count(recs)
        bad = [r for r in recs if not r["answer_extracted"]]
        # сколько из не извлечённых официально содержат любое финальное число вообще
        has_ans = 0
        for r in bad:
            m = None
            for m in LOOSE.finditer(r["model_answer"]):
                pass
            if m and parse_num(m.group(1)) is not None:
                has_ans += 1
        print(f"{name:24} {off:6} {diag:6} {diag - off:11} {len(bad):16} {has_ans:24}")
    print("\n'диагн-офиц' — сколько задач стало бы засчитано при более мягком формате (потеря формата,")
    print("а не арифметики). Отобранные варианты и итоговое сравнение test1319 считаются по официальной метрике.")


if __name__ == "__main__":
    main()
