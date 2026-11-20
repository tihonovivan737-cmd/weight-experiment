#!/usr/bin/env python3
"""Сводка финальной проверки на удержанном наборе из 400 задач.

Считает по трём прогонам (выбранный вариант, исходная модель, w8):
  * верно, извлечение FINAL, обрывы, средняя длина ответа;
  * попарно: дельта верных, исправлено/сломано, точный двусторонний критерий `McNemar`
    на всех 400 задачах и отдельно на пересечении задач, где FINAL есть у обеих моделей
    (это «чистое качество» без оформления);
  * долю прироста, объяснимую оформлением: сколько задач вариант оформил, а база нет,
    и сколько из них верны.

Набор использован один раз; повторных прогонов не делается. Скрипт только читает.
Вывод: runs/analysis/holdout6_summary.txt
"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/holdout6"
LABELS = ["w50_finalrow_rotation15_smallmargin", "orig", "w8"]
PRETTY = {"w50_finalrow_rotation15_smallmargin": "выбранный вариант w50",
          "orig": "исходная модель", "w8": "w8 (контрольная точка)"}
PAIRS = [("orig", "w50_finalrow_rotation15_smallmargin"),
         ("w8", "w50_finalrow_rotation15_smallmargin"),
         ("orig", "w8")]


def load(label):
    p = OUT / label / "results.jsonl"
    if not p.exists():
        return None
    rows = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["index"]] = r
    return rows


def fin(r):
    return (r.get("n_final_lines") or 0) >= 1


def mcnemar(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)


def overall(rows):
    n = len(rows)
    vals = list(rows.values())
    return dict(n=n,
                correct=sum(1 for r in vals if r.get("correct")),
                extracted=sum(1 for r in vals if fin(r)),
                truncated=sum(1 for r in vals if r.get("truncated")),
                mean_tokens=sum(r.get("output_tokens") or 0 for r in vals) / max(n, 1))


def main():
    data = {lab: load(lab) for lab in LABELS}
    missing = [lab for lab, d in data.items() if not d]
    lines = ["Финальная проверка на удержанном наборе (400 задач train-пула, использован один раз).",
             "Набор: runs/splits/holdout6.json, хеш индексов 261affefc6c27b51…", ""]
    if missing:
        lines.append(f"ПОКА НЕТ ПРОГОНОВ: {', '.join(missing)} — сводка неполная.")
        lines.append("")
    lines.append(f"{'модель':24s} {'задач':>6} {'верно':>7} {'точность':>9} {'с FINAL':>9} {'обрывов':>8} {'среднее токенов':>16}")
    for lab in LABELS:
        d = data[lab]
        if not d:
            lines.append(f"{PRETTY[lab]:24s} — прогона ещё нет")
            continue
        o = overall(d)
        lines.append(f"{PRETTY[lab]:24s} {o['n']:>6} {o['correct']:>7} {100*o['correct']/o['n']:>8.2f}% "
                     f"{o['extracted']:>7} ({100*o['extracted']/o['n']:.1f}%) {o['truncated']:>4} {o['mean_tokens']:>11.1f}")
    lines.append("")
    for bk, vk in PAIRS:
        base, var = data[bk], data[vk]
        if not base or not var:
            lines.append(f"пара {PRETTY[bk]} -> {PRETTY[vk]}: данных нет")
            continue
        common = sorted(set(base) & set(var))
        c = sum(1 for i in common if var[i]["correct"])
        b = sum(1 for i in common if base[i]["correct"])
        gain = sum(1 for i in common if var[i]["correct"] and not base[i]["correct"])
        loss = sum(1 for i in common if base[i]["correct"] and not var[i]["correct"])
        inter = [i for i in common if fin(base[i]) and fin(var[i])]
        qb = sum(1 for i in inter if base[i]["correct"])
        qv = sum(1 for i in inter if var[i]["correct"])
        qg = sum(1 for i in inter if var[i]["correct"] and not base[i]["correct"])
        ql = sum(1 for i in inter if base[i]["correct"] and not var[i]["correct"])
        fmt_var = [i for i in common if fin(var[i]) and not fin(base[i])]
        fmt_base = [i for i in common if fin(base[i]) and not fin(var[i])]
        fv_ok = sum(1 for i in fmt_var if var[i]["correct"])
        fb_ok = sum(1 for i in fmt_base if base[i]["correct"])
        lines.append(f"пара {PRETTY[bk]} -> {PRETTY[vk]} (общих задач {len(common)})")
        lines.append(f"  верно: {b} -> {c} (Δ {c-b:+d}); исправлено {gain}, сломано {loss}, "
                     f"p = {mcnemar(gain, loss):.4f}")
        lines.append(f"  оформление: с FINAL {sum(1 for i in common if fin(base[i]))} -> "
                     f"{sum(1 for i in common if fin(var[i]))}; оформил только вариант {len(fmt_var)} "
                     f"(верных {fv_ok}), оформила только база {len(fmt_base)} (верных {fb_ok})")
        lines.append(f"  качество на пересечении FINAL ({len(inter)} задач): {qb} ({100*qb/len(inter):.1f}%) -> "
                     f"{qv} ({100*qv/len(inter):.1f}%), Δ {qv-qb:+d}; переходы {qg}/{ql}, "
                     f"p = {mcnemar(qg, ql):.4f}")
        lines.append("")
    text = "\n".join(lines)
    (ROOT / "runs/analysis/holdout6_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
