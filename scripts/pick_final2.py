#!/usr/bin/env python3
"""
ПРЕДЗАРЕГИСТРИРОВАННОЕ правило выбора победителя серии «волна 6» (журнал §23).
Правило зафиксировано ДО того, как увидены итоговые числа волны 6 (на момент фиксации был
известен только полный прогон w25 на search; пороги вычислены только по референсам).

Вход: runs/analysis/wave6_table.json (scripts/wave6_table.py): полные прогоны 250 search +
100 dev каждого варианта, сравнённые и с исходной моделью, и с контрольной точкой w8.

Метрики:
  C  = верно всего (по 350 отоброчным задачам);
  E  = FINAL извлечён;  T = обрывов;
  M_реф = «математическая» дельта: верно среди задач, где FINAL извлекаем и в варианте,
          и в референсе (форматный компонент исключён); считается отдельно vs исходной (Mo)
          и vs w8 (Mw8).

Обязательные условия (все четыре):
  1) Mo >= +2 и Mw8 >= +2     — математика улучшилась против ОБОИХ референсов;
  2) C >= C_w8 + 2            — итоговая точность не ниже контрольной точки;
  3) T <= T_w8                — обрывов не больше, чем у контрольной точки;
  4) ни на одном наборе против любого референса нет значимого ухудшения
     (delta_correct < 0 при mcnemar p < alpha = 0,05).
Ранжирование прошедших: наибольший min(Mo, Mw8); затем наибольший C; затем наименьший
footprint (values_actually_changed из manifest.json). Если прошедших нет — победителем
остаётся контрольная точка w8_z_gate_mild_format.

Победитель фиксируется этим скриптом (runs/selection_wave6.txt, runs/GATE_FINAL2) ДО любых
прогонов удержанного пула runs/splits/holdout6.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
VAR = ROOT / "runs/variants"
SPLITS = ("search", "dev")
REFS = ("orig", "w8")


def ref_totals():
    out = {}
    for k in REFS:
        paths = {"orig": [ROOT / f"runs/baseline_{s}/results.jsonl" for s in SPLITS],
                 "w8": [VAR / "w8_z_gate_mild_format" / s / "results.jsonl" for s in SPLITS]}[k]
        C = E = T = N = 0
        for p in paths:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                N += 1
                C += 1 if r.get("correct") else 0
                E += 1 if r.get("answer_extracted") else 0
                T += 1 if r.get("truncated") else 0
        out[k] = {"N": N, "C": C, "E": E, "T": T}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="runs/analysis/wave6_table.json")
    ap.add_argument("--out", default="runs/selection_wave6.txt")
    ap.add_argument("--gate", default="runs/GATE_FINAL2")
    ap.add_argument("--min-math-gain", type=int, default=2)
    ap.add_argument("--min-total-gain", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    table = json.loads((ROOT / a.table).read_text(encoding="utf-8"))
    ref = ref_totals()
    print("Референсы (350 отборочных задач):")
    for k in REFS:
        r = ref[k]
        print(f"  {k:<6} верно {r['C']}/{r['N']}  FINAL {r['E']}  обрывов {r['T']}")
    print(f"\nУсловия: Mo >= +{a.min_math_gain}, Mw8 >= +{a.min_math_gain}, "
          f"C >= {ref['w8']['C'] + a.min_total_gain}, T <= {ref['w8']['T']}, alpha = {a.alpha}\n")

    rows = []
    for v, d in sorted(table.items()):
        if not all(s in d for s in SPLITS):
            continue
        C = sum(d[s]["stats"]["correct"] for s in SPLITS)
        E = sum(d[s]["stats"]["extracted"] for s in SPLITS)
        T = sum(d[s]["stats"]["truncated"] for s in SPLITS)
        Mo = sum(d[s]["vs"]["orig"]["delta_math"] for s in SPLITS)
        Mw8 = sum(d[s]["vs"]["w8"]["delta_math"] for s in SPLITS)
        pmath = [d[s]["vs"][k]["mcnemar_p_math"] for s in SPLITS for k in REFS]
        regress = [f"{s}/{k}" for s in SPLITS for k in REFS
                   if d[s]["vs"][k]["delta_correct"] < 0 and d[s]["vs"][k]["mcnemar_p"] < a.alpha]
        man = {}
        mp = VAR / v / "manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text(encoding="utf-8")).get("totals", {})
        checks = {"1_math_both_refs": Mo >= a.min_math_gain and Mw8 >= a.min_math_gain,
                  "2_total_not_below_w8": C >= ref["w8"]["C"] + a.min_total_gain,
                  "3_truncation": T <= ref["w8"]["T"],
                  "4_no_significant_regression": not regress}
        rows.append({"variant": v, "C": C, "E": E, "T": T, "Mo": Mo, "Mw8": Mw8,
                     "p_math_min": min(pmath), "regressions": regress, "checks": checks,
                     "values_changed": man.get("values_actually_changed"),
                     "score": min(Mo, Mw8),
                     "ok": all(checks.values())})

    rows.sort(key=lambda r: (-r["score"], -r["C"],
                             r["values_changed"] if r["values_changed"] is not None else 10**12))
    print(f"{'variant':26}{'C/350':>7}{'FINAL':>7}{'обр':>5}{'Δмат orig':>10}{'Δмат w8':>9}"
          f"{'min p мат':>10}{'знач. ухудш.':>16}{'измен. значений':>16}  вердикт")
    for r in rows:
        print(f"{r['variant']:26}{r['C']:>7}{r['E']:>7}{r['T']:>5}{r['Mo']:>10}{r['Mw8']:>9}"
              f"{r['p_math_min']:>10}{','.join(r['regressions']) or '—':>16}"
              f"{str(r['values_changed']):>16}  "
              f"{'ПРОХОДИТ' if r['ok'] else 'нет: ' + ','.join(k for k, x in r['checks'].items() if not x)}")

    passing = [r for r in rows if r["ok"]]
    if passing:
        b = passing[0]
        line = (f"{b['variant']}   # C={b['C']}/350, FINAL {b['E']}, обрывов {b['T']}, "
                f"Δмат: {b['Mo']:+d} к исходной, {b['Mw8']:+d} к w8 "
                f"(min p_math={b['p_math_min']}), значений изменено {b['values_changed']}")
        print(f"\nПОБЕДИТЕЛЬ: {b['variant']}")
    else:
        line = ("w8_z_gate_mild_format   # победителем остаётся контрольная точка: "
                "ни один вариант волны 6 не выполнил все четыре условия")
        print("\nНи один вариант не прошёл правило — победителем остаётся контрольная точка.")

    (ROOT / a.out).write_text(line + "\n", encoding="utf-8")
    print(f"Список: {a.out}")
    if a.commit:
        (ROOT / a.gate).write_text("ok\n", encoding="utf-8")
        print(f"Gate: {a.gate}")


if __name__ == "__main__":
    main()
