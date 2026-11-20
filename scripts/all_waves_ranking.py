#!/usr/bin/env python3
"""Сводка по всем волнам 6–11: кто был ближе всего к четырём условиям отбора.

Никаких прогонов модели: читаются только сохранённые таблицы
runs/analysis/wave{6,8,9,10,11}_table.json (экран 350 = dev100 + search250)
и манифесты вариантов (база, след правки). Официальные 1319 задач не читаются.

Печать: C/350, обрывы, FINAL, Δверных и Δматематика против исходной модели и
против w8, наименьший Δмат, и список нарушенных условий gate.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "runs" / "variants"
TABLES = ROOT / "runs" / "analysis"

# пороги pick_final2.py: Mo >= +2, Mw8 >= +2, C >= C_w8 + 2, T <= T_w8
C_W8, T_W8 = 216, 18
MIN_MATH, MIN_TOTAL = 2, 2
TABLE_FILES = ["wave6_table.json", "wave8_table.json", "wave9_table.json",
               "wave10_table.json", "wave11_table.json"]


def manifest(name: str) -> dict | None:
    p = VARIANTS / name / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def base_of(m: dict) -> str:
    b = Path(m.get("base_model_dir") or "")
    n = b.name
    if n == "model":
        n = b.parent.name
    if n.startswith("qwen3.5"):
        return "orig"
    if n == "w8_z_gate_mild_format":
        return "w8"
    return n


def main() -> None:
    rows = []
    seen: set[str] = set()
    for tf in TABLE_FILES:
        p = TABLES / tf
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for name, got in d.items():
            if name in seen or "combined" not in got:
                continue
            seen.add(name)
            comb = got["combined"]
            st = comb["stats"]
            vo = comb["vs"].get("orig") or {}
            vw = comb["vs"].get("w8") or {}
            m = manifest(name) or {}
            tot = m.get("totals") or {}
            rows.append({
                "variant": name,
                "wave": tf.replace("_table.json", "").replace("wave", ""),
                "base": base_of(m) if m else "?",
                "C": st["correct"],
                "T": st["truncated"],
                "F": st["extracted"],
                "len": st["mean_out_tokens"],
                "Do": vo.get("delta_correct"),
                "Dw": vw.get("delta_correct"),
                "Mo": vo.get("delta_math"),
                "Mw8": vw.get("delta_math"),
                "p_math": min([x for x in (vo.get("mcnemar_p_math"),
                                           vw.get("mcnemar_p_math")) if x is not None]
                              or [None]),
                "foot": tot.get("values_actually_changed"),
                "tensors": tot.get("tensors_changed"),
            })

    def fails(r: dict) -> list[str]:
        out = []
        if (r["Mo"] or 0) < MIN_MATH:
            out.append(f"1) Δмат orig {r['Mo']} < +{MIN_MATH}")
        if (r["Mw8"] or 0) < MIN_MATH:
            out.append(f"2) Δмат w8 {r['Mw8']} < +{MIN_MATH}")
        if r["C"] < C_W8 + MIN_TOTAL:
            out.append(f"3) C {r['C']} < {C_W8 + MIN_TOTAL}")
        if r["T"] > T_W8:
            out.append(f"4) обрывы {r['T']} > {T_W8}")
        return out

    rows.sort(key=lambda r: (-(min(r["Mo"] or -99, r["Mw8"] or -99)), -r["C"],
                             (r["foot"] or 10**12)))
    print(f"Отборочный экран 350 задач (dev100 + search250). Референсы: "
          f"orig C=202 T=24 F=310; w8 C={C_W8} T={T_W8} F=332.")
    print(f"Условия gate: Δмат ≥ +{MIN_MATH} против обоих референсов, "
          f"C ≥ {C_W8 + MIN_TOTAL}, обрывы ≤ {T_W8}, p (Макнемар) ≥ 0,05.")
    print(f"Волн сводки: {', '.join(t.replace('_table.json', '') for t in TABLE_FILES)}; "
          f"вариантов с полным экраном 350: {len(rows)}\n")
    hdr = (f"{'вариант':<34}{'волна':>6}{'база':>6}{'C/350':>7}{'обр':>5}{'FINAL':>7}"
           f"{'Δмат o':>8}{'Δмат w8':>9}{'Δверн o':>9}{'Δверн w8':>9}"
           f"{'след':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<34}{r['wave']:>6}{r['base']:>6}{r['C']:>7}{r['T']:>5}"
              f"{r['F']:>7}{r['Mo']:>8}{r['Mw8']:>9}{r['Do']:>9}{r['Dw']:>9}"
              f"{(r['foot'] or 0):>12}")
    print("\nнарушенные условия (по возрастанию числа совпадений с gate):")
    for r in rows:
        f = fails(r)
        mark = "ПРОШЁЛ" if not f else "; ".join(f)
        print(f"  {r['variant']:<34} {mark}")


if __name__ == "__main__":
    main()
