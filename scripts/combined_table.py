#!/usr/bin/env python3
"""Компактная таблица суммарной метрики (dev100 + search250 = 350 train-задач)
по всем runs/variants/*/combined.json. Сортировка: Δ по 350, затем p Макнемара,
затем объём правки весов (меньше — лучше) из manifest.json.

Только чтение готовых отчётов; новых прогонов не делает.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
VARIANTS = ROOT / "runs/variants"

rows = []
missing = []
for d in sorted(VARIANTS.iterdir()):
    if not d.is_dir():
        continue
    cj = d / "combined.json"
    if not cj.exists():
        missing.append(d.name)
        continue
    rep = json.loads(cj.read_text(encoding="utf-8"))
    per = rep.get("per_split", {})
    comb = rep.get("combined", {})
    if not comb:
        missing.append(d.name + " (нет combined)")
        continue
    mf = d / "manifest.json"
    vals = None
    if mf.exists():
        vals = json.loads(mf.read_text(encoding="utf-8")).get("totals", {}).get("values_actually_changed")

    def cell(sp):
        s = per.get(sp)
        if not s:
            return "—"
        vs = s.get("variant_stats", {})
        return f"{vs.get('correct')}(Δ{s.get('delta_correct')},p{s.get('mcnemar_p'):.3f})"

    cvs = comb.get("variant_stats", {})
    rows.append({
        "variant": d.name,
        "dev": cell("dev"),
        "search": cell("search"),
        "comb_correct": cvs.get("correct"),
        "comb_acc": cvs.get("accuracy"),
        "comb_delta": comb.get("delta_correct"),
        "comb_fixed": len(comb.get("fixed", [])),
        "comb_broken": len(comb.get("broken", [])),
        "comb_p": comb.get("mcnemar_p"),
        "extr": f"{comb.get('base_stats', {}).get('extracted')}->{cvs.get('extracted')}",
        "trunc": f"{comb.get('base_stats', {}).get('truncated')}->{cvs.get('truncated')}",
        "tokens": f"{comb.get('base_stats', {}).get('mean_out_tokens')}->{cvs.get('mean_out_tokens')}",
        "values_changed": vals,
    })

rows.sort(key=lambda r: (-(r["comb_delta"] or 0), r["comb_p"] if r["comb_p"] is not None else 1.0,
                         r["values_changed"] if r["values_changed"] is not None else 10**12))

hdr = (f"{'вариант':<26}{'dev100':<26}{'search250':<28}{'350':>5}{'Δ350':>6}{'ф/с':>8}"
       f"{'p':>8}{'извл':>12}{'обрывы':>10}{'токенов':>16}{'значений':>12}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['variant']:<26}{r['dev']:<26}{r['search']:<28}{r['comb_correct']:>5}{r['comb_delta']:>6}"
          f"{str(r['comb_fixed']) + '/' + str(r['comb_broken']):>8}{r['comb_p']:>8.4f}{r['extr']:>12}"
          f"{r['trunc']:>10}{r['tokens']:>16}{str(r['values_changed']):>12}")

if missing:
    print("\nБез combined.json:", ", ".join(missing))

out = ROOT / "runs/analysis/combined_table.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nJSON: {out}")
