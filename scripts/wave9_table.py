#!/usr/bin/env python
"""Сводная таблица волны 9: 250 search + 100 dev, сравнение независимо с
исходной моделью (runs/baseline_*) и с контрольной точкой w8_z_gate_mild_format.
Методика и пороги совпадают с scripts/wave6_table.py (раздел 24 журнала);
здесь только список вариантов и имена выходных файлов отличаются."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

VARIANTS = ["w38_z_corner_early10008_late08742", "w39_z_late_release1075", "w40_z_early_extra094"]


def main():
    out = {}
    for split in ("search", "dev"):
        bases = {k: W.load(str(p).format(split=split)) for k, p in W.REFS.items()}
        for v in VARIANTS:
            path = W.V / v / split / "results.jsonl"
            if not path.exists():
                continue
            rows = W.load(path)
            if len(rows) != W.EXPECT[split]:
                continue  # прогон ещё не завершён
            out.setdefault(v, {})[split] = {"stats": W.stats(rows),
                                            "vs": {k: W.cmp(rows, bases[k], k) for k in bases}}
    refs_all = {}
    for k, p in W.REFS.items():
        merged = W.load(str(p).format(split="search"))
        merged.update(W.load(str(p).format(split="dev")))
        refs_all[k] = merged
    for v in VARIANTS:
        got = out.get(v, {})
        if "search" in got and "dev" in got:
            rows = W.load(W.V / v / "search" / "results.jsonl")
            rows.update(W.load(W.V / v / "dev" / "results.jsonl"))
            if len(rows) == 350:
                got["combined"] = {"stats": W.stats(rows),
                                   "vs": {k: W.cmp(rows, refs_all[k], k) for k in refs_all}}
    (ROOT / "runs/analysis/wave9_table.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["референсы на 350 задачах отбора: исходная 202 верно / FINAL 310 / обрывов 24; "
             "w8_z_gate_mild_format 216 верно / FINAL 332 / обрывов 18"]
    for split in ("search", "dev", "combined"):
        lines.append(f"\n=== {split} ===")
        lines.append(f"{'вариант':<26}{'верно':>6}{'FINAL':>6}{'обр':>5}{'верно+FINAL':>12} "
                     f"{'Δorig':>7} {'p':>7} {'Δмат':>6}  {'Δw8':>6} {'p':>7} {'Δмат':>6}")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            s = d["stats"]; o = d["vs"]["orig"]; w = d["vs"]["w8"]
            lines.append(f"{v:<26}{s['correct']:>6}{s['extracted']:>6}{s['truncated']:>5}"
                         f"{s['correct_with_final']:>12} {o['delta_correct']:>7} {o['mcnemar_p']:>7}"
                         f" {o['delta_math']:>6}  {w['delta_correct']:>6} {w['mcnemar_p']:>7} {w['delta_math']:>6}")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            for k in ("orig", "w8"):
                c = d["vs"][k]
                lines.append(f"    {v:<26} vs {k:<5} только-математика (FINAL есть в обоих, n={c['math_subset_n']}): "
                             f"{c['math_subset_variant']} vs {c['math_subset_base']} "
                             f"Δ{c['delta_math']:+d} (починил {c['math_fixed']}, сломал {c['math_broken']}, "
                             f"p={c['mcnemar_p_math']})")
        lines.append("перевороты (формат/математика):")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            for k in ("orig", "w8"):
                c = d["vs"][k]
                lines.append(f"  {v:<26} vs {k:<5} фикс {c['fixed']:>3} (форм {c['fixed_format']:>3}"
                             f" / мат {c['fixed_math']:>3})  слом {c['broken']:>3}"
                             f" (форм {c['broken_format']:>3} / мат {c['broken_math']:>3})")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/wave9_table.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
