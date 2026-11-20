#!/usr/bin/env python
"""Сводная таблица волны 13: 250 search + 100 dev.

Два варианта (§43, §44):
  w50_finalrow_rotation15_smallmargin           — пререгистрированный знак s = +0,015
  w51_finalrow_rotation15_smallmargin_negsign   — тот же механизм, знак исправлен измерением (§44.2)
Методика и пороги совпадают с scripts/wave6_table.py (раздел 24 журнала); здесь отличаются
список вариантов, выходные имена и референсы. Референс контрольной точки — w8.

Только чтение results.jsonl; генерации не запускаются.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

VARIANTS = [
    "w50_finalrow_rotation15_smallmargin",
    "w51_finalrow_rotation15_smallmargin_negsign",
]

REFS = dict(W.REFS)


def main():
    out = {}
    for split in ("search", "dev"):
        bases = {k: W.load(str(p).format(split=split)) for k, p in REFS.items()}
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
    for k, p in REFS.items():
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
    (ROOT / "runs/analysis/wave13_table.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["референсы на 350 задачах отбора: исходная 202 верно / FINAL 310 / обрывов 24; "
             "w8_z_gate_mild_format 216 верно / FINAL 332 / обрывов 18 (та же выборка)"]
    keys = list(REFS.keys())
    for split in ("dev", "search", "combined"):
        lines.append(f"\n=== {split} ===")
        head = f"{'вариант':<44}{'верно':>6}{'FINAL':>6}{'обр':>5}{'верно+FINAL':>12}{'сред. длина':>12}"
        for k in keys:
            head += f" {k + ' Δверн':>10} {'p':>7} {'Δмат':>6}"
        lines.append(head)
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                lines.append(f"{v:<44}нет завершённого прогона ({split})")
                continue
            s = d["stats"]
            row = (f"{v:<44}{s['correct']:>6}{s['extracted']:>6}{s['truncated']:>5}"
                   f"{s['correct_with_final']:>12}{s['mean_out_tokens']:>12.1f}")
            for k in keys:
                c = d["vs"][k]
                row += f" {c['delta_correct']:>10} {c['mcnemar_p']:>7} {c['delta_math']:>6}"
            lines.append(row)
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            for k in keys:
                c = d["vs"][k]
                lines.append(f"    {v:<44} vs {k:<5} только-математика "
                             f"(FINAL есть в обоих, n={c['math_subset_n']}): "
                             f"{c['math_subset_variant']} vs {c['math_subset_base']} "
                             f"Δ{c['delta_math']:+d} (починил {c['math_fixed']}, "
                             f"сломал {c['math_broken']}, p={c['mcnemar_p_math']})")
        lines.append("перевороты (формат/математика):")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            for k in keys:
                c = d["vs"][k]
                lines.append(f"  {v:<44} vs {k:<5} фикс {c['fixed']:>3}"
                             f" (форм {c['fixed_format']:>3} / мат {c['fixed_math']:>3})"
                             f"  слом {c['broken']:>3}"
                             f" (форм {c['broken_format']:>3} / мат {c['broken_math']:>3})")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/wave13_table.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
