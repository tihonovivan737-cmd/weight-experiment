#!/usr/bin/env python
"""Сводная таблица волны 10: 250 search + 100 dev, сравнение независимо с
исходной моделью (runs/baseline_*), контрольной точкой w8_z_gate_mild_format и
основой сборки w36_z_late_extra094.
Методика и пороги совпадают с scripts/wave6_table.py (раздел 24 журнала); здесь
только список вариантов, дополнительные референсы и имена выходных файлов отличаются.
Правило выбора (scripts/pick_final2.py) использует только референсы orig и w8 —
дополнительный референс w36 приведён для отчёта и не влияет на правило."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

VARIANTS = ["w41_format_embed132", "w42_format_embed144",
            "w43_format_embed144_latedose", "w44_mlp_branch_late105"]
EXTRA_REFS = {"w36": W.V / "w36_z_late_extra094" / "{split}" / "results.jsonl"}


def main():
    refs = dict(W.REFS)
    refs.update(EXTRA_REFS)
    out = {}
    for split in ("search", "dev"):
        bases = {k: W.load(str(p).format(split=split)) for k, p in refs.items()}
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
    for k, p in refs.items():
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
    (ROOT / "runs/analysis/wave10_table.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["референсы на 350 задачах отбора: исходная 202 верно / FINAL 310 / обрывов 24; "
             "w8_z_gate_mild_format 216 верно / FINAL 332 / обрывов 18; "
             "w36_z_late_extra094 219 верно / FINAL 329 / обрывов 21"]
    keys = list(refs.keys())
    for split in ("search", "dev", "combined"):
        lines.append(f"\n=== {split} ===")
        head = f"{'вариант':<28}{'верно':>6}{'FINAL':>6}{'обр':>5}{'верно+FINAL':>12}{'сред. длина':>12}"
        for k in keys:
            head += f" {k + ' Δверн':>10} {'p':>7} {'Δмат':>6}"
        lines.append(head)
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            s = d["stats"]
            row = (f"{v:<28}{s['correct']:>6}{s['extracted']:>6}{s['truncated']:>5}"
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
                lines.append(f"    {v:<28} vs {k:<5} только-математика "
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
                lines.append(f"  {v:<28} vs {k:<5} фикс {c['fixed']:>3}"
                             f" (форм {c['fixed_format']:>3} / мат {c['fixed_math']:>3})"
                             f"  слом {c['broken']:>3}"
                             f" (форм {c['broken_format']:>3} / мат {c['broken_math']:>3})")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/wave10_table.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
