#!/usr/bin/env python
"""Сводная таблица волны 14: 250 search + 100 dev.

Три arms (§45.1), каждый с одним рычагом:
  w52_frozen10_dose003_negsign        — замороженные 10 строк, s = -0,030 (проверка линейности дозы)
  w53_frozen10_dose003_mirrorpos      — то же, s = +0,030 (зеркальный парный контроль)
  w54_vhead_late19_head1_amp105       — scale lam = 1,05 на строках 256..511 v_proj слоя 19

Методика совпадает с scripts/wave6_table.py (раздел 24 журнала) и scripts/wave13_table.py:
здесь отличаются список вариантов, выходные имена и референсы. Референсы — исходная модель
и контрольная точка w8, оцениваются раздельно.

Дополнительно печатается сравнение с пререгистрированными предсказаниями §45.3
(средний запас по позиции: arm A -0,226, arm B +1,048; arm C — замер без знака).

Только чтение results.jsonl; генерации не запускается.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

VARIANTS = [
    "w52_frozen10_dose003_negsign",
    "w53_frozen10_dose003_mirrorpos",
    "w54_vhead_late19_head1_amp105",
]

# Пререгистрированные предсказания §45.3, фиксируются до прогона.
PRED = {
    "w52_frozen10_dose003_negsign": ("средний запас -0,226 логита; ожидается выход за коридоры "
                                     "(формат ниже 336, обрывов не меньше 18)",
                                     "ухудшение"),
    "w53_frozen10_dose003_mirrorpos": ("средний запас +1,048 логита; ожидается формат около "
                                       "верхней границы 348 и обрывы 2-3",
                                       "улучшение"),
    "w54_vhead_late19_head1_amp105": ("знак из весов не выводится; регистрируется как замер, "
                                      "|Δматематика| <= 3 и |Δформат| <= 3 против w8",
                                      "замер"),
}

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
    (ROOT / "runs/analysis/wave14_table.json").write_text(
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

    lines.append("\n=== сопоставление с пререгистрированными предсказаниями §45.3 ===")
    for v in VARIANTS:
        pred, dirn = PRED[v]
        d = out.get(v, {}).get("combined")
        if not d:
            lines.append(f"{v}: предсказание — {pred}; фактически — нет завершённого прогона")
            continue
        s = d["stats"]
        cw8 = d["vs"]["w8"]
        fact = (f"фактически: формат {s['extracted']} (Δ{cw8['delta_extracted']:+d} против w8), "
                f"обрывы {s['truncated']} (Δ{cw8['delta_truncated']:+d}), "
                f"математика {s['correct']} (Δ{cw8['delta_correct']:+d} против w8, "
                f"Δ{cw8['delta_math']:+d} только-математика)")
        lines.append(f"{v}: предсказание ({dirn}) — {pred}; {fact}")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/wave14_table.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
