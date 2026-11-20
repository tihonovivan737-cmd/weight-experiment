#!/usr/bin/env python3
"""Дополнение к факторному плану 3×3 (раздел 32 журнала): заполненная ячейка (1,0000 × 1,0000).

Ячейка (EARLY≈1,0000 ; LATE≈1,0000) уже была измерена ранее как w16_final_rows12_only
(доза in_proj_z 1,0000 во всех 18 линейных слоях, только строки embed_tokens FINAL ×1,2 —
та же правка строк, что входит в контрольную точку w8). Сравнивается variant-vs-variant,
поэтому общий слой правки строк сокращается и контраст чистый по дозе in_proj_z.

ВАЖНО: это пост-фактум дополнение. Замороженный критерий §31.6 (≥3 контраста одного знака
и |средний| ≥ 4 задачи) относится к трём контрастам на ось из предрегистрированных прогонов
волны 9; дополнительные контрасты выводятся отдельным блоком и в замороженный тест не входят.
Только чтение данных; ничего не перезаписывает.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

V = W.V
EXTRA = "w16_final_rows12_only"
lines = []

rows_extra = W.load(V / EXTRA / "search" / "results.jsonl")
rows_extra |= W.load(V / EXTRA / "dev" / "results.jsonl")
if len(rows_extra) != 350:
    raise SystemExit(f"у {EXTRA} {len(rows_extra)} задач вместо 350 — ячейку не использовать")

s = W.stats(rows_extra)
lines.append(f"Ячейка (1,0000 × 1,0000) = {EXTRA}: верных {s['correct']} / FINAL {s['extracted']} "
             f"/ обрывов {s['truncated']} / средняя длина {s['mean_out_tokens']} токена")
lines.append("")


def cell_contrast(a_name: str, b_name: str, title: str):
    a = W.load(V / a_name / "search" / "results.jsonl") | W.load(V / a_name / "dev" / "results.jsonl")
    b = W.load(V / b_name / "search" / "results.jsonl") | W.load(V / b_name / "dev" / "results.jsonl")
    if len(a) != 350 or len(b) != 350:
        lines.append(f"  {title:<46} — не измерено")
        return None
    c = W.cmp(a, b, a_name)
    lines.append(f"  {title:<46} Δверных {c['delta_correct']:+4d} (p={c['mcnemar_p']:.5f})  "
                 f"Δмат {c['delta_math']:+4d} (p_math={c['mcnemar_p_math']:.5f})  "
                 f"[{W.stats(a)['correct']}/{W.stats(a)['extracted']}/{W.stats(a)['truncated']} против "
                 f"{W.stats(b)['correct']}/{W.stats(b)['extracted']}/{W.stats(b)['truncated']}]")
    return c


lines.append("Дополнительные контрасты через ячейку w16 (1,0000 × 1,0000):")
late = cell_contrast("w37_z_early_release1075", EXTRA, "EARLY≈1,0008: w37 − w16 (углубление LATE)")
early = cell_contrast(EXTRA, "w39_z_late_release1075", "LATE≈1,0008: w16 − w39 (углубление EARLY)")

lines.append("")
lines.append("Оценка взаимодействия (2×2 на верхнем левом углу сетки, верные из 350):")
grid = {"w16": 209, "w37": 212, "w39": 207, "w8": 216}
try:
    for name in ("w16", "w37", "w39", "w8"):
        rows = W.load(V / name / "search" / "results.jsonl") | W.load(V / name / "dev" / "results.jsonl")
        grid[name] = W.stats(rows)["correct"]
except FileNotFoundError:
    pass
inter = (grid["w8"] + grid["w16"]) - (grid["w37"] + grid["w39"])
lines.append(f"  (w8 + w16) − (w37 + w39) = {grid['w8']} + {grid['w16']} − {grid['w37']} − {grid['w39']} = {inter:+d}")
lines.append("  отрицательное значение = дозы EARLY и LATE мешают друг другу (отрицательное взаимодействие);")
lines.append("  нулевое = эффекты складываются; положительное = дозы взаимно усиливают математику.")

text = "\n".join(lines)
(ROOT / "runs/analysis/factorial9_extra.txt").write_text(text + "\n", encoding="utf-8")
print(text)
