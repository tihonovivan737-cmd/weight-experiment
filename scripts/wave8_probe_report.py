#!/usr/bin/env python3
"""Доза → активация: что реально сдвинулось в слоях после волны 8.

Сравнение с w8 по одним и тем же 63 задачам. Группировка по группам слоёв
(ранние/средние 11 и поздние 7), потому что именно на них работали операции w36 и w37.
Команда `gain = RMS ветки / RMS остаточного потока на входе блока` (scripts/probe_layers.py:239).
"""
import json
from pathlib import Path

EARLY = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13]
LATE = [14, 16, 17, 18, 20, 21, 22]
VARIANTS = ["w36_z_late_extra094", "w37_z_early_release1075", "w35_z_dominant_damp078"]
REF = "runs/analysis/probe_w8_model.json"

out = []
ref = json.loads(Path(REF).read_text(encoding="utf-8"))["layers"]


def group_pct(lay, keys, group):
    """Σ по группе слоёв: вариант/эталон по величине, предельно чувствительной к дозе."""
    a = b = 0.0
    for l in group:
        a += ref[str(l)][keys[0]]
        b += lay[str(l)][keys[1]]
    return a, b, (b / a - 1) * 100


for v in VARIANTS:
    lay = json.loads(Path(f"runs/analysis/probe6_{v}.json").read_text(encoding="utf-8"))["layers"]
    out.append(f"=== {v} против w8 (одни и те же 63 задачи) ===")
    for label, group in (("EARLY (11 слоёв)", EARLY), ("LATE (7 слоёв)", LATE)):
        for metric in ("gain_attn", "mean_z_gate", "gain_mlp", "rms_attn_branch"):
            a, b, pct = group_pct(lay, (metric, metric), group)
            out.append(f"  {label:16s} {metric:17s} {a:8.4f} -> {b:8.4f}  {pct:+6.2f} %")
    out.append("  послойно gain_attn (w8 -> вариант):")
    row = []
    for l in EARLY + LATE:
        x, y = ref[str(l)]["gain_attn"], lay[str(l)]["gain_attn"]
        row.append(f"L{l}:{y / x - 1:+.1%}")
    out.append("    " + "  ".join(row))
    zx = [lay[str(l)]["z_gate_head"] for l in EARLY + LATE]
    out.append(f"  головки с |silu(z)|>0,8: {sum(1 for hh in zx for x in hh if x > 0.8)} "
               f"(у w8 было 29)")
    out.append("")

text = "\n".join(out)
Path("runs/analysis/wave8_probe_report.txt").write_text(text, encoding="utf-8")
print(text)
