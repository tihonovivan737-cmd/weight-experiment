#!/usr/bin/env python3
"""Доза → активация: что сдвинулось в слоях после волны 10 (форматная ветка).

Сравнение с базой по одним и тем же 63 задачам search (те же индексы и тот же промпт, что в
scripts/probe_layers.py). Для w41, w42, w44 база — проба w8; для w43 база — проба w36,
поскольку w43 построен поверх w36. Команда `gain = RMS ветки / RMS остаточного потока на входе
блока` (scripts/probe_layers.py:239).

Смысл проверки: правка строк embed_tokens не меняет относительную долю ветвей внутри блока,
она меняетagnitude самих эмбеддингов. Поэтому ожидается другой отклик, чем у оси in_proj_z:
не перераспределение внимания и MLP, а общий сдвиг масштаба остаточного потока (rms_attn_branch
и gain_* меняются согласованно во всех группах слоёв, включая необработанные).

Выход: runs/analysis/wave10_probe_report.txt
"""
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
EARLY = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13]
LATE = [14, 16, 17, 18, 20, 21, 22]
REFS = {
    "w8": ROOT / "runs/analysis/probe_w8_model.json",
    "w36": ROOT / "runs/analysis/probe6_w36_z_late_extra094.json",
}
VARIANTS = [
    ("w41_format_embed132", "w8"),
    ("w42_format_embed144", "w8"),
    ("w43_format_embed144_latedose", "w36"),
    ("w44_mlp_branch_late105", "w8"),
]
HEAD_THRESHOLD = 0.8

out = []


def group_pct(base_layers, lay, metric, group):
    """Σ по группе слоёв: база -> вариант и относительный сдвиг в процентах."""
    a = b = 0.0
    for l in group:
        a += base_layers[str(l)][metric]
        b += lay[str(l)][metric]
    return a, b, (b / a - 1) * 100


def head_count(lay):
    return sum(1 for l in EARLY + LATE for x in lay[str(l)]["z_gate_head"] if x > HEAD_THRESHOLD)


for variant, ref_key in VARIANTS:
    vpath = ROOT / f"runs/analysis/probe6_{variant}.json"
    rpath = REFS[ref_key]
    if not vpath.is_file():
        out.append(f"=== {variant}: пробы нет ({vpath.name}) — пропуск ===\n")
        continue
    if not rpath.is_file():
        out.append(f"=== {variant}: эталонной пробы {rpath.name} нет — пропуск ===\n")
        continue
    lay = json.loads(vpath.read_text(encoding="utf-8"))["layers"]
    base = json.loads(rpath.read_text(encoding="utf-8"))["layers"]
    out.append(f"=== {variant} против {ref_key} (одни и те же 63 задачи) ===")
    for label, group in (("EARLY (11 слоёв)", EARLY), ("LATE (7 слоёв)", LATE)):
        for metric in ("gain_attn", "mean_z_gate", "gain_mlp", "rms_attn_branch", "rms_mlp_branch"):
            if metric not in lay[str(EARLY[0])] or metric not in base[str(EARLY[0])]:
                continue
            a, b, pct = group_pct(base, lay, metric, group)
            out.append(f"  {label:16s} {metric:17s} {a:8.4f} -> {b:8.4f}  {pct:+6.2f} %")
    out.append("  послойно gain_attn (база -> вариант):")
    out.append("    " + "  ".join(
        f"L{l}:{lay[str(l)]['gain_attn'] / base[str(l)]['gain_attn'] - 1:+.1%}" for l in EARLY + LATE))
    out.append("  послойно gain_mlp (база -> вариант):")
    out.append("    " + "  ".join(
        f"L{l}:{lay[str(l)]['gain_mlp'] / base[str(l)]['gain_mlp'] - 1:+.1%}" for l in EARLY + LATE))
    out.append(f"  головки с |silu(z)|>{HEAD_THRESHOLD}: {head_count(lay)} "
               f"(у базы {head_count(base)})")
    out.append("")

text = "\n".join(out)
(ROOT / "runs/analysis/wave10_probe_report.txt").write_text(text, encoding="utf-8")
print(text)
