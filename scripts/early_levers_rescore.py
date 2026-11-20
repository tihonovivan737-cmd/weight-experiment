#!/usr/bin/env python3
"""Повторный пересчёт ранних рычагов (волны 1–6) на фиксированной выборке
250 search + 100 dev = 350 задач. Никаких новых прогонов модели не выполняется:
используются только сохранённые results.jsonl. Официальные 1319 задач не читаются.

Вывод: runs/analysis/early_levers_rescore.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

V = W.V

# Ранние варианты, чьи прогоны search+dev сохранены.
NAMES = [
    "v1_Alog_decay_half",
    "v2_Alog_decay_double",
    "v3_dt_bias_shift_down",
    "v6_attn_sharpen",
    "v7_attn_smooth",
    "v4_beta_amplify",
    "v5_beta_soften",
    "v8_mlp_amp",
    "v9_mlp_damp",
    "v10_attn_branch_amp",
    "w4_branch_damp",
    "w2_bold_damp",
    "w3_final_amp_bold_damp",
    "w5_z_gate_damp",
    "w6_dt_bias_up",
    "w26_norm_amp093",
    "w24_fullgate_open085",
    "w28_fullgate_close115",
    "w33_fullgate_open_closed_heads",
    "w30_qknorm_sharp108",
    "w32_qknorm104_on_w8",
    "w31_z_dominant_restore106",
    "w34_restore_plus_qknorm104",
    "w35_z_dominant_damp078",
    "w25_z_headsel_damp088",
]


def combined(variant: str):
    p = V / variant
    for split in ("search", "dev"):
        if not (p / split / "results.jsonl").exists():
            return None
    rows = W.load(str(p / "search" / "results.jsonl")) | W.load(str(p / "dev" / "results.jsonl"))
    return rows if len(rows) == 350 else None


def base_of(variant: str) -> str:
    """Эффективная база по manifest.json (не по имени варианта и не по рецепту)."""
    mf = V / variant / "manifest.json"
    if not mf.exists():
        return "orig"
    base = json.loads(mf.read_text(encoding="utf-8")).get("base_model_dir") or ""
    return "w8" if "w8_z_gate_mild_format" in base else "orig"


# Варианты волны 1b, отобранные по экрану dev100 (§10): полный search-прогон для них
# не запускался, поэтому они пересчитаны только на 100 задачах dev.
DEV_ONLY = [
    "v1_Alog_decay_half",
    "v3_dt_bias_shift_down",
    "v4_beta_amplify",
    "v5_beta_soften",
    "v6_attn_sharpen",
    "v7_attn_smooth",
    "v8_mlp_amp",
    "v10_attn_branch_amp",
    "w4_branch_damp",
    "w2_bold_damp",
    "w3_final_amp_bold_damp",
    "w5_z_gate_damp",
    "w6_dt_bias_up",
]


def dev100(variant: str):
    """Экран dev100 (§10): у вариантов волны 1b сохранён только прогон dev."""
    p = V / variant / "dev" / "results.jsonl"
    if not p.exists():
        return None
    rows = W.load(str(p))
    return rows if len(rows) == 100 else None


def main() -> int:
    refs = {
        "orig": W.load(str(ROOT / "runs/baseline_search/results.jsonl"))
        | W.load(str(ROOT / "runs/baseline_dev/results.jsonl")),
        "w8": combined("w8_z_gate_mild_format"),
    }
    refs_dev = {
        "orig": W.load(str(ROOT / "runs/baseline_dev/results.jsonl")),
        "w8": dev100("w8_z_gate_mild_format"),
    }
    print("Повторный пересчёт ранних рычагов на 350 задачах (250 search + 100 dev).")
    print("Новых прогонов модели нет; официальные 1319 задач не использованы.")
    print()
    hdr = (
        f"{'вариант':32} {'база':5} {'C':>4} {'FINAL':>6} {'обр':>4} {'длина':>7} "
        f"{'Δбаза':>6} {'p':>8} {'Δмат':>5} | {'Δw8':>5} {'p(w8)':>8} {'Δмат(w8)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in NAMES:
        rows = combined(n)
        if rows is None:
            note = "только экран dev100 — см. вторую таблицу" if n in DEV_ONLY else "нет сохранённых результатов"
            print(f"{n:32} {note}")
            continue
        bname = base_of(n)
        s = W.stats(rows)
        c = W.cmp(rows, refs[bname], bname)
        cw8 = W.cmp(rows, refs["w8"], "w8")
        print(
            f"{n:32} {bname:5} {s['correct']:4} {s['correct_with_final']:6} {s['truncated']:4} "
            f"{s['mean_out_tokens']:7.1f} {c['delta_correct']:+6} {c['mcnemar_p']:8.5f} "
            f"{c['delta_math']:+5} | {cw8['delta_correct']:+5} {cw8['mcnemar_p']:8.5f} "
            f"{cw8['delta_math']:+8}"
        )
    print("-" * len(hdr))
    for k in ("w8", "orig"):
        s = W.stats(refs[k])
        print(
            f"{('w8 (контроль)' if k == 'w8' else 'orig (исходная)'):32} {k:5} {s['correct']:4} "
            f"{s['correct_with_final']:6} {s['truncated']:4} {s['mean_out_tokens']:7.1f}"
        )

    print()
    print("Экран dev100 (100 задач dev, §10) для вариантов волны 1b; полные 350 для них не запускались.")
    print(f"{'вариант':32} {'база':5} {'C':>4} {'FINAL':>6} {'обр':>4} {'длина':>7} "
          f"{'Δбаза':>6} {'p':>8} {'Δмат':>5} | {'Δw8':>5} {'p(w8)':>8} {'Δмат(w8)':>8}")
    print("-" * len(hdr))
    for n in DEV_ONLY:
        rows = dev100(n)
        if rows is None:
            print(f"{n:32} — нет сохранённого dev100")
            continue
        bname = base_of(n)
        s = W.stats(rows)
        c = W.cmp(rows, refs_dev[bname], bname)
        cw8 = W.cmp(rows, refs_dev["w8"], "w8")
        print(
            f"{n:32} {bname:5} {s['correct']:4} {s['correct_with_final']:6} {s['truncated']:4} "
            f"{s['mean_out_tokens']:7.1f} {c['delta_correct']:+6} {c['mcnemar_p']:8.5f} "
            f"{c['delta_math']:+5} | {cw8['delta_correct']:+5} {cw8['mcnemar_p']:8.5f} "
            f"{cw8['delta_math']:+8}"
        )
    for k in ("w8", "orig"):
        s = W.stats(refs_dev[k])
        print(
            f"{('w8 (контроль, dev100)' if k == 'w8' else 'orig (исходная, dev100)'):32} {k:5} "
            f"{s['correct']:4} {s['correct_with_final']:6} {s['truncated']:4} {s['mean_out_tokens']:7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
