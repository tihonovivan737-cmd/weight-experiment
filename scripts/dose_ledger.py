#!/usr/bin/env python3
"""Дозовая ведомость: фактические (измеренные по байтам) накопленные множители z-дозы.

Для каждого варианта читается `linear_attn.in_proj_z.weight`, `in_proj_a.weight`,
`in_proj_b.weight` и сравнивается с исходной моделью послойно и поголовно (головка =
128 строк для in_proj_z, 1 строка для in_proj_a/b). Средний множитель считается по
значениям, которые реально изменились (|b-a| > 0); если значение не изменилось, оно не
входит в среднее, поэтому «средний множитель слоя» показывает именно дозу, а не артефакт
округления. Никаких градиентов — только чтение safetensors.

Зачем: рецепт описывает замысел, манифест сборки — только Δ относительно базы сборки
(которая сама уже содержит дозу w8). Истинная накопленная доза относительно ИСХОДНОЙ
модели измеряется только здесь. Выход: runs/analysis/dose_ledger.txt
"""
import json
import numpy as np
import torch
from safetensors import safe_open

ORIG = "models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors"
EARLY = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13]
LATE = [14, 16, 17, 18, 20, 21, 22]
VARIANTS = [
    "w8_z_gate_mild_format",
    "w25_z_headsel_damp088",
    "w29_z_total087",
    "w35_z_dominant_damp078",
    "w36_z_late_extra094",
    "w37_z_early_release1075",
    "w38_z_corner_early10008_late08742",
    "w39_z_late_release1075",
    "w40_z_early_extra094",
]

def blocks(name, layer):
    return f"model.language_model.layers.{layer}.{name}.weight"

def ratios(a, b):
    """поэлементный множитель b/a (a из исходной модели)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(a) > 1e-12, b / a, np.nan)

def main():
    ro = safe_open(ORIG, "pt")
    lines = ["Дозовая ведомость z-семейства: измеренный множитель значения варианта к исходной модели",
             "EARLY = 11 ранних/средних LIN слоёв, LATE = 7 поздних LIN слоёв (состав дозы w8, §24.3)", ""]
    for v in VARIANTS:
        path = f"runs/variants/{v}/model/model.safetensors-00001-of-00001.safetensors"
        rv = safe_open(path, "pt")
        per_layer = {}
        heads_hot = {}
        ab = {}
        for l in range(24):
            if l in (3, 7, 11, 15, 19, 23):
                continue
            a = ro.get_tensor(blocks("linear_attn.in_proj_z", l)).to(torch.float64).numpy()
            b = rv.get_tensor(blocks("linear_attn.in_proj_z", l)).to(torch.float64).numpy()
            r = ratios(a, b)
            ch = np.abs(b - a) > 0
            per_layer[l] = float(np.nanmean(r[ch])) if ch.any() else 1.0
            # поголовно: 16 головок по 128 строк
            hot = []
            for h in range(16):
                rh = r[h * 128:(h + 1) * 128]
                chh = (np.abs(b - a)[h * 128:(h + 1) * 128] > 0)
                if chh.any():
                    mh = float(np.nanmean(rh[chh]))
                    if abs(mh - 1.0) > 1e-3:
                        hot.append((h, round(mh, 4)))
            if hot:
                heads_hot[l] = hot
            ra = np.concatenate([ro.get_tensor(blocks("linear_attn.in_proj_a", l)).to(torch.float64).numpy(),
                                 ro.get_tensor(blocks("linear_attn.in_proj_b", l)).to(torch.float64).numpy()])
            rb = np.concatenate([rv.get_tensor(blocks("linear_attn.in_proj_a", l)).to(torch.float64).numpy(),
                                 rv.get_tensor(blocks("linear_attn.in_proj_b", l)).to(torch.float64).numpy()])
            rch = np.abs(rb - ra) > 0
            ab[l] = float(np.nanmean(ratios(ra, rb)[rch])) if rch.any() else 1.0
        e = [per_layer[l] for l in EARLY]
        la = [per_layer[l] for l in LATE]
        lines.append(f"=== {v} ===")
        lines.append(f"  in_proj_z EARLY: среднее {np.mean(e):.4f}  мин {min(e):.4f}  макс {max(e):.4f}")
        lines.append(f"  in_proj_z LATE : среднее {np.mean(la):.4f}  мин {min(la):.4f}  макс {max(la):.4f}")
        lines.append("  in_proj_z по слоям: " + "  ".join(f"L{l}:{per_layer[l]:.4f}" for l in range(24) if l not in (3, 7, 11, 15, 19, 23)))
        lines.append("  in_proj_a/b по слоям: " + "  ".join(f"L{l}:{ab[l]:.4f}" for l in range(24) if l not in (3, 7, 11, 15, 19, 23)))
        if heads_hot:
            lines.append("  головки с изменённой дозой: " + "; ".join(
                f"L{l}:" + ",".join(f"h{h}={m}" for h, m in hs) for l, hs in sorted(heads_hot.items())))
        else:
            lines.append("  головки с изменённой дозой: все 16 головок каждого LIN слоя получают одинаковый множитель")
        lines.append("")
    out = "runs/analysis/dose_ledger.txt"
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nСохранено: {out}")

if __name__ == "__main__":
    main()
