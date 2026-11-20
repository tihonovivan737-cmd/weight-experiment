#!/usr/bin/env python3
"""Дозовая ведомость волны 10 (форматная ветка): накопленные множители, измеренные по файлам.

Для каждого варианта относительно ИСХОДНОЙ модели:
  1) строки 95429 ('FINAL') и 64697 (' FINAL') embed_tokens — отношения L2-норм и средний
     поэлементный множитель по реально изменившимся значениям;
  2) post_attention_layernorm.weight в 7 LATE-слоях (рычаг w44);
  3) контрольная проверка: линейные in_proj_z EARLY/LATE и in_proj_a/b совпадают с базой
     сборки (значит волна 10 действительно не трогает дозу ворот);
  4) контрольная проверка: input_layernorm, linear_attn.norm.weight, A_log, dt_bias,
     conv1d.weight побайтово равны основе.

Никаких градиентов — только чтение safetensors. Выход: runs/analysis/dose_ledger_wave10.txt
"""
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path("/home/mpiscil/weight-experiment")
ORIG = str(ROOT / "models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors")
EARLY = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13]
LATE = [14, 16, 17, 18, 20, 21, 22]
ROWS = [95429, 64697]
ROW_TOKENS = {95429: "FINAL", 64697: " FINAL"}
VARIANTS = [
    ("w41_format_embed132", "w8_z_gate_mild_format"),
    ("w42_format_embed144", "w8_z_gate_mild_format"),
    ("w43_format_embed144_latedose", "w36_z_late_extra094"),
    ("w44_mlp_branch_late105", "w8_z_gate_mild_format"),
]
SHARD = "model.safetensors-00001-of-00001.safetensors"


def path_of(variant: str) -> str:
    if variant == "orig":
        return ORIG
    return str(ROOT / "runs/variants" / variant / "model" / SHARD)


def bl(name: str, layer: int) -> str:
    return f"model.language_model.layers.{layer}.{name}.weight"


def f64(t):
    return t.to(torch.float64).numpy()


def mean_ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """средний множитель b/a по значениям, которые реально изменились"""
    changed = np.abs(b - a) > 0
    n = int(changed.sum())
    if n == 0:
        return 1.0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(a) > 1e-12, b / a, np.nan)
    return float(np.nanmean(r[changed])), n


def main() -> None:
    out = [ "Дозовая ведомость волны 10: измеренные накопленные множители к ИСХОДНОЙ модели",
            "EARLY = 11 слоёв [0,1,2,4,5,6,8,9,10,12,13], LATE = 7 слоёв [14,16,17,18,20,21,22]", ""]

    with safe_open(ORIG, "pt") as ro:
        e_orig = ro.get_slice("model.language_model.embed_tokens.weight")
        orig_rows = {i: f64(e_orig[i]) for i in ROWS}
        z_orig = {l: f64(ro.get_tensor(bl("linear_attn.in_proj_z", l))) for l in EARLY + LATE}
        ln2_orig = {l: f64(ro.get_tensor(bl("post_attention_layernorm", l))) for l in EARLY + LATE}

    for variant, base in VARIANTS:
        with safe_open(path_of(variant), "pt") as rv:
            sl = rv.get_slice("model.language_model.embed_tokens.weight")
            lines = [f"=== {variant} (основа сборки: {base}) ==="]
            for i in ROWS:
                v = f64(sl[i])
                a = orig_rows[i]
                m, n = mean_ratio(a, v)
                lines.append(f"  embed_tokens строка {i:>6} {ROW_TOKENS[i]:>7}: "
                             f"норма {np.linalg.norm(a):.4f} -> {np.linalg.norm(v):.4f} "
                             f"(×{np.linalg.norm(v) / np.linalg.norm(a):.4f}); "
                             f"поэлементно ×{m:.4f} (изменено {n} из {v.size})")
            ln2 = {l: f64(rv.get_tensor(bl("post_attention_layernorm", l))) for l in EARLY + LATE}
            for group, layers in (("EARLY", EARLY), ("LATE", LATE)):
                ms = [mean_ratio(ln2_orig[l], ln2[l]) for l in layers]
                changed = sum(c for _, c in ms)
                lines.append(f"  post_attention_layernorm {group}: ×{np.mean([m for m, _ in ms]):.4f} "
                             f"(изменено значений {changed})")
            z = {l: f64(rv.get_tensor(bl("linear_attn.in_proj_z", l))) for l in EARLY + LATE}
            for group, layers in (("EARLY", EARLY), ("LATE", LATE)):
                ms = [mean_ratio(z_orig[l], z[l]) for l in layers]
                lines.append(f"  in_proj_z {group} против ИСХОДНОЙ: ×{np.mean([m for m, _ in ms]):.4f}")
            with safe_open(path_of(base), "pt") as rb:
                ident = True
                for l in EARLY + LATE:
                    for name in ("input_layernorm", "post_attention_layernorm",
                                 "linear_attn.norm", "linear_attn.conv1d"):
                        a = f64(rb.get_tensor(bl(name, l)))
                        b = f64(rv.get_tensor(bl(name, l)))
                        if not np.array_equal(a, b):
                            ident = False
                            lines.append(f"  ОТЛИЧИЕ от основы: {bl(name, l)}")
                    for name in ("linear_attn.A_log", "linear_attn.dt_bias"):
                        a = f64(rb.get_tensor(f"model.language_model.layers.{l}.{name}"))
                        b = f64(rv.get_tensor(f"model.language_model.layers.{l}.{name}"))
                        if not np.array_equal(a, b):
                            ident = False
                            lines.append(f"  ОТЛИЧИЕ от основы: {name} слой {l}")
                z_changed = [l for l in EARLY + LATE
                             if not np.array_equal(f64(rb.get_tensor(bl("linear_attn.in_proj_z", l))), z[l])]
                emb_changed = [l for l in EARLY + LATE
                               if not np.array_equal(f64(rb.get_tensor(bl("linear_attn.in_proj_a", l))),
                                                     f64(rv.get_tensor(bl("linear_attn.in_proj_a", l))))
                               or not np.array_equal(f64(rb.get_tensor(bl("linear_attn.in_proj_b", l))),
                                                     f64(rv.get_tensor(bl("linear_attn.in_proj_b", l))))]
            lines.append(f"  побайтовое совпадение с основой по LN1/LN2/linear_attn.norm/conv1d/A_log/dt_bias: "
                         f"{'ДА' if ident else 'НЕТ'}")
            lines.append(f"  слои с изменённым in_proj_z относительно основы: {z_changed or 'нет'}")
            lines.append(f"  слои с изменённым in_proj_a/b относительно основы: {emb_changed or 'нет'}")
            out.append("\n".join(lines))
    text = "\n\n".join(out) + "\n"
    (ROOT / "runs/analysis/dose_ledger_wave10.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
