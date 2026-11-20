#!/usr/bin/env python
"""Счёт вклада ветвей в остаточный поток по послойному зонду (63 задачи).

Определения взяты из scripts/probe_layers.py:95-99,232-242:
  gain_attn = rms_attn_branch / rms_residual_pre   — сколько RMS добавляет attention-ветка
                                                 к остаточному потоку блока за один шаг
  gain_mlp  = rms_mlp_branch  / rms_residual_pre   — то же для MLP
Каждый из 24 блоков проходится ровно один раз на каждый генерируемый токен, поэтому
сумма gain по блокам — это доля правки остаточного потока, приходящаяся на ветку.
Никаких градиентов и обучения: только активации прямого прохода."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULLS = [3, 7, 11, 15, 19, 23]
LIN = [l for l in range(24) if l not in FULLS]
KEYS = ["probe_base_qwen3.5-0.8b.json", "probe_w8_model.json"]


def load(name):
    d = json.load(open(ROOT / "runs/analysis" / name))["layers"]
    return {l: d[str(l)] for l in range(24)}


for name in KEYS:
    L = load(name)
    steps = {l: L[l]["n_steps_mean"] for l in range(24)}
    tot_steps = sum(steps.values())
    ga = {l: L[l]["gain_attn"] for l in range(24)}
    gm = {l: L[l]["gain_mlp"] for l in range(24)}
    sum_ga_full = sum(ga[l] for l in FULLS)
    sum_ga_lin = sum(ga[l] for l in LIN)
    sum_gm = sum(gm[l] for l in range(24))
    tot = sum_ga_full + sum_ga_lin + sum_gm
    print(f"=== {name}")
    print(f"  шагов генерации (сумма по 24 блокам, 63 задачи): {tot_steps:.1f}  "
          f"средняя длина задачи {tot_steps/24:.1f} токена")
    print(f"  сумма gain_attn: FULL {sum_ga_full:.4f}  LIN {sum_ga_lin:.4f}   сумма gain_mlp: {sum_gm:.4f}")
    print(f"  доля правки остаточного потока: FULL-внимание {100*sum_ga_full/tot:.1f} %  "
          f"LIN-внимание {100*sum_ga_lin/tot:.1f} %  MLP {100*sum_gm/tot:.1f} %")
    print("  по FULL-слоям: " + "  ".join(
        f"L{l}: gain_attn {ga[l]:.4f} ({100*ga[l]/tot:.1f} % от всей правки, "
        f"гейт {L[l]['mean_gate_sigmoid']:.3f})" for l in FULLS))
    print("  по LIN-слоям (gain_attn): " + " ".join(f"L{l}:{ga[l]:.3f}" for l in LIN))
    print("  по LIN-слоям (gain_mlp):  " + " ".join(f"L{l}:{gm[l]:.3f}" for l in LIN))

b, w = load(KEYS[0]), load(KEYS[1])
print("=== base -> w8, изменение gain по блокам")
for l in range(24):
    da = 100 * (w[l]["gain_attn"] / b[l]["gain_attn"] - 1)
    dm = 100 * (w[l]["gain_mlp"] / b[l]["gain_mlp"] - 1)
    tag = "FULL" if l in FULLS else "LIN "
    print(f"  L{l:2d} {tag} gain_attn {b[l]['gain_attn']:.4f} -> {w[l]['gain_attn']:.4f} "
          f"({da:+.1f} %)   gain_mlp {dm:+.1f} %")
for tag, S in (("FULL", FULLS), ("LIN", LIN)):
    db = sum(b[l]["gain_attn"] for l in S)
    dw = sum(w[l]["gain_attn"] for l in S)
    dm = sum(b[l]["gain_mlp"] for l in S)
    dwm = sum(w[l]["gain_mlp"] for l in S)
    print(f"  {tag}: сумма gain_attn {db:.4f} -> {dw:.4f} ({100*(dw/db-1):+.2f} %), "
          f"сумма gain_mlp по этим же блокам {100*(dwm/dm-1):+.2f} %")
