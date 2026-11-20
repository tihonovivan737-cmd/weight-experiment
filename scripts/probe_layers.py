#!/usr/bin/env python3
"""Прямые измерения активаций Qwen3.5-0.8B на задачах GSM8K (без градиентов).

Скрипт ставит forward-хуки на блоки 24 языковых слоёв и считает:
  * RMS остаточного потока до блока, RMS выхода attention-ветви и RMS выхода MLP
    -> относительный «усиление» ветвей (branch gain) по слоям;
  * для linear_attention (GatedDeltaNet): силу вентиля silu(z) по 16 value-головам,
    beta = sigmoid(in_proj_b), softplus(in_proj_a + dt_bias) (сила записи/затухания);
  * для full_attention: логиты выхода gate (вторая половина каждого head-блока в
    q_proj), sigmoid(gate), RMS q/k/v.

Замеры делаются на собственных генерациях модели (greedy, та же схема промпта,
что и в eval_gsm8k.py), только на разрешённых отборочных индексах search.
Никаких gold-ответов не читает.

Пример:
  .venv/bin/python scripts/probe_layers.py --model models/qwen3.5-0.8b
  .venv/bin/python scripts/probe_layers.py --model runs/variants/w8_z_gate_mild_format/model
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/home/mpiscil/weight-experiment")


def load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_gsm8k", ROOT / "scripts" / "eval_gsm8k.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rms(x: torch.Tensor, axis=-1) -> float:
    return float(x.to(torch.float32).pow(2).mean(axis=axis).sqrt().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="json с результатом")
    ap.add_argument("--n", type=int, default=64, help="сколько search-задач")
    ap.add_argument("--stride", type=int, default=4, help="шаг по search-индексам")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    model_dir = Path(args.model)
    out_path = Path(args.out) if args.out else ROOT / "runs" / "analysis" / f"{args.tag}_{model_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ev = load_eval_module()
    splits = json.loads((ROOT / "runs" / "splits" / "splits.json").read_text(encoding="utf-8"))
    search = splits["search_indices"]
    idxs = search[:: args.stride][: args.n]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True
    ).to(args.device)
    model.eval()

    layers = {}
    for name, mod in model.named_modules():
        m = re.fullmatch(r"model\.language_model\.layers\.(\d+)", name)
        if m:
            layers[int(m.group(1))] = mod
    assert len(layers) == 24, f"найдено слоёв: {len(layers)}"

    stats = {}

    def new_entry(ltype):
        return {
            "type": ltype,
            "pre": [],            # RMS остатка до блока (по шагам генерации)
            "attn": [],           # RMS выхода attention-ветви
            "mlp": [],            # RMS выхода MLP
            "norm_out": [],       # RMS выхода input_layernorm (вход внимания)
            "post_ln_out": [],    # RMS выхода post_attention_layernorm (вход MLP)
            "z_gate_head": [],    # (n_steps, n_heads) среднее |silu(z)|
            "beta_head": [],      # (n_steps, n_heads) среднее beta
            "softplus_head": [],  # (n_steps, n_heads) среднее softplus(a+dt_bias)
            "head_out_rms": [],   # (n_steps, n_heads) RMS вклада головы на входе out_proj
            "gate_logit_head": [],# full: (n_steps, n_heads) средний логит gate
            "gate_sigmoid_head": [],
            "q_rms": [], "k_rms": [], "v_rms": [],
        }

    for i, mod in layers.items():
        lin = hasattr(mod, "linear_attn")
        stats[i] = new_entry("linear_attention" if lin else "full_attention")
        att = mod.linear_attn if lin else mod.self_attn

        mod.register_forward_pre_hook(
            lambda m, inp, s=stats[i]: s["pre"].append(rms(inp[0]))
        )
        att.register_forward_hook(
            lambda m, inp, out, s=stats[i]: s["attn"].append(rms(out[0] if isinstance(out, tuple) else out))
        )
        mod.mlp.register_forward_hook(
            lambda m, inp, out, s=stats[i]: s["mlp"].append(rms(out))
        )
        mod.input_layernorm.register_forward_hook(
            lambda m, inp, out, s=stats[i]: s["norm_out"].append(rms(out))
        )
        mod.post_attention_layernorm.register_forward_hook(
            lambda m, inp, out, s=stats[i]: s["post_ln_out"].append(rms(out))
        )

        if lin:
            hd = att.head_v_dim

            # z до norm: in_proj_z -> [B,T,16*head_v_dim], head-major
            def z_hook(m, inp, out, s=stats[i], hd=hd):
                zz = out.to(torch.float32).view(out.shape[0], out.shape[1], -1, hd)
                s["z_gate_head"].append(
                    torch.nn.functional.silu(zz).abs().mean(dim=(0, 1, 3)).cpu().numpy()
                )

            att.in_proj_z.register_forward_hook(z_hook)

            # вход out_proj = RMSNorm(core) * silu(z): фактический вклад каждой головы
            def op_in_hook(m, inp, out, s=stats[i], hd=hd):
                v = inp[0].to(torch.float32).view(inp[0].shape[0], inp[0].shape[1], -1, hd)
                s["head_out_rms"].append(
                    v.pow(2).mean(dim=3).sqrt().mean(dim=(0, 1)).cpu().numpy()
                )

            att.out_proj.register_forward_hook(op_in_hook)

            att.in_proj_b.register_forward_hook(
                lambda m, inp, out, s=stats[i]: s["beta_head"].append(
                    torch.sigmoid(out.to(torch.float32)).mean(dim=(0, 1)).cpu().numpy()
                )
            )

            def a_hook(m, inp, out, s=stats[i], att=att):
                val = torch.nn.functional.softplus(out.to(torch.float32) + att.dt_bias.to(torch.float32))
                s["softplus_head"].append(val.mean(dim=(0, 1)).cpu().numpy())

            att.in_proj_a.register_forward_hook(a_hook)
        else:
            def full_gate_hook(m, inp, out, s=stats[i]):
                # q_proj -> view(B,T,8,512) -> chunk по последним 256: head-блоки
                g = out.to(torch.float32).view(out.shape[0], out.shape[1], -1, 512)
                gate = g[..., 256:]
                s["gate_logit_head"].append(gate.mean(dim=(0, 1, 3)).cpu().numpy())
                s["gate_sigmoid_head"].append(torch.sigmoid(gate).mean(dim=(0, 1, 3)).cpu().numpy())

            att.q_proj.register_forward_hook(full_gate_hook)
            att.k_proj.register_forward_hook(
                lambda m, inp, out, s=stats[i]: s["k_rms"].append(rms(out))
            )
            att.v_proj.register_forward_hook(
                lambda m, inp, out, s=stats[i]: s["v_rms"].append(rms(out))
            )

    abs_decay = {}
    for i, mod in layers.items():
        if hasattr(mod, "linear_attn"):
            a = mod.linear_attn.A_log.detach().to(torch.float32)
            abs_decay[i] = float(a.exp().mean())

    import pandas as pd

    train = pd.read_parquet(
        ROOT / "datasets" / "gsm8k" / "main" / "train-00000-of-00001.parquet"
    )

    pad_id = processor.tokenizer.pad_token_id
    n_done = 0
    task_lens = []
    with torch.no_grad():
        for gi in idxs:
            before = len(stats[0]["pre"])
            item = train.loc[int(gi)]
            inputs = ev.build_inputs(processor, item["question"])
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=pad_id,
            )
            task_lens.append(len(stats[0]["pre"]) - before)
            n_done += 1
            if n_done % 8 == 0:
                print(f"[probe] {n_done}/{len(idxs)}", flush=True)

    print("[probe] длины генераций (шаги):", task_lens[:12], "...", flush=True)

    # -------------------- агрегация (по задачам: среднее по всем шагам и по хвосту)
    TAIL = 25

    def seg(arr, k=TAIL):
        """Возвращает (среднее по всем шагам всех задач, среднее по последним k шагам каждой задачи)."""
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return None, None
        if a.shape[0] != sum(task_lens):
            segs = [a]
        else:
            segs = np.split(a, np.cumsum(task_lens)[:-1])
        means = np.mean([s.mean(axis=0) for s in segs], axis=0)
        tails = np.mean([s[-k:].mean(axis=0) for s in segs], axis=0)
        return means, tails

    per_layer = {}
    for i in range(24):
        s = stats[i]
        pre_all, pre_tail = seg(s["pre"])
        attn_all, attn_tail = seg(s["attn"])
        mlp_all, mlp_tail = seg(s["mlp"])
        z_all, z_tail = seg(s["z_gate_head"])
        rec = {
            "type": s["type"],
            "n_steps_mean": float(np.mean(task_lens)),
            "rms_residual_pre": float(pre_all),
            "rms_residual_pre_tail": float(pre_tail),
            "rms_attn_branch": float(attn_all),
            "rms_attn_branch_tail": float(attn_tail),
            "rms_mlp_branch": float(mlp_all),
            "rms_mlp_branch_tail": float(mlp_tail),
            "gain_attn": float(attn_all / pre_all),
            "gain_mlp": float(mlp_all / pre_all),
            "gain_attn_tail": float(attn_tail / pre_tail),
            "gain_mlp_tail": float(mlp_tail / pre_tail),
            "rms_ln_in": float(seg(s["norm_out"])[0]),
            "rms_post_ln": float(seg(s["post_ln_out"])[0]),
        }
        if s["type"] == "linear_attention":
            b_all, b_tail = seg(s["beta_head"])
            sp_all, _ = seg(s["softplus_head"])
            rec["z_gate_head"] = [float(x) for x in z_all]
            rec["z_gate_head_tail"] = [float(x) for x in z_tail]
            rec["beta_head"] = [float(x) for x in b_all]
            rec["beta_head_tail"] = [float(x) for x in b_tail]
            rec["softplus_a_dt_head"] = [float(x) for x in sp_all]
            rec["mean_z_gate"] = float(np.mean(z_all))
            rec["mean_z_gate_tail"] = float(np.mean(z_tail))
            rec["mean_beta"] = float(np.mean(b_all))
            rec["mean_abs_decay"] = float(abs_decay[i])
        else:
            gl_all, _ = seg(s["gate_logit_head"])
            gs_all, _ = seg(s["gate_sigmoid_head"])
            rec["gate_logit_head"] = [float(x) for x in gl_all]
            rec["gate_sigmoid_head"] = [float(x) for x in gs_all]
            rec["mean_gate_logit"] = float(np.mean(gl_all))
            rec["mean_gate_sigmoid"] = float(np.mean(gs_all))
            rec["rms_k"] = float(seg(s["k_rms"])[0])
            rec["rms_v"] = float(seg(s["v_rms"])[0])
        per_layer[i] = rec

    out = {
        "tag": args.tag,
        "model_dir": str(model_dir),
        "n_tasks": n_done,
        "search_indices_used": idxs,
        "max_new_tokens": args.max_new,
        "dtype": args.dtype,
        "device": args.device,
        "note": (
            "Замеры на собственных генерациях модели (greedy), промпт как в eval_gsm8k.py. "
            "tail-40 = последние 40 шагов генерации (хвост ответа, зона FINAL)."
        ),
        "layers": per_layer,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------------------- человекочитаемая таблица
    lines = [f"# Замеры активаций: {model_dir.name} ({n_done} задач, среднее по всем шагам генерации)"]
    lines.append(
        f"{'слой':>4} {'тип':>6} {'rmsRest':>8} {'rmsAttn':>8} {'rmsMLP':>8} "
        f"{'gainAttn':>8} {'gainMLP':>8} {'meanZ':>7} {'meanBeta':>8} {'decay':>7} {'gAttnTail':>9} {'gMLPTail':>8}"
    )
    for i in range(24):
        r = per_layer[i]
        z = r.get("mean_z_gate")
        b = r.get("mean_beta")
        d = r.get("mean_abs_decay")
        lines.append(
            f"{i:>4} {'LIN' if r['type'] == 'linear_attention' else 'FULL':>6} "
            f"{r['rms_residual_pre']:>8.4f} {r['rms_attn_branch']:>8.4f} {r['rms_mlp_branch']:>8.4f} "
            f"{r['gain_attn']:>8.4f} {r['gain_mlp']:>8.4f} "
            f"{(z if z is not None else float('nan')):>7.4f} {(b if b is not None else float('nan')):>8.4f} "
            f"{(d if d is not None else float('nan')):>7.4f} "
            f"{r['gain_attn_tail']:>9.4f} {r['gain_mlp_tail']:>8.4f}"
        )
    lines.append("")
    for i in range(24):
        r = per_layer[i]
        if r["type"] == "full_attention":
            lines.append(
                f"full {i:>2}: gate logit {r['mean_gate_logit']:+.4f} -> sigmoid {r['mean_gate_sigmoid']:.4f}; "
                f"rms k {r['rms_k']:.4f} v {r['rms_v']:.4f}"
            )
        else:
            zh = r["z_gate_head"]
            lines.append(
                f"lin  {i:>2}: silu(z) по головам "
                + " ".join(f"{v:.3f}" for v in zh)
            )
    print("\n".join(lines))
    print("сохранено:", out_path)


if __name__ == "__main__":
    main()
