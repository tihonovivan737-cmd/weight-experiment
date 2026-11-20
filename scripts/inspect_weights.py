"""Статистика весов Qwen3.5-0.8B без их изменения (только чтение).

Пишет runs/analysis/weight_stats.json:
  - список всех тензоров (имя, форма, dtype, кол-во элементов, mean/std/absmax/L2),
  - сводку по типам слоёв,
  - отдельно: эмбеддинги цифровых токенов и токенов формата ответа.
Model_dir по умолчанию — исходная копия в models/ (не изменяется).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path("/home/mpiscil/weight-experiment")


def tensor_stats(t: torch.Tensor) -> dict:
    """Статистика в float32 по чанкам: полная fp32-копия тензора не требуется."""
    flat = t.reshape(-1)
    n = flat.numel()
    step = 1 << 20
    s = ss = 0.0
    absmax = 0.0
    for i in range(0, n, step):
        c = flat[i : i + step].float()
        s += float(c.sum())
        ss += float((c * c).sum())
        am = float(c.abs().max())
        if am > absmax:
            absmax = am
    mean = s / n
    var = max(ss / n - mean * mean, 0.0)
    return {
        "numel": n,
        "mean": round(mean, 6),
        "std": round(var ** 0.5, 6),
        "absmax": round(absmax, 6),
        "l2": round(ss ** 0.5, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/qwen3.5-0.8b"))
    ap.add_argument("--out", default=str(ROOT / "runs/analysis/weight_stats.json"))
    args = ap.parse_args()

    model_dir = Path(args.model)
    st_file = next(iter(sorted(model_dir.glob("*.safetensors"))))
    cfg = json.loads((model_dir / "config.json").read_text())

    tensors = {}
    with safe_open(st_file, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            rec = {"shape": list(t.shape), "dtype": str(t.dtype)}
            rec.update(tensor_stats(t))
            tensors[k] = rec

    # группировка по типу параметра
    groups: dict[str, dict] = {}
    for k, v in tensors.items():
        tail = k.split(".")[-1]
        g = groups.setdefault(tail, {"count": 0, "numel": 0, "l2_sum": 0.0, "std_mean": []})
        g["count"] += 1
        g["numel"] += v["numel"]
        g["l2_sum"] += v["l2"]
        g["std_mean"].append(v["std"])
    for g in groups.values():
        g["std_mean"] = round(float(np.mean(g["std_mean"])), 6)

    # --- токены, важные для GSM8K -------------------------------------
    vocab = json.loads((model_dir / "vocab.json").read_text())
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    digit_ids = {}
    for s in [str(d) for d in range(10)] + [f" {d}" for d in range(10)]:
        if s in vocab:
            digit_ids[s] = vocab[s]
    format_tokens = ["FINAL", "ĠFINAL", "FINAL:", "ĠFINAL:", "Answer", "ĠAnswer",
                     "**", "Ġthe", "ĠTherefore", "ĠTherefore:", "\\n", "Ġanswer"]
    format_ids = {s: vocab[s] for s in format_tokens if s in vocab}

    emb_key = None
    for k in tensors:
        if k.endswith("embed_tokens.weight"):
            emb_key = k
            break
    emb_info = {}
    if emb_key:
        with safe_open(st_file, framework="pt") as f:
            emb = f.get_tensor(emb_key)  # [vocab, hidden]
        row_l2 = emb.float().norm(dim=1)
        row_std = emb.float().std(dim=1)
        def row_info(ids: dict[str, int]) -> dict:
            out = {}
            for name, i in ids.items():
                out[name] = {"id": i, "l2": round(float(row_l2[i]), 4),
                             "std": round(float(row_std[i]), 6)}
            return out
        vocab_l2 = row_l2.numpy()
        order = np.argsort(-vocab_l2)
        emb_info = {
            "key": emb_key,
            "shape": list(emb.shape),
            "vocab_l2_mean": round(float(vocab_l2.mean()), 4),
            "vocab_l2_median": round(float(np.median(vocab_l2)), 4),
            "vocab_l2_p95": round(float(np.percentile(vocab_l2, 95)), 4),
            "digits": row_info(digit_ids),
            "format_tokens": row_info(format_ids),
            "top10_l2_tokens": [
                {"id": int(i), "l2": round(float(vocab_l2[i]), 4)} for i in order[:10]
            ],
        }
        id2tok = {v: k for k, v in vocab.items()}
        emb_info["top10_l2_tokens"] = [
            {**t, "token": id2tok.get(t["id"], "?")} for t in emb_info["top10_l2_tokens"]
        ]
        del emb, row_l2, row_std

    out = {
        "model_dir": str(model_dir),
        "weights_file": st_file.name,
        "architectures": cfg.get("architectures"),
        "n_tensors": len(tensors),
        "total_numel": sum(v["numel"] for v in tensors.values()),
        "groups": groups,
        "embedding": emb_info,
        "tensors": tensors,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Тензоров: {len(tensors)}; параметров: {out['total_numel']:,}")
    print("Группы (по имени параметра):")
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["numel"]):
        print(f"  {name:<28} n={g['count']:<3} numel={g['numel']:>10,} std_mean={g['std_mean']}")
    print("\nЭмбеддинги цифровых токенов:")
    for k, v in emb_info.get("digits", {}).items():
        print(f"  {k!r:>6} id={v['id']:>7} l2={v['l2']} std={v['std']}")
    print(f"vocab L2: mean={emb_info.get('vocab_l2_mean')} median={emb_info.get('vocab_l2_median')} p95={emb_info.get('vocab_l2_p95')}")
    print("Топ-10 по L2:", [(t['token'], t['l2']) for t in emb_info.get('top10_l2_tokens', [])])
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
