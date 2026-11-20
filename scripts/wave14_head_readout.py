#!/usr/bin/env python3
# Волна 14, замер до запуска: вклад value-голов FULL-слоёв в логит ответа.
# Только чтение, только CPU. Путь: строки v_proj головы -> columns o_proj той же головы
# -> финальная нормализация -> строка embed_tokens для токена ответа.
# Для каждой головы считается средняя по 63 задачам величина
# || A_h^T (n_final * e_task) ||, где A_h — столбцы o_proj этой головы,
# n_final — веса model.language_model.norm.weight, e_task — строка embed_tokens
# для токена, который должен решить ответ (токен золотого ответа), и отдельно
# для токена FINAL. Отношение показывает, какой вклад читается в логит.
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/variants/w8_z_gate_mild_format/model"
FULL = [3, 7, 11, 15, 19, 23]
FINAL_ROWS = [95429, 64697]
HEAD_DIM = 256
N_HEADS = 2


def read_header(fp):
    n = struct.unpack("<Q", fp.read(8))[0]
    return 8 + n, json.loads(fp.read(n).decode("utf-8"))


def get_tensor(fp, hs, h, name):
    info = h[name]
    do, ds = hs + info["data_offsets"][0], hs + info["data_offsets"][1]
    fp.seek(do)
    raw = fp.read(ds - do)
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
    return t.reshape(info["shape"])




shard = sorted(BASE.glob("*.safetensors"))[0]

sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util

spec = importlib.util.spec_from_file_location("flp", ROOT / "scripts" / "final_logit_probe.py")
flp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flp)

# Токены золотых ответов по 63 задачам (список индексов из probe_w8_model.json).
import pandas as pd

ds = pd.read_parquet(flp.DATASET / "train-00000-of-00001.parquet")
idx = json.load(open(ROOT / "runs/analysis/probe_w8_model.json", encoding="utf-8"))["search_indices_used"]
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(ROOT / "models/qwen3.5-0.8b")
answer_tokens = []
for i in idx:
    ans = str(ds.iloc[i]["answer"]).strip()
    last = [ln for ln in ans.splitlines() if ln.strip()][-1]
    piece = last.split()[-1]
    ids = tok.encode(piece, add_special_tokens=False)
    if ids:
        answer_tokens.append(ids[0])

with open(shard, "rb") as fp:
    hs, h = read_header(fp)
    n_final = get_tensor(fp, hs, h, "model.language_model.norm.weight")
    emb = get_tensor(fp, hs, h, "model.language_model.embed_tokens.weight")
    e_final = emb[FINAL_ROWS[0]]
    e_final = e_final / torch.linalg.norm(e_final)
    rows = []
    for L in FULL:
        v = get_tensor(fp, hs, h, f"model.language_model.layers.{L}.self_attn.v_proj.weight")
        o = get_tensor(fp, hs, h, f"model.language_model.layers.{L}.self_attn.o_proj.weight")
        for hc in range(N_HEADS):
            A = o[:, hc * HEAD_DIM:(hc + 1) * HEAD_DIM]  # 1024 x 256
            g_ans = 0.0
            for t in answer_tokens:
                e = emb[t]
                e = e / torch.linalg.norm(e)
                g = A.T @ (n_final * e)
                g_ans += float(torch.linalg.norm(g))
            g_final = float(torch.linalg.norm(A.T @ (n_final * e_final)))
            g_ans /= len(answer_tokens)
            block = v[hc * HEAD_DIM:(hc + 1) * HEAD_DIM]
            rows.append({
                "layer": L,
                "head": hc,
                "gain_answer": round(g_ans, 4),
                "gain_final": round(g_final, 4),
                "ratio_final_over_answer": round(g_final / g_ans, 4),
                "block_l2": round(float(torch.linalg.norm(block)), 4),
                "block_rows": [hc * HEAD_DIM, (hc + 1) * HEAD_DIM - 1],
            })

rows.sort(key=lambda r: -r["gain_answer"])
out = {
    "base": str(BASE),
    "n_answer_tokens": len(answer_tokens),
    "layers": FULL,
    "heads_per_layer": N_HEADS,
    "rows_sorted_by_gain_answer": rows,
    "note": "ЗАМЕР. Кладка q/k/v в FULL-слоях однозначна: v_proj = 6 слоёв x 2 головы x 256 строк, "
           "столбцы o_proj для головы h = h*256..h*256+255. linear_attn.in_proj_qkv (6144 строк) "
           "здесь не трактуется, потому что порядок q/k/v внутри него не подтверждён.",
}
Path(ROOT / "runs/analysis/wave14_head_readout.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Замер value-голов FULL-слоёв (база w8), токенов ответа: {len(answer_tokens)}")
print("| слой | голова | строки v_proj | вклад в логит ответа | вклад в FINAL | отношение | L2 блока |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['layer']} | {r['head']} | {r['block_rows'][0]}-{r['block_rows'][1]} | "
          f"{r['gain_answer']:.4f} | {r['gain_final']:.4f} | {r['ratio_final_over_answer']:.4f} | "
          f"{r['block_l2']:.4f} |")
top_late = [r for r in rows if r["layer"] in (19, 23)]
print("Поздние FULL-слои (19, 23):")
for r in top_late:
    print(f"  слой {r['layer']} голова {r['head']} строки {r['block_rows'][0]}-{r['block_rows'][1]} "
          f"вклад ответа {r['gain_answer']:.4f} FINAL {r['gain_final']:.4f}")
