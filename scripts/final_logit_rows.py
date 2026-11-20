#!/usr/bin/env python3
"""Замер строк logit-весов с малым запасом против FINAL (волна 13, подготовительный зонд).

Строит ровно те же 63 search-задачи, что `probe_w8_model.json`, и прогоняет их через w8
(прямой проход, без градиентов, gold-ответы не используются — только текст ответа, который
реально выдал w8). На позициях, где этот текст действительно пишет токен FINAL, считает по
всему словарю:

  * запас margin_i = logit_i − logit_FINAL для каждой строки i;
  * сколько строк словаря имеют запас < порога (проверка оценки «6.2 % строк с запасом < 1»,
    которая в §41.7 была записана без артефакта);
  * какие конкретные строки обгоняют FINAL и на скольких позициях.

Ничего не правится: скрипт только измеряет. Данные — только разрешённые (search из 63 задач).

Запуск (единственный GPU-прогон, GPU1):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/final_logit_rows.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/home/mpiscil/weight-experiment")
MODEL = ROOT / "runs/variants/w8_z_gate_mild_format/model"
OUT_JSON = ROOT / "runs/analysis/final_logit_rows.json"
OUT_TXT = ROOT / "runs/analysis/final_logit_rows.txt"


def load_probe_module():
    """Переиспользовать конфигурацию волны 10: те же строки FINAL и тот же набор задач."""
    spec = importlib.util.spec_from_file_location("flp", ROOT / "scripts" / "final_logit_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=63)
    ap.add_argument("--max-answer-tokens", type=int, default=420)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin-threshold", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--model", default=str(MODEL))
    args = ap.parse_args()

    flp = load_probe_module()
    flp.device = args.device
    eval_mod = flp.load_eval_module()
    flp.eval_mod = eval_mod
    items = flp.build_tasks(args.n_tasks)
    if len(items) < 40:
        raise SystemExit(f"задач для замеров: {len(items)} — недостаточно")

    rows = flp.ROWS
    row_tokens = flp.ROW_TOKENS
    model_path = Path(args.model)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    vocab = processor.tokenizer.get_vocab()
    inv = {v: k for k, v in vocab.items()}
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True).to(args.device)
    model.eval()

    per_pos = []
    row_stats = {}
    n_tasks_done = 0

    with torch.no_grad():
        for idx, question, answer in items:
            inputs = eval_mod.build_inputs(processor, question)
            ids = inputs["input_ids"][0].to(args.device)
            ans_ids = processor.tokenizer.encode(answer, add_special_tokens=False)[:args.max_answer_tokens]
            if not ans_ids:
                continue
            seq = torch.cat([ids, torch.tensor(ans_ids, device=args.device, dtype=ids.dtype)], dim=-1)
            attn = torch.ones_like(seq).unsqueeze(0)
            logits = model(input_ids=seq.unsqueeze(0), attention_mask=attn).logits[0]
            tgt = seq[1:]
            n = min(int(logits.shape[0]) - 1, int(tgt.shape[0]))
            sub = logits[:n].float()
            f_val = sub[:, rows].max(dim=-1).values
            is_final = (tgt[:n] == rows[0]) | (tgt[:n] == rows[1])
            n_tasks_done += 1
            for p in range(n):
                if not bool(is_final[p]):
                    continue
                fv = float(f_val[p])
                margin = sub[p] - fv
                lt1 = int((margin < args.margin_threshold).sum())
                top_val, top_id = torch.topk(margin, args.topk)
                rec = {
                    "task": idx,
                    "pos": p,
                    "final_logit": round(fv, 3),
                    "rows_lt1": lt1,
                    "top": [[int(t), round(float(v), 3), inv.get(int(t), "")]
                            for t, v in zip(top_id.tolist(), top_val.tolist())],
                }
                per_pos.append(rec)
                for t, v in zip(top_id.tolist(), top_val.tolist()):
                    if float(v) >= 0.0:
                        s = row_stats.setdefault(int(t), {"token": inv.get(int(t), ""),
                                                          "n_positions": 0, "margins": []})
                        s["n_positions"] += 1
                        s["margins"].append(round(float(v), 3))
            del logits, sub, f_val, seq, ids, attn
            torch.cuda.empty_cache()

    if not per_pos:
        raise SystemExit("позиций с фактической записью FINAL не найдено")

    lt1 = [r["rows_lt1"] for r in per_pos]
    dist = {}
    for r in per_pos:
        dist[r["rows_lt1"]] = dist.get(r["rows_lt1"], 0) + 1
    uniq = len(row_stats)
    by_freq = sorted(row_stats.items(), key=lambda kv: (-kv[1]["n_positions"], kv[0]))

    data = {
        "variant": "w8",
        "model": str(model_path),
        "n_tasks": n_tasks_done,
        "max_answer_tokens": args.max_answer_tokens,
        "final_rows": rows,
        "final_row_tokens": {str(k): v for k, v in row_tokens.items()},
        "margin_threshold": args.margin_threshold,
        "positions": len(per_pos),
        "mean_rows_lt1": round(sum(lt1) / len(lt1), 2),
        "share_rows_lt1": round(sum(lt1) / len(lt1) / len(vocab), 6),
        "max_rows_lt1": max(lt1),
        "min_rows_lt1": min(lt1),
        "hist_rows_lt1": dist,
        "unique_beating_rows": uniq,
        "rows": [{"row": t, "token": row_stats[t]["token"], "n_positions": row_stats[t]["n_positions"],
                  "mean_margin": round(sum(row_stats[t]["margins"]) / len(row_stats[t]["margins"]), 3)}
                 for t, _ in by_freq],
        "per_position": per_pos,
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Замер строк logit-весов против FINAL на позициях фактической записи FINAL (w8).",
        f"Задач: {n_tasks_done}, позиций записи FINAL: {len(per_pos)}, порог запаса: {args.margin_threshold}.",
        f"Строк словаря с запасом < порога: среднее {data['mean_rows_lt1']}, "
        f"доля {data['share_rows_lt1'] * 100:.2f} % (оценка §41.7 была 6.2 %), "
        f"мин {data['min_rows_lt1']}, макс {data['max_rows_lt1']}, "
        f"уникальных строк-конкурентов: {uniq}.",
        "",
        f"{'row':>7} {'токен':<16} {'позиций':>8} {'сред. запас':>12}",
    ]
    for r in data["rows"][:40]:
        lines.append(f"{r['row']:>7} {r['token']:<16} {r['n_positions']:>8} {r['mean_margin']:>12.3f}")
    lines.append("")
    lines.append("Распределение числа строк с запасом < порога по позициям (первые 12):")
    for k, v in sorted(dist.items())[:12]:
        lines.append(f"  {k:>6} строк — {v:>4} позиций")
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
