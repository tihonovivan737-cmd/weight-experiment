#!/usr/bin/env python3
"""Механизм форматной дозы: что именно делает усиление строк «FINAL» в embed_tokens.

Правка tied-эмбеддингов (строки 95429 «FINAL» и 64697 « FINAL») меняет не только входную
эмбеддинг-строку токена, но и строку выходного проектора: логит tok FINAL = (e_FINAL · h).
Значит доза ×1,32 или ×1,44 масштабирует логит ПРОПОРЦИОНАЛЬНО его текущему значению, а не
добавляет постоянную прибавку: там, где логит отрицательный, та же доза его понижает.

Скрипт прогоняет ОДНУ И ТУ ЖЕ последовательность токенов (промпт + ответ, который реально
выдал w8) через каждое модельное состояние и считает по всем позициям:
  * средний логит лучшей из двух строк FINAL и средний логит выбранного (argmax) токена;
  * запас margin = логит argmin ... точнее (логит argmax − логит FINAL);
  * сколько позиций, где argmax — токен FINAL, хотя в эталонном тексте на этом месте FINAL не
    стоит (контрфактический «преждевременный FINAL» при жадном разборе);
  * логит FINAL и запас на тех позициях, где эталонный текст действительно пишет FINAL.
Знак и величина логита FINAL на этих позициях объясняют, почему лестница доз работает против
верных ответов. Градиентов нет: только прямой проход, gold-ответы не используются.

Пример:
  .venv/bin/python scripts/final_logit_probe.py
  .venv/bin/python scripts/final_logit_probe.py --variants w8 w41 w42 --max-tokens 500
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/home/mpiscil/weight-experiment")
DATASET = ROOT / "datasets/gsm8k/main"
ROWS = [95429, 64697]
ROW_TOKENS = {95429: "FINAL", 64697: " FINAL"}
VARIANTS = {
    "w8": "runs/variants/w8_z_gate_mild_format/model",
    "w41": "runs/variants/w41_format_embed132/model",
    "w42": "runs/variants/w42_format_embed144/model",
    "w43": "runs/variants/w43_format_embed144_latedose/model",
    "w44": "runs/variants/w44_mlp_branch_late105/model",
}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_gsm8k", ROOT / "scripts" / "eval_gsm8k.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_tasks(n_tasks: int):
    """Те же 63 search-задачи, что в probe_layers.py, и тот же текст ответа, что выдал w8."""
    idxs = [int(i) for i in json.loads(
        (ROOT / "runs/analysis/probe_w8_model.json").read_text(encoding="utf-8")
    )["search_indices_used"]]
    answers = {}
    for line in (ROOT / "runs/variants/w8_z_gate_mild_format/search/results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        rec = json.loads(line)
        answers[int(rec["index"])] = rec["model_answer"]
    df = pd.read_parquet(DATASET / "train-00000-of-00001.parquet")
    items = []
    for i in idxs[:n_tasks]:
        ans = answers.get(i)
        if not ans:
            continue
        items.append((i, str(df.iloc[i]["question"]), ans))
    return items


def probe(model, processor, items, max_answer_tokens: int) -> dict:
    agg = {
        "n_tasks": 0, "positions": 0,
        "final_logit": [], "top1_logit": [], "margin": [], "ref_logit": [],
        "argmax_final_positions": 0, "premature_final_positions": 0, "true_final_positions": 0,
        "true_final_logit": [], "true_final_margin": [],
    }
    with torch.no_grad():
        for idx, question, answer in items:
            inputs = eval_mod.build_inputs(processor, question)
            ids = inputs["input_ids"][0].to(device)
            ans_ids = processor.tokenizer.encode(answer, add_special_tokens=False)[:max_answer_tokens]
            if not ans_ids:
                continue
            seq = torch.cat([ids, torch.tensor(ans_ids, device=device, dtype=ids.dtype)], dim=-1)
            attn = torch.ones_like(seq).unsqueeze(0)
            logits = model(input_ids=seq.unsqueeze(0), attention_mask=attn).logits[0]
            tgt = seq[1:]
            n = min(int(logits.shape[0]) - 1, int(tgt.shape[0]))
            sub = logits[:n].float()
            top_val, top_id = sub.max(dim=-1)
            frow = sub[:, ROWS]
            f_val, f_pos = frow.max(dim=-1)
            ref_val = sub.gather(-1, tgt[:n].unsqueeze(-1)).squeeze(-1)

            is_final_token = (tgt[:n] == ROWS[0]) | (tgt[:n] == ROWS[1])
            argmax_is_final = (top_id == ROWS[0]) | (top_id == ROWS[1])

            agg["n_tasks"] += 1
            agg["positions"] += n
            agg["final_logit"].append(float(f_val.mean()))
            agg["top1_logit"].append(float(top_val.mean()))
            agg["margin"].append(float((top_val - f_val).mean()))
            agg["ref_logit"].append(float(ref_val.mean()))
            agg["argmax_final_positions"] += int(argmax_is_final.sum())
            agg["premature_final_positions"] += int((argmax_is_final & ~is_final_token).sum())
            agg["true_final_positions"] += int(is_final_token.sum())
            if bool(is_final_token.any()):
                sel = is_final_token
                agg["true_final_logit"].append(float(f_val[sel].mean()))
                agg["true_final_margin"].append(float((top_val - f_val)[sel].mean()))
            del logits, sub, top_val, top_id, frow, f_val, ref_val, seq, ids, attn
    return agg


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    global eval_mod, device
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--n-tasks", type=int, default=63)
    ap.add_argument("--max-answer-tokens", type=int, default=420)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    eval_mod = load_eval_module()
    items = build_tasks(args.n_tasks)
    if len(items) < 40:
        raise SystemExit(f"задач для пробы: {len(items)} — недостаточно")

    data = {}
    for key in args.variants:
        rel = VARIANTS.get(key)
        if not rel:
            print(f"{key}: неизвестный вариант — пропуск")
            continue
        path = ROOT / rel
        if not path.is_dir():
            print(f"{key}: модель не найдена ({rel}) — пропуск")
            continue
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True).to(device)
        model.eval()
        data[key] = probe(model, processor, items, args.max_answer_tokens)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{key:4} готово: задач {data[key]['n_tasks']}, позиций {data[key]['positions']}")

    lines = [
        "Проба логитов строки FINAL (прямой проход, без градиентов).",
        f"Один и тот же текст (ответ w8) прогнан через каждое состояние; {len(items)} задач search.",
        "logit FINAL — средний логит лучшей из строк 95429/64697; argmax — логит выбранного токена;",
        "margin = argmax − FINAL (большой запас = токен FINAL далеко от решения);",
        "преждевременный FINAL — позиции, где argmax был бы токеном FINAL, хотя в тексте его там нет.",
        "",
        f"{'вариант':<6} {'позиций':>8} {'логит FINAL':>12} {'логит argmax':>13} {'margin':>9} "
        f"{'логит этал. токена':>18} {'argmax=FINAL':>13} {'прежд. FINAL':>13}",
    ]
    for k, a in data.items():
        lines.append(
            f"{k:<6} {a['positions']:>8} {mean(a['final_logit']):>12.3f} {mean(a['top1_logit']):>13.3f} "
            f"{mean(a['margin']):>9.3f} {mean(a['ref_logit']):>18.3f} "
            f"{a['argmax_final_positions']:>13} {a['premature_final_positions']:>13}")
    lines.append("")
    lines.append("Позиции, где эталонный текст действительно пишет FINAL:")
    lines.append(f"{'вариант':<6} {'таких позиций':>14} {'логит FINAL там':>16} {'margin там':>12}")
    for k, a in data.items():
        lines.append(f"{k:<6} {a['true_final_positions']:>14} {mean(a['true_final_logit']):>16.3f} "
                     f"{mean(a['true_final_margin']):>12.3f}")
    base = "w8" if "w8" in data else next(iter(data))
    lines.append("")
    lines.append(f"Сдвиг против {base} (логит FINAL по всем позициям / по позициям записи FINAL / "
                 "преждевременные FINAL):")
    for k, a in data.items():
        if k == base:
            continue
        b = data[base]
        dl = mean(a["final_logit"]) - mean(b["final_logit"])
        dt = (mean(a["true_final_logit"]) - mean(b["true_final_logit"])
              if a["true_final_logit"] and b["true_final_logit"] else float("nan"))
        lines.append(f"  {k:<6} Δлогит FINAL {dl:+8.3f}  Δлогит на позициях записи FINAL {dt:+8.3f}  "
                     f"преждевременных FINAL {a['premature_final_positions'] - b['premature_final_positions']:+5d}")
    text = "\n".join(lines) + "\n"
    (ROOT / "runs/analysis/final_logit_probe.txt").write_text(text, encoding="utf-8")
    (ROOT / "runs/analysis/final_logit_probe.json").write_text(
        json.dumps({"n_tasks": len(items), "max_answer_tokens": args.max_answer_tokens,
                    "rows": ROWS, "row_tokens": {str(k): v for k, v in ROW_TOKENS.items()},
                    "variants": data}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
