"""Дешёвый прокси-метрика вместо генерации: teacher-forced NLL эталонного решения.

Смысл: прогоняем ОДИН forward по (промпт + эталонное решение GSM8K) и считаем
  - mean NLL на токен эталонного решения  -> nll_ref;
  - logprob финального числа (хвост после «####») -> answer_logprob, answer_prob;
  - совпал ли argmax на позиции финального числа с эталонным токеном -> answer_top1_ok.
Это в ~10 раз дешевле генерации, поэтому годится как ФИЛЬТР (отбраковать явно
сломанные варианты) и как количественный diagnostic. Решения об улучшении принимаются
только по accuracy в генерации, не по прокси.

Промпт совпадает со scripts/eval_gsm8k.py (тот же PROMPT_SUFFIX и chat template).
Результаты дописываются построчно (proxy.jsonl) — прерванный прогон возобновляем.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/home/mpiscil/weight-experiment")
DATASET = ROOT / "datasets/gsm8k/main"
SPLITS_FILE = ROOT / "runs/splits/splits.json"

PROMPT_SUFFIX = (
    "\n\nSolve the problem step by step. "
    "Finish with a final line exactly in this format: "
    "FINAL: <number>"
)


def load_split_indices(split: str) -> list[int]:
    if split in ("test", "train"):
        df = pd.read_parquet(DATASET / f"{split}-00000-of-00001.parquet")
        return list(range(len(df)))
    return list(json.loads(SPLITS_FILE.read_text(encoding="utf-8"))[f"{split}_indices"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/qwen3.5-0.8b"))
    ap.add_argument("--split", default="search", choices=["test", "train", "search", "dev", "smoke"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-ref-tokens", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "proxy.jsonl"

    if args.split in ("test", "train"):
        df = pd.read_parquet(DATASET / f"{args.split}-00000-of-00001.parquet")
    else:
        df = pd.read_parquet(DATASET / "train-00000-of-00001.parquet")
    indices = load_split_indices(args.split)
    if args.limit:
        indices = indices[: args.limit]

    done = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(int(json.loads(line)["index"]))
                except Exception:
                    continue
    remaining = [i for i in indices if i not in done]
    print(f"Задач: {len(indices)}, готово: {len(done)}, осталось: {len(remaining)}", flush=True)

    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()
    tok = processor.tokenizer

    with results.open("a", encoding="utf-8", buffering=1) as fh:
        for idx in remaining:
            row = df.loc[idx]
            question = str(row["question"])
            ref_full = str(row["answer"])
            messages = [{"role": "user", "content": question + PROMPT_SUFFIX}]
            prompt_ids = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            # хвост с финальным числом: всё, начиная с последнего «####»
            cut = ref_full.rfind("####")
            ref_head = ref_full[:cut]          # рассуждение
            ref_tail = ref_full[cut:]          # "#### 42"

            head_text = ref_head if ref_head.strip() else ""
            ids_prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            ids_head = torch.tensor([tok(head_text, add_special_tokens=False)["input_ids"]],
                                    dtype=torch.long, device=device)
            ids_tail = torch.tensor([tok(ref_tail, add_special_tokens=False)["input_ids"]],
                                    dtype=torch.long, device=device)
            n_head, n_tail = ids_head.shape[1], ids_tail.shape[1]
            if n_head + n_tail > args.max_ref_tokens:
                # обрезаем рассуждение, хвост сохраняем целиком
                keep = args.max_ref_tokens - n_tail
                ids_head = ids_head[:, :keep]
                n_head = ids_head.shape[1]
            full = torch.cat([ids_prompt, ids_head, ids_tail], dim=1)
            p = ids_prompt.shape[1]
            total = full.shape[1]
            if total < 4:
                continue
            attention_mask = torch.ones_like(full)

            t0 = time.monotonic()
            with torch.inference_mode():
                out = model(input_ids=full, attention_mask=attention_mask,
                            use_cache=False)
                logits = out.logits  # [1, total, vocab]
            nll_sum = 0.0
            n_tok = 0
            # loss по токенам эталонного решения: позиция t предсказывается из t-1
            chunk = 64
            ans_logprob = 0.0
            ans_tokens = 0
            top1_ok = 0
            top1_first = None
            for start in range(p, total, chunk):
                end = min(start + chunk, total)
                lg = logits[0, start - 1 : end - 1, :].float()
                tgt = full[0, start:end]
                logp = torch.log_softmax(lg, dim=-1)
                tok_logp = logp.gather(1, tgt[:, None]).squeeze(1)
                nll_sum += float(-tok_logp.sum())
                n_tok += tgt.numel()
                # отдельный учёт хвоста с ответом
                if start >= p + n_head:
                    ans_logprob += float(tok_logp.sum())
                    ans_tokens += tgt.numel()
                    am = logp.argmax(dim=-1)
                    top1_ok += int((am == tgt).sum())
                    if top1_first is None:
                        top1_first = int(am[0])
            del logits, out
            secs = time.monotonic() - t0
            rec = {
                "index": int(idx),
                "ref_tokens_scored": n_tok,
                "nll_ref": round(nll_sum / max(1, n_tok), 5),
                "answer_logprob_per_token": round(ans_logprob / max(1, ans_tokens), 5),
                "answer_prob_per_token": round(math.exp(ans_logprob / max(1, ans_tokens)), 6),
                "answer_tokens": ans_tokens,
                "answer_top1_matches": int(top1_ok),
                "answer_top1_first_token": top1_first,
                "seconds": round(secs, 2),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if True:
                print(f"[{idx}] nll={rec['nll_ref']} ans_p={rec['answer_prob_per_token']} "
                      f"top1={rec['answer_top1_matches']}/{rec['answer_tokens']} {rec['seconds']}s",
                      flush=True)

    recs = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(recs)
    summ = {
        "n": n,
        "mean_nll_ref": round(sum(r["nll_ref"] for r in recs) / n, 5),
        "mean_answer_prob": round(sum(r["answer_prob_per_token"] for r in recs) / n, 5),
        "answer_top1_all_match": sum(1 for r in recs
                                     if r["answer_top1_matches"] == r["answer_tokens"]),
        "answer_top1_partial": sum(1 for r in recs if r["answer_top1_matches"] > 0),
    }
    (out_dir / "proxy_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print("ИТОГ:", json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
