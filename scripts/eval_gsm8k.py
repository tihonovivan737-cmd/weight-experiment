"""Оценка модели на GSM8K с возобновляемым прогоном и полной фиксацией метаданных.

Заменения относительно scripts/gsm8k_baseline.py (исходный скрипт не изменён):
  1. Модель по умолчанию берётся из копии внутри проекта
     (models/qwen3.5-0.8b), а не из /home/mpiscil/models — чтобы прогон был
     самовоспроизводим внутри проекта. Копия сверена с оригиналом по sha256.
  2. dtype по умолчанию bfloat16 вместо float32: в 2 раза меньше памяти и
     быстрее на CPU (промеры зафиксированы в runs/env_probe).
  3. Добавлено: подбор устройства, число потоков, батчинг, подсчёт токенов,
     признак обрыва генерации (truncated), классификация ошибок извлечения,
     summary-файл, хеши модели/датасета, версии всех библиотек.
  4. Промпт и метрика совпадут с исходным скриптом (тот же PROMPT_SUFFIX,
     тот же regex извлечения, сравнение с допуском 1e-6), чтобы результаты
     были сопоставимы.

Результаты дописываются в results.jsonl после каждой задачи: прогон можно
прервать и продолжить той же командой.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path("/home/mpiscil/weight-experiment")
DATASET = ROOT / "datasets/gsm8k/main"
SPLITS_FILE = ROOT / "runs/splits/splits.json"

PROMPT_SUFFIX = (
    "\n\nSolve the problem step by step. "
    "Finish with a final line exactly in this format: "
    "FINAL: <number>"
)


# ------------------------------------------------------------------ утилиты


def sha256_file(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_number(value):
    value = value.strip().replace(",", "").replace("$", "")
    try:
        return float(value)
    except ValueError:
        return None


def expected_answer(text):
    return normalize_number(text.rsplit("####", 1)[-1])


FINAL_RE = re.compile(r"(?im)^\s*FINAL:\s*(.+?)\s*$")


def predicted_answer(text):
    """Как в исходном скрипте: берём последнюю строку FINAL: <число>.

    Дополнительно возвращаем диагноз, если извлечь не удалось.
    """
    matches = re.findall(r"(?im)^\s*FINAL:\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return normalize_number(matches[-1]), None, len(matches)
    loose = FINAL_RE.findall(text)
    if loose:
        return None, "final_line_not_number", len(loose)
    return None, "no_final_line", 0


def env_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tokenizers": transformers.__version__ and __import__("tokenizers").__version__,
    }
    try:
        import safetensors

        versions["safetensors"] = safetensors.__version__
    except Exception:
        pass
    versions["omp_num_threads"] = torch.get_num_threads()
    versions["cpu_count"] = os.cpu_count()
    versions["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            versions["gpu_names"] = [
                torch.cuda.get_device_properties(i).name
                for i in range(torch.cuda.device_count())
            ]
        except Exception:
            pass
    return versions


def pick_device(requested: str, model_dir: Path) -> str:
    """GPU доступен только если на нём реально есть свободные >4 ГБ."""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            if free > 4 * 1024**3:
                print(f"[device] cuda:{i} свободная VRAM {free/1e9:.2f} ГБ из {total/1e9:.2f}")
                return f"cuda:{i}"
            print(f"[device] cuda:{i} свободно только {free/1e9:.2f} ГБ — пропускаем")
    except Exception as exc:  # pragma: no cover
        print(f"[device] проверка VRAM не удалась: {exc}")
    return "cpu"


def cuda_mem(device: str) -> dict:
    """Снимок памяти CUDA: свободная VRAM всего GPU (видит и чужие процессы) + резерв torch."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    idx = int(device.split(":")[-1]) if ":" in device else 0
    free, total = torch.cuda.mem_get_info(idx)
    props = torch.cuda.get_device_properties(idx)
    return {
        "device_index": idx,
        "device_name": props.name,
        "gpu_total_mib": round(props.total_memory / 2**20),
        "gpu_free_mib": round(free / 2**20),
        "gpu_used_by_others_mib": round((total - free) / 2**20),
        "torch_allocated_mib": round(torch.cuda.memory_allocated(idx) / 2**20),
        "torch_reserved_mib": round(torch.cuda.memory_reserved(idx) / 2**20),
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated(idx) / 2**20),
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved(idx) / 2**20),
    }


def load_split_indices(args) -> list[int]:
    if args.split == "test":
        df = pd.read_parquet(DATASET / "test-00000-of-00001.parquet")
        return list(range(len(df)))
    if args.split == "train":
        df = pd.read_parquet(DATASET / "train-00000-of-00001.parquet")
        return list(range(len(df)))
    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    if args.split in ("search", "dev", "smoke"):
        return list(splits[f"{args.split}_indices"])
    if args.split == "custom":
        return parse_indices(args.indices)
    raise SystemExit(f"Неизвестный split: {args.split}")


def parse_indices(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def model_dir_info(model_dir: Path) -> dict:
    info = {"model_dir": str(model_dir)}
    cfg = model_dir / "config.json"
    if cfg.exists():
        info["config_sha256"] = sha256_file(cfg)
    st = sorted(model_dir.glob("*.safetensors")) + sorted(
        model_dir.glob("model.safetensors-*of*.safetensors")
    )
    seen = set()
    weights = {}
    for p in st:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        weights[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    info["weights"] = weights
    return info


# ------------------------------------------------------------------ вывод


def read_done(results_file: Path) -> dict:
    done = {}
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # оборванная строка после падения — игнорируем
                done[rec["index"]] = rec
    return done


def write_summary(out_dir: Path, done: dict, total: int, config: dict):
    recs = list(done.values())
    n = len(recs)
    correct = sum(1 for r in recs if r["correct"])
    extracted = sum(1 for r in recs if r["answer_extracted"])
    trunc = sum(1 for r in recs if r.get("truncated"))
    secs = sum(r.get("seconds", 0.0) for r in recs)
    toks = sum(r.get("output_tokens", 0) for r in recs)
    diag = {}
    for r in recs:
        d = r.get("extraction_error")
        if d:
            diag[d] = diag.get(d, 0) + 1
    summary = {
        "completed": n,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "answer_extracted": extracted,
        "extraction_rate": round(extracted / n, 4) if n else None,
        "truncated": trunc,
        "total_seconds": round(secs, 1),
        "mean_seconds": round(secs / n, 2) if n else None,
        "mean_output_tokens": round(toks / n, 1) if n else None,
        "tokens_per_second": round(toks / secs, 2) if secs else None,
        "extraction_errors": diag,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "config": config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


# ------------------------------------------------------------------ генерация


def build_inputs(processor, question: str) -> dict:
    messages = [{"role": "user", "content": question + PROMPT_SUFFIX}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v for k, v in inputs.items()}


def pad_batch(items: list[dict], pad_id: int) -> dict:
    """Лев паддинг: новые токены всегда в конце строки."""
    max_len = max(int(d["input_ids"].shape[-1]) for d in items)
    keys = set()
    for d in items:
        keys.update(d.keys())
    batch = {}
    for k in keys:
        tensors = [d[k] for d in items]
        first = tensors[0]
        if first.dtype == torch.long and k in ("input_ids", "attention_mask"):
            fill = pad_id if k == "input_ids" else 0
            padded = []
            for t in tensors:
                pad = torch.full(
                    (int(first.shape[0]), max_len - int(t.shape[-1])),
                    fill,
                    dtype=t.dtype,
                )
                padded.append(torch.cat([pad, t], dim=-1))
            batch[k] = torch.cat(padded, dim=0)
        else:
            # прочие ключи (mm_token_type_ids и т.п.) pad'им нулями по той же схеме
            padded = []
            for t in tensors:
                if t.shape[-1] == max_len:
                    padded.append(t)
                else:
                    pad = torch.zeros(
                        *t.shape[:-1], max_len - int(t.shape[-1]), dtype=t.dtype
                    )
                    padded.append(torch.cat([pad, t], dim=-1))
            batch[k] = torch.cat(padded, dim=0)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/qwen3.5-0.8b"))
    ap.add_argument("--split", default="test",
                    choices=["test", "train", "search", "dev", "smoke", "custom"])
    ap.add_argument("--indices", default="", help="для --split custom: '0-9,20'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    model_dir = Path(args.model)
    if not model_dir.exists():
        raise SystemExit(f"Модель не найдена: {model_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"

    indices = load_split_indices(args)
    if args.split in ("test", "train"):
        df = pd.read_parquet(DATASET / f"{args.split}-00000-of-00001.parquet")
    else:
        df = pd.read_parquet(DATASET / "train-00000-of-00001.parquet")
    if args.limit:
        indices = indices[: args.limit]

    device = pick_device(args.device, model_dir)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    config = {
        "model_dir": str(model_dir),
        "model_info": model_dir_info(model_dir),
        "dataset_dir": str(DATASET),
        "split": args.split,
        "n_indices": len(indices),
        "indices_sha256": hashlib.sha256(
            ",".join(map(str, indices)).encode()
        ).hexdigest(),
        "device": device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_threads": torch.get_num_threads(),
        "prompt_suffix": PROMPT_SUFFIX,
        "note": args.note,
        "env": env_versions(),
    }

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        comparable = {k: old.get(k) for k in config}
        if comparable != config:
            raise SystemExit(
                "Настройки отличаются от предыдущего прогона в этой папке. "
                "Создай новую папку, чтобы не смешивать прогоны."
            )
    else:
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    done = read_done(results_file)
    remaining = [i for i in indices if i not in done]
    print(
        f"Задач в срезе: {len(indices)}; уже сделано: {len(done)}; осталось: {len(remaining)}",
        flush=True,
    )
    write_summary(out_dir, done, len(indices), config)
    if not remaining:
        print("Все задачи уже выполнены.")
        return

    print(f"Загружаю модель ({args.dtype}) на {device} ...", flush=True)
    t_load = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()
    load_time = time.monotonic() - t_load
    mem_after_load = cuda_mem(device)
    print(
        f"Модель загружена за {load_time:.1f} с | VRAM: "
        + json.dumps(mem_after_load, ensure_ascii=False),
        flush=True,
    )

    def save_runtime():
        runtime = {
            "device": device,
            "load_seconds": round(load_time, 1),
            "after_model_load": mem_after_load,
            "peak": cuda_mem(device),
            "env_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        (out_dir / "runtime.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save_runtime()

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    bs = max(1, args.batch_size)
    with results_file.open("a", encoding="utf-8", buffering=1) as output:
        for start in range(0, len(remaining), bs):
            chunk = remaining[start : start + bs]
            prepared = []
            for index in chunk:
                row = df.loc[index]
                prepared.append(
                    {
                        "index": int(index),
                        "question": str(row["question"]),
                        "reference": str(row["answer"]),
                        "inputs": build_inputs(processor, str(row["question"])),
                    }
                )
            batch = pad_batch([p["inputs"] for p in prepared], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            input_len = int(batch["input_ids"].shape[-1])

            started = time.monotonic()
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
            except torch.cuda.OutOfMemoryError:
                save_runtime()
                raise SystemExit(
                    f"OOM на {device}: не хватает VRAM. Прогон приостановлен; "
                    f"свободная память: {cuda_mem(device)}. "
                    "Продолжи той же командой после освобождения памяти."
                )
            elapsed = time.monotonic() - started

            for pos, p in enumerate(prepared):
                new_tokens = generated[pos][input_len:]
                answer_text = processor.decode(new_tokens, skip_special_tokens=True)
                n_new = int(new_tokens.shape[0])
                # обрыв = генерация дошла до лимита и не завершилась EOS
                eos_ids = {processor.tokenizer.eos_token_id}
                if getattr(processor.tokenizer, "im_end_token_id", None) is not None:
                    eos_ids.add(processor.tokenizer.im_end_token_id)
                truncated = n_new >= args.max_new_tokens and (
                    int(new_tokens[-1]) not in {e for e in eos_ids if e is not None}
                )
                expected = expected_answer(p["reference"])
                predicted, extraction_error, n_final = predicted_answer(answer_text)
                correct = (
                    expected is not None
                    and predicted is not None
                    and abs(expected - predicted) < 1e-6
                )
                record = {
                    "index": p["index"],
                    "question": p["question"],
                    "reference_solution": p["reference"],
                    "model_answer": answer_text,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    "answer_extracted": predicted is not None,
                    "extraction_error": extraction_error,
                    "n_final_lines": n_final,
                    "input_tokens": input_len,
                    "output_tokens": n_new,
                    "truncated": truncated,
                    "seconds": round(elapsed / len(prepared), 2),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[p["index"]] = record

            summary = write_summary(out_dir, done, len(indices), config)
            print(
                f"[{summary['completed']}/{len(indices)}] верно {summary['correct']} "
                f"({100*summary['accuracy']:.1f}%) | извлечено {summary['answer_extracted']} "
                f"| обрывов {summary['truncated']} | {elapsed/len(prepared):.2f} с/задача",
                flush=True,
            )
            if summary["completed"] % 25 == 0:
                save_runtime()

    save_runtime()
    print("Готово.", flush=True)


if __name__ == "__main__":
    main()
