import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

HOME = Path.home()
MODEL_DIR = HOME / "models/qwen3.5-0.8b"
DATA_FILE = (
    HOME / "weight-experiment/datasets/gsm8k/main/"
    "test-00000-of-00001.parquet"
)
OUTPUT_DIR = HOME / "weight-experiment/runs/baseline"
RESULTS_FILE = OUTPUT_DIR / "results.jsonl"
CONFIG_FILE = OUTPUT_DIR / "config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Единый формат нужен для автоматической проверки.
PROMPT_SUFFIX = (
    "\n\nSolve the problem step by step. "
    "Finish with a final line exactly in this format: "
    "FINAL: <number>"
)

MAX_NEW_TOKENS = 512


def normalize_number(value):
    value = value.strip().replace(",", "").replace("$", "")
    try:
        return float(value)
    except ValueError:
        return None


def expected_answer(text):
    # В GSM8K итоговый ответ указан после ####.
    return normalize_number(text.rsplit("####", 1)[-1])


def predicted_answer(text):
    # Считаем только явно обозначенный итоговый ответ.
    matches = re.findall(
        r"(?im)^\s*FINAL:\s*\$?(-?[\d,]+(?:\.\d+)?)",
        text,
    )
    return normalize_number(matches[-1]) if matches else None


def main():
    if not MODEL_DIR.exists():
        raise SystemExit(f"Модель не найдена: {MODEL_DIR}")
    if not DATA_FILE.exists():
        raise SystemExit(f"Датасет не найден: {DATA_FILE}")

    data = pd.read_parquet(DATA_FILE)

    completed = set()
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["index"])

    config = {
        "model": str(MODEL_DIR),
        "dataset": str(DATA_FILE),
        "device": "cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "prompt_suffix": PROMPT_SUFFIX,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "total_questions": len(data),
    }

    if CONFIG_FILE.exists():
        old_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if old_config != config:
            raise SystemExit(
                "Настройки отличаются от предыдущего запуска. "
                "Создай новую папку результатов, чтобы не смешивать прогоны."
            )
    else:
        CONFIG_FILE.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Всего задач: {len(data)}; уже обработано: {len(completed)}")
    print("Загружаю модель на CPU...", flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to("cpu")
    model.eval()

    with RESULTS_FILE.open("a", encoding="utf-8", buffering=1) as output:
        for index, row in data.iterrows():
            index = int(index)
            if index in completed:
                continue

            question = str(row["question"])
            messages = [
                {
                    "role": "user",
                    "content": question + PROMPT_SUFFIX,
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            input_length = inputs["input_ids"].shape[-1]
            answer_text = processor.decode(
                generated[0][input_length:],
                skip_special_tokens=True,
            )

            expected = expected_answer(str(row["answer"]))
            predicted = predicted_answer(answer_text)
            correct = (
                expected is not None
                and predicted is not None
                and abs(expected - predicted) < 1e-6
            )

            record = {
                "index": index,
                "question": question,
                "reference_solution": str(row["answer"]),
                "model_answer": answer_text,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "answer_extracted": predicted is not None,
                "seconds": round(time.monotonic() - started, 2),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index + 1}/{len(data)}] "
                f"{'верно' if correct else 'неверно'} | "
                f"ожидалось: {expected}, получено: {predicted} | "
                f"{record['seconds']} сек.",
                flush=True,
            )


if __name__ == "__main__":
    main()
