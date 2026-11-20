"""Таксономия ошибок прогона GSM8K (только чтение results.jsonl).

Классы:
  extracted_wrong        — FINAL извлечён, число неверное
  no_final_line          — строки FINAL нет (часто обрыв генерации)
  final_not_number       — FINAL есть, но не число
  truncated              — генерация упёрлась в лимит токенов (пересекается с классами выше)
Дополнительные признаки для extracted_wrong:
  - предсказанное число встречается в эталонном решении (промежуточная величина вместо финальной);
  - предсказанное = эталонное * k, k в {2,3,4,5,10,100} или /k (типичные потери множителя);
  - предсказанное = эталонное + сумма каких-то чисел вопроса (пропуск шага) — грубая эвристика;
  - наличие повторного цикла (троекратное повтор одного и того же отрезка).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def numbers_of(text: str) -> list[float]:
    out = []
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        v = m.replace(",", "")
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def has_repetition(text: str, min_len: int = 40, times: int = 3) -> bool:
    """Грубый детектор зацикливания: один и тот же отрезок повторяется >= times раз."""
    low = re.sub(r"\s+", " ", text).strip()
    n = len(low)
    if n < min_len * times:
        return False
    # ищем самый частый подстроковый «цикл» из набора окон
    seen = Counter()
    step = max(1, n // 400)
    for L in (60, 120, 200):
        for i in range(0, n - L, step):
            seen[low[i : i + L]] += 1
    for seg, c in seen.most_common(20):
        if c >= times:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results.jsonl прогона")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    recs = []
    for line in Path(args.results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if args.limit:
        recs = recs[: args.limit]
    n = len(recs)
    if not n:
        raise SystemExit("пусто")

    cls = Counter()
    feats = Counter()
    wrong_examples = []
    for r in recs:
        exp, pred = r["expected"], r["predicted"]
        trunc = bool(r.get("truncated"))
        if pred is None:
            cls[r.get("extraction_error") or "no_prediction"] += 1
        elif exp is not None and abs(exp - pred) < 1e-6:
            cls["correct"] += 1
        else:
            cls["extracted_wrong"] += 1
            ref = r["reference_solution"]
            refnums = numbers_of(ref)
            qnums = numbers_of(r["question"])
            if pred in refnums:
                feats["pred_is_intermediate_in_reference"] += 1
            hit = None
            for k in (2, 3, 4, 5, 6, 10, 100, 0.5, 0.25, 0.1):
                if exp and abs(pred - exp * k) < 1e-6:
                    hit = f"pred=expected*{k}"
                    break
            if hit:
                feats[hit] += 1
            if any(abs(pred - (exp + q)) < 1e-6 for q in qnums):
                feats["pred=expected+num_from_question"] += 1
            if len(str(int(pred)) if float(pred).is_integer() else str(pred)) != \
               len(str(int(exp)) if float(exp).is_integer() else str(exp)):
                feats["different_digit_length"] += 1
            wrong_examples.append(r)
        if trunc:
            feats["truncated"] += 1
            cls["  (из них truncated)"] += 1
        if has_repetition(r["model_answer"]):
            feats["repetition_loop"] += 1

    print(f"Проверено записей: {n}")
    for k, v in cls.most_common():
        print(f"  {k:<28} {v:>5}  ({100*v/n:.1f}%)")
    print("Признаки:")
    for k, v in feats.most_common():
        print(f"  {k:<34} {v:>5}")

    print("\nПримеры extracted_wrong:")
    for r in wrong_examples[: args.show]:
        refnums = numbers_of(r["reference_solution"])
        print(f"  idx {r['index']}: expected={r['expected']} pred={r['predicted']} "
              f"trunc={r.get('truncated')} pred_in_ref={r['predicted'] in refnums} "
              f"| {r['question'][:80]}")

    # сводка по длине ответов
    toks = [r.get("output_tokens", 0) for r in recs]
    print(f"\noutput_tokens: mean={sum(toks)/n:.1f} max={max(toks)} "
          f"на_limit={sum(1 for t in toks if t >= 1024)}")


if __name__ == "__main__":
    main()
