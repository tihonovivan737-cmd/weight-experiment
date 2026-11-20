"""Парное сравнение двух прогонов по одинаковому набору индексов.

Выводит: accuracy обоих, число исправленных/сломанных задач, точный тест Макнемара
(двусторонний биномиальный), статистику извлечения ответов и обрывов, примеры
изменённых случаев. Ничего не меняет на диске, кроме опционального --report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(r["index"])] = r
    return out


def stats(recs: list[dict]) -> dict:
    n = len(recs)
    c = sum(1 for r in recs if r["correct"])
    ex = sum(1 for r in recs if r.get("answer_extracted"))
    tr = sum(1 for r in recs if r.get("truncated"))
    secs = sum(r.get("seconds", 0) for r in recs)
    toks = sum(r.get("output_tokens", 0) for r in recs)
    return {
        "n": n, "correct": c, "accuracy": round(c / n, 4) if n else None,
        "extracted": ex, "truncated": tr,
        "mean_out_tokens": round(toks / n, 1) if n else None,
        "mean_seconds": round(secs / n, 2) if n else None,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Точный двусторонний тест Макнемара. b: стало верно, c: стало неверно."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    # P(X <= m) для Binomial(n, 0.5)
    p_low = sum(math.comb(n, k) for k in range(0, m + 1)) / (2 ** n)
    return min(1.0, 2 * p_low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="results.jsonl базового прогона")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--base_name", default="baseline")
    ap.add_argument("--variant_name", default="variant")
    ap.add_argument("--show", type=int, default=12, help="сколько изменённых примеров показать")
    ap.add_argument("--report", default=None, help="куда сохранить json-отчёт")
    args = ap.parse_args()

    A = load(Path(args.base))
    B = load(Path(args.variant))
    common = sorted(set(A) & set(B))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    if not common:
        raise SystemExit("Нет общих индексов — сравнивать нечего")

    ra = [A[i] for i in common]
    rb = [B[i] for i in common]
    sa, sb = stats(ra), stats(rb)

    fixed = [i for i in common if not A[i]["correct"] and B[i]["correct"]]
    broken = [i for i in common if A[i]["correct"] and not B[i]["correct"]]
    same_ok = sum(1 for i in common if A[i]["correct"] and B[i]["correct"])
    same_bad = sum(1 for i in common if not A[i]["correct"] and not B[i]["correct"])
    text_changed = sum(1 for i in common if A[i]["model_answer"] != B[i]["model_answer"])
    p = mcnemar_exact(len(fixed), len(broken))

    print(f"Сравнение: {args.base_name} -> {args.variant_name} (общих задач: {len(common)})")
    if only_a or only_b:
        print(f"  Внимание: только в base {len(only_a)}, только в variant {len(only_b)}")
    for label, s in ((args.base_name, sa), (args.variant_name, sb)):
        print(f"  {label:<22} acc={s['correct']}/{s['n']} ({100*s['accuracy']:.2f}%) "
              f"извлечено={s['extracted']} обрывов={s['truncated']} "
              f"токенов={s['mean_out_tokens']} сек/задача={s['mean_seconds']}")
    d = sb["correct"] - sa["correct"]
    print(f"  Дельта: {d:+d} задач ({100*(sb['accuracy']-sa['accuracy']):+.2f} п.п.)")
    print(f"  Исправлено: {len(fixed)}  Сломано: {len(broken)}  "
          f"Верно в обоих: {same_ok}  Неверно в обоих: {same_bad}")
    print(f"  Точный тест Макнемара: p = {p:.4g} "
          f"({'значимо' if p < 0.05 else 'не значимо'} на уровне 0.05)")
    print(f"  Ответы модели различаются хотя бы в одном токене: {text_changed}/{len(common)}")

    def brief(i: int, old: dict, new: dict) -> str:
        o, n = old[i], new[i]
        return (f"idx {i}: exp={o['expected']} pred_был={o['predicted']} pred_стал={n['predicted']} "
                f"trunc_стал={n.get('truncated')} err_стал={n.get('extraction_error')} "
                f"tok={o.get('output_tokens')}->{n.get('output_tokens')} | {o['question'][:90]}")

    if fixed:
        print("\n  --- Исправлено (было неверно -> стало верно) ---")
        for i in fixed[: args.show]:
            print("   ", brief(i, A, B))
    if broken:
        print("\n  --- Сломано (было верно -> стало неверно) ---")
        for i in broken[: args.show]:
            print("   ", brief(i, A, B))

    if args.report:
        Path(args.report).write_text(json.dumps({
            "base": args.base, "variant": args.variant,
            "base_name": args.base_name, "variant_name": args.variant_name,
            "n_common": len(common), "base_stats": sa, "variant_stats": sb,
            "delta_correct": d, "fixed": fixed, "broken": broken,
            "same_correct": same_ok, "same_wrong": same_bad,
            "answers_differ": text_changed, "mcnemar_p": p,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nОтчёт: {args.report}")


if __name__ == "__main__":
    main()
