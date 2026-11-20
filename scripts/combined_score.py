#!/usr/bin/env python3
"""Итоговый скор варианта по объединению двух train-наборов (dev100 + search250 = 350 задач).

Наборы не пересекаются, поэтому объединение — это просто конкатенация парных сравнений.
Для каждой задачи нужны записи baseline и варианта в обоих наборах; задачи, отсутствующие
с любой стороны, исключаются и учитываются в 'missing'.

Использование:
  .venv/bin/python scripts/combined_score.py --variant w8_z_gate_mild_format \
      --report runs/variants/w8_z_gate_mild_format/combined.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SPLITS = {
    "dev": ("runs/baseline_dev/results.jsonl", "dev"),
    "search": ("runs/baseline_search/results.jsonl", "search"),
}


def load(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["index"])] = r
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    p_low = sum(math.comb(n, k) for k in range(0, m + 1)) / (2 ** n)
    return min(1.0, 2 * p_low)


def stats(recs: list[dict]) -> dict:
    n = len(recs)
    c = sum(1 for r in recs if r["correct"])
    ex = sum(1 for r in recs if r.get("answer_extracted"))
    tr = sum(1 for r in recs if r.get("truncated"))
    toks = sum(r.get("output_tokens", 0) for r in recs)
    secs = sum(r.get("seconds", 0.0) for r in recs)
    return {
        "n": n,
        "correct": c,
        "accuracy": round(c / n, 4) if n else None,
        "extracted": ex,
        "truncated": tr,
        "mean_out_tokens": round(toks / n, 1) if n else None,
        "mean_seconds": round(secs / n, 2) if n else None,
    }


def pair(base: dict[int, dict], var: dict[int, dict]) -> dict:
    common = sorted(set(base) & set(var))
    missing = {
        "only_base": sorted(set(base) - set(var)),
        "only_variant": sorted(set(var) - set(base)),
    }
    b = [base[i] for i in common]
    v = [var[i] for i in common]
    fixed = [i for i in common if not base[i]["correct"] and var[i]["correct"]]
    broken = [i for i in common if base[i]["correct"] and not var[i]["correct"]]
    cb = sum(1 for r in b if r["correct"])
    cv = sum(1 for r in v if r["correct"])
    return {
        "n_common": len(common),
        "missing": missing,
        "base_stats": stats(b),
        "variant_stats": stats(v),
        "delta_correct": cv - cb,
        "fixed": fixed,
        "broken": broken,
        "mcnemar_p": mcnemar_exact(len(fixed), len(broken)),
        "answers_differ": sum(1 for i in common
                              if base[i].get("model_answer") != var[i].get("model_answer")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="имя варианта в runs/variants/")
    ap.add_argument("--splits", default="dev,search", help="какие наборы объединять")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    vdir = Path("runs/variants") / args.variant
    per_split: dict[str, dict] = {}
    merged_base: dict[int, dict] = {}
    merged_var: dict[int, dict] = {}

    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if sp not in SPLITS:
            raise SystemExit(f"неизвестный набор: {sp}")
        base_path = Path(SPLITS[sp][0])
        var_path = vdir / SPLITS[sp][1] / "results.jsonl"
        if not base_path.exists():
            raise SystemExit(f"нет baseline-файла: {base_path}")
        if not var_path.exists():
            print(f"{sp}: ПРОПУСК, нет {var_path}")
            continue
        base = load(base_path)
        var = load(var_path)
        per_split[sp] = pair(base, var)
        # наборы не пересекаются; смещение индексов не нужно, но храним как (sp, index)
        for i, r in base.items():
            merged_base[(sp, i)] = r
        for i, r in var.items():
            merged_var[(sp, i)] = r

    if not merged_var:
        raise SystemExit("нет ни одного прогона варианта")

    combined = pair(merged_base, merged_var)

    lines = [f"Сводка варианта {args.variant} (наборы: {args.splits})"]
    for sp, d in per_split.items():
        lines.append(
            f"  {sp:7} n={d['n_common']:4}  base {d['base_stats']['correct']:4} "
            f"({d['base_stats']['accuracy'] * 100:5.2f}%)  var {d['variant_stats']['correct']:4} "
            f"({d['variant_stats']['accuracy'] * 100:5.2f}%)  Δ{d['delta_correct']:+3}  "
            f"ф{len(d['fixed']):3}/с{len(d['broken']):3}  p={d['mcnemar_p']:.4f}  "
            f"извл {d['base_stats']['extracted']}->{d['variant_stats']['extracted']}  "
            f"обр {d['base_stats']['truncated']}->{d['variant_stats']['truncated']}  "
            f"ток {d['base_stats']['mean_out_tokens']}->{d['variant_stats']['mean_out_tokens']}"
        )
    lines.append(
        f"  ВМЕСТЕ n={combined['n_common']:4}  base {combined['base_stats']['correct']:4} "
        f"({combined['base_stats']['accuracy'] * 100:5.2f}%)  var {combined['variant_stats']['correct']:4} "
        f"({combined['variant_stats']['accuracy'] * 100:5.2f}%)  Δ{combined['delta_correct']:+3}  "
        f"ф{len(combined['fixed']):3}/с{len(combined['broken']):3}  p={combined['mcnemar_p']:.4f}  "
        f"извл {combined['base_stats']['extracted']}->{combined['variant_stats']['extracted']}  "
        f"обр {combined['base_stats']['truncated']}->{combined['variant_stats']['truncated']}  "
        f"ток {combined['base_stats']['mean_out_tokens']}->{combined['variant_stats']['mean_out_tokens']}"
    )
    if combined["missing"]["only_base"] or combined["missing"]["only_variant"]:
        lines.append(f"  ПРОПУЩЕННЫЕ ЗАДАЧИ: {combined['missing']}")
    text = "\n".join(lines)
    print(text)

    if args.report:
        Path(args.report).write_text(json.dumps({
            "variant": args.variant,
            "splits": args.splits,
            "per_split": per_split,
            "combined": combined,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Отчёт: {args.report}")


if __name__ == "__main__":
    main()
