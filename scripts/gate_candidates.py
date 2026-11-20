#!/usr/bin/env python3
"""Отбор кандидатов для следующего этапа по метрикам экранных прогонов.

Правило отбора (фиксируется ДО просмотра результатов следующего этапа):
  1. берём только варианты с уже посчитанным compare_dev.json (dev100 = 100 задач train);
  2. метрика отбора = delta_correct (accuracy по фиксированному правилу извлечения FINAL),
     потому что именно accuracy является целевой метрикой; метрики формата
     (extracted/truncated/токены) выводятся как диагностика и на отбор не влияют;
  3. вариант проходит, если delta_correct >= --min-delta;
  4. сортировка: delta_correct по убыванию, при равенстве -- меньший p Макнемара;
  5. не более --max вариантов;
  6. варианты из --include добавляются отдельно (шумовые/идентичные референсы),
     даже если они не проходят порог.

Никакие результаты теста (1319 задач) в отборе не участвуют.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="runs/baseline_dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--min-delta", type=int, default=2)
    ap.add_argument("--include", nargs="*", default=[],
                    help="варианты, добавляемые как референсы независимо от порога")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    rows = []
    for cmp_path in sorted(Path(args.baseline).parent.glob("variants/*/compare_dev.json")):
        name = cmp_path.parent.name
        d = json.loads(cmp_path.read_text(encoding="utf-8"))
        rows.append({
            "variant": name,
            "delta": int(d["delta_correct"]),
            "p": float(d["mcnemar_p"]),
            "fixed": len(d["fixed"]),
            "broken": len(d["broken"]),
            "acc": d["variant_stats"]["accuracy"],
            "base_acc": d["base_stats"]["accuracy"],
            "extracted": d["variant_stats"]["extracted"],
            "base_extracted": d["base_stats"]["extracted"],
            "truncated": d["variant_stats"]["truncated"],
            "base_truncated": d["base_stats"]["truncated"],
            "mean_tokens": round(d["variant_stats"]["mean_out_tokens"], 1),
            "answers_differ": d["answers_differ"],
        })
    if not rows:
        print("нет compare_dev.json ни у одного варианта")
        return 1

    rows.sort(key=lambda r: (-r["delta"], r["p"]))
    base_acc = rows[0]["base_acc"]
    print(f"baseline {args.baseline}: {base_acc * 100:.1f}% (сравнение по 100 общим задачам)")
    print(f"{'вариант':26}{'Δ':>4}{'фикс':>5}{'слом':>5}{'p':>9}{'извл':>6}{'обрыв':>6}{'ток':>7}{'отл.отв':>8}  отбор")
    selected: list[str] = []
    for r in rows:
        ok = r["delta"] >= args.min_delta and len(selected) < args.max
        tag = "ДА" if ok else "-"
        if ok:
            selected.append(r["variant"])
        print(f"{r['variant']:26}{r['delta']:>+4}{r['fixed']:>5}{r['broken']:>5}{r['p']:>9.4f}"
              f"{r['extracted']:>6}{r['truncated']:>6}{r['mean_tokens']:>7}{r['answers_differ']:>8}  {tag}")

    forced = [v for v in args.include if v not in selected]
    for v in forced:
        selected.append(v)
        print(f"{v:26}{'':>4}{'':>5}{'':>5}{'':>9}{'':>6}{'':>6}{'':>7}{'':>8}  ДОБАВЛЕН (референс)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Список вариантов для следующего этапа.",
        f"# Отбор по dev100 (train): delta_correct >= {args.min_delta}, максимум {args.max}, "
        "сортировка по Δ (при равенстве — по p Макнемара).",
        "# Метрики формата приведены как диагностика и на отбор не влияли.",
        "# Результаты теста 1319 в отборе не использовались.",
    ]
    if args.note:
        lines.append(f"# {args.note}")
    by_name = {r["variant"]: r for r in rows}
    for v in selected:
        if v in by_name:
            r = by_name[v]
            lines.append(f"{v}  # dev100: {r['acc'] * 100:.0f}% (база {r['base_acc'] * 100:.0f}%), Δ{r['delta']:+d}, p={r['p']:.3f}, "
                         f"извл {r['extracted']}/{r['base_extracted']}, обрывов {r['truncated']}/{r['base_truncated']}")
        else:
            lines.append(f"{v}  # референс вне порога отбора")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Список:", out)
    print("Отобрано:", ", ".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
