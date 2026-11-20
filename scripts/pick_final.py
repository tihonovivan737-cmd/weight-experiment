#!/usr/bin/env python3
"""Механический выбор финального кандидата по пред-зарегистрированному правилу (журнал §14, §14.2).

Пул: все варианты в runs/variants/, у которых есть combined.json с обоими наборами (dev + search).
Критерии:
  1) суммарный Δ по 350 задачам >= MIN_DELTA (по умолчанию +5);
  2) нет значимого ухудшения ни на одном наборе: p >= 0.05 и delta >= 0 по каждому набору
     (ухудшением считается delta < 0 при p < 0.05);
Ранжирование: наибольший суммарный Δ; далее меньший суммарный p; далее меньшее число изменённых значений.

Пишет runs/selection_final.txt и (при --commit) runs/GATE_FINAL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
VAR = ROOT / "runs/variants"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-delta", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="runs/selection_final.txt")
    ap.add_argument("--gate", default="runs/GATE_FINAL")
    ap.add_argument("--commit", action="store_true", help="создать gate-файл и запустить финальный прогон нельзя автоматически")
    args = ap.parse_args()

    rows = []
    for v in sorted(VAR.iterdir()):
        if not v.is_dir():
            continue
        c = v / "combined.json"
        if not c.exists():
            continue
        d = json.loads(c.read_text(encoding="utf-8"))
        comb = d["combined"]
        per = d.get("per_split", {})
        if "dev" not in per or "search" not in per:
            continue
        man = {}
        mp = v / "manifest.json"
        if mp.exists():
            man = json.loads(mp.read_text(encoding="utf-8")).get("totals", {})
        regress = [sp for sp, s in per.items()
                   if s["delta_correct"] < 0 and s["mcnemar_p"] < args.alpha]
        rows.append({
            "variant": v.name,
            "combined_delta": comb["delta_correct"],
            "combined_p": round(comb["mcnemar_p"], 4),
            "combined_correct": comb["variant_stats"]["correct"],
            "n": comb["n_common"],
            "dev_delta": per["dev"]["delta_correct"],
            "dev_p": round(per["dev"]["mcnemar_p"], 4),
            "search_delta": per["search"]["delta_correct"],
            "search_p": round(per["search"]["mcnemar_p"], 4),
            "regressions": regress,
            "values_changed": man.get("values_actually_changed"),
            "ok": (comb["delta_correct"] >= args.min_delta and not regress),
        })

    rows.sort(key=lambda r: (-r["combined_delta"], r["combined_p"],
                             r["values_changed"] if r["values_changed"] is not None else 10**12))

    print(f"Кандидаты с полным покрытием 350 train-задач (порог Δ >= {args.min_delta}, alpha={args.alpha}):")
    print(f"{'variant':24} {'вместе':>10} {'Δ':>4} {'p':>7} {'dev Δ/p':>16} {'search Δ/p':>18} {'знач. ухудш.':>13} {'значений':>9}  вердикт")
    for r in rows:
        print(f"{r['variant']:24} {r['combined_correct']}/{r['n']:3} {r['combined_delta']:+4} {r['combined_p']:7} "
              f"{r['dev_delta']:+4}/{r['dev_p']:<7} {r['search_delta']:+4}/{r['search_p']:<7} "
              f"{','.join(r['regressions']) or '—':>13} {str(r['values_changed']):>9}  "
              f"{'ПРОЙДЕТ' if r['ok'] else 'нет'}")

    passing = [r for r in rows if r["ok"]]
    lines = []
    if passing:
        best = passing[0]
        lines.append(f"{best['variant']}   # суммарный Δ{best['combined_delta']:+d} по {best['n']} train-задачам, p={best['combined_p']}")
        print(f"\nВЫБРАН: {best['variant']}")
    else:
        if not rows:
            print("\nНет вариантов с полным покрытием обоих наборов — выбор невозможен.")
            raise SystemExit(1)
        best = rows[0]
        lines.append(f"{best['variant']}   # РАЗВЕДОЧНЫЙ: значимого улучшения нет; суммарный Δ{best['combined_delta']:+d}, p={best['combined_p']}")
        print(f"\nЗначимого улучшения нет. Разведочный кандидат: {best['variant']}")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Список: {args.out}")
    if args.commit:
        Path(args.gate).write_text("ok\n", encoding="utf-8")
        print(f"Gate: {args.gate}")


if __name__ == "__main__":
    main()
