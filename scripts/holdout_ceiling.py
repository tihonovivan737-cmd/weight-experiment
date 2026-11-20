#!/usr/bin/env python3
"""Потолок «только формата» на удержанном наборе из 400 задач.

Для каждой из трёх моделей считается:
  * верно сейчас;
  * задачи без строки `FINAL` и среди них те, где эталонное число уже написано в последних
    400 символах (то есть «посчитал, но не оформил»);
  * задачи с неверным ответом при наличии `FINAL` и среди них те, где эталонное число
    есть в хвосте ДО строки ответа (то есть «посчитал, но выбрал другое число»);
  * потолок — сколько было бы верно при идеальном выборе уже написанного числа.

Смысл: если потолок у вариантов примерно одинаков, а различается только число верных,
значит различие объясняется выбором/оформлением, а не способностью посчитать.

Только чтение существующих файлов результатов.
Вывод: runs/analysis/holdout_ceiling.txt
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAIL = 400
LABELS = [("w50_finalrow_rotation15_smallmargin", "выбранный вариант w50"),
          ("orig", "исходная модель"),
          ("w8", "w8 (контрольная точка)")]


def forms_of(expected):
    try:
        v = float(str(expected).replace(",", ""))
    except ValueError:
        return []
    if abs(v - round(v)) < 1e-9:
        i = int(round(v))
        return [str(i), f"{i:,}"]
    return [str(v)]


def appears(text, forms):
    for f in forms:
        for m in re.finditer(re.escape(f), text):
            a, b = m.start(), m.end()
            left = text[a - 1] if a else ""
            right = text[b] if b < len(text) else ""
            if left.isdigit() or left == ".":
                continue
            if right.isdigit():
                continue
            return True
    return False


def final_line_span(text):
    """Позиция последней строки, начинающейся с FINAL, или None."""
    best = -1
    for m in re.finditer(r"(?im)^([^\n]*?FINAL[^\n]*)$", text):
        best = m.start()
    return best if best >= 0 else None


def main():
    lines = [f"Удержанный набор 400 задач, хвост = последние {TAIL} символов.", ""]
    lines.append(f"{'модель':30s} {'верно':>6} {'без FINAL':>10} {'из них число в хвосте':>22} "
                 f"{'неверно с FINAL':>16} {'из них число в хвосте':>22} {'потолок':>8}")
    got = 0
    for lab, pretty in LABELS:
        p = ROOT / "runs/holdout6" / lab / "results.jsonl"
        if not p.exists():
            lines.append(f"{pretty:30s} — прогона ещё нет")
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rows)
        corr = sum(1 for r in rows if r.get("correct"))
        nofin = [r for r in rows if (r.get("n_final_lines") or 0) == 0]
        wrong_fin = [r for r in rows if not r.get("correct") and (r.get("n_final_lines") or 0) >= 1]
        hit_nof = 0
        for r in nofin:
            f = forms_of(r.get("expected"))
            if f and appears((r.get("model_answer") or "")[-TAIL:], f):
                hit_nof += 1
        hit_wf = 0
        for r in wrong_fin:
            text = r.get("model_answer") or ""
            pos = final_line_span(text)
            body = text[:pos] if pos is not None else text
            f = forms_of(r.get("expected"))
            if f and appears(body[-TAIL:], f):
                hit_wf += 1
        lines.append(f"{pretty:30s} {corr:>6} {len(nofin):>10} {hit_nof:>22} "
                     f"{len(wrong_fin):>16} {hit_wf:>22} {corr + hit_nof + hit_wf:>8}"
                     f"   (задач {n})")
        got += 1
    lines.append("")
    lines.append("Если потолок у моделей близок, а «верно» различается — различие объясняется "
                 "оформлением и выбором уже написанного числа, а не способностью посчитать.")
    text = "\n".join(lines)
    (ROOT / "runs/analysis/holdout_ceiling.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
