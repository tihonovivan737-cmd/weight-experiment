#!/usr/bin/env python3
"""Проверка гипотезы H55: теряется ли ответ на извлечении или на рассуждении.

Для каждой модели по dev (100) и search (250) считаются неверные строки, и в них —
встречается ли эталонное число в хвосте генерации. Хвост = последние TAIL символов
до строки FINAL (если FINAL есть) или последние TAIL символов всего текста (если FINAL
нет). Число ищется как отдельное вхождение: слева и справа не цифра и не точка.

Заодно делит неверные строки на две группы: без строки FINAL (сбой формата) и с
ненулевым числом извлечённых строк (рассуждение дало не тот ответ).

Только чтение существующих результатов; новых прогонов не делается.
Вывод: runs/analysis/gold_in_tail.txt
"""
import json
import re
import sys
from pathlib import Path

TAIL = 400
MODELS = {
    "исходная модель": "runs/baseline_{sp}/results.jsonl",
    "w8": "runs/variants/w8_z_gate_mild_format/{sp}/results.jsonl",
    "w50": "runs/variants/w50_finalrow_rotation15_smallmargin/{sp}/results.jsonl",
    "w78": "runs/variants/w78_finalrow_six_mid_rows_on_w8/{sp}/results.jsonl",
    "w79": "runs/variants/w79_finalrow_ten_rows_double_step_on_w8/{sp}/results.jsonl",
    "w81": "runs/variants/w81_finalrow_four_small_rows_step15_on_w8/{sp}/results.jsonl",
    "w82": "runs/variants/w82_wave11_head_lever_on_w8/{sp}/results.jsonl",
    "w83": "runs/variants/w83_mlp_layer_lever_on_w8/{sp}/results.jsonl",
}
SPLITS = ("dev", "search")


def gold_variants(expected):
    """Эталонное число в видах с запятой и без, целая часть без дробного нуля."""
    try:
        v = float(str(expected).replace(",", ""))
    except ValueError:
        return []
    out = set()
    if abs(v - round(v)) < 1e-9:
        i = int(round(v))
        out.add(str(i))
        out.add(f"{i:,}".replace(",", ","))
    else:
        out.add(str(v))
    return sorted(out)


def appears_in(text, forms):
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


def scan(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    st = dict(n=len(rows), wrong=0, wrong_no_final=0, tail_in_no_final=0,
              wrong_with_final=0, tail_in_with_final=0, tail_in_any_wrong=0)
    for r in rows:
        if r.get("correct"):
            continue
        st["wrong"] += 1
        text = r.get("model_answer") or ""
        forms = gold_variants(r.get("expected"))
        if not forms:
            continue
        lines = [ln for ln in text.splitlines() if ln.strip().upper().startswith("FINAL")]
        if lines:
            body = text[:text.rfind(lines[-1])]
            st["wrong_with_final"] += 1
            hit = appears_in(body[-TAIL:], forms)
            if hit:
                st["tail_in_with_final"] += 1
        else:
            st["wrong_no_final"] += 1
            if appears_in(text[-TAIL:], forms):
                st["tail_in_no_final"] += 1
        if appears_in(text[-TAIL:], forms):
            st["tail_in_any_wrong"] += 1
    return st


def main():
    out = []
    out.append(f"Хвост = последние {TAIL} символов до строки FINAL (или всего текста, если FINAL нет).")
    out.append("")
    out.append(f"{'модель':18s} {'сплит':8s} {'неверных':>9} {'без FINAL':>10} {'нужное число в хвосте':>24} "
               f"{'с FINAL':>8} {'в хвосте':>9} {'всего верных':>13}")
    for label, tpl in MODELS.items():
        tot = dict(wrong=0, nof=0, hit_nof=0, wf=0, hit_wf=0, n=0, correct=0)
        for sp in SPLITS:
            p = Path(tpl.format(sp=sp))
            if not p.exists():
                out.append(f"{label:18s} {sp:8s} — файла нет: {p}")
                continue
            s = scan(p)
            tot["wrong"] += s["wrong"]; tot["nof"] += s["wrong_no_final"]
            tot["hit_nof"] += s["tail_in_no_final"]; tot["wf"] += s["wrong_with_final"]
            tot["hit_wf"] += s["tail_in_with_final"]; tot["n"] += s["n"]
            tot["correct"] += s["n"] - s["wrong"]
            out.append(f"{label:18s} {sp:8s} {s['wrong']:>9} {s['wrong_no_final']:>10} "
                       f"{s['tail_in_no_final']:>24} {s['wrong_with_final']:>8} "
                       f"{s['tail_in_with_final']:>9} {s['n'] - s['wrong']:>13}")
        if tot["n"]:
            out.append(f"{label:18s} {'вместе':8s} {tot['wrong']:>9} {tot['nof']:>10} "
                       f"{tot['hit_nof']:>24} {tot['wf']:>8} {tot['hit_wf']:>9} {tot['correct']:>13}")
        out.append("")
    text = "\n".join(out)
    Path("runs/analysis/gold_in_tail.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
