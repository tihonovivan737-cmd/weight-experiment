#!/usr/bin/env python3
"""Указатель рецептов серии: имя, полоса, формулировка гипотезы, измеренный исход.

Берётся из recipes/*.json (поля name, band, hypothesis, ops) и из результатов прогона
(dev/results.jsonl, search/results.jsonl в каталоге варианта). Нужен для итогового отчёта:
что именно проверялось, с каким предсказанием и чем кончилось.

Только чтение. Вывод: runs/analysis/recipe_index.txt
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "recipes"
VARS = ROOT / "runs/variants"


def stats(name):
    out = {}
    for sp in ("dev", "search"):
        p = VARS / name / sp / "results.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        out[sp] = dict(correct=sum(1 for r in rows if r.get("correct")),
                       extracted=sum(1 for r in rows if (r.get("n_final_lines") or 0) >= 1),
                       truncated=sum(1 for r in rows if r.get("truncated")),
                       n=len(rows))
    return out


def tensors_of(recipe):
    s = set()
    for o in recipe.get("ops", []):
        for t in o.get("tensors", []):
            parts = [x for x in str(t).split(".") if not x.isdigit()]
            s.add(".".join(parts[-3:]))
    return ", ".join(sorted(s)) if s else "—"


def main():
    lines = ["Указатель рецептов: гипотеза, целевые тензоры, полоса, измеренный исход.", ""]
    n_rec = n_meas = 0
    for f in sorted(REC.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            lines.append(f"{f.stem}: файл не читается")
            continue
        name = r.get("name") or f.stem
        n_rec += 1
        st = stats(name)
        both = st.get("dev") and st.get("search")
        if st:
            n_meas += 1
        if both:
            res = (f"вместе {st['dev']['correct'] + st['search']['correct']}/"
                   f"{st['dev']['extracted'] + st['search']['extracted']}/"
                   f"{st['dev']['truncated'] + st['search']['truncated']} "
                   f"(dev {st['dev']['correct']}/{st['dev']['extracted']}/{st['dev']['truncated']}, "
                   f"search {st['search']['correct']}/{st['search']['extracted']}/{st['search']['truncated']})")
        elif st:
            sp = next(iter(st))
            res = f"только {sp}: {st[sp]['correct']}/{st[sp]['extracted']}/{st[sp]['truncated']} из {st[sp]['n']}"
        else:
            res = "не измерен (нет файлов результатов)"
        hyp = " ".join(str(r.get("hypothesis", "")).split())
        lines.append(f"### {name}")
        lines.append(f"файл: recipes/{f.name} | полоса: {r.get('band') or '—'} | тензоры: {tensors_of(r)}")
        lines.append(f"гипотеза: {hyp[:300]}{'…' if len(hyp) > 300 else ''}")
        lines.append(f"исход: {res}")
        lines.append("")
    lines.insert(2, f"Рецептов: {n_rec}, из них с результатами хотя бы одного разреза: {n_meas}.")
    lines.insert(3, "")
    text = "\n".join(lines)
    (ROOT / "runs/analysis/recipe_index.txt").write_text(text + "\n", encoding="utf-8")
    print(f"записано: runs/analysis/recipe_index.txt | рецептов {n_rec}, с результатами {n_meas}, строк {len(lines)}")


if __name__ == "__main__":
    main()
