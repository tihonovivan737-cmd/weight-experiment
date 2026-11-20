#!/usr/bin/env python3
"""Таблица кредит-гейта по всем вариантам, у которых есть оба разреза `dev` и `search`.

Правила отбора (зафиксированы в журнале):
  кредит-гейт серии — четыре условия наcombined 350 задачах и на dev:
    1) извлечение FINAL не меньше 336;
    2) математика не меньше 216;
    3) обрывы не больше 14;
    4) математика на dev не меньше 66.
  Выбор победителя: сначала большее число выполненных условий, затем большая математика
  вместе, затем меньшее число обрывов, затем большее извлечение.

Варианты без обоих разрезов (ранняя экранная эпоха: только dev или только smoke) в таблицу
не попадают — их нельзя сравнивать на одних и тех же 350 задачах; они перечислены отдельно.

Только чтение существующих результатов.
Вывод: runs/analysis/credit_table.txt
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARS = ROOT / "runs/variants"
SPLITS = ("dev", "search")


def rows_of(path: Path):
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


def stats(rows):
    return dict(
        n=len(rows),
        correct=sum(1 for r in rows if r.get("correct")),
        extracted=sum(1 for r in rows if (r.get("n_final_lines") or 0) >= 1),
        truncated=sum(1 for r in rows if r.get("truncated")),
    )


def main():
    have, missing = [], []
    for m in sorted(VARS.glob("*/manifest.json")):
        name = m.parent.name
        paths = {sp: VARS / name / sp / "results.jsonl" for sp in SPLITS}
        if all(p.exists() and p.stat().st_size > 0 for p in paths.values()):
            per = {sp: stats(rows_of(p)) for sp, p in paths.items()}
            comb = {k: sum(per[sp][k] for sp in SPLITS) for k in ("n", "correct", "extracted", "truncated")}
            checks = [comb["extracted"] >= 336, comb["correct"] >= 216,
                      comb["truncated"] <= 14, per["dev"]["correct"] >= 66]
            have.append((sum(checks), comb["correct"], -comb["truncated"], comb["extracted"],
                         name, per, comb, checks))
        else:
            missing.append(name)

    have.sort(key=lambda t: (-t[0], -t[1], t[2], -t[3]))
    out = ["Кредит-гейт: извлечение >= 336, математика >= 216, обрывы <= 14, математика dev >= 66.",
           "Порядок сортировки: выполненные условия, математика вместе, обрывы (меньше — лучше), извлечение.",
           ""]
    out.append(f"{'вариант':46s} {'условий':>8} {'вместе':>14} {'dev':>10} {'search':>13}  условия")
    for score, math, negtr, ext, name, per, comb, checks in have:
        flags = "".join("+" if c else "." for c in checks)
        out.append(f"{name:46s} {score:>6} из 4 "
                   f"{comb['correct']:>4}/{comb['extracted']}/{comb['truncated']:<4} "
                   f"{per['dev']['correct']:>4}/{per['dev']['extracted']}/{per['dev']['truncated']:<3}   "
                   f"{per['search']['correct']:>4}/{per['search']['extracted']}/{per['search']['truncated']:<3}  "
                   f"{flags}")
    out.append("")
    out.append("Условия столбца флагов по порядку: извлечение, математика, обрывы, математика dev.")
    out.append("")
    out.append(f"Вариантов с обоими разрезами: {len(have)}. Без них (сравнение невозможно): {len(missing)}")
    out.append("  " + ", ".join(missing))
    best = have[0]
    bname, bscore, bcomb = best[4], best[0], best[6]
    out.append("")
    out.append(f"Первый по правилу отбора: {bname} — условий {bscore} из 4, "
               f"вместе {bcomb['correct']}/{bcomb['extracted']}/{bcomb['truncated']}")
    text = "\n".join(out)
    (ROOT / "runs/analysis/credit_table.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
