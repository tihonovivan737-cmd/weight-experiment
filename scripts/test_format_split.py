#!/usr/bin/env python3
"""Разбор официального набора (1319 задач) на «формат» и «качество» — только по уже
имеющимся результатам, без новых прогонов и без использования этих чисел для отбора.

Для каждой пары (база, вариант) считается:
  * доля верных целиком;
  * пересечение задач, где строка FINAL есть у обеих моделей, и доля верных на нём
    (это «чистое качество»: обе модели дошли до строки ответа);
  * сколько задач база не оформила, а вариант оформил, и наоборот («формат»);
  * парные переходы внутри пересечения и точный двусторонний критерий `McNemar`
    (информационно, поправка на множественность не применяется).

Вывод: runs/analysis/test_format_split.txt
"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "исходная модель": "runs/baseline_test/results.jsonl",
    "w8_z_gate_mild_format": "runs/variants/w8_z_gate_mild_format/test/results.jsonl",
    "w50_finalrow_rotation15_smallmargin": "runs/variants/w50_finalrow_rotation15_smallmargin/test/results.jsonl",
}
PAIRS = [("исходная модель", "w8_z_gate_mild_format"),
         ("исходная модель", "w50_finalrow_rotation15_smallmargin"),
         ("w8_z_gate_mild_format", "w50_finalrow_rotation15_smallmargin")]


def load(path):
    out = {}
    for ln in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["index"]] = r
    return out


def has_final(row):
    return (row.get("n_final_lines") or 0) >= 1


def mcnemar(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)


def overall(rows):
    n = len(rows)
    return (sum(1 for r in rows.values() if r.get("correct")),
            sum(1 for r in rows.values() if has_final(r)),
            sum(1 for r in rows.values() if r.get("truncated")),
            n)


def main():
    data = {k: load(v) for k, v in FILES.items()}
    out = ["Официальный набор 1319 задач: только уже полученные оценки, отбора не было.", ""]
    for k in FILES:
        c, e, t, n = overall(data[k])
        out.append(f"{k:40s} верно {c} ({100*c/n:.2f}%) | с FINAL {e} ({100*e/n:.2f}%) | обрывов {t}")
    out.append("")
    for bk, vk in PAIRS:
        base, var = data[bk], data[vk]
        common = sorted(set(base) & set(var))
        inter = [i for i in common if has_final(base[i]) and has_final(var[i])]
        bc = sum(1 for i in inter if base[i]["correct"])
        vc = sum(1 for i in inter if var[i]["correct"])
        gained = sum(1 for i in inter if var[i]["correct"] and not base[i]["correct"])
        lost = sum(1 for i in inter if base[i]["correct"] and not var[i]["correct"])
        fmt_only_var = [i for i in common if has_final(var[i]) and not has_final(base[i])]
        fmt_only_base = [i for i in common if has_final(base[i]) and not has_final(var[i])]
        # сколько верных появилось именно за счёт оформления
        var_correct_from_format = sum(1 for i in fmt_only_var if var[i]["correct"] and not base[i]["correct"])
        base_correct_from_format = sum(1 for i in fmt_only_base if base[i]["correct"] and not var[i]["correct"])
        out.append(f"пары: {bk} -> {vk}")
        out.append(f"  общих задач {len(common)}, с FINAL у обеих {len(inter)}")
        out.append(f"  всего верных: {sum(1 for i in common if base[i]['correct'])} -> "
                   f"{sum(1 for i in common if var[i]['correct'])}")
        out.append(f"  качество на пересечении: {bc} ({100*bc/len(inter):.1f}%) -> "
                   f"{vc} ({100*vc/len(inter):.1f}%), Δ {vc-bc:+d}")
        out.append(f"  парные переходы внутри пересечения: лучше {gained}, хуже {lost}, "
                   f"p = {mcnemar(gained, lost):.4f}")
        out.append(f"  оформил только вариант: {len(fmt_only_var)} задач, из них верных "
                   f"{var_correct_from_format}; оформила только база: {len(fmt_only_base)}, "
                   f"из них верных {base_correct_from_format}")
        out.append("")
    text = "\n".join(out)
    (ROOT / "runs/analysis/test_format_split.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
