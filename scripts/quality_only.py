#!/usr/bin/env python3
"""Качественная разбивка: сравнение только по строкам, где у ОБЕИХ моделей есть строка FINAL.

Отсекает сбои формата: остаётся только то, где обе модели дошли до строки ответа,
и значит разница объясняется арифметикой, а не оформлением.
Считается отдельно: доля верных на пересечении и парные переходы (исправлено/сломано)
внутри пересечения, с точным двусторонним критерием МакНемара.

Только чтение существующих результатов.
Использование: .venv/bin/python scripts/quality_only.py <имя варианта> [база ...]
Базы по умолчанию: исходная модель и w8.
Вывод: runs/analysis/quality_only_<имя>.txt
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("dev", "search")
DEFAULT_BASES = {
    "исходная модель": ["runs/baseline_dev/results.jsonl", "runs/baseline_search/results.jsonl"],
    "w8": ["runs/variants/w8_z_gate_mild_format/dev/results.jsonl",
           "runs/variants/w8_z_gate_mild_format/search/results.jsonl"],
    "w50": ["runs/variants/w50_finalrow_rotation15_smallmargin/dev/results.jsonl",
            "runs/variants/w50_finalrow_rotation15_smallmargin/search/results.jsonl"],
    "w83": ["runs/variants/w83_mlp_layer_lever_on_w8/dev/results.jsonl",
            "runs/variants/w83_mlp_layer_lever_on_w8/search/results.jsonl"],
}


def by_index(paths):
    """{index: row} по всем сплитам базы или варианта."""
    out = {}
    for p in paths:
        f = ROOT / p
        if not f.exists():
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[r["index"]] = r
    return out


def has_final(row):
    return (row.get("n_final_lines") or 0) >= 1


def mcnemar_exact(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)
    return p


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "w82_wave11_head_lever_on_w8"
    var_paths = [f"runs/variants/{name}/dev/results.jsonl", f"runs/variants/{name}/search/results.jsonl"]
    bases = DEFAULT_BASES if len(sys.argv) <= 2 else {a: [a] for a in sys.argv[2:]}
    var = by_index(var_paths)
    lines = [f"Качественная разбивка для {name}: только задачи, где строка FINAL есть у обеих моделей."]
    lines.append("")
    for blabel, bpaths in bases.items():
        base = by_index(bpaths)
        common = sorted(set(var) & set(base))
        inter = [i for i in common if has_final(var[i]) and has_final(base[i])]
        bc = sum(1 for i in inter if base[i]["correct"])
        vc = sum(1 for i in inter if var[i]["correct"])
        gained = [i for i in inter if var[i]["correct"] and not base[i]["correct"]]
        lost = [i for i in inter if base[i]["correct"] and not var[i]["correct"]]
        both_final = len(inter)
        p = mcnemar_exact(len(gained), len(lost))
        lines.append(f"против {blabel}:")
        lines.append(f"  общих задач {len(common)}, из них с FINAL у обеих {both_final}")
        lines.append(f"  верных на пересечении: база {bc} ({100 * bc / both_final:.1f}%), "
                     f"вариант {vc} ({100 * vc / both_final:.1f}%), Δ {vc - bc:+d}")
        lines.append(f"  парные переходы внутри пересечения: стало лучше {len(gained)}, "
                     f"стало хуже {len(lost)}, p = {p:.4f}")
        lines.append(f"  только формат (есть FINAL у варианта, нет у базы): "
                     f"{sum(1 for i in common if has_final(var[i]) and not has_final(base[i]))}, "
                     f"обратно: {sum(1 for i in common if has_final(base[i]) and not has_final(var[i]))}")
        lines.append(f"  верных по всем общим задачам: база {sum(1 for i in common if base[i]['correct'])}, "
                     f"вариант {sum(1 for i in common if var[i]['correct'])}")
        lines.append("")
    text = "\n".join(lines)
    out = ROOT / f"runs/analysis/quality_only_{name}.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
