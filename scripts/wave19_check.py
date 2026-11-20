#!/usr/bin/env python3
# Проверка полос П64.3 (записаны в §64.9 до сборки) для w70_qknorm110_on_w8.
# Только измеренные числа; официальные 1319 задач не используются.
import json, statistics
from pathlib import Path

V = "w70_qknorm110_on_w8"
BASES = {"orig": {"dev": "runs/baseline_dev/results.jsonl", "search": "runs/baseline_search/results.jsonl"},
         "w8":   {"dev": "runs/variants/w8_z_gate_mild_format/dev/results.jsonl",
                  "search": "runs/variants/w8_z_gate_mild_format/search/results.jsonl"}}
BANDS = {  # математика, формат, обрывы
    "dev":   {"math": (65, 68), "format": (95, 99), "trunc": (1, 6)},
    "search": {"math": (147, 152), "format": (233, 238), "trunc": (11, 14)},
    "combined": {"math": (212, 216), "format": (333, 338), "trunc": (12, 16)},
}
LEN = {"search": {"mean": 352.5, "median": 307.5}, "dev": {"mean": 321.6, "median": 275.0}}
GATE = {"combined_format": 336, "combined_math": 216, "combined_trunc": 14, "dev_math": 66}

def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l]

def stats(rs):
    n = len(rs)
    return {"n": n,
            "math": sum(1 for r in rs if r["correct"]),
            "format": sum(1 for r in rs if r["answer_extracted"]),
            "trunc": sum(1 for r in rs if r["truncated"]),
            "mean": round(statistics.mean(r["output_tokens"] for r in rs), 1),
            "median": round(statistics.median(r["output_tokens"] for r in rs), 1),
            "max": max(r["output_tokens"] for r in rs)}

out = {}
for sp in ("dev", "search"):
    f = Path(f"runs/variants/{V}/{sp}/results.jsonl")
    if not f.exists():
        print(f"{sp}: результатов нет"); continue
    v = stats(rows(f))
    if sp == "dev" and v["n"] != 100: print(f"{sp}: не 100 строк"); continue
    if sp == "search" and v["n"] != 250: print(f"{sp}: не 250 строк"); continue
    out[sp] = v
    for lab, base in BASES.items():
        b = stats(rows(base[sp]))
        print(f"{sp} против {lab}: {b['math']}/{b['format']}/{b['trunc']} -> {v['math']}/{v['format']}/{v['trunc']} "
              f"(математика {v['math']-b['math']:+d}, формат {v['format']-b['format']:+d}, обрывы {v['trunc']-b['trunc']:+d})")
    for k, band in BANDS[sp].items():
        ok = band[0] <= out[sp][k] <= band[1]
        print(f"полоса П64.3 {sp} {k} {band[0]}–{band[1]}: {out[sp][k]} — {'ВЫПОЛНЕНА' if ok else 'НЕ ВЫПОЛНЕНА'}")
    lim = LEN[sp]
    for key in ("mean", "median"):
        ok = out[sp][key] <= lim[key]
        print(f"предсказание длины {sp} {key} ≤ {lim[key]}: {out[sp][key]} — {'ВЫПОЛНЕНА' if ok else 'НЕ ВЫПОЛНЕНА'}")

if "dev" in out and "search" in out:
    comb = {"math": out["dev"]["math"] + out["search"]["math"],
            "format": out["dev"]["format"] + out["search"]["format"],
            "trunc": out["dev"]["trunc"] + out["search"]["trunc"]}
    for k, band in BANDS["combined"].items():
        ok = band[0] <= comb[k] <= band[1]
        print(f"полоса П64.3 combined {k} {band[0]}–{band[1]}: {comb[k]} — {'ВЫПОЛНЕНА' if ok else 'НЕ ВЫПОЛНЕНА'}")
    flags = {"combined_format": comb["format"] >= GATE["combined_format"],
             "combined_math": comb["math"] >= GATE["combined_math"],
             "combined_trunc": comb["trunc"] <= GATE["combined_trunc"],
             "dev_math": out["dev"]["math"] >= GATE["dev_math"]}
    print("условие кредитования:", flags, "->", "ЗАСЧИТАНО" if all(flags.values()) else "НЕ ЗАСЧИТАНО")
