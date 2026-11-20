#!/usr/bin/env python3
"""
Итог по пред-зарегистрированному удержанному пулу (400 ранее не виденных train-задач):
исходная модель vs контрольная точка w8 vs победитель волны 6.
Метрики: верно, FINAL извлечён, обрывы, Δмат (верно среди задач с FINAL в обоих прогонах),
разложение переворотов на форматные и математические, точный тест Макнемара.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
H = ROOT / "runs/holdout6"
SPL = json.loads((ROOT / "runs/splits/holdout6.json").read_text(encoding="utf-8"))
IDX = sorted(int(i) for i in SPL["holdout_indices"])


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["index"])] = r
    return rows


def stats(rows):
    n = len(rows)
    c = sum(1 for r in rows.values() if r.get("correct"))
    e = sum(1 for r in rows.values() if r.get("answer_extracted"))
    t = sum(1 for r in rows.values() if r.get("truncated"))
    ot = [int(r.get("output_tokens") or 0) for r in rows.values()]
    lc = [len(r.get("model_answer") or "") for r in rows.values() if r.get("correct")]
    return {"n": n, "correct": c, "extracted": e, "truncated": t,
            "mean_out_tokens": round(sum(ot) / max(n, 1), 1),
            "mean_chars_correct": round(sum(lc) / max(len(lc), 1), 1)}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(0, m + 1)) / (2 ** n))


def cmp(rows, base):
    ids = sorted(set(rows) & set(base))
    a, b = stats({i: rows[i] for i in ids}), stats({i: base[i] for i in ids})
    fixed = [i for i in ids if not base[i].get("correct") and rows[i].get("correct")]
    broken = [i for i in ids if base[i].get("correct") and not rows[i].get("correct")]
    fx_f = sum(1 for i in fixed if not base[i].get("answer_extracted") or base[i].get("truncated"))
    br_f = sum(1 for i in broken if not rows[i].get("answer_extracted") or rows[i].get("truncated"))
    S = [i for i in ids if rows[i].get("answer_extracted") and base[i].get("answer_extracted")]
    mv = sum(1 for i in S if rows[i].get("correct"))
    mb = sum(1 for i in S if base[i].get("correct"))
    mf = sum(1 for i in S if not base[i].get("correct") and rows[i].get("correct"))
    mbr = sum(1 for i in S if base[i].get("correct") and not rows[i].get("correct"))
    return {"n_common": len(ids), "variant": a, "base": b,
            "delta_correct": a["correct"] - b["correct"],
            "delta_extracted": a["extracted"] - b["extracted"],
            "delta_truncated": a["truncated"] - b["truncated"],
            "fixed": len(fixed), "broken": len(broken),
            "fixed_format": fx_f, "fixed_math": len(fixed) - fx_f,
            "broken_format": br_f, "broken_math": len(broken) - br_f,
            "math_subset_n": len(S), "math_subset_variant": mv, "math_subset_base": mb,
            "delta_math": mv - mb, "mcnemar_p": round(mcnemar(len(fixed), len(broken)), 5),
            "mcnemar_p_math": round(mcnemar(mf, mbr), 5),
            "fixed_ids": fixed, "broken_ids": broken}


def main():
    tags = ["orig", "w8"]
    win = (ROOT / "runs/selection_wave6.txt").read_text(encoding="utf-8").strip().split()[0]
    if win != "w8_z_gate_mild_format":
        tags.append(win)
    runs = {}
    for t in tags:
        p = H / t / "results.jsonl"
        if not p.exists():
            print(f"НЕТ ПРОГОНА: {p}")
            return 1
        runs[t] = load(p)
    for t, r in runs.items():
        ids = sorted(r)
        hs = hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest()
        ok = ids == IDX and hs == SPL["indices_sha256"]
        print(f"{t:<28} задач {len(ids)}  совпадает с удержанным пулом: {ok}  (sha {hs[:12]})")
        if not ok:
            print(f"  ОШИБКА: прогон {t} не совпадает с пред-зарегистрированным пулом")
            return 1

    pairs = [("w8", "orig")] + [(w, "orig") for w in tags[2:]] + [(w, "w8") for w in tags[2:]]
    out = {"pool": {"name": SPL["name"], "n": len(IDX), "seed": SPL["seed"],
                    "indices_sha256": SPL["indices_sha256"]},
           "stats": {t: stats(r) for t, r in runs.items()}, "pairs": {}}
    lines = ["", "Сводка по удержанному пулу (400 задач, не участвовавших в отборе):",
             f"{'модель':<28}{'верно':>7}{'доля':>8}{'FINAL':>7}{'обрывы':>8}{'токенов':>9}{'ср. длина верных':>17}"]
    for t in tags:
        s = stats(runs[t])
        lines.append(f"{t:<28}{s['correct']:>7}{100*s['correct']/s['n']:>7.2f}%{s['extracted']:>7}"
                     f"{s['truncated']:>8}{s['mean_out_tokens']:>9}{s['mean_chars_correct']:>17}")
    for v, b in pairs:
        c = cmp(runs[v], runs[b])
        out["pairs"][f"{v}_vs_{b}"] = c
        lines.append("")
        lines.append(f"{v} против {b}: верно {c['variant']['correct']} vs {c['base']['correct']} "
                     f"(Δ{c['delta_correct']:+d}, Макнемар p={c['mcnemar_p']})")
        lines.append(f"  FINAL извлечён {c['variant']['extracted']} vs {c['base']['extracted']} "
                     f"(Δ{c['delta_extracted']:+d}); обрывов {c['variant']['truncated']} vs "
                     f"{c['base']['truncated']} (Δ{c['delta_truncated']:+d})")
        lines.append(f"  математика (FINAL в обоих прогонах, n={c['math_subset_n']}): "
                     f"{c['math_subset_variant']} vs {c['math_subset_base']} "
                     f"(Δ{c['delta_math']:+d}, Макнемар по математике p={c['mcnemar_p_math']})")
        lines.append(f"  исправлено {c['fixed']} (формат {c['fixed_format']}, математика {c['fixed_math']}); "
                     f"сломано {c['broken']} (формат {c['broken_format']}, математика {c['broken_math']})")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/analysis/holdout6_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "runs/holdout6_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
