"""
Сводка волны 6: каждый вариант сравнивается НЕЗАВИСИМО с исходной моделью и с контрольной
точкой w8, по четырём показателям, и перевороты разлагаются на «формат» и «математика».

Форматный переворот: задача перестала/начала давать извлекаемый FINAL (или упёрлась в обрыв).
Математический переворот: FINAL извлекаем в обоих прогонах, но ответ стал верным/неверным.

Только чтение results.jsonl; генерации не запускаются.
"""
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
V = ROOT / "runs/variants"
W8 = "w8_z_gate_mild_format"
REFS = {"orig": ROOT / "runs/baseline_{split}/results.jsonl", "w8": V / W8 / "{split}/results.jsonl"}
EXPECT = {"search": 250, "dev": 100}
VARIANTS = ["w25_z_headsel_damp088", "w24_fullgate_open085", "w28_fullgate_close115",
            "w29_z_total087", "w26_norm_amp093", "w27_beta_sharpen_late130", "w30_qknorm_sharp108",
            "w31_z_dominant_restore106", "w32_qknorm104_on_w8", "w33_fullgate_open_closed_heads",
            "w34_restore_plus_qknorm104"]


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
    ce = sum(1 for r in rows.values() if r.get("correct") and r.get("answer_extracted"))
    ot = [int(r.get("output_tokens") or 0) for r in rows.values()]
    return {"n": n, "correct": c, "extracted": e, "truncated": t, "correct_with_final": ce,
            "correct_without_final": c - ce, "mean_out_tokens": round(sum(ot) / max(n, 1), 1)}


def cmp(rows, base, bname):
    ids = sorted(set(rows) & set(base))
    a, b = stats({i: rows[i] for i in ids}), stats({i: base[i] for i in ids})
    fixed = [i for i in ids if not base[i].get("correct") and rows[i].get("correct")]
    broken = [i for i in ids if base[i].get("correct") and not rows[i].get("correct")]
    def cls_fix(i):
        return "format" if not base[i].get("answer_extracted") or base[i].get("truncated") else "math"
    def cls_brk(i):
        return "format" if not rows[i].get("answer_extracted") or rows[i].get("truncated") else "math"
    # «математическая» дельта: только задачи, где FINAL извлекаем в ОБОИХ прогонах
    S = [i for i in ids if rows[i].get("answer_extracted") and base[i].get("answer_extracted")]
    mv = sum(1 for i in S if rows[i].get("correct"))
    mb = sum(1 for i in S if base[i].get("correct"))
    mf = sum(1 for i in S if not base[i].get("correct") and rows[i].get("correct"))
    mbr = sum(1 for i in S if base[i].get("correct") and not rows[i].get("correct"))
    p_math = 1.0
    if mf + mbr:
        from math import comb
        n = mf + mbr; k = min(mf, mbr)
        p_math = min(1.0, 2.0 * sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n))
    fx = {"format": sum(1 for i in fixed if cls_fix(i) == "format"),
          "math": sum(1 for i in fixed if cls_fix(i) == "math")}
    br = {"format": sum(1 for i in broken if cls_brk(i) == "format"),
          "math": sum(1 for i in broken if cls_brk(i) == "math")}
    b_, f_ = len(broken), len(fixed)
    p = 1.0
    if b_ + f_:
        try:
            from math import comb, exp
            n = b_ + f_
            k = min(b_, f_)
            tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
            p = min(1.0, 2.0 * tail)
        except Exception:
            p = float("nan")
    return {"ref": bname, "n_common": len(ids), "variant": a, "base": b,
            "delta_correct": a["correct"] - b["correct"],
            "math_subset_n": len(S), "math_subset_variant": mv, "math_subset_base": mb,
            "delta_math": mv - mb, "math_fixed": mf, "math_broken": mbr, "mcnemar_p_math": round(p_math, 5),
            "delta_extracted": a["extracted"] - b["extracted"],
            "delta_truncated": a["truncated"] - b["truncated"],
            "fixed": len(fixed), "broken": len(broken), "fixed_format": fx["format"],
            "fixed_math": fx["math"], "broken_format": br["format"], "broken_math": br["math"],
            "mcnemar_p": round(p, 5), "fixed_ids": fixed, "broken_ids": broken}


def main():
    out = {}
    for split in ("search", "dev"):
        bases = {k: load(str(p).format(split=split)) for k, p in REFS.items()}
        for v in VARIANTS:
            path = V / v / split / "results.jsonl"
            if not path.exists():
                continue
            rows = load(path)
            if len(rows) != EXPECT[split]:
                continue  # прогон ещё не завершён
            out.setdefault(v, {})[split] = {"stats": stats(rows),
                                            "vs": {k: cmp(rows, bases[k], k) for k in bases}}
    # объединённая выборка отбора (search 250 + dev 100 = 350) для полностью пройденных вариантов
    refs_all = {}
    for k, p in REFS.items():
        merged = load(str(p).format(split="search"))
        merged.update(load(str(p).format(split="dev")))
        refs_all[k] = merged
    for v in VARIANTS:
        got = out.get(v, {})
        if "search" in got and "dev" in got:
            rows = load(V / v / "search" / "results.jsonl")
            rows.update(load(V / v / "dev" / "results.jsonl"))
            if len(rows) == 350:
                got["combined"] = {"stats": stats(rows),
                                   "vs": {k: cmp(rows, refs_all[k], k) for k in refs_all}}
    (ROOT / "runs/analysis/wave6_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = []
    for split in ("search", "dev", "combined"):
        lines.append(f"\n=== {split} ===")
        lines.append(f"{'вариант':<26}{'верно':>6}{'FINAL':>6}{'обр':>5}{'верно+FINAL':>12} "
                     f"{'Δorig':>7} {'p':>7} {'Δмат':>6}  {'Δw8':>6} {'p':>7} {'Δмат':>6}")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            s = d["stats"]; o = d["vs"]["orig"]; w = d["vs"]["w8"]
            lines.append(f"{v:<26}{s['correct']:>6}{s['extracted']:>6}{s['truncated']:>5}"
                         f"{s['correct_with_final']:>12} {o['delta_correct']:>7} {o['mcnemar_p']:>7}"
                         f" {o['delta_math']:>6}  {w['delta_correct']:>6} {w['mcnemar_p']:>7} {w['delta_math']:>6}")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            for k in ("orig", "w8"):
                c = d["vs"][k]
                lines.append(f"    {v:<26} vs {k:<5} только-математика (FINAL есть в обоих, n={c['math_subset_n']}): "
                             f"{c['math_subset_variant']} vs {c['math_subset_base']} "
                             f"Δ{c['delta_math']:+d} (починил {c['math_fixed']}, сломал {c['math_broken']}, "
                             f"p={c['mcnemar_p_math']})")
        lines.append("перевороты (формат/математика):")
        for v in VARIANTS:
            d = out.get(v, {}).get(split)
            if not d:
                continue
            for k in ("orig", "w8"):
                c = d["vs"][k]
                lines.append(f"  {v:<26} vs {k:<5} фикс {c['fixed']:>3} (форм {c['fixed_format']:>3}"
                             f" / мат {c['fixed_math']:>3})  слом {c['broken']:>3}"
                             f" (форм {c['broken_format']:>3} / мат {c['broken_math']:>3})")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/wave6_table.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
