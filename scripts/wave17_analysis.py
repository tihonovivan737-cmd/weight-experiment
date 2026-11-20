#!/usr/bin/env python3
"""Волна 17 (§63.6): сводка для w68 по полосам П63.1.

Четыре вида чисел:
  A. все задачи: math (верные), format (извлечённые), trunc, средняя и медиана длины;
  B. только неверные задачи: количество, средняя и медиана длины;
  C. восстановительные множества по определениям §62.4:
     O — оригинал, W — w8, X — вариант; U = {W неверна}, G = {W верна};
     Wb = {O верна и W неверна}, Wg = {O неверна и W верна};
     A = {X верна и W неверна}, B = {W верна и X неверна};
     p — гипергеометрическая «k или больше» для A внутри Wb на U
     и для B внутри Wg на G;
  D. решающий тест направленности (§63.7): A делится на A_m = A и Wb и
     A_n = A и (U без Wb); B делится на B_m = B и Wg и B_n = B и (G без Wg).
     Вариант, который отменяет правку w8, должен собирать победы в Wb и поражения
     в Wg, то есть A_n и B_n около ожидания или меньше. Вариант, который усиливает
     правку w8, должен собирать победы вне Wb и поражения вне Wg, то есть
     A_n и B_n выше ожидания со значимым p. Прежняя формулировка «A внутри Wg»
     дает тождественно пустое пересечение и не используется.
Полосы записаны до сборки и до прогона (§63.6); файл ничего не запускает.
"""
import json, statistics, sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
V = sys.argv[1] if len(sys.argv) > 1 else "w68_inproj_a_damp090_on_w8"

BANDS = {
    "search": {"math": (146, 154), "format": (232, 240), "trunc": (10, 16)},
    "dev": {"math": (62, 68), "format": (93, 98), "trunc": (2, 6)},
}
COMBINED = {"math": (208, 222), "format": (325, 338), "trunc": (12, 22)}
CREDIT = {"combined_format_min": 336, "combined_math_min": 216, "trunc_max": 14, "dev_math_min": 66}
LEN_PRED = {"search_mean_max": 352.5, "dev_mean_max": 321.6, "search_median_max": 307.5}
BASE_LEN = {"search": {"w8_mean": 352.5}, "dev": {"w8_mean": 321.6}}

def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def stats(rs):
    n = len(rs)
    ok = sum(1 for r in rs if r.get("correct"))
    ex = sum(1 for r in rs if r.get("answer_extracted"))
    tr = sum(1 for r in rs if r.get("truncated"))
    lens = [int(r.get("output_tokens", 0)) for r in rs]
    wrong = [int(r.get("output_tokens", 0)) for r in rs if not r.get("correct")]
    return {
        "n": n, "math": ok, "format": ex, "trunc": tr,
        "mean": round(sum(lens) / n, 1) if n else 0,
        "median": round(statistics.median(lens), 1) if lens else 0,
        "wrong_n": len(wrong),
        "wrong_mean": round(sum(wrong) / len(wrong), 1) if wrong else 0,
        "wrong_median": round(statistics.median(wrong), 1) if wrong else 0,
        "max": max(lens) if lens else 0,
    }

def hp(N, K, n, k):
    if not n or not K or n > N or K > N or k > min(n, K):
        return None
    tot = comb(N, n)
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(n, K) + 1)) / tot

def recovery(rs_o, rs_w, rs_x):
    def key(r):
        return r["index"]
    O = {key(r): bool(r.get("correct")) for r in rs_o}
    W = {key(r): bool(r.get("correct")) for r in rs_w}
    X = {key(r): bool(r.get("correct")) for r in rs_x}
    common = [i for i in W if i in O and i in X]
    U = [i for i in common if not W[i]]
    G = [i for i in common if W[i]]
    Wb = [i for i in U if O[i]]
    Wg = [i for i in G if not O[i]]
    A = [i for i in U if X[i]]
    B = [i for i in G if not X[i]]
    sWb, sWg = set(Wb), set(Wg)
    a_m = [i for i in A if i in sWb]
    b_m = [i for i in B if i in sWg]
    a_n = [i for i in A if i not in sWb]
    b_n = [i for i in B if i not in sWg]
    U2, G2 = len(U) - len(Wb), len(G) - len(Wg)
    return {
        "common": len(common), "U": len(U), "G": len(G), "Wb": len(Wb), "Wg": len(Wg),
        "U_minus_Wb": U2, "G_minus_Wg": G2,
        "A": len(A), "B": len(B),
        "mirror_side": {
            "A_m": len(a_m), "exp_A_m": round(len(A) * len(Wb) / len(U), 1) if U else 0,
            "p_A_m": hp(len(U), len(Wb), len(A), len(a_m)),
            "B_m": len(b_m), "exp_B_m": round(len(B) * len(Wg) / len(G), 1) if G else 0,
            "p_B_m": hp(len(G), len(Wg), len(B), len(b_m)),
        },
        "other_side": {
            "A_n": len(a_n), "exp_A_n": round(len(A) * U2 / len(U), 1) if U else 0,
            "p_A_n": hp(len(U), U2, len(A), len(a_n)),
            "B_n": len(b_n), "exp_B_n": round(len(B) * G2 / len(G), 1) if G else 0,
            "p_B_n": hp(len(G), G2, len(B), len(b_n)),
        },
    }

out = {"variant": V, "bands": "П63.1 (§63.6), записаны до сборки и до прогона"}
combined = {"math": 0, "format": 0, "trunc": 0}
for sp in ("search", "dev"):
    p = ROOT / "runs/variants" / V / sp / "results.jsonl"
    if not p.exists():
        out[sp] = {"missing": True}
        continue
    rs = rows(p)
    st = stats(rs)
    hist = Counter((int(r.get("output_tokens", 0)) // 64) * 64 for r in rs)
    out[sp] = {
        "stats": st,
        "hist": {str(k): hist[k] for k in range(64, 1025, 64)},
        "bands": {k: (st[k], BANDS[sp][k][0] <= st[k] <= BANDS[sp][k][1]) for k in ("math", "format", "trunc")},
        "len_pred": {"mean_ok": st["mean"] < BASE_LEN[sp]["w8_mean"],
                     "median_ok": sp != "search" or st["median"] <= LEN_PRED["search_median_max"]},
        "w8_stats": {"math": None, "format": None, "trunc": None},
    }
    ww = ROOT / "runs/variants/w8_z_gate_mild_format" / sp / "results.jsonl"
    oo = ROOT / "runs" / ("baseline_%s" % sp) / "results.jsonl"
    if ww.exists():
        wst = stats(rows(ww))
        out[sp]["w8_stats"] = {"math": wst["math"], "format": wst["format"],
                               "trunc": wst["trunc"], "mean": wst["mean"]}
    if oo.exists() and ww.exists():
        out[sp]["recovery"] = recovery(rows(oo), rows(ww), rows(p))
    combined["math"] += st["math"]; combined["format"] += st["format"]; combined["trunc"] += st["trunc"]

out["combined"] = combined
out["combined_bands"] = {k: (combined[k], COMBINED[k][0] <= combined[k] <= COMBINED[k][1]) for k in COMBINED}
out["credit"] = {
    "format_ok": combined["format"] >= CREDIT["combined_format_min"],
    "math_ok": combined["math"] >= CREDIT["combined_math_min"],
    "trunc_ok": combined["trunc"] <= CREDIT["trunc_max"],
    "dev_math_ok": out.get("dev", {}).get("stats", {}).get("math", 0) >= CREDIT["dev_math_min"],
}
out["len_pred_combined"] = {
    "search_mean_below_w8": out.get("search", {}).get("len_pred", {}).get("mean_ok"),
    "dev_mean_below_w8": out.get("dev", {}).get("len_pred", {}).get("mean_ok"),
}
out["w8_reference"] = {"combined": {"math": 216, "format": 332, "trunc": 18}}
print(json.dumps(out, ensure_ascii=False, indent=1))
