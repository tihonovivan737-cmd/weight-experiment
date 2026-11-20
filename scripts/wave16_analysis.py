#!/usr/bin/env python3
"""Волна 16 (§62.6): сводка для w67 по полосам П62.1.

Три вида чисел:
  A. все задачи: math (верные), format (извлечённые), trunc, средняя и медиана длины;
  B. только неверные задачи: количество, средняя и медиана длины;
  C. восстановительные множества по определениям §62.4:
     O — оригинал, W — w8, X — вариант; U = {W неверна}, G = {W верна};
     Wb = {O верна и W неверна}, Wg = {O неверна и W верна};
     A = {X верна и W неверна}, B = {W верна и X неверна};
     p — гипергеометрическая «k или больше» отдельно для A внутри Wb на U
     и для B внутри Wg на G.
Полосы записаны до сборки и до прогона (§62.6); файл ничего не запускает.
"""
import json, statistics
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
V = "w67_inproj_a_amp110_on_w8"

BANDS = {
    "search": {"math": (148, 156), "format": (231, 241), "trunc": (8, 16)},
    "dev": {"math": (63, 69), "format": (93, 98), "trunc": (2, 6)},
}
COMBINED = {"math": (211, 225), "format": (324, 340), "trunc": (10, 22)}
CREDIT = {"combined_format_min": 336, "combined_math_min": 216, "trunc_max": 14, "dev_math_min": 66}
CLOSE = {"combined_math_max": 209, "trunc_min_close": 17}
LEN_PRED = {"search_mean_max": 352.5, "dev_mean_max": 321.6, "search_median_max": 307.5}
BASE_LEN = {"search": {"orig_mean": 398.0, "w8_mean": 352.5}, "dev": {"orig_mean": None, "w8_mean": 321.6}}

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
    if n > N or K > N or k > min(n, K):
        return None
    tot = comb(N, n)
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(k, n, K) + 1)) / tot

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
    a_in = [i for i in A if i in set(Wb)]
    b_in = [i for i in B if i in set(Wg)]
    pa = hp(len(U), len(Wb), len(A), len(a_in))
    pb = hp(len(G), len(Wg), len(B), len(b_in))
    return {
        "common": len(common), "U": len(U), "G": len(G), "Wb": len(Wb), "Wg": len(Wg),
        "A": len(A), "B": len(B), "A_in_Wb": len(a_in), "B_in_Wg": len(b_in),
        "exp_A": round(len(A) * len(Wb) / len(U), 1) if U else 0,
        "exp_B": round(len(B) * len(Wg) / len(G), 1) if G else 0,
        "p_A": pa, "p_B": pb,
    }

out = {"variant": V, "bands": "П62.1 (§62.6), записаны до сборки и до прогона"}
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
        "hist": {str(k): hist[k] for k in sorted(hist)},
        "bands": {k: (st[k], BANDS[sp][k][0] <= st[k] <= BANDS[sp][k][1]) for k in ("math", "format", "trunc")},
        "len_pred": {"mean_ok": st["mean"] < BASE_LEN[sp]["w8_mean"],
                     "median_ok": sp != "search" or st["median"] <= LEN_PRED["search_median_max"]},
    }
    combined["math"] += st["math"]; combined["format"] += st["format"]; combined["trunc"] += st["trunc"]
    oo = ROOT / "runs" / ("baseline_%s" % sp) / "results.jsonl"
    ww = ROOT / "runs/variants/w8_z_gate_mild_format" / sp / "results.jsonl"
    r = recovery(rows(oo), rows(ww), rows(p)) if oo.exists() and ww.exists() else {"missing": True}
    out[sp]["recovery"] = r

out["combined"] = combined
out["combined_bands"] = {k: (combined[k], COMBINED[k][0] <= combined[k] <= COMBINED[k][1]) for k in COMBINED}
out["credit"] = {
    "format_ok": combined["format"] >= CREDIT["combined_format_min"],
    "math_ok": combined["math"] >= CREDIT["combined_math_min"],
    "trunc_ok": combined["trunc"] <= CREDIT["trunc_max"],
    "dev_math_ok": out["dev"].get("stats", {}).get("math", 0) >= CREDIT["dev_math_min"],
}
out["close"] = {
    "math_closed": combined["math"] <= CLOSE["combined_math_max"],
    "trunc_closed": combined["trunc"] >= CLOSE["trunc_min_close"],
}
out["w8_reference"] = {"combined": {"math": 216, "format": 332, "trunc": 14}}
print(json.dumps(out, ensure_ascii=False, indent=1))
