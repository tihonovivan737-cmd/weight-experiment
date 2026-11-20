# Проверка полос П67.1 (записаны в §67.5 до сборки) для w72_finalrow_plus_zlate_on_w8.
import json, math, statistics
from pathlib import Path

V = "w72_finalrow_plus_zlate_on_w8"
REF = {"dev": (321.6, 276.5), "search": (352.5, 307.5)}   # w8, только как свидетельство

def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def mcn(f, b):
    nn = f + b
    if nn == 0: return 1.0
    return min(1.0, sum(math.comb(nn, k) for k in range(0, min(f, b) + 1)) * 2 / 2 ** nn)

def st(rs):
    return (sum(1 for r in rs if r["correct"]),
            sum(1 for r in rs if r["answer_extracted"]),
            sum(1 for r in rs if r["truncated"]))

def lens(rs):
    tok = sorted(int(r["output_tokens"]) for r in rs)
    return statistics.mean(tok), statistics.median(tok), tok[int(0.9 * (len(tok) - 1))], max(tok)

base = Path("runs/variants") / V
dev = rows(base / "dev" / "results.jsonl"); sea = rows(base / "search" / "results.jsonl")
assert len(dev) == 100 and len(sea) == 250
for split, rs in (("dev", dev), ("search", sea)):
    for lab, bp in (("orig", f"runs/baseline_{split}/results.jsonl"),
                    ("w8", f"runs/variants/w8_z_gate_mild_format/{split}/results.jsonl")):
        brs = rows(bp); idx = {r["index"]: r for r in brs}
        f = sum(1 for r in rs if r["correct"] and not idx[r["index"]]["correct"])
        b = sum(1 for r in rs if idx[r["index"]]["correct"] and not r["correct"])
        a, c = st(brs), st(rs)
        print(f"{split} против {lab}: {a[0]}/{a[1]}/{a[2]} -> {c[0]}/{c[1]}/{c[2]} "
              f"(математика {c[0]-a[0]:+d}, извлечение {c[1]-a[1]:+d}, обрывы {c[2]-a[2]:+d}) "
              f"исправлено {f} сломано {b} McNemar p = {mcn(f,b):.4f}")

dc, de, dt = st(dev); sc, se, tt = st(sea)
print(f"combined: {dc+sc}/{de+se}/{dt+tt}")
print("кредитное условие:", {"combined_format": de+se >= 336, "combined_math": dc+sc >= 216,
                             "combined_trunc": dt+tt <= 14, "dev_math": dc >= 66})
bands = {"dev math": (dc, 58, 66), "dev format": (de, 93, 98), "dev trunc": (dt, 1, 7),
         "search math": (sc, 145, 155), "search format": (se, 230, 242), "search trunc": (tt, 8, 19),
         "combined math": (dc+sc, 209, 219), "combined format": (de+se, 326, 340), "combined trunc": (dt+tt, 10, 22)}
for k, (v, lo, hi) in bands.items():
    print(f"полоса П67.1 {k}: {v} — {'ВЫПОЛНЕНА' if lo <= v <= hi else 'НЕ ВЫПОЛНЕНА'}")
print("длины, только свидетельство о распределении (без решения):")
for split, rs in (("dev", dev), ("search", sea)):
    m, md, d90, mx = lens(rs)
    am, ad = REF[split]
    print(f"  {split}: среднее {m:.1f} (w8 {am}) медиана {md:.1f} (w8 {ad}) "
          f"дециль 90 {d90} максимум {mx}")
# отдельные признаки, которые нельзя путать с числом строк
for split, rs in (("dev", dev), ("search", sea)):
    both = sum(1 for r in rs if r["answer_extracted"] and r["truncated"])
    print(f"  {split}: строки с извлечением и обрывом одновременно: {both}")
