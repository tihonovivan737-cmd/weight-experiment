# Проверка полос П71.1 (записаны в §71.2 до сборки) для w76_z_flatten_top5_on_w8.
import json, math, statistics
from pathlib import Path

V = "w76_z_flatten_top5_on_w8"
REF = {"dev": (321.6, 276.5), "search": (352.5, 307.5)}   # w8, только как свидетельство
EXP = {"dev": (312, 336, 276, 296), "search": (348, 370, 298, 316)}  # ожидаемый диапазон из §71.2

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
gate = {"combined_format": de+se >= 336, "combined_math": dc+sc >= 216,
        "combined_trunc": dt+tt <= 14, "dev_math": dc >= 66}
print("кредитное условие:", gate, "=", sum(gate.values()), "из 4")
bands = {"dev math": (dc, 62, 67), "dev format": (de, 92, 96), "dev trunc": (dt, 2, 6),
         "search math": (sc, 144, 152), "search format": (se, 227, 238), "search trunc": (tt, 12, 20),
         "combined math": (dc+sc, 206, 219), "combined format": (de+se, 321, 335), "combined trunc": (dt+tt, 14, 25)}
for k, (v, lo, hi) in bands.items():
    print(f"полоса П71.1 {k}: {v} — {'ВЫПОЛНЕНА' if lo <= v <= hi else 'НЕ ВЫПОЛНЕНА'}")

print("условия закрытия П71.1:")
print("  1.", "сработало" if dc >= 66 and de+se >= 332 else "не сработало",
      f"(математика dev {dc} при границе 66, извлечение вместе {de+se} при границе 332)")
print("  2.", "сработало" if de+se <= 326 else "не сработало",
      f"(извлечение вместе {de+se} при границе 326)")
print("  3.", "сработало" if dc <= 62 else "не сработало", f"(математика dev {dc} при границе 62)")
print("  4.", "сработало" if sum(gate.values()) == 4 else "не сработало",
      f"(кредитное условие {sum(gate.values())} из 4)")

print("длины, только свидетельство о распределении (без решения):")
for split, rs in (("dev", dev), ("search", sea)):
    m, md, d90, mx = lens(rs)
    lo_m, hi_m, lo_md, hi_md = EXP[split]
    am, ad = REF[split]
    print(f"  {split}: среднее {m:.1f} (w8 {am}, ожидание {lo_m}–{hi_m}) медиана {md:.1f} "
          f"(w8 {ad}, ожидание {lo_md}–{hi_md}) дециль 90 {d90} максимум {mx}")
for split, rs in (("dev", dev), ("search", sea)):
    both = sum(1 for r in rs if r["answer_extracted"] and r["truncated"])
    print(f"  {split}: строки с извлечением и обрывом одновременно: {both}")
print("Значения McNemar — только информация, не часть решения по полосам П71.1.")
