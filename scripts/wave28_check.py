# Проверка полос П73.1 (записаны в §73.5 до сборки) для w78_finalrow_six_mid_rows_on_w8.
import json, math, statistics
from pathlib import Path

V = "w78_finalrow_six_mid_rows_on_w8"
REF = {"dev": (321.6, 276.5), "search": (352.5, 307.5)}   # w8, только как свидетельство
EXP = {"dev": (312, 336, 272, 296, 510), "search": (348, 370, 298, 316, 580)}  # коридоры §73.5, только свидетельство

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
        dif = sum(1 for r in rs if (r["model_answer"] or "") != (idx[r["index"]]["model_answer"] or ""))
        print(f"{split} против {lab}: {a[0]}/{a[1]}/{a[2]} -> {c[0]}/{c[1]}/{c[2]} "
              f"(математика {c[0]-a[0]:+d}, извлечение {c[1]-a[1]:+d}, обрывы {c[2]-a[2]:+d}) "
              f"исправлено {f} сломано {b} отличающиеся ответы {dif} McNemar p = {mcn(f,b):.4f}")

dc, de, dt = st(dev); sc, se, tt = st(sea)
print(f"combined: {dc+sc}/{de+se}/{dt+tt}")
gate = {"combined_format": de+se >= 336, "combined_math": dc+sc >= 216,
        "combined_trunc": dt+tt <= 14, "dev_math": dc >= 66}
print("кредитное условие:", gate, "=", sum(gate.values()), "из 4")
bands = {"dev math": (dc, 61, 66), "dev format": (de, 93, 98), "dev trunc": (dt, 1, 5),
         "search math": (sc, 145, 155), "search format": (se, 231, 242), "search trunc": (tt, 8, 16),
         "combined math": (dc+sc, 209, 218), "combined format": (de+se, 327, 340), "combined trunc": (dt+tt, 9, 21)}
for k, (v, lo, hi) in bands.items():
    print(f"полоса П73.1 {k}: {v} — {'ВЫПОЛНЕНА' if lo <= v <= hi else 'НЕ ВЫПОЛНЕНА'}")

print("условия закрытия П73.2:")
print("  1.", "сработало" if sum(gate.values()) == 4 else "не сработало",
      f"(кредитное условие {sum(gate.values())} из 4)")
print("  2.", "сработало" if de+se >= 336 and dc+sc <= 213 else "не сработало",
      f"(извлечение вместе {de+se} при границе 336, математика вместе {dc+sc} при границе 213)")
print("  3.", "сработало" if de+se <= 325 else "не сработало",
      f"(извлечение вместе {de+se} при границе 325)")
print("  4.", "сработало" if dc >= 66 and de+se <= 331 else "не сработало",
      f"(математика dev {dc} при границе 66, извлечение вместе {de+se} при границе 331)")
print("  5.", "сработало" if dt >= 7 or dt+tt >= 22 else "не сработало",
      f"(обрывы dev {dt} при границе 7, обрывы вместе {dt+tt} при границе 22; в §73.6 пункт 5 "
      f"прочитан как нарушение полосы, граница 21)")

print("длины, только свидетельство о распределении (без решения):")
for split, rs in (("dev", dev), ("search", sea)):
    m, md, d90, mx = lens(rs)
    lo_m, hi_m, lo_md, hi_md, hi_d90 = EXP[split]
    am, ad = REF[split]
    print(f"  {split}: среднее {m:.1f} (w8 {am}, коридор {lo_m}–{hi_m}) медиана {md:.1f} "
          f"(w8 {ad}, коридор {lo_md}–{hi_md}) дециль 90 {d90} (коридор не выше {hi_d90}) максимум {mx}")
for split, rs in (("dev", dev), ("search", sea)):
    both = sum(1 for r in rs if r["answer_extracted"] and r["truncated"])
    print(f"  {split}: строки с извлечением и обрывом одновременно: {both}")
print("Значения McNemar — только информация, не часть решения по полосам П73.1.")
