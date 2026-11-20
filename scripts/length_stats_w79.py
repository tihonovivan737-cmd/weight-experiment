#!/usr/bin/env python3
# Информационные длины ответов по срезам dev и search (ключ output_tokens).
# Дециль: сортировка значений, затем tok[int(0.9 * (len(tok) - 1))] — та же convention,
# что в scripts/wave26_check.py. both = задачи, достигшие потолка 1024 на обоих срезах.
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
CAP = 1024


def recs(p: Path):
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def lens(rs):
    tok = sorted(r["output_tokens"] for r in rs)
    if not tok:
        return None
    d90 = tok[int(0.9 * (len(tok) - 1))]
    mean = sum(tok) / len(tok)
    med = tok[len(tok) // 2] if len(tok) % 2 else (tok[len(tok)//2 - 1] + tok[len(tok)//2]) / 2
    cap = sum(1 for t in tok if t >= CAP)
    return len(tok), mean, med, d90, tok[-1], cap


def main(names):
    sets = {}
    for n in names:
        for sp in ("dev", "search"):
            p = ROOT / "runs/variants" / n / sp / "results.jsonl"
            if not p.is_file():
                p = ROOT / "runs/baseline" / ""  # не используется
                p = ROOT / f"runs/baseline_{sp}" / "results.jsonl" if n == "orig" else None
                if p is None:
                    continue
            if p is None or not p.is_file():
                print(f"НЕТ ФАЙЛА: {p}")
                continue
            sets[(n, sp)] = recs(p)
    for n in names:
        row = {}
        for sp in ("dev", "search"):
            s = lens(sets.get((n, sp), []))
            row[sp] = s
        both = 0
        if (n, "dev") in sets and (n, "search") in sets:
            ids = {(n, "dev"), (n, "search")}
            di = {r["index"] for r in sets[(n, "dev")] if r["output_tokens"] >= CAP}
            si = {r["index"] for r in sets[(n, "search")] if r["output_tokens"] >= CAP}
            both = len(di & si)
        print(f"{n}: dev {row['dev']} | search {row['search']} | both {both}")


main(["orig", "w8_z_gate_mild_format", "w50_finalrow_rotation15_smallmargin",
      "w74_finalrow_plus_two_rows198_248046_on_w8", "w76_finalrow_core_six_on_w8",
      "w77_finalrow_plus_8rows_on_w8"])
