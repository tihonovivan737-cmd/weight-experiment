"""
Диссекция ошибок по results.jsonl (только чтение, без GPU и без обращения к test-задачам
с целью подбора вариантов: используется лишь для отчёта о измерениях).
Считает: верно, извлечён FINAL, обрезки, "золото спрятано в обрезанном тексте",
near-miss (|pred-gold|<=1 или относительная ошибка <=1 %), длина ответа по группам.
"""
import argparse
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/home/mpiscil/weight-experiment")
DAT = ROOT / "datasets/gsm8k/main"
FINAL_RE = re.compile(r"FINAL:\s*\[?\s*(-?\d[\d,\.]*)", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d[\d,\.]*")


def num(s):
    if s is None:
        return None
    t = str(s).replace(",", "").strip().rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


def answer_text(a):
    if isinstance(a, dict):
        return str(a.get("answer", ""))
    return str(a)


def gold_numbers(a):
    t = answer_text(a)
    tail = t.split("####")[-1] if "####" in t else t
    vals = [num(x) for x in NUM_RE.findall(tail.replace(",", ""))]
    vals = [v for v in vals if v is not None]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", required=True, help="train (индексы из splits.json) | test")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(DAT / ("train-00000-of-00001.parquet" if a.split == "train"
                                else "test-00000-of-00001.parquet"))
    rows = [json.loads(l) for l in Path(a.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    n = corr = extracted = trunc = gold_hidden = nearmiss = 0
    len_ok = len_bad_final = len_bad_nofinal = 0
    bad_final = bad_nofinal = 0
    for r in rows:
        gi = int(r["index"]); gold = gold_numbers(df.loc[gi]["answer"])
        gv = gold[-1] if gold else None
        gen = r.get("model_answer") or r.get("generated") or ""
        tr = bool(r.get("truncated"))
        ext = bool(r.get("answer_extracted"))
        pr = num(r.get("predicted"))
        n += 1
        ok = bool(r.get("correct"))
        m = list(FINAL_RE.finditer(gen))
        pred = pr if pr is not None else (num(m[-1].group(1)) if m else None)
        if ext:
            extracted += 1
        correct_by_pred = pred is not None and gv is not None and abs(pred - gv) < 1e-6
        if ok:
            corr += 1
        if not ext:
            bad_nofinal += 1
        if tr:
            trunc += 1
            hidden = [v for v in gold if re.search(r"(?<!\d)" + re.sub(r"\.", r"\\.", f"{v:g}") + r"(?!\d)", gen)]
            if gv is not None and hidden:
                gold_hidden += 1
        if ok:
            len_ok += len(gen)
        elif pred is not None:
            bad_final += 1
            len_bad_final += len(gen)
        if pred is not None and gv is not None and not ok:
            if abs(pred - gv) <= 1 or (gv != 0 and abs(pred - gv) / abs(gv) <= 0.01):
                nearmiss += 1
    rep = {
        "results": str(a.results), "label": a.label, "split": a.split, "n": n,
        "correct": corr, "final_extracted": extracted, "no_final": n - extracted,
        "truncated": trunc, "gold_hidden_in_truncated_text": gold_hidden,
        "wrong_with_final": bad_final, "wrong_without_final": bad_nofinal,
        "near_miss_within_1pct": nearmiss,
        "mean_len_correct": round(len_ok / max(corr, 1), 1),
        "mean_len_wrong_with_final": round(len_bad_final / max(bad_final, 1), 1),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
