#!/usr/bin/env python3
"""Попарный обзор серии: каждый измеренный вариант против исходной модели и против `w8`.

Для каждого варианта с обоими разрезами (`dev` 100 и `search` 250) считается на общих 350
задачах: верно/извлечение/обрывы у варианта, дельта верных против референса, парные переходы
(исправлено и сломано), точный двусторонний критерий `McNemar` и отдельно дельта качества
на пересечении задач, где `FINAL` есть у обеих моделей. В конце — сводка: сколько вариантов
вообще вышли за уровень шума (p < 0,05) и сколько из них — в лучшую сторону.

Только чтение существующих результатов; новых прогонов нет.
Вывод: runs/analysis/paired_survey.txt
"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARS = ROOT / "runs/variants"
REF = {
    "исходная модель": [ROOT / "runs/baseline_dev/results.jsonl", ROOT / "runs/baseline_search/results.jsonl"],
    "w8": [ROOT / "runs/variants/w8_z_gate_mild_format/dev/results.jsonl",
           ROOT / "runs/variants/w8_z_gate_mild_format/search/results.jsonl"],
}


def load_pair(name, paths):
    out = {}
    for p in paths:
        if not p.exists():
            return None
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[r["index"]] = r
    return out


def var_rows(name):
    out = {}
    for sp in ("dev", "search"):
        p = VARS / name / sp / "results.jsonl"
        if not p.exists():
            return None
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[r["index"]] = r
    return out


def fin(r):
    return (r.get("n_final_lines") or 0) >= 1


def mcnemar(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)


def cmp_rows(base, var):
    common = sorted(set(base) & set(var))
    c = sum(1 for i in common if var[i]["correct"])
    b = sum(1 for i in common if base[i]["correct"])
    e = sum(1 for i in common if fin(var[i]))
    et = sum(1 for i in common if base[i].get("correct") and not var[i]["correct"])
    gain = sum(1 for i in common if var[i]["correct"] and not base[i]["correct"])
    tr = sum(1 for i in common if var[i].get("truncated"))
    inter = [i for i in common if fin(base[i]) and fin(var[i])]
    iq_b = sum(1 for i in inter if base[i]["correct"])
    iq_v = sum(1 for i in inter if var[i]["correct"])
    qg = sum(1 for i in inter if var[i]["correct"] and not base[i]["correct"])
    ql = sum(1 for i in inter if base[i]["correct"] and not var[i]["correct"])
    return dict(n=len(common), correct=c, base_correct=b, extracted=e, truncated=tr,
                gain=gain, loss=et, p=mcnemar(gain, et), inter=len(inter),
                q_base=iq_b, q_var=iq_v, q_gain=qg, q_loss=ql, q_p=mcnemar(qg, ql))


def main():
    refs = {k: load_pair(k, v) for k, v in REF.items()}
    lines = ["Попарный обзор: каждый вариант с обоими разрезами против исходной модели и против `w8`.",
             "Числа: верно / извлечение / обрывы на 350 задачах; p — точный критерий `McNemar`.", ""]
    rows = []
    for m in sorted(VARS.glob("*/manifest.json")):
        name = m.parent.name
        var = var_rows(name)
        if not var or len(var) < 340:
            continue
        rows.append((name, {k: cmp_rows(refs[k], var) for k in refs if refs[k]}))

    lines.append(f"{'вариант':46s} | {'против исходной':>28} | {'против w8':>28} | качество на пересечении (против w8)")
    sig_orig = sig_w8 = better_orig = better_w8 = 0
    for name, d in rows:
        o, w = d.get("исходная модель"), d.get("w8")
        parts = []
        for r in (o, w):
            parts.append(f"{r['correct']:>3} ({r['correct'] - r['base_correct']:+3d}) p={r['p']:.3f}" if r else "—")
            if r and r["p"] < 0.05:
                if r is o:
                    sig_orig += 1
                    better_orig += 1 if r["correct"] > r["base_correct"] else 0
                else:
                    sig_w8 += 1
                    better_w8 += 1 if r["correct"] > r["base_correct"] else 0
        q = f"{w['q_var']:>3} против {w['q_base']:>3} ({w['q_var'] - w['q_base']:+d}), p={w['q_p']:.3f}"
        lines.append(f"{name:46s} | {parts[0]:>28} | {parts[1]:>28} | {q}")
    lines.append("")
    lines.append(f"Всего вариантов в обзоре: {len(rows)}.")
    lines.append(f"Значимых отличий (p < 0,05) от исходной модели: {sig_orig}, из них в лучшую сторону: {better_orig}.")
    lines.append(f"Значимых отличий (p < 0,05) от w8: {sig_w8}, из них в лучшую сторону: {better_w8}.")
    best_q = max(rows, key=lambda t: t[1]["w8"]["q_var"] - t[1]["w8"]["q_base"])
    lines.append(f"Наибольший плюс качества против w8: {best_q[0]} "
                 f"({best_q[1]['w8']['q_var'] - best_q[1]['w8']['q_base']:+d} задач, "
                 f"p = {best_q[1]['w8']['q_p']:.4f}).")
    worst_q = min(rows, key=lambda t: t[1]["w8"]["q_var"] - t[1]["w8"]["q_base"])
    lines.append(f"Наибольшая потеря качества против w8: {worst_q[0]} "
                 f"({worst_q[1]['w8']['q_var'] - worst_q[1]['w8']['q_base']:+d} задач, "
                 f"p = {worst_q[1]['w8']['q_p']:.4f}).")
    text = "\n".join(lines)
    (ROOT / "runs/analysis/paired_survey.txt").write_text(text + "\n", encoding="utf-8")
    print("\n".join(lines[:6] + ["…"] + lines[-8:]))


if __name__ == "__main__":
    main()
