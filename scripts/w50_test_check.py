#!/usr/bin/env python3
# Сводка по официальному набору GSM8K (1319 задач): w50 против исходной модели и против w8.
# Только оценка уже измеренного варианта; отбор новых вариантов по этому срезу не производится.
import json
from math import comb
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
V = "w50_finalrow_rotation15_smallmargin"
FILES = {
    "orig": ROOT / "runs/baseline_test/results.jsonl",
    "w8": ROOT / "runs/variants/w8_z_gate_mild_format/test/results.jsonl",
    "w50": ROOT / f"runs/variants/{V}/test/results.jsonl",
}
OUT = ROOT / "runs/analysis/w50_test_report.txt"


def load(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["index"]] = r
    return rows


def st(rs):
    n = len(rs)
    c = sum(1 for r in rs.values() if r["correct"])
    f = sum(1 for r in rs.values() if r["answer_extracted"])
    t = sum(1 for r in rs.values() if r["truncated"])
    return n, c, f, t


def mcn(f, b):
    nn = f + b
    if nn == 0:
        return 1.0
    kmin = min(f, b)
    p = sum(comb(nn, k) for k in range(0, kmin + 1)) * 2 / 2 ** nn
    return min(1.0, p)


def pair(a_name, b_name, A, B):
    common = sorted(set(A) & set(B))
    fixed = [i for i in common if not A[i]["correct"] and B[i]["correct"]]
    broken = [i for i in common if A[i]["correct"] and not B[i]["correct"]]
    pa, pb = st(A), st(B)
    pp_a = 100.0 * pa[1] / pa[0]
    pp_b = 100.0 * pb[1] / pb[0]
    lines = [
        f"{b_name} против {a_name} (официальный набор, {len(common)} общих задач)",
        f"  {a_name}: {pa[1]} из {pa[0]} = {pp_a:.2f}% | извлечено FINAL {pa[2]} | обрывы {pa[3]}",
        f"  {b_name}: {pb[1]} из {pb[0]} = {pp_b:.2f}% | извлечено FINAL {pb[2]} | обрывы {pb[3]}",
        f"  Исправлено: {len(fixed)}  Сломано: {len(broken)}  Разность: {len(fixed) - len(broken)}",
        f"  Разница в процентных пунктах: {pp_b - pp_a:+.2f}",
        f"  McNemar точный p = {mcn(len(fixed), len(broken)):.4f} (справочно, полосы решают)",
    ]
    exa = [i for i in common if A[i]["answer_extracted"] and B[i]["answer_extracted"]]
    lost = [i for i in exa if A[i]["predicted"] != B[i]["predicted"]]
    exl = [i for i in common if A[i]["answer_extracted"] and not B[i]["answer_extracted"]]
    exn = [i for i in common if not A[i]["answer_extracted"] and B[i]["answer_extracted"]]
    lines.append(
        f"  Оба извлекли FINAL: {len(exa)}; извлечение потеряно: {len(exl)}; получено: {len(exn)}; "
        f"значение ответа различается при обоих извлечениях: {len(lost)}"
    )
    return lines


def main():
    data = {}
    for name, path in FILES.items():
        if not path.exists():
            print(f"НЕТ ФАЙЛА: {path}")
            return
        data[name] = load(path)
    lines = []
    for name in ("orig", "w8", "w50"):
        n, c, f, t = st(data[name])
        lines.append(
            f"{name}: задач {n} | верно {c} ({100.0 * c / n:.2f}%) | FINAL извлечено {f} | обрывы {t}"
        )
    lines.append("")
    lines += pair("orig", "w50", data["orig"], data["w50"])
    lines.append("")
    lines += pair("w8", "w50", data["w8"], data["w50"])
    lines.append("")
    lines.append("Срез официальный: оценка, не отбор; новые варианты по нему не подбираются.")
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"Отчёт: {OUT}")


main()
