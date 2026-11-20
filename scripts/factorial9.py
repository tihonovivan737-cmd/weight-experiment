#!/usr/bin/env python
"""Анализ факторного плана 3×3 «доза EARLY × доза LATE» (раздел 31 журнала).

Сетка — накопленный множитель `linear_attn.in_proj_z` в 11 ранних/средних LIN-слоях (EARLY)
и в 7 поздних LIN-слоях (LATE). Каждая ячейка — уже построенный вариант; восемь из девяти
ячеек измеримы, клетка (1,0008 × 1,0008) остаётся незаполненной.

Главный эффект каждой оси считается как средняя разность по соседним контрастам при
фиксированном значении второй оси (предрегистрировано в §31.6): на 350 задачах одиночный
перепад в 3…5 задач — уровень шума (§22), поэтому вывод строится на знаке нескольких
независимых контрастов, а не на одном сравнении.

Только чтение results.jsonl; генерации не запускаются. Выход: runs/analysis/factorial9.txt
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

EARLY = [1.0008, 0.9300, 0.8742]          # A1 > A2 > A3 (A2 — контрольная точка w8)
LATE = [1.0008, 0.9300, 0.8742]           # B1 > B2 > B3

# ячейка -> вариант (orig = исходная модель, прогоны runs/baseline_*)
CELL = {
    (1.0008, 1.0008): None,                       # не собирался
    (1.0008, 0.9300): "w37_z_early_release1075",
    (1.0008, 0.8742): "w38_z_corner_early10008_late08742",
    (0.9300, 1.0008): "w39_z_late_release1075",
    (0.9300, 0.9300): "w8_z_gate_mild_format",
    (0.9300, 0.8742): "w36_z_late_extra094",
    (0.8742, 1.0008): None,                       # не собирался
    (0.8742, 0.9300): "w40_z_early_extra094",
    (0.8742, 0.8742): "w29_z_total087",
}
ORIG = "orig"


def combined(name):
    """350 строк результата варианта (search 250 + dev 100) или None, если прогон неполон."""
    if name == ORIG:
        rows = W.load(str(ROOT / "runs/baseline_search/results.jsonl"))
        rows.update(W.load(str(ROOT / "runs/baseline_dev/results.jsonl")))
        return rows if len(rows) == 350 else None
    try:
        rows = W.load(W.V / name / "search/results.jsonl")
        rows.update(W.load(W.V / name / "dev/results.jsonl"))
    except FileNotFoundError:
        return None
    return rows if len(rows) == 350 else None


def pair(rows_a, rows_b):
    """Δверных, McNemar p и отдельный математический контраст (FINAL есть в обоих прогонах)."""
    c = W.cmp(rows_a, rows_b, "b")
    return c["delta_correct"], c["mcnemar_p"], c["delta_math"], c["mcnemar_p_math"]


def contrast(rows_a, rows_b, label):
    d, p, dm, pm = pair(rows_a, rows_b)
    return {"label": label, "delta": d, "p": p, "delta_math": dm, "p_math": pm}


def main():
    data = {}
    for key, v in CELL.items():
        if v is None:
            continue
        rows = combined(v)
        if rows is not None:
            data[key] = (v, rows)

    lines = ["Факторный план 3×3: накопленная доза in_proj_z EARLY (строки) × LATE (столбцы)",
             "Ячейка: вариант | верных из 350 | FINAL | обрывы | верных+FINAL",
             ""]
    for a in EARLY:
        row = [f"EARLY {a:.4f}"]
        for b in LATE:
            got = data.get((a, b))
            if got is None:
                v = CELL.get((a, b))
                row.append("(нет данных)" if v else "[не собирался]")
                continue
            v, rows = got
            s = W.stats(rows)
            row.append(f"{v.split('_')[0]} {s['correct']}/{s['extracted']}/{s['truncated']}"
                       f"/{s['correct_with_final']}")
        lines.append("  ".join(f"{c:<28}" for c in row))

    lines.append("")

    late_pairs = [((0.9300, 0.8742), (0.9300, 0.9300), "EARLY 0,9300: w36 − w8"),
                  ((1.0008, 0.8742), (1.0008, 0.9300), "EARLY 1,0008: w38 − w37"),
                  ((0.8742, 0.8742), (0.8742, 0.9300), "EARLY 0,8742: w29 − w40")]
    early_pairs = [((0.9300, 0.9300), (1.0008, 0.9300), "LATE 0,9300: w8 − w37"),
                   ((0.8742, 0.9300), (0.9300, 0.9300), "LATE 0,9300: w40 − w8"),
                   ((0.9300, 0.8742), (1.0008, 0.8742), "LATE 0,8742: w36 − w38")]

    def block(title, pairs):
        res = []
        lines.append(title)
        for ka, kb, label in pairs:
            if ka not in data or kb not in data:
                lines.append(f"  {label:<34} — не измерено")
                continue
            c = contrast(data[ka][1], data[kb][1], label)
            res.append(c)
            lines.append(f"  {label:<34} Δверных {c['delta']:+4d} (p={c['p']:.5f})  "
                         f"Δмат {c['delta_math']:+4d} (p={c['p_math']:.5f})")
        if res:
            mean = sum(c["delta"] for c in res) / len(res)
            meanm = sum(c["delta_math"] for c in res) / len(res)
            same = sum(1 for c in res if c["delta"] > 0), sum(1 for c in res if c["delta"] < 0)
            lines.append(f"  --> средний эффект {mean:+.2f} задачи по {len(res)} контрастам "
                         f"(плюс {same[0]} / минус {same[1]}); средний Δмат {meanm:+.2f}")
            lines.append(f"  --> критерий §31.6 (≥3 контраста одного знака и |средний| ≥ 4): "
                         f"{'ось реальна' if (max(same) >= 3 and abs(mean) >= 4) else 'ось = шум'}")
        lines.append("")
        return res

    late = block("LATE-ось (углубление дозы в поздних слоях):", late_pairs)
    early = block("EARLY-ось (углубление дозы в ранних/средних слоях):", early_pairs)

    lines.append("Взаимодействие осей (диагональ):")
    need = [(1.0008, 0.8742), (1.0008, 0.9300), (0.9300, 0.8742), (0.9300, 0.9300)]
    if all(k in data for k in need):
        left = sum(W.stats(data[k][1])["correct"] for k in ((1.0008, 0.8742), (0.9300, 0.9300)))
        right = sum(W.stats(data[k][1])["correct"] for k in ((1.0008, 0.9300), (0.9300, 0.8742)))
        lines.append(f"  (w38 + w8) − (w37 + w36) = {left} − {right} = {left - right:+d} "
                     f"(положительно — оси складываются, отрицательно — конфликтуют)")
    else:
        lines.append("  — не измерено (нужны w38, w37, w36, w8)")

    lines.append("")
    lines.append("Проверка предрегистрированных предсказаний §31.4:")
    for key, label, pred in (((1.0008, 0.8742), "H32a (w38)", "≥220 верных, FINAL ≥332, обрывы ≤20; иначе оси конфликтуют"),
                             ((0.9300, 1.0008), "H32b (w39)", "≤212 верных и не лучше w37; иначе оптимум LATE между 0,8742 и 1,0008"),
                             ((0.8742, 0.9300), "H32c (w40)", "FINAL ≥334, обрывы ≤14, верных ≤216; если ≈w29 — форматный эффект целиком ранний")):
        got = data.get(key)
        if got is None:
            lines.append(f"  {label}: [{pred}] — прогон ещё не завершён")
            continue
        s = W.stats(got[1])
        lines.append(f"  {label} = {got[0]}: верных {s['correct']}, FINAL {s['extracted']}, "
                     f"обрывов {s['truncated']}, верных+FINAL {s['correct_with_final']}, "
                     f"средняя длина {s['mean_out_tokens']} токена")
        lines.append(f"      предсказание: {pred}")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/analysis/factorial9.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
