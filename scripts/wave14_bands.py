#!/usr/bin/env python
"""Проверка волны 14 по заранее зарегистрированным коридорам (§45.2).

Коридоры записаны ДО прогона и не пересматриваются post hoc. Оцениваются только
250 search + 100 dev (350 строк), официальный test не участвует.

Коридоры §45.2 (три arms, контрольная точка w8):
  формат FINAL (answer_extracted)      336 .. 348      (w8 = 332)
  обрывы (truncated)                   0 .. 14         (w8 = 18)
  математика (correct)                 216 .. 222      (w8 = 216)
  индикатор преждевременного FINAL     <= 1            (w8 = 0)

Индикатор преждевременного FINAL на 350 строках — прокси: строки с
n_final_lines >= 2. На пробной выборке 63 задач (§43.2) w8 имел ровно 1 такую позицию.

Печатаются и пререгистрированные предсказания §45.3, чтобы вердикт сравнивался
с тем, что было заявлено до прогона.

Только чтение результатов; генерации не запускается.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANDS = {
    "format_band": (336, 348),
    "truncation_band": (0, 14),
    "math_band": (216, 222),
    "premature_final_max": 1,
}
W8 = "w8_z_gate_mild_format"

PRED = {
    "w52_frozen10_dose003_negsign": "предсказано ухудшение (средний запас -0,226 логита): "
                                    "выход за коридоры формата и обрывов",
    "w53_frozen10_dose003_mirrorpos": "предсказано улучшение (средний запас +1,048 логита): "
                                      "формат около 348, обрывы 2-3",
    "w54_vhead_late19_head1_amp105": "предсказание знака нет (замер): |Δматематика| <= 3, "
                                     "|Δформат| <= 3 против w8",
}


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def premature(rows):
    return sum(1 for r in rows if int(r.get("n_final_lines") or 0) >= 2)


def main():
    table = json.loads((ROOT / "runs/analysis/wave14_table.json").read_text(encoding="utf-8"))
    rows350 = load(ROOT / "runs/variants" / W8 / "search" / "results.jsonl")
    rows350 += load(ROOT / "runs/variants" / W8 / "dev" / "results.jsonl")
    base = {"format": sum(1 for r in rows350 if r.get("answer_extracted")),
            "trunc": sum(1 for r in rows350 if r.get("truncated")),
            "math": sum(1 for r in rows350 if r.get("correct")),
            "prem": premature(rows350)}
    lines = ["коридоры §45.2 (зарегистрированы до прогона):",
             f"  формат {BANDS['format_band'][0]}..{BANDS['format_band'][1]}  "
             f"обрывы {BANDS['truncation_band'][0]}..{BANDS['truncation_band'][1]}  "
             f"математика {BANDS['math_band'][0]}..{BANDS['math_band'][1]}  "
             f"преждевременный FINAL <= {BANDS['premature_final_max']}",
             f"  контрольная точка w8 на 350 строках: формат {base['format']} / "
             f"обрывы {base['trunc']} / математика {base['math']} / "
             f"преждевременный FINAL (прокси n_final_lines>=2) {base['prem']}"]
    if not table:
        lines.append("нет данных: прогоны ещё не завершены")
    for v in PRED:
        c = table.get(v, {}).get("combined")
        if not c:
            lines.append(f"\n{v}: нет завершённого combined-прогона ({PRED[v]})")
            continue
        s = c["stats"]
        rows = load(ROOT / "runs/variants" / v / "search" / "results.jsonl")
        rows += load(ROOT / "runs/variants" / v / "dev" / "results.jsonl")
        got = {"format": s["extracted"], "trunc": s["truncated"], "math": s["correct"],
               "prem": premature(rows)}
        chk = (BANDS["format_band"][0] <= got["format"] <= BANDS["format_band"][1],
               BANDS["truncation_band"][0] <= got["trunc"] <= BANDS["truncation_band"][1],
               BANDS["math_band"][0] <= got["math"] <= BANDS["math_band"][1],
               got["prem"] <= BANDS["premature_final_max"])
        names = ["формат", "обрывы", "математика", "преждевременный FINAL"]
        parts = [f"{n} {got[k]} {'OK' if ok else 'НЕ OK'}"
                 for n, k, ok in zip(names, ("format", "trunc", "math", "prem"), chk)]
        lines.append(f"\n{v}: " + " | ".join(parts))
        lines.append(f"  ИТОГ по коридорам: {'все выполнены' if all(chk) else 'не выполнены'} "
                     f"({sum(chk)} из 4)")
        lines.append(f"  сдвиг против w8: формат {got['format'] - base['format']:+d}, "
                     f"обрывы {got['trunc'] - base['trunc']:+d}, "
                     f"математика {got['math'] - base['math']:+d}, "
                     f"преждевременный FINAL {got['prem'] - base['prem']:+d}")
        lines.append(f"  {PRED[v]}")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/wave14_bands.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
