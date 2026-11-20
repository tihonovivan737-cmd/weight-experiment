#!/usr/bin/env python3
"""Генератор рецептов волны 15 (§49.3, §51.3, §52).

Только детерминированные байтовые правки: op="scale" одной строкой mlp.gate_proj
слоя 23. Основание — w8 (контрольная точка), оригинал не трогается.

Arm'ы:
  w55_gold16_l23_amp130  — 16 строк по золоту, lam = 1.30 (пререгистрирован в §50.3)
  w56_gold16_l23_amp070  — те же строки, lam = 0.70 (зеркальный знак)
  w57_gold16_l23_amp160  — те же строки, lam = 1.60 (монотонность дозы)
  w58_gold16_l23_damp053 — те же строки, lam = 0.53 (направленно верное подавление, §51.3)
  w59_flippot16_l23_amp163 — 16 строк по flip_potential, lam = 1.63 (единственное
                             направленно верное усиление, §52)

Строки берутся из runs/analysis/wave15_gate_probe.json — readings повторного прогона
(одна запись маржи на задачу). Строки внутри рецепта уникальны, каждый arm —
отдельный вариант-копия шарда w8.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
PROBE = ROOT / "runs/analysis/wave15_gate_probe.json"
RECIPES = ROOT / "recipes"

TENSOR = "model.language_model.layers.23.mlp.gate_proj.weight"
BASE = "w8_z_gate_mild_format"
LAYER = 23

ARMS = [
    ("w55_gold16_l23_amp130", "gold", 1.30,
     "Усиление 16 строк с наибольшим средним вкладом в золотой ответ (П50.2 опровергнута: "
     "flip_potential этих строк отрицателен, сумма -26.7132, то есть на неверных задачах они "
     "голосуют за написанное число сильнее, чем за золотое). Arm остаётся как "
     "пререгистрированный контроль: ожидаемое направление знака — ухудшение."),
    ("w56_gold16_l23_amp070", "gold", 0.70,
     "Тот же список 16 строк, обратный знак дозы. Единственный из пререгистрированной тройки "
     "направленно осмысленный arm: подавление строк, голосующих за неверное число."),
    ("w57_gold16_l23_amp160", "gold", 1.60,
     "Тот же список 16 строк, доза 1.60: проверка монотонности дозы на механизме scale "
     "в пределах линейного режима gelu (a_j = gelu(lam*g_j)*u_j)."),
    ("w58_gold16_l23_damp053", "gold", 0.53,
     "Тот же список 16 строк, lam = 0.53 = 1 - 12.5/26.7132. Ожидаемое изменение среднего "
     "запасa на неверных задачах (1 - 0.53) x (-26.7132) = -12.56 логита, что равно медиане "
     "маржи 12.5: по смыслу запас к неверному числу уменьшается ровно на медиану."),
    ("w59_flippot16_l23_amp163", "flip", 1.63,
     "16 строк с наибольшим положительным flip_potential (П51.2 опровергнута: сумма "
     "положительных flip_potential по L23 равна 30.7016 при 1451 положительных строках, "
     "сумма по этим 16 строкам 19.9831). Доза 1.63 = 1 + 12.5/19.9831: ожидаемое изменение "
     "среднего запаса 0.63 x 19.9831 = 12.59 логита, то есть не меньше медианы маржи 12.5. "
     "Это единственное направленное усиление, которое может переписать ответ в пользу верного."),
]


def rows_for(kind: str) -> list[int]:
    d = json.loads(PROBE.read_text(encoding="utf-8"))
    l23 = [r for r in d["rows"] if r.get("layer") == LAYER][0]
    key = "top_by_gold" if kind == "gold" else "top_by_flip"
    rows = [int(x["row"]) for x in l23[key]]
    if len(set(rows)) != len(rows):
        raise SystemExit(f"неуникальные строки в {key}: {rows}")
    return rows


def main() -> None:
    for name, kind, lam, hypothesis in ARMS:
        rows = rows_for(kind)
        ops = []
        for r in rows:
            ops.append({
                "tensors": [TENSOR],
                "op": "scale",
                "lam": lam,
                "rows": [r],
                "note": f"wave15 {kind} row {r} lam {lam}",
            })
        recipe = {
            "name": name,
            "hypothesis": hypothesis,
            "expected_effect": (
                "Детерминированная правка 16 строк mlp.gate_proj слоя 23, "
                f"lam = {lam}. Оценочные коридоры: формат 336-348, обрывы <= 14, "
                "математика 216-222, преждевременный FINAL <= 1. Решает dev (100 задач, "
                "не участвовал в отборе строк)."
            ),
            "base_variant": BASE,
            "op": "scale",
            "tensor": TENSOR,
            "rows": rows,
            "lam": lam,
            "formula": f"y_i = lam * x_i для строки i; в линейном режиме dmargin = (lam-1) * sum_flip(top16)",
            "expected_diff_rows": len(rows),
            "ops": ops,
        }
        out = RECIPES / f"{name}.json"
        out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{out.name}: rows={rows} lam={lam} ops={len(ops)}")


if __name__ == "__main__":
    main()
