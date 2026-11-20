#!/usr/bin/env python
"""Генератор рецепта головного рычага (журнал §82.6, полосы П82.1).

Одно переменное сравнение — направление дозы по головкам тензора
model.language_model.layers.*.linear_attn.in_proj_z.weight в трёх слоях 20, 21, 22.
Четыре головки (громкие по волне 11) демпфируются множителем 0,94, остальные двенадцать
множителем 1,0208. Сумарная доза по слою нейтральна:
4 * ln(0,94) + 12 * ln(1,0208) = -0,247501 + 0,247039 = -0,000462 (с точностью округления ноль).

Блок головки h — строки от h * 128 до h * 128 + 127 (правило head_block из П82.1).
Списки головок берутся из измеренного файла runs/analysis/wave11_head_sets.txt, вручную
не оцениваются. Операнд только scale, восстановление нормы не применяется.

Рецепт: recipes/w82_wave11_head_lever_on_w8.json
База сборки — контрольная точка w8_z_gate_mild_format, читается только на чтение.
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEAS = ROOT / "runs/analysis/wave11_head_sets.txt"
TENSOR = "model.language_model.layers.*.linear_attn.in_proj_z.weight"
LAYERS = [20, 21, 22]
LAM_LOUD = 0.94
LAM_REST = 1.0208
HEAD_DIM = 128
BAND = "П82.1"
NAME = "w82_wave11_head_lever_on_w8"
PLANNED_BAND = ("dev 59..66 / 90..98 / 0..6; search 147..157 / 228..244 / 9..17; "
                "combined 206..220 / 320..348 / 9..23")

NUM = re.compile(r"^\s*слой\s+(\d+):\s*громкие\s+\[([^\]]+)\]\s+\(([^)]+)\)\s+\|\s*тихие\s+\[([^\]]+)\]")
VAL = re.compile(r"\(([^\]]+)\)\s+\|\s*тихие\s+\[([^\]]+)\]")


def parse_head_sets(path: Path):
    """Слои -> (список громких головок, значения |silu(z)| по головке)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = NUM.match(line)
        if not m:
            continue
        layer = int(m.group(1))
        heads = [int(v) for v in m.group(2).split(",")]
        vals = [float(v) for v in m.group(3).split(",")]
        out[layer] = (heads, dict(zip(heads, vals)))
    return out


def head_block(h: int):
    return [h * HEAD_DIM + i for i in range(HEAD_DIM)]


def main():
    sets = parse_head_sets(MEAS)
    missing = [l for l in LAYERS if l not in sets]
    if missing:
        raise SystemExit(f"нет слоёв в измеренном файле: {missing}")

    ops = []
    print(f"измеренный файл: {MEAS.relative_to(ROOT)}")
    for layer in LAYERS:
        loud, vals = sets[layer]
        if len(loud) != 4:
            raise SystemExit(f"слой {layer}: ожидалось 4 громкие головки, получено {len(loud)}")
        dose = 4 * math.log(LAM_LOUD) + 12 * math.log(LAM_REST)
        print(f"слой {layer}: громкие {loud} | доза по слою {dose:+.6f}")
        for h in range(16):
            lam = LAM_LOUD if h in loud else LAM_REST
            rows = head_block(h)
            if h in loud:
                note = (f"слой {layer}, головка {h} — громкая по волне 11, "
                        f"среднее |silu(z)| {vals[h]:.4f}, множитель {lam}; блок строк "
                        f"{rows[0]}..{rows[-1]} по правилу head_block; операнд scale, "
                        f"без восстановления нормы; полоса П82.1 записана до сборки (§82.6)")
            else:
                note = (f"слой {layer}, головка {h} — не входит в четыре громкие, "
                        f"множитель {lam}; блок строк {rows[0]}..{rows[-1]} по правилу "
                        f"head_block; сумарная доза по слою нейтральна; полоса П82.1 "
                        f"записана до сборки (§82.6)")
            ops.append({
                "tensors": [TENSOR],
                "op": "scale",
                "lam": lam,
                "layers": [layer],
                "rows": rows,
                "note": note,
            })

    recipe = {
        "name": NAME,
        "hypothesis": (
            "H55 (волна 31, журнал §82.6): одно переменное — направление дозы по головкам "
            "in_proj_z в слоях 20, 21, 22. Четыре громкие головки (20: 6, 9, 11, 13; "
            "21: 5, 8, 10, 14; 22: 0, 2, 7, 12) демпфируются до 0,94, остальные двенадцать "
            "поднимаются до 1,0208, поэтому сумарная доза по слою нейтральна. Проверяемая "
            "гипотеза: качество можно изменить не величиной дозы, а её распределением по "
            "головкам, при неизменной сумарной дозе. База — w8, операнд только scale, "
            "восстановление нормы не применяется. Никаких изменений строк embed_tokens "
            "и никаких других тензоров."
        ),
        "expected_effect": (
            f"48 блоков строк по {HEAD_DIM} строк (16 головок на три слоя), всего "
            f"{48 * HEAD_DIM} строк и {48 * HEAD_DIM * 1024} планируемых изменённых значений "
            f"до округления до bfloat16. Сумарная доза по слою "
            f"{4 * math.log(LAM_LOUD) + 12 * math.log(LAM_REST):+.6f}. Ожидание по полосам "
            f"П82.1, записанным до сборки (§82.6): {PLANNED_BAND}. Условие прохода: combined "
            f"математика не меньше 212, combined извлечение FINAL не меньше 332, обрывы не "
            f"больше 13, dev математика не меньше 59. Решение о качестве принимается на dev "
            f"и search; официальный набор из 1319 задач используется только для оценки."
        ),
        "band": BAND,
        "ops": ops,
    }

    out = ROOT / "recipes" / f"{NAME}.json"
    if out.exists():
        raise SystemExit(f"файл уже существует, не перезаписываю: {out}")
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")

    types = {tuple(sorted(o["tensors"])) for o in ops}
    print("тип tensors:", "list" if all(isinstance(o["tensors"], list) for o in ops) else "не list")
    print(f"файл: {out.relative_to(ROOT)}, ops {len(ops)}")
    print("строк:", sum(len(o["rows"]) for o in ops))
    print("планируемых изменённых значений:", sum(len(o["rows"]) * 1024 for o in ops))
    print("полоса рецепта:", recipe["band"])
    print("значение `band` в рецепте совпадает с полосой П82.1 из журнала §82.6.")


main()
