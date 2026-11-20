#!/usr/bin/env python3
# Рецепты волны 26. w76: одиночное плечо — тот же измеренный головной рычаг z и та же
# конфигурация выравнивания w45 (громкие головы 0,94, тихие головы 1,0208), но покрытие
# расширено с 48 до 80 блоков: пять слоёв с наибольшей измеренной неоднородностью голов
# по runs/analysis/wave11_head_sets.txt — 22 (8,47 раза), 21 (7,70), 4 (6,96), 16 (6,23),
# 8 (6,04). Ops берутся дословно из рецепта w45, чтобы изменялся только один признак.
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "recipes" / "w45_z_headsel_flatten.json"
TENSOR = "model.language_model.layers.*.linear_attn.in_proj_z.weight"
LAYERS = [4, 8, 16, 21, 22]
NAME = "w76_z_flatten_top5_on_w8"
BAND = "П71.1 (журнал §71.2, записаны до сборки)"


def layer_of(op):
    lay = op.get("layers")
    return lay[0] if isinstance(lay, list) else lay


def head_rows(head):
    return list(range(head * 128, head * 128 + 128))


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    ops = []
    for o in src["ops"]:
        if layer_of(o) not in LAYERS:
            continue
        heads = sorted({r // 128 for r in o["rows"]})
        rows = []
        for h in heads:
            rows.extend(head_rows(h))
        ops.append({
            "op": o["op"],
            "tensors": [TENSOR],
            "layers": [layer_of(o)],
            "rows": sorted(rows),
            "lam": o["lam"],
            "note": f"перенесено дословно из w45: слой {layer_of(o)}, "
                    f"головы {heads}, ламбда {o['lam']}, строк {len(rows)}",
        })
    ops.sort(key=lambda o: (o["layers"][0], o["lam"]))

    rows_loud = sum(len(o["rows"]) for o in ops if o["lam"] < 1.0)
    rows_quiet = sum(len(o["rows"]) for o in ops if o["lam"] > 1.0)
    total_rows = rows_loud + rows_quiet

    recipe = {
        "name": NAME,
        "hypothesis": (
            "Одиночное плечо по полосам П71.1: тот же измеренный головной рычаг z и та же "
            "конфигурация выравнивания w45 (громкие головы 0,94, тихие головы 1,0208), но покрытие "
            "расширено с 48 до 80 блоков: пять слоёв с наибольшей измеренной неоднородностью голов "
            "по disk-файлу wave11_head_sets.txt — 22 (8,47 раза), 21 (7,70), 4 (6,96), 16 (6,23), "
            "8 (6,04). Изменяется только покрытие: 80 блоков по 128 строк "
            f"(всего {total_rows} строк) против 48 у w75 и 288 у w45. Проверяется, растёт ли "
            "математика dev вместе с покрытием и остаётся ли профиль формата на уровне w8."
        ),
        "expected_effect": (
            "Полосы П71.1: dev математика 62–67, извлечение 92–96, обрывы 2–6; "
            "search математика 144–152, извлечение 227–238, обрывы 12–20; "
            "вместе математика 206–219, извлечение 321–335, обрывы 14–25. "
            "Средний лог-усилитель каждого слоя остаётся на уровне w8 "
            "(4·ln0,94 + 12·ln1,0208 = 0), то есть движет только распределение по головам. "
            "Значения McNemar — только информация. Базой является w8, база указана явно."
        ),
        "band": BAND,
        "tensors": [TENSOR],
        "ops": ops,
    }
    out = ROOT / "recipes" / f"{NAME}.json"
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("рецепт:", out)
    print("ops:", len(ops), "строк громких:", rows_loud, "строк тихих:", rows_quiet,
          "всего строк:", total_rows, "блоков:", total_rows // 128)
    print("планируемых изменённых значений:", total_rows * 1024)
    for o in ops:
        print("  слой", o["layers"][0], "ламбда", o["lam"], "строк", len(o["rows"]),
              "голов", sorted({r // 128 for r in o["rows"]}))


main()
