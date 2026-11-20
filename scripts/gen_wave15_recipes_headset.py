#!/usr/bin/env python3
# Рецепты волны 25. w75: одиночное плечо — головной рычаг z, конфигурация выравнивания
# w45 (громкие головы 0,94, тихие головы 1,0208), покрытие ограничено тремя слоями с
# наибольшей измеренной неоднородностью голов (20, 21, 22). Ops берутся дословно из
# рецепта w45, чтобы изменялся только один признак — покрытие.
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "recipes" / "w45_z_headsel_flatten.json"
TENSOR = "model.language_model.layers.*.linear_attn.in_proj_z.weight"
LAYERS = [20, 21, 22]
NAME = "w75_z_flatten_l202122_on_w8"
BAND = "П70.1 (журнал §70.2, записаны до сборки)"


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
            "Одиночное плечо по условиям П70.1: тот же измеренный головной рычаг z и та же "
            "конфигурация выравнивания w45 (громкие головы 0,94, тихие головы 1,0208), но покрытие "
            "сужено до трёх слоёв с наибольшей измеренной неоднородностью голов — 4,28 раза для слоя 20, "
            "7,70 для слоя 21, 8,47 для слоя 22. Изменяется только покрытие: 48 блоков по 128 строк "
            f"(всего {total_rows} строк) против 288 блоков у w45 и 12 блоков у w73. Проверяется, "
            "переживает ли носитель качества сужение покрытия и снижается ли при этом стоимость формата."
        ),
        "expected_effect": (
            f"Полосы П70.1: dev математика 62–67, извлечение 92–97, обрывы 2–7; "
            "search математика 142–152, извлечение 227–238, обрывы 14–21; "
            "вместе математика 204–219, извлечение 319–335, обрывы 16–28. "
            "Средний лог-усилитель каждого из трёх слоёв остаётся на уровне w8 "
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
        print("  слой", o["layers"][0], "ламм", o["lam"], "строк", len(o["rows"]),
              "голов", sorted({r // 128 for r in o["rows"]}))


main()
