#!/usr/bin/env python3
"""Генератор рецепта w81 (волна 30, журнал §81.5–§81.6).

Один изменённый признак по сравнению с w50: состав строк. Берутся только четыре
малые строки 12650, 361, 695, 332; шесть средних строк (2972, 271, 1088, 1510,
498, 328) исключены. Шаг остаётся плюс 0,015, оператор `tilt`, база w8, тот же
тензор `model.language_model.embed_tokens.weight`, то же восстановление нормы.
Факторы переносятся из файла recipe w50 без пересчёта: тот же базовый тензор, та же
строка, тот же шаг, значит факторы те же.
Полоса П81.1 записана в журнале §81.6 до сборки.
"""
import json
from pathlib import Path

BASE_RECIPE = Path("recipes/w50_finalrow_rotation15_smallmargin.json")
OUT = Path("recipes/w81_finalrow_four_small_rows_step15_on_w8.json")

BAND = "П81.1"
NAME = "w81_finalrow_four_small_rows_step15_on_w8"
TENSOR = ["model.language_model.embed_tokens.weight"]
SMALL_ROWS = [12650, 361, 695, 332]
STEP = 0.015

HYPOTHESIS = (
    "H56 (волна 30, журнал §81.4–§81.6): выполняется пункт 3 полос П74.2 — "
    "удвоение шага не добавило формата (333 против 335), ответ по рычагу из десяти "
    "строк насытился, серия формата закрывается на w50. Здесь меняется только состав: "
    "четыре строки с малым запасом (12650, 361, 695, 332) без шести средних строк "
    "(2972, 271, 1088, 1510, 498, 328), шаг тот же плюс 0,015, база w8, оператор "
    "`tilt`, тот же тензор `model.language_model.embed_tokens.weight`, тот же "
    "пересчёт факторов. Проверяемая гипотеза H54: качество держится за этими четырьмя "
    "малыми строками, поэтому без шести средних строк качество остаётся на уровне w8 "
    "или выше (по полосе — вместе не ниже 216), а формат падает к порогу кредита. "
    "Сравнение с w50 остаётся одно переменным по составу. Официальный набор 1319 задач "
    "не использовался и не используется для выбора варианта."
)

EXPECTED_EFFECT = (
    "Четыре малые строки дают средний ожидаемый запас после поворота: строка 12650 "
    "по w50 была -0,090, строка 361 +0,101, строка 695 +0,244, строка 332 +0,357 "
    "(по заметкам w50) — то есть вклад меньше, чем у шести средних строк, и формат "
    "ожидаемо будет ниже w50. Полосы П81.1 записаны в журнале §81.6 до сборки: "
    "dev 59–67 / 90–98 / 0–6; search 143–158 / 220–244 / 9–17; combined "
    "202–220 / 310–346 / 10–21. Условие успеха: combined формат не ниже 320 и combined "
    "математика не ниже 216 и обрывы не выше 14 и dev математика не ниже 62. Условие "
    "отказа: combined математика не выше 205 при формате не ниже 315 — тогда малые "
    "строки качество не держат и гипотеза H54 снимается. Значок p справочный. "
    "Решение о качестве принимается на dev и search; официальный набор из 1319 задач "
    "используется только для оценки."
)


def main() -> int:
    if OUT.exists():
        print("файл рецепта уже существует — не перезаписываем:", str(OUT))
        return 1
    rec = json.loads(BASE_RECIPE.read_text(encoding="utf-8"))
    ops = []
    for op in rec["ops"]:
        rows = [int(r) for r in op["rows"]]
        if not all(r in SMALL_ROWS for r in rows):
            continue
        new = dict(op)
        new["tensors"] = TENSOR
        new["s"] = STEP
        new["note"] = (op.get("note", "") +
                       " ; состав по пункту 3 полос П74.2: только четыре малые строки, "
                       "шесть средних строк исключены, шаг не менялся")
        ops.append(new)
    if len(ops) != 4:
        print("собрано операций:", len(ops), "— ожидалось 4, остановка")
        return 1
    order = sorted(range(len(ops)), key=lambda i: -SMALL_ROWS.index(int(ops[i]["rows"][0])))
    ops = [ops[i] for i in order]
    out = {
        "name": NAME,
        "band": BAND,
        "hypothesis": HYPOTHESIS,
        "expected_effect": EXPECTED_EFFECT,
        "tensors": TENSOR,
        "ops": ops,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("список строк по порядку (как в §81.6):", [int(o["rows"][0]) for o in ops])
    print("тип tensors:", "list" if all(isinstance(o["tensors"], list) for o in ops) else "НЕ list")
    print("файл:", str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
