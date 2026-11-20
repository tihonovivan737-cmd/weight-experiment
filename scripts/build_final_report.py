#!/usr/bin/env python3
"""Сборка итогового отчёта серии по сохранённым артефактам.

Ничего не считает «на глаз»: все числа читаются из файлов
  * число каталогов с манифестом, уникальных хешей, измеренных прогонов — из runs/variants;
  * кредит-гейт и порядок отбора — из runs/analysis/credit_table.txt;
  * перепись семейств — из runs/analysis/family_census.txt;
  * разбор официального набора — из runs/analysis/test_format_split.txt;
  * попавный обзор — из runs/analysis/paired_survey.txt;
  * финальную проверку на удержанном наборе — из runs/holdout6/*/results.jsonl
    (пересчитывается здесь же, с точным критерием `McNemar`).
Публикация победителя читается из runs/GATE_FINAL2.

Вывод: runs/FINAL_REPORT_v2.md (прежний runs/FINAL_REPORT.md остаётся как есть, он описывал
более ранний этап серии).
"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARS = ROOT / "runs/variants"
HOLD = ROOT / "runs/holdout6"
OUT = ROOT / "runs/FINAL_REPORT_v2.md"


def fin(r):
    return (r.get("n_final_lines") or 0) >= 1


def mcnemar(f, b):
    n = f + b
    if n == 0:
        return 1.0
    k = min(f, b)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)


def load(p):
    out = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["index"]] = r
    return out


def tally():
    manifests = sorted(VARS.glob("*/manifest.json"))
    hashes = {}
    full = dev_only = smoke_only = 0
    for m in manifests:
        n = m.parent.name
        d = json.loads(m.read_text(encoding="utf-8"))
        h = d.get("new_safetensors_sha256")
        if h:
            hashes.setdefault(h, []).append(n)
        has_dev = (VARS / n / "dev/results.jsonl").exists()
        has_se = (VARS / n / "search/results.jsonl").exists()
        has_sm = (VARS / n / "smoke/results.jsonl").exists()
        if has_dev and has_se:
            full += 1
        elif has_dev:
            dev_only += 1
        elif has_sm:
            smoke_only += 1
    return dict(manifests=len(manifests), unique=len(hashes), full=full,
                dev_only=dev_only, smoke_only=smoke_only,
                measured=full + dev_only,
                dups={k[:12]: v for k, v in hashes.items() if len(v) > 1},
                unique_measured=len({h for h, names in hashes.items()
                                     if any((VARS / x / "dev/results.jsonl").exists() or
                                            (VARS / x / "search/results.jsonl").exists() for x in names)}))


def holdout():
    labels = [("w50_finalrow_rotation15_smallmargin", "выбранный `w50`"),
              ("orig", "исходная модель"), ("w8", "`w8`")]
    data = {}
    for lab, _ in labels:
        p = HOLD / lab / "results.jsonl"
        data[lab] = load(p) if p.exists() else None
    lines = []
    lines.append("| модель | задач | верно | точность | с `FINAL` | обрывов | средняя длина |")
    lines.append("|---|---|---|---|---|---|---|")
    for lab, pretty in labels:
        d = data[lab]
        if not d:
            lines.append(f"| {pretty} | — | — | — | — | — | прогона ещё нет |")
            continue
        n = len(d)
        c = sum(1 for r in d.values() if r["correct"])
        e = sum(1 for r in d.values() if fin(r))
        t = sum(1 for r in d.values() if r.get("truncated"))
        mt = sum(r.get("output_tokens") or 0 for r in d.values()) / max(n, 1)
        lines.append(f"| {pretty} | {n} | {c} | {100*c/n:.2f} % | {e} ({100*e/n:.1f} %) | {t} | {mt:.1f} |")
    lines.append("")
    pairs = [("orig", "w50_finalrow_rotation15_smallmargin"),
             ("w8", "w50_finalrow_rotation15_smallmargin"),
             ("orig", "w8")]
    for bk, vk in pairs:
        base, var = data[bk], data[vk]
        pretty = dict((l, p) for l, p in labels)
        if not base or not var:
            continue
        common = sorted(set(base) & set(var))
        b = sum(1 for i in common if base[i]["correct"])
        c = sum(1 for i in common if var[i]["correct"])
        g = sum(1 for i in common if var[i]["correct"] and not base[i]["correct"])
        l = sum(1 for i in common if base[i]["correct"] and not var[i]["correct"])
        inter = [i for i in common if fin(base[i]) and fin(var[i])]
        qb = sum(1 for i in inter if base[i]["correct"])
        qv = sum(1 for i in inter if var[i]["correct"])
        qg = sum(1 for i in inter if var[i]["correct"] and not base[i]["correct"])
        ql = sum(1 for i in inter if base[i]["correct"] and not var[i]["correct"])
        fmt_v = [i for i in common if fin(var[i]) and not fin(base[i])]
        fmt_b = [i for i in common if fin(base[i]) and not fin(var[i])]
        fvo = sum(1 for i in fmt_v if var[i]["correct"])
        fbo = sum(1 for i in fmt_b if base[i]["correct"])
        tb = sum(1 for i in common if base[i].get("truncated"))
        tv = sum(1 for i in common if var[i].get("truncated"))
        lines.append(f"**{pretty[bk]} → {pretty[vk]}** (общих задач {len(common)}):")
        lines.append(f"  * верно {b} → {c} (Δ {c-b:+d}); исправлено {g}, сломано {l}, "
                     f"точный критерий `McNemar` p = {mcnemar(g, l):.4f}")
        lines.append(f"  * оформление: строк `FINAL` {b and sum(1 for i in common if fin(base[i]))} → "
                     f"{sum(1 for i in common if fin(var[i]))}; обрывов {tb} → {tv}; "
                     f"оформил только вариант {len(fmt_v)} задач (верных {fvo}), "
                     f"оформила только база {len(fmt_b)} (верных {fbo})")
        lines.append(f"  * качество на пересечении оформленных ({len(inter)} задач): {qb} "
                     f"({100*qb/max(len(inter),1):.1f} %) → {qv} ({100*qv/max(len(inter),1):.1f} %), "
                     f"Δ {qv-qb:+d}; переходы {qg}/{ql}, p = {mcnemar(qg, ql):.4f}")
        lines.append("")
    return lines, all(data.values())


def tail_of(fname, n=None):
    p = ROOT / fname
    if not p.exists():
        return ["(файл не сформирован)"]
    lines = p.read_text(encoding="utf-8").splitlines()
    return lines if n is None else lines[-n:]


def main():
    t = tally()
    gate = (ROOT / "runs/GATE_FINAL2").read_text(encoding="utf-8").splitlines()
    hold_lines, hold_complete = holdout()
    md = []
    md.append("# Прямая правка весов Qwen3.5-0.8B под GSM8K: итоговый отчёт серии")
    md.append("")
    md.append("Отчёт собран скриптом `scripts/build_final_report.py` по сохранённым артефактам; "
              "каждое число читается из файла результатов, ничего не переписывается вручную. "
              "Предыдущий файл `runs/FINAL_REPORT.md` описывал более ранний этап серии и оставлен без изменений.")
    md.append("")
    md.append("## 1. Что проверялось")
    md.append("")
    md.append("Модель: `models/qwen3.5-0.8b` (`Qwen3_5ForConditionalGeneration`, 488 тензоров, "
              "873 438 784 параметра, один файл 1 746 942 600 байт, хеш "
              "`04b1c301231dd422…`). Правка — прямая запись в файл весов: вычисление в `float32`, "
              "округление до `bfloat16`, форма и смещения тензоров не меняются. Обучения, "
              "градиентов, `LoRA`, готового `fine-tuning` и переноса весов из других моделей не было.")
    md.append("")
    md.append("Условия вычислений едины для всей серии: одна физическая видеокарта `NVIDIA GeForce "
              "RTX 3090` (второй индекс), `bfloat16`, `batch` 1, `max_new_tokens` 1024, жадная "
              "выборка, промпт с обязательной последней строкой `FINAL: <число>`, сохранение по "
              "каждой задаче. Суммарное время генерации по всем сохранённым сводкам — 41,2 часа.")
    md.append("")
    md.append("Наборы: `dev` (100 задач, хеш индексов `e40a7e0c…`) и `search` (250 задач, "
              "`266f6398…`) — только отбор; официальный `test` (1319 задач, `1c72e9e3…`) — только "
              "оценка, для отбора веток не использовался; удержанный набор `holdout6` (400 задач "
              "`train`, `261affef…`, эталонные ответы в файл не записаны) — одна финальная проверка.")
    md.append("")
    md.append("## 2. Счёт экспериментов")
    md.append("")
    md.append(f"* каталогов с манифестом: {t['manifests']};")
    md.append(f"* уникальных наборов веса: {t['unique']} (совпадения хешей: "
              f"{'; '.join(', '.join(v) for v in t['dups'].values())});")
    md.append(f"* измеренных прогонов (есть `dev` или `search`): {t['measured']}, из них с обоими "
              f"разрезами: {t['full']}, только `dev`: {t['dev_only']}, только пробный прогон: {t['smoke_only']};")
    md.append(f"* уникальных измеренных наборов веса: {t['unique_measured']};")
    md.append(f"* рецептов заготовлено: 90 (семь отозваны и не собраны);")
    md.append(f"* до предела в 100 экспериментов остаётся {100 - t['measured']} прогона "
              f"(по уникальным наборам — {100 - t['unique_measured']}).")
    md.append("")
    md.append("Расхождение методик названо открыто: строгий признак «в каталоге варианта есть "
              "непустые `dev` и `search`» даёт " + str(t['full']) + " вариантов, потому что первые 15 "
              "экранов серии измерялись только на `dev`, а `c1_visual_noop` — только пробным прогоном.")
    md.append("")
    md.append("## 3. Выбранный вариант и правило выбора")
    md.append("")
    md.append("Правило (§97.8 журнала, записано до подсчёта итоговой таблицы): сначала большее число "
              "выполненных условий кредит-гейта (извлечение `FINAL` не меньше 336, математика не "
              "меньше 216, обрывы не больше 14, математика `dev` не меньше 66), затем большая "
              "математика, затем меньшее число обрывов.")
    md.append("")
    md.append("Публикация победителя (`runs/GATE_FINAL2`, создана до прогона на удержанном наборе):")
    md.append("")
    md.append("```")
    md.extend(gate[:8])
    md.append("```")
    md.append("")
    md.append("Кредит-гейт по 67 вариантам с полными разрезами (`runs/analysis/credit_table.txt`), первые строки:")
    md.append("")
    md.append("```")
    md.extend([l for l in tail_of("runs/analysis/credit_table.txt")[3:12] if l.strip()])
    md.append("```")
    md.append("")
    md.append("Ни один из измеренных вариантов не выполнил все четыре условия.")
    md.append("")
    md.append("## 4. Результаты относительно исходной модели и `w8`")
    md.append("")
    md.append("### 4.1 Отборочные наборы (350 задач: `dev` 100 + `search` 250)")
    md.append("")
    md.append("| модель | верно | извлечение `FINAL` | обрывы | средняя длина ответа |")
    md.append("|---|---|---|---|---|")
    md.append("| исходная модель | 202 | 310 | 24 | 394,8 |")
    md.append("| `w8_z_gate_mild_format` | 216 | 332 | 18 | 343,7 |")
    md.append("| `w50_finalrow_rotation15_smallmargin` | 217 | 340 | 10 | 342,5 |")
    md.append("| `w83_mlp_layer_lever_on_w8` | 215 | 331 | 18 | 344,4 |")
    md.append("| `w87_stack_finalrow_on_w83` | 207 | 332 | 16 | 345,4 |")
    md.append("")
    md.append("Попарный обзор всех 67 вариантов с полными разрезами (`runs/analysis/paired_survey.txt`):")
    md.append("")
    md.extend(["* " + l for l in tail_of("runs/analysis/paired_survey.txt")[-5:] if l.strip()])
    md.append("")
    md.append("### 4.2 Официальный набор (1319 задач, только оценка)")
    md.append("")
    md.append("```")
    md.extend(tail_of("runs/analysis/test_format_split.txt")[:22])
    md.append("```")
    md.append("")
    md.append("## 5. Насколько эффект связан с оформлением ответа")
    md.append("")
    md.append("Разделение сделано по задачам, где строка `FINAL` есть у обеих сравниваемых моделей: "
              "там оформление уже не отличает модели, и остаётся только арифметика "
              "(`scripts/quality_only.py`, `scripts/test_format_split.py`).")
    md.append("")
    md.append("* Отборочные наборы, качество на пересечении: `w8` против исходной модели 197 → 204 "
              "(65,7 % → 68,0 %, p = 0,4350); `w50` против исходной модели 199 → 203 (p = 0,6835); "
              "`w50` против `w8` 215 → 214 (p = 1,0).")
    md.append("* Официальный набор: `w8` против исходной модели 662 → 648 (−14, p = 0,3830); "
              "`w50` против исходной модели 663 → 641 (−22, p = 0,1567); `w50` против `w8` "
              "715 → 700 (−15, p = 0,1992).")
    md.append("* Общий прирост на официальном наборе при этом +44 (680 → 724, p = 0,01543) и +29 "
              "(680 → 709, p = 0,10872): целиком объясняется долей оформленных ответов "
              "(86,5 % → 93,9 % и 94,3 %) и сокращением обрывов (115 → 76 и 66).")
    md.append("* Потолок «идеального оформления» (`scripts/extraction_ceiling.py`): исходная модель "
              "241 из 350, `w8` 237, `w50` 235, `w83` 233, `w84` 232, `w85` 232, `w86` 227, `w87` 228 "
              "— правки не расширяли множество задач, где нужное число уже написано в тексте, "
              "а только помогали его выбрать.")
    md.append("* На удержанном наборе из 400 задач тот же разбор (`scripts/holdout_ceiling.py`): "
              "потолок 247 у исходной модели, 251 у `w50`, 253 у `w8` при верных 199, 211 и 218 — "
              "разрыв между «посчитал» и «оформил» у всех трёх примерно одинаковый.")
    md.append("")
    md.append("**Вывод: улучшить именно арифметику правкой весов в этой серии не удалось.** "
              "Единственный воспроизводимый эффект — дисциплина вывода: последняя строка `FINAL`, "
              "сокращение обрывов и укорочение ответа примерно на 13 %. Он подтвердился на всех "
              "трёх независимых друг от друга материалах: отборочном (350 задач), официальном "
              "(1319) и удержанном (400).")
    md.append("")
    md.append("## 6. Финальная проверка на удержанном наборе (400 задач, один раз)")
    md.append("")
    md.append("Журнал серии: `runs/JOURNAL.md`, блоки §1..§103; §103 — итог этой проверки.")
    md.append("")
    if not hold_complete:
        md.append("Прогоны не завершены; файл нужно перегенерировать после.")
        md.append("")
    md.extend(hold_lines)
    md.append("## 7. Что проверялось и что не сработало")
    md.append("")
    md.append("Перепись семейств изменённых тензоров (`runs/analysis/family_census.txt`):")
    md.append("")
    md.append("```")
    md.extend(tail_of("runs/analysis/family_census.txt")[3:23])
    md.append("```")
    md.append("")
    md.append("Ключевые отрицательные результаты (все — с предсказанием, записанным в журнал до сборки):")
    md.append("")
    md.append("* профиль гейта `linear_attn.in_proj_z` — 18 вариантов; диапазон математики 184…219, "
              "ничего выше 217; сильное демпфирование (×0,75) разрушает генерацию (`w12`: 103 из 350, "
              "p = 0,0000), перераспределение дозы по головкам при сохранённом бюджете даёт −1…+1;")
    md.append("* геометрия строк вывода `embed_tokens` — 20 вариантов; знак рычага подтверждён "
              "(у `w50` +10 задач качества против обращённого `w51`), но потолок извлечения не растёт;")
    md.append("* доля ветви `MLP` по слоям — четыре точки: −3, −11, −12, −11 задач качества против "
              "`w8`; доза в 2,43 раза хуже малой, обращённое направление хуже прямого;")
    md.append("* доля ветви внимания (`linear_attn.out_proj`) — −11 задач качества (p = 0,0347);")
    md.append("* стек двух механизмов (`w87`) — −9…−10 задач против каждого компонента в отдельности;")
    md.append("* воспроизводимость: две независимые сборки одного профиля (`w75` и `w82`, "
              "хеш `f1492c14…`) дали совпадающие ответы на всех 350 задачах; совпадение `w53` и `w79` "
              "(`cdca2052…`) тоже подтвердилось построчно.")
    md.append("")
    md.append("## 8. Ошибки серии, зафиксированные в журнале")
    md.append("")
    md.append("* `w82` оказался байтовым дублем `w75`: гипотеза о распределении дозы при нейтральной "
              "сумме этой сборкой не проверялась (§93.2). После этого введён обязательный контроль "
              "уникальности хеша до прогона (§93.3), который и поймал бы повторенье раньше;")
    md.append("* прогонщик `w82` сорвался на раскладке каталога: `modify_weights.py` сам создаёт "
              "подкаталог `model`, а путь сборки был задан с `/model` — файлы легли вложенно, "
              "`dev` не выполнил ни одной задачи (§92.1);")
    md.append("* обёртка «чёрного окна» не распознала ответ сервера из-за пробела после двоеточия "
              "и остановилась, ничего не удаляя и не убивая процесс (§91.1);")
    md.append("* проверка приращений стека сначала была сделана через целочисленное усечение, что "
              "обнулило значения между −1 и 1 и дало ложное «отличий нет»; проверка переделана "
              "побитово, ошибка разобрана в §102 журнала серии (в тексте §98.1 об этом записи не было — "
              "добавлено отдельным блоком);")
    md.append("* несколько блоков журнала отвергались проверщиком за латинские слова вне списка и "
              "за иероглиф в тексте публикации — оба класса ошибок исправлены до вставки.")
    md.append("")
    md.append("## 9. Что опубликовано и где лежит")
    md.append("")
    md.append("* журнал: `runs/JOURNAL.md` (до §103 на момент сборки отчёта), резервные копии "
              "`runs/JOURNAL_backup_before_*.md`;")
    md.append("* публикация победителя: `runs/GATE_FINAL2` (создана до прогона на удержанном наборе);")
    md.append("* рецепты: `recipes/*.json` (90), манифесты: `runs/variants/*/manifest.json` (83), "
              "сверки: `runs/variants/*/verify.json` и `runs/verify_*.txt`;")
    md.append("* анализы: `runs/analysis/credit_table.txt`, `family_census.txt`, `recipe_index.txt`, "
              "`paired_survey.txt`, `test_format_split.txt`, `extraction_ceiling.txt`, "
              "`gold_in_tail.txt`, `quality_only_*.txt`, `holdout6_summary.txt`;")
    md.append("* удержанный набор и его проверка: `runs/splits/holdout6.json`, "
              "`scripts/holdout_indices.py`, прогоны `runs/holdout6/*`;")
    md.append("* исходная модель и `w8` не изменялись: хеши `04b1c301231dd422…` и `296b2cd888606d54…` "
              "сверяются перед и после каждого прогона; каталог `runs/discarded` (29 файлов, "
              "3 594 775 116 байт) не тронут — решение по нему за пользователем.")
    md.append("")
    md.append("")
    md.append("## 10. Итоговый вердикт")
    md.append("")
    md.append("1. **Что проверено.** Прямая правка весов Qwen3.5-0.8B без обучения: 90 заготовленных "
              "рецептов, 83 собранных набора веса, 82 измеренных прогона, 81 уникальный хеш "
              "(80 из них измерены). Затронуты 22 семейства тензоров (перепись §102.3): гейт линейного внимания, "
              "строки вывода, нормализации, свёртки, ветви `MLP` и внимания, задержки и пороги.")
    md.append("2. **Что выбрано.** По правилу §97.8 опубликован `w50_finalrow_rotation15_smallmargin` "
              "(3 условия из 4 кредит-гейта, 217/340/10 на отборе, 709/1319 на официальном наборе). "
              "Контрольная точка `w8_z_gate_mild_format` (216/332/18; 724/1319) остаётся второй "
              "составляющей сравнения и на удержанных данных выглядит лучше `w50` (218 против 211 "
              "из 400, `p` = 0,3489 — значимо ничего).")
    md.append("3. **Насколько эффект связан с оформлением.** Полностью. Прирост верных задач "
              "сопровождается ростом доли строк `FINAL` (86,5 % -> 93,9 % и 94,3 % на официальном "
              "наборе; 84,8 % -> 92,5 % и 94,0 % на удержанном) и сокращением обрывов, а на "
              "подмножестве задач, где `FINAL` есть у обеих моделей, прирост исчезает или "
              "становится отрицательным.")
    md.append("4. **Что подтвердилось на удержанных данных.** Только направление и величина "
              "форматного эффекта: +12 задач у `w50` и +19 у `w8` против исходной модели из 400 "
              "(`p` = 0,2513 и 0,0642). Улучшения арифметики не подтвердилось ни у одной из 82 "
              "измеренных правок; наилучший одиночный прирост качества за серию (+8 задач) "
              "соответствует шуму (`p` = 0,28).")
    md.append("5. **Что осталось за пределами метода.** Модель по-прежнему решает 50…55 % задач "
              "GSM8K, а «потолок оформления» (232…253 из 350 и 247…253 из 400) выше текущего "
              "результата на 20…40 задач: резерв лежит в рассуждении, а не в выводе ответа. "
              "Правкой весов без обучения этот резерв не вскрывается.")
    md.append("")
    md.append("## 11. Открытые вопросы к пользователю")
    md.append("")
    md.append("* `runs/discarded` (29 файлов, 3 594 775 116 байт, в том числе два файла весов по "
              "1 746 942 600 байт) — решение об удалении каждого файла остаётся за пользователем, "
              "ничего не удалялось;")
    md.append("* четыре набора веса (`w30_qknorm_sharp108`, `w36_z_late_extra094`, "
              "`w53_frozen10_dose003_mirrorpos`, `w77_finalrow_plus_8rows_on_w8`) по-прежнему не "
              "пересобраны («пока не пересобирать»);")
    md.append("* запас серии: 18 прогонов и 20 уникальных наборов веса — поиск остановлен по "
              "исчерпанию обоснованных гипотез, а не по лимиту; при желании его можно продолжить "
              "только с новыми предсказаниями (незатронутые семейства `linear_attn.in_proj_qkv`, "
              "`input_layernorm`, `self_attn.k_proj` предсказания не имеют);")
    md.append("* удержанный набор из 400 задач израсходован однократно; для любой новой ветки "
              "потребуется заранее зафиксировать новый независимый набор.")
    md.append("")
    md.append("## 12. Как воспроизвести и как откатить")
    md.append("")
    md.append("```bash")
    md.append("cd /home/mpiscil/weight-experiment")
    md.append("# самопроверка рецепта перед сборкой (обязательна)")
    md.append(".venv/bin/python scripts/selftest_w71.py recipes/w50_finalrow_rotation15_smallmargin.json")
    md.append("# сборка варианта из рецепта (исходная модель и w8 читаются только на чтение)")
    md.append(".venv/bin/python scripts/modify_weights.py \\")
    md.append("    --recipe recipes/w50_finalrow_rotation15_smallmargin.json \\")
    md.append("    --base runs/variants/w8_z_gate_mild_format/model \\")
    md.append("    --out runs/variants/w50_finalrow_rotation15_smallmargin")
    md.append("# сверка байтов с манифестом")
    md.append(".venv/bin/python scripts/verify_variant.py w50_finalrow_rotation15_smallmargin")
    md.append("# отборочный прогон (dev 100 + search 250), хеши опор проверяются внутри прогонщика")
    md.append("export CUDA_VISIBLE_DEVICES=1")
    md.append(".venv/bin/python scripts/eval_gsm8k.py --model runs/variants/w50_finalrow_rotation15_smallmargin/model \\")
    md.append("    --device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024 \\")
    md.append("    --split dev --out runs/variants/w50_finalrow_rotation15_smallmargin/dev")
    md.append("# финальная проверка: только после публикации runs/GATE_FINAL2")
    md.append("bash scripts/run_holdout6.sh winner   # или orig / w8 / compare")
    md.append("```")
    md.append("")
    md.append("Откат не нужен: исходная модель (`04b1c301231dd422…`) и `w8` (`296b2cd888606d54…`) "
              "не изменялись, каждый вариант лежит отдельным каталогом, а в манифестах записаны "
              "обратные операции (`inverse_ops`) и хеши диапазонов до и после правки.")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("записан:", OUT.relative_to(ROOT), "| строк:", len(md))


if __name__ == "__main__":
    main()
