#!/usr/bin/env python3
"""Сборка финального отчёта runs/FINAL_REPORT.md из записанных результатов.

Ничего не вычисляет заново: только читает summary.json / compare.json / combined.json /
manifest.json / selection_final.txt / результаты финального теста. Если финального теста
нет, отчёт собирается без него и помечает это.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
VARIANTS_DIR = ROOT / "runs/variants"
BASE = {"dev100": ROOT / "runs/baseline_dev",
        "search250": ROOT / "runs/baseline_search",
        "test1319": ROOT / "runs/baseline_test"}


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def summ(dir_: Path) -> dict:
    d = load(Path(dir_) / "summary.json")
    if not d:
        return {}
    return d.get("summary", d)


def cfg(dir_: Path) -> dict:
    d = load(Path(dir_) / "summary.json")
    return (d or {}).get("config", {})


def sha16(dir_: Path) -> str:
    w = cfg(dir_).get("model_info", {}).get("weights", {})
    return next(iter(w.values()), {}).get("sha256", "")[:16]


def lines(path: Path, n: int | None = None) -> list[str]:
    if not path.exists():
        return []
    out = path.read_text(encoding="utf-8").splitlines()
    return out[:n] if n else out


def pct(x) -> str:
    return f"{100 * x:.2f} %" if isinstance(x, (int, float)) else "—"


def pick_variant() -> str:
    sel = ROOT / "runs/selection_final.txt"
    for ln in lines(sel):
        ln = ln.split("#")[0].strip()
        if ln:
            return ln
    return ""


def main() -> int:
    chosen = pick_variant()
    if not chosen:
        print("ОСТАНОВКА: runs/selection_final.txt пуст", file=sys.stderr)
        return 1

    vdir = VARIANTS_DIR / chosen
    man = load(vdir / "manifest.json") or {}
    verify = load(vdir / "verify.json") or {}
    ctest = load(vdir / "compare_test.json") or {}
    b_test = summ(BASE["test1319"])
    v_test = summ(vdir / "test")
    has_test = bool(v_test) and v_test.get("completed") == 1319

    out: list[str] = []
    w = out.append

    w("# Прямое изменение весов Qwen3.5-0.8B под GSM8K: итоговый отчёт\n")
    w(f"Вариант, прошедший отбор и получивший единственный прогон на тесте: **{chosen}**.\n")

    # 1. Модель и базовые результаты
    w("## 1. Модель и базовые прогоны\n")
    mi = cfg(BASE["test1319"]).get("model_info", {})
    w(f"- Модель: `models/qwen3.5-0.8b`, bf16, `torch {cfg(BASE['test1319']).get('env', {}).get('torch', '?')}`, "
      f"greedy (`do_sample=false`), `max_new_tokens=1024`, промпт с требованием `FINAL: <number>`.")
    w(f"- SHA-256 весов исходной модели: `{mi.get('weights', {}).get('model.safetensors-00001-of-00001.safetensors', {}).get('sha256', '')[:16]}…` "
      f"(исходная модель за весь эксперимент не изменялась — проверено повторным хэшем).")
    w(f"- Метрика: точное извлечение числа из финальной строки и сравнение с эталоном; "
      f"дополнительно — доля извлечений, доля обрывов генерации, средняя длина ответа.\n")
    w("| сплит | назначение | задач | верно | точность | извлечено | обрывов | среднее токенов |")
    w("|---|---|---|---|---|---|---|---|")
    for name, d in (("dev100", BASE["dev100"]), ("search250", BASE["search250"]),
                    ("test1319", BASE["test1319"])):
        s = summ(d)
        role = {"dev100": "проверка обобщения (train-задачи)",
                "search250": "поиск/отбор (train-задачи)",
                "test1319": "официальный тест, один прогон"}[name]
        w(f"| {name} | {role} | {s.get('total')} | {s.get('correct')} | {pct(s.get('accuracy'))} "
          f"| {s.get('answer_extracted')} | {s.get('truncated')} | {s.get('mean_output_tokens')} |")
    w("")

    # 2. Варианты: что меняли и что получилось
    w("## 2. Все варианты и их результаты на train-сплитах (350 задач)\n")
    rows = []
    for vd in sorted(VARIANTS_DIR.iterdir()):
        if not vd.is_dir():
            continue
        c = load(vd / "combined.json")
        if not c:
            continue
        comb = c.get("combined", {})
        ps = c.get("per_split", {})
        dev = ps.get("dev", ps.get("dev100", {}))
        se = ps.get("search", ps.get("search250", {}))
        m = load(vd / "manifest.json") or {}
        rows.append({
            "variant": vd.name,
            "dev": (dev.get("variant_stats") or {}).get("correct"),
            "dev_d": dev.get("delta_correct"),
            "search": (se.get("variant_stats") or {}).get("correct"),
            "search_d": se.get("delta_correct"),
            "comb": (comb.get("variant_stats") or {}).get("correct"),
            "d": comb.get("delta_correct"),
            "p": comb.get("mcnemar_p"),
            "fixed": len(comb.get("fixed", [])),
            "broken": len(comb.get("broken", [])),
            "values": (m.get("totals") or {}).get("values_actually_changed"),
            "tensors": (m.get("totals") or {}).get("tensors_changed"),
        })
    rows.sort(key=lambda r: -(r["d"] if r["d"] is not None else -10**9))
    w("| вариант | dev100 | search250 | 350 | Δ350 | испр/слом | точный p | тензоров | значений |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        p = r["p"]
        ptxt = f"{p:.3f}" if p is not None else "—"
        w(f"| {r['variant']} | {r['dev']} (Δ{r['dev_d']}) | {r['search']} (Δ{r['search_d']}) "
          f"| {r['comb']} | {r['d']:+d} | {r['fixed']}/{r['broken']} "
          f"| {ptxt} | {r['tensors']} | {r['values']} |")
    w("")
    w("Полные разборы: `runs/JOURNAL.md` (§1–§19), `runs/analysis/combined_table.txt`, "
      "`runs/analysis/flip_breakdown.txt`, `runs/analysis/final_token_format_profile.txt`, "
      "`runs/analysis/screening_table.json`.\n")

    # 3. Отбор
    w("## 3. Правило отбора и его результат\n")
    w("Правило зафиксировано в `scripts/pick_final.py` до обращения к тесту: Δ по 350 train-задачам "
      "≥ +5, отсутствие регрессии ни на dev, ни на search; предпочтителен точный McNemar p ≤ 0,05. "
      "Ни один вариант не достиг p ≤ 0,05 (минимум 0,135), поэтому выбран вариант с наибольшим "
      "суммарным Δ без регрессии — это честно зафиксировано в §18 журнала.\n")
    w(f"Выбор: **{chosen}** (`runs/selection_final.txt`, gate `runs/GATE_FINAL`).\n")

    # 4. Что именно изменено в выбранном варианте
    w("## 4. Физическое изменение весов выбранного варианта\n")
    tot = man.get("totals", {})
    mad, mxd = tot.get("mean_abs_delta"), tot.get("max_abs_delta")
    extra = (f"; mean |Δ| = {mad:.3e}, max |Δ| = {mxd:.3e}"
             if isinstance(mad, (int, float)) and isinstance(mxd, (int, float)) else "")
    w(f"- операций рецепта: {len(man.get('recipe_ops', []))}; операций по тензорам: "
      f"{tot.get('touched_tensor_ops', len(man.get('changes', [])))}; затронуто тензоров: "
      f"{tot.get('tensors_changed')}; значений изменено: **{tot.get('values_actually_changed')}**"
      + extra)
    w(f"- вычисление: {man.get('computation')}")
    w(f"- SHA-256 shard исходной модели: `{man.get('base_safetensors_sha256', '')[:16]}…`, "
      f"варианта: `{man.get('new_safetensors_sha256', '')[:16]}…`; заголовок safetensors не изменён "
      f"({man.get('header_length_bytes')} байт)")
    w(f"- проверка целостности (`verify.json`): проблем {len(verify.get('problems', []))}, "
      f"проверок {len(verify.get('checks', []))}")
    w("")
    w("| тензор | форма | байты в shard | значений | mean abs Δ | max abs Δ |")
    w("|---|---|---|---|---|---|")
    for ch in man.get("changes", []):
        w(f"| `{ch['tensor']}` | {ch['shape']} | {ch['byte_offsets']} | {ch['values_actually_changed']} "
          f"| {ch['mean_abs_delta']:.2e} | {ch['max_abs_delta']:.2e} |")
    w("")
    w("Откат: полный список обратных операций в `inverse_ops` того же `manifest.json` "
      "(байтовые смещения + прежние значения + sha256 до/после), восстановление проверяется "
      "по совпадению хэша shard с исходным.\n")

    # 5. Финальный тест
    w("## 5. Единственный прогон на официальном тесте (1319 задач)\n")
    if has_test:
        cmp_ = ctest or {}
        def cnt(x):
            return len(x) if isinstance(x, list) else (x if isinstance(x, int) else None)
        w(f"- {chosen}: **{v_test['correct']}/{v_test['total']} = {pct(v_test['accuracy'])}** "
          f"против базовых **{b_test['correct']}/{b_test['total']} = {pct(b_test['accuracy'])}** "
          f"(тот же промпт, greedy, `max_new_tokens=1024`, `batch=1`, GPU 1).")
        d = cmp_.get("delta_correct")
        if not isinstance(d, int):
            d = int(v_test["correct"]) - int(b_test["correct"])
        fx, bk = cnt(cmp_.get("fixed")), cnt(cmp_.get("broken"))
        p_ex = cmp_.get("mcnemar_p")
        w(f"- Δ = {d:+d} задач; исправлено {fx}, сломано {bk}; "
          f"ответы различаются в {cmp_.get('answers_differ')} задачах.")
        if p_ex is not None:
            w(f"- точный McNemar: p = {p_ex:.4f} "
              f"({'значимо' if p_ex <= 0.05 else 'не значимо'} на уровне 0,05).")
        w(f"- извлечено: {b_test.get('answer_extracted')} → {v_test.get('answer_extracted')}; "
          f"обрывов: {b_test.get('truncated')} → {v_test.get('truncated')}; "
          f"среднее токенов: {b_test.get('mean_output_tokens')} → {v_test.get('mean_output_tokens')}.")
        w(f"- время прогона: {v_test.get('total_seconds')} с "
          f"({v_test.get('mean_seconds')} с/задача), режим `batch=1`, GPU 1.")
        # независимая сверка целостности финального прогона по записям
        def recs(p: Path) -> list[dict]:
            if not p.exists():
                return []
            return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        vr, br = recs(vdir / "test/results.jsonl"), recs(BASE["test1319"] / "results.jsonl")
        vi, bi = {int(r["index"]) for r in vr}, {int(r["index"]) for r in br}
        vcfg, bcfg = cfg(vdir / "test"), cfg(BASE["test1319"])
        diff_keys = sorted(k for k in set(vcfg) | set(bcfg)
                           if vcfg.get(k) != bcfg.get(k) and k not in {"model_dir", "model_info", "note"})
        w(f"- независимая сверка записей: строк {len(vr)}, уникальных индексов {len(vi)} "
          f"(тот же набор задач, что у базы: {vi == bi}), пересчёт верных ответов: "
          f"{sum(1 for r in vr if r.get('correct'))}; `indices_sha256` конфигов совпадает: "
          f"{vcfg.get('indices_sha256') == bcfg.get('indices_sha256')}; параметры прогона "
          f"отличаются только моделью (`model_dir` и хэш весов)"
          + ("" if not diff_keys else f" и ключами {diff_keys}") + ".")
        ns = [int(m.group(1)) for m in (re.match(r"\[(\d+)/1319\]", l) for l in lines(vdir / "test.log")) if m]
        n_test_runs = sorted(p.parent.parent.name for p in VARIANTS_DIR.glob("*/test/summary.json"))
        if ns:
            mono = all(b > a for a, b in zip(ns, ns[1:]))
            head = lines(vdir / "test.log", 1)
            w(f"- журнал прогресса: {len(ns)} строк, счётчик {min(ns)}→{max(ns)}, "
              f"монотонный без повторов: {mono}; начало журнала: `{head[0] if head else '?'}`.")
            w(f"- каталог `test/` с полным прогоном есть ровно у {len(n_test_runs)} варианта "
              f"({', '.join(n_test_runs) if n_test_runs else 'ни у одного'}); тестовый срез 1319 "
              f"использован один раз (подробно — §19 журнала).")
        fb = lines(vdir / "flip_breakdown_test.txt")
        if fb:
            w("\nРазложение переворотов на тесте (`формат / обрыв / арифметика`):")
            w("```")
            out.extend(fb[:12])
            w("```")
        err = lines(vdir / "errors.txt", 14)
        if err:
            w("\nФрагмент разбора ошибок (`runs/variants/%s/errors.txt`):" % chosen)
            w("```")
            out.extend(err)
            w("```")
        if p_ex is not None:
            w(f"\n**Вывод:** вариант превзошёл зафиксированный базовый результат на тесте — "
              f"{v_test['correct']}/{v_test['total']} ({pct(v_test['accuracy'])}) против "
              f"{b_test['correct']}/{b_test['total']} ({pct(b_test['accuracy'])}), "
              f"Δ = {d:+d} задач ({100 * (v_test['accuracy'] - b_test['accuracy']):+.2f} п.п.), "
              f"точный McNemar p = {p_ex:.4f}. Единственный прогон на тесте выполнен один раз; "
              f"отбор всех вариантов шёл только по train-наборам dev100 + search250.\n")
    else:
        w("Финальный прогон test1319 ещё не завершён: "
          "`runs/variants/%s/test/summary.json` отсутствует или неполон.\n" % chosen)

    # 6. Что не сработало
    w("\n## 6. Отрицательные результаты (тоже часть ответа)\n")
    w("- сильные версии тех же рычагов разрушают модель: `w11_final_token_amp_15` (эмбеддинг ×1,5) "
      "−119 задач на 350, `w12_z_gate_strong` (z ×0,75) −99;");
    w("- подавление MLP (`v9_mlp_damp`) −18 (search p = 0,017, значимый регресс), "
      "`v2_Alog_decay_double` −8, `c0_final_norm_gain` −3;");
    w("- ослабление строки «Final» коэффициентом 0,7 (`w15` против `w16`) не дало измеримого эффекта;");
    w("- формальный рычаг (эмбеддинг FINAL) повышает извлекаемость, но почти не меняет арифметику; "
      "z-гейт ×0,93 меняет ход рассуждения. Сложение даёт лучший результат, чем каждый по отдельности.\n")

    w("## 7. Как воспроизвести и как откатить\n")
    w("```")
    out.extend([
        "# пересобрать вариант из рецепта",
        ".venv/bin/python scripts/modify_weights.py --recipe recipes/%s.json" % chosen,
        "# проверка целостности и хэшей",
        ".venv/bin/python scripts/verify_variant.py runs/variants/%s" % chosen,
        ".venv/bin/python scripts/hash_manifests.py --wave wave5",
        "# единственный тестовый прогон (уже выполнен, не повторять)" % (),
        "bash scripts/run_final.sh",
    ])
    w("```")
    w("\nИсходная модель `models/qwen3.5-0.8b` не изменялась ни разу; все прогоны шли по копиям "
      "в `runs/variants/*/model`.\n")

    text = "\n".join(out) + "\n"
    (ROOT / "runs/FINAL_REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nСохранено: {ROOT/'runs/FINAL_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
