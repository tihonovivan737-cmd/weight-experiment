#!/usr/bin/env python3
"""Упаковка материалов эксперимента для независимого аудита.

Ничего не удаляет и не изменяет в рабочем дереве: всё копируется в отдельный каталог
подготовки (по умолчанию /tmp/wex_audit_stage), оттуда собирается tar.gz.

Исключено по требованию задания: файлы весов (*.safetensors), токенизаторы и прочие
файлы моделей (копируются в каждый runs/variants/<имя>/model), датасет целиком
(parquet-файлы), виртуальное окружение (.venv). Вместо них в архив кладутся хеши,
размеры и манифесты, чтобы выводы проверялись без весов.

Манифесты крупнее 4 МиБ копируются облегчёнными: из поля changes вырезаются массивы
values_before/values_after (это пофайловые списки значений, до 1,2 ГиБ на вариант);
остальные поля, смещения, хеши диапазонов до и после сохраняются. В файле появляется
блок _upakovka с размером и хешем исходного файла, чтобы подмену нельзя было скрыть.

Перед упаковкой каталог проходит проверку на секреты; найденное замазывается только
в копии архива, исходники не трогаются, факты пишутся в SECURITY_SCAN.txt.

Запуск:  .venv/bin/python scripts/make_audit_archive.py [--dry]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
STAGE_ROOT = Path(os.environ.get("AUDIT_STAGE", "/tmp/wex_audit_stage"))
TOP = "weight-experiment-audit-2026-11-19"
STAGE = STAGE_ROOT / TOP
MANIFEST_LIMIT = 4 * 1024 * 1024

SKIP_NAMES = {".venv", "__pycache__", ".cache"}
SKIP_FILENAMES = {"tokenizer.json", "vocab.json", "merges.txt", "tokenizer.json.metadata"}
SKIP_EXT = {".safetensors", ".pt", ".bin", ".gguf", ".lock", ".pyc"}
SECRET_PATTERNS = [
    ("api-ключ sk-", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("ключ huggingface", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("ключ github", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("ключ aws", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("ключ slack", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("присвоенный ключ в тексте", re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[\"']?\s*[:=]\s*[\"'][^\"']{12,}[\"']")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}")),
    ("закрытый ключ", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def sha256_file(path: Path, chunk=1 << 22):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def want_copy(path: Path):
    """Правило: что вообще может попасть в архив."""
    parts = set(path.parts)
    if parts & SKIP_NAMES:
        return False
    if path.suffix in SKIP_EXT:
        return False
    if path.name in SKIP_FILENAMES:
        return False
    if path.name in {"nohup.out"}:
        return False
    return True


def slim_manifest(src: Path, dst: Path):
    """Копирует манифест, при необходимости вырезая пофайловые списки значений."""
    size = src.stat().st_size
    if size <= MANIFEST_LIMIT:
        shutil.copy2(src, dst)
        return dict(mode="полный копия", size=size, sha256=sha256_file(src))
    data = json.loads(src.read_text(encoding="utf-8"))
    removed = 0
    ch = data.get("changes")
    if isinstance(ch, list):
        for entry in ch:
            if isinstance(entry, dict):
                for key in ("values_before", "values_after"):
                    if key in entry:
                        entry.pop(key)
                        removed += 1
    data["_upakovka"] = {
        "файл_оригинала": str(src.relative_to(ROOT)),
        "размер_оригинала_байт": size,
        "sha256_оригинала": sha256_file(src),
        "что_вырезано": f"массивы values_before/values_after, полей удалено: {removed}",
        "почему": "объём одного такого манифеста достигал 1,2 ГиБ; для аудита сохранены "
                  "смещения, форма, dtype, параметры операций и хеши диапазонов до и после",
        "срезано_скриптом": "scripts/make_audit_archive.py",
    }
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return dict(mode="облегчённая копия", size=size, sha256=data["_upakovka"]["sha256_оригинала"],
                удалённых_массивов=removed)


def copy_tree(rel: str):
    src = ROOT / rel
    if not src.exists():
        return []
    if src.is_file() and (src.suffix in SKIP_EXT or src.name in SKIP_FILENAMES):
        return []
    done = []
    if src.is_file():
        dst = STAGE / TOP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.name == "manifest.json":
            slim_manifest(src, dst)
        else:
            shutil.copy2(src, dst)
        return [(rel, dst)]
    for path in sorted(src.rglob("*")):
        if path.is_dir() or not want_copy(path):
            continue
        rp = str(path.relative_to(ROOT))
        dst = STAGE / TOP / rp
        if path.name == "manifest.json" and path.parent.name != "discarded":
            dst.parent.mkdir(parents=True, exist_ok=True)
            slim_manifest(path, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
        done.append((rp, dst))
    return done


def collect_includes():
    """Явный список того, что берём (кроме явно исключённого весами и окружением)."""
    items = [
        "runs/JOURNAL.md",
        "runs/FINAL_REPORT.md",
        "runs/FINAL_REPORT_v2.md",
        "runs/INTERIM_REPORT.md",
        "runs/CLEANUP_REPORT.md",
        "runs/DISCARDED_REVIEW.md",
        "runs/GATE_FINAL",
        "runs/GATE_FINAL2",
        "runs/GATE_DEV3",
        "runs/GATE_WAVE2",
        "runs/GATE_WAVE3",
        "runs/splits",
        "runs/analysis",
        "runs/logs",
        "runs/baseline_dev",
        "runs/baseline_search",
        "runs/baseline_test",
        "runs/holdout6",
        "runs/discarded",
        "recipes",
        "scripts",
        "datasets/gsm8k/main/.gitignore",
    ]
    blocks = sorted(ROOT.glob("runs/journal_block_*.md"))
    backups = sorted(ROOT.glob("runs/JOURNAL_backup_before_*.md"))
    items += [str(b.relative_to(ROOT)) for b in blocks + backups]
    # варианты: манифесты, сверки, сводки, результаты — но НЕ каталог model
    for vdir in sorted((ROOT / "runs/variants").iterdir()):
        if not vdir.is_dir():
            continue
        for path in sorted(vdir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if len(rel.parts) > 3 and rel.parts[3] == "model":
                continue
            if path.suffix in SKIP_EXT or ".cache" in path.parts:
                continue
            items.append(str(rel))
    return items


def write_env_files():
    env_dir = STAGE / TOP / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    summ = json.loads((ROOT / "runs/holdout6/w8/summary.json").read_text(encoding="utf-8"))
    (env_dir / "versions.json").write_text(json.dumps({
        "окружение_из_сводки_прогона": summ["config"]["env"],
        "настройки_генерации": {k: summ["config"][k] for k in
                                ("device", "dtype", "batch_size", "max_new_tokens", "do_sample",
                                 "num_threads", "prompt_suffix", "dataset_dir", "split",
                                 "n_indices", "indices_sha256")},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    try:
        pip = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=180).stdout
        (env_dir / "pip_freeze.txt").write_text(pip, encoding="utf-8")
    except Exception as exc:
        (env_dir / "pip_freeze.txt").write_text(f"не удалось получить список пакетов: {exc}\n",
                                                encoding="utf-8")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    (env_dir / "gpu.txt").write_text(gpu, encoding="utf-8")
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    (env_dir / "cpu.txt").write_text(cpu, encoding="utf-8")
    (env_dir / "как_воспроизвести.txt").write_text(
        "Порядок воспроизведения (без файлов весов их нужно взять из исходного репозитория;\n"
        "хеш файла весов исходной модели: "
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696,\n"
        "размер 1746942600 байт):\n\n"
        "1) самопроверка рецепта:   .venv/bin/python scripts/selftest_w71.py recipes/<имя>.json\n"
        "2) сборка варианта:        .venv/bin/python scripts/modify_weights.py --recipe recipes/<имя>.json \\\n"
        "                              --base runs/variants/w8_z_gate_mild_format/model --out runs/variants/<имя>\n"
        "3) сверка байтов:          .venv/bin/python scripts/verify_variant.py <имя>\n"
        "4) прогоны dev+search:     bash scripts/run_w87_disk.sh   (пример последнего прогонщика; "
        "там же контроль хешей опор и порогов)\n"
        "5) сводка варианта:        .venv/bin/python scripts/combined_score.py --variant <имя> "
        "--report runs/variants/<имя>/combined.json\n"
        "6) разборы против баз:     .venv/bin/python scripts/compare_runs.py --base <файл результатов базы> "
        "--variant <файл варианта> --report <файл отчёта>\n"
        "7) финальные таблицы:      .venv/bin/python scripts/credit_table.py; scripts/paired_survey.py; "
        "scripts/family_census.py; scripts/recipe_index.py\n"
        "8) проверка на удержанном наборе (УЖЕ ИЗРАСХОДОВАНА, повтор заблокирован скриптом): "
        "bash scripts/run_holdout6.sh winner|orig|w8|compare\n", encoding="utf-8")


def write_hashes_files(copied):
    hashes_dir = STAGE / TOP / "hashes"
    hashes_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rp, dst in copied:
        if not dst.exists():
            continue
        rows.append(f"{sha256_file(dst)}  {rp}")
    (STAGE / TOP / "FILES.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sizes = [f"{os.path.getsize(dst)}  {rp}" for rp, dst in copied if dst.exists()]
    (STAGE / TOP / "FILES.sizes.txt").write_text("\n".join(sizes) + "\n", encoding="utf-8")

    # хеши весов: что на диске — пересчёт, чего нет — запись из манифеста
    out = ["Список файлов весов и их хеши. Файлы весов в архив НЕ включены.\n"]
    orig = next((ROOT / "models/qwen3.5-0.8b").glob("*.safetensors"))
    out.append(f"исходная модель: {orig.name} | {orig.stat().st_size} байт | sha256 {sha256_file(orig)}")
    for mdir in sorted((ROOT / "runs/variants").iterdir()):
        if not mdir.is_dir():
            continue
        mf = mdir / "manifest.json"
        rec = None
        if mf.exists():
            try:
                rec = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                rec = None
        files = sorted(p for p in mdir.rglob("*.safetensors"))
        on_disk = ""
        if files:
            f = files[0]
            h = sha256_file(f)
            ok = "совпадает с манифестом" if rec and h == rec.get("new_safetensors_sha256") else "РАСХОЖДЕНИЕ"
            on_disk = f"есть на диске ({f.stat().st_size} байт) sha256 {h} — {ok}"
        else:
            on_disk = "нет на диске"
        man = (rec or {}).get("new_safetensors_sha256", "нет записи")
        base = (rec or {}).get("base_safetensors_sha256", "нет записи")
        out.append(f"{mdir.name:44s} | записанный хеш: {man} | база: {base[:16]}… | {on_disk}")
    for ddir in sorted((ROOT / "runs/discarded").iterdir()):
        if not ddir.is_dir():
            continue
        files = sorted(p for p in ddir.rglob("*.safetensors"))
        if files:
            f = files[0]
            out.append(f"discarded/{ddir.name:34s} | {f.stat().st_size} байт | sha256 {sha256_file(f)} | в архив не включён")
    (hashes_dir / "weights.sha256.txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    ds = ["Файлы датасета GSM8K (main) в архив НЕ включены; вот их контрольные суммы и размер:"]
    for p in sorted((ROOT / "datasets/gsm8k/main").glob("*.parquet")):
        ds.append(f"{p.name}: {p.stat().st_size} байт | sha256 {sha256_file(p)}")
    sp = json.loads((ROOT / "runs/splits/splits.json").read_text(encoding="utf-8"))
    ds.append("")
    ds.append("Разбиения (runs/splits/splits.json): ключы и числа задач:")
    for k in sorted(sp):
        v = sp[k]
        ds.append(f"  {k}: {len(v) if isinstance(v, list) else v}")
    (hashes_dir / "dataset_and_splits.sha256.txt").write_text("\n".join(ds) + "\n", encoding="utf-8")


def scan_secrets(copied):
    report = []
    changed = 0
    for rp, dst in copied:
        if not dst.exists() or dst.stat().st_size > 60 * 1024 * 1024:
            continue
        try:
            text = dst.read_text(encoding="utf-8")
        except Exception:
            continue
        if "\0" in text[:4096]:
            continue
        hits = []
        for label, pat in SECRET_PATTERNS:
            for m in pat.finditer(text):
                hits.append((label, m.start(), m.group(0)))
        if not hits:
            continue
        for label, pos, value in hits:
            text = text.replace(value, value[:6] + "…[ЗАМАЗАНО]" if len(value) > 10 else "[ЗАМАЗАНО]")
        dst.write_text(text, encoding="utf-8")
        changed += 1
        report.append(f"{rp}: найдено {len(hits)} совпадений ({', '.join(sorted({h[0] for h in hits}))}) — "
                      f"в копии архива значения замазаны, исходный файл не изменён")
    head = ["Проверка включённых файлов на секреты (ключи, токены, пароли, закрытые ключи).",
            f"Файлов проверено: {len(copied)}; изменено копий: {changed}.", ""]
    (STAGE / TOP / "SECURITY_SCAN.txt").write_text("\n".join(head + report + ["Итог: " +
                                  ("секретов не найдено" if not report else
                                   "найдены замаскированные значения; см. список выше")]) + "\n",
                                  encoding="utf-8")
    return changed, report


def write_readme(copied, secret_changed, secret_report):
    counts = collections_summary(copied)
    n_manifests = sum(1 for rp, _ in copied if rp.endswith("manifest.json"))
    n_results = sum(1 for rp, _ in copied if rp.endswith("results.jsonl"))
    lines = []
    a = lines.append
    a("# Архив для независимого аудита: программная правка весов Qwen3.5-0.8B под GSM8K\n")
    a(f"Собрано: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} (Asia/Krasnoyarsk). ")
    a("Собирающий скрипт: `scripts/make_audit_archive.py` (в архиве — `scripts/`). ")
    a("Исходное рабочее дерево: `/home/mpiscil/weight-experiment`. При упаковке ничего не "
      "удалялось и не изменялось: все файлы только копировались; облегчённые манифесты — это "
      "копии, в них записаны размер и sha256 оригинала.\n")

    a("## 1. Хронология эксперимента\n")
    a("| этап | что происходило | где смотреть |")
    a("|---|---|---|")
    a("| 2026-11-10 | исходная модель загружена (файл весов 1 746 942 600 байт, "
      "sha256 `04b1c301…`), базовые прогоны `test` (680/1319), `dev` (59/100), `search` (143/250); "
      "первые 22 варианта собраны в тот же день | `hashes/`, `runs/baseline_*`, `runs/JOURNAL.md` §1… |")
    a("| 2026-11-10 22:40 | контрольная точка `w8_z_gate_mild_format` (гейт линейного внимания + "
      "усиление строк `FINAL`), позже 724/1319 на официальном наборе | `runs/variants/w8_z_gate_mild_format/` |")
    a("| 2026-11-11…11-13 | волны 1–12: экран `dev100`, затем пары разрезов; 15+10+4 манифеста; "
      "промежуточный отчёт | `runs/INTERIM_REPORT.md`, `runs/FINAL_REPORT.md` (11.11), журнал §1…§40 |")
    a("| 2026-11-15 | волна 13: поворот строк-конкурентов, `w50` (217/340/10) и обращённый `w51`; "
      "зафиксированы коридоры и правило отбора | журнал §43…§44, `runs/GATE_FINAL` (тогдашнее решение: победа за `w8`) |")
    a("| 2026-11-16…11-17 | волны 14–30: гейты, головы, строки вывода, свёртки, нормализации; "
      "официальный тест `w50` (709/1319) | журнал §45…§80, `runs/analysis/w50_test_report.txt` |")
    a("| 2026-11-18 | разрешённая очистка диска (58 файлов весов удалены по решению пользователя, "
      "107 ГиБ → 12 ГиБ), разбор `runs/discarded`, остановка из-за порога памяти | "
      "`runs/CLEANUP_REPORT.md`, `runs/DISCARDED_REVIEW.md`, журнал §84…§89 |")
    a("| 2026-11-19 | неудачная попытка запуска через «чёрное окно», сбой раскладки сборки, "
      "измерение `w82`…`w87`, ось «доля ветви по слоям» закрыта, публикация победителя, "
      "одиночная проверка на удержанном наборе 400 задач, итоговый отчёт | журнал §90…§104, "
      "`runs/GATE_FINAL2`, `runs/holdout6/`, `runs/FINAL_REPORT_v2.md` |")
    a("")
    a("Число манифестов по датам создания: " + counts["по датам"] + "\n")

    a("## 2. Состав архива\n")
    a("| раздел | содержимое | объём |")
    a("|---|---|---|")
    a("| `journal/JOURNAL.md` | полный журнал эксперимента (6 995 строк, 104 блока, единственный "
      "финальный маркер) | 0,8 МиБ |")
    a("| `runs/journal_block_*.md`, `runs/JOURNAL_backup_before_*.md` | блоки до вставки и "
      "резервные копии журнала перед вставкой (28 файлов) — доказательство, что журнал "
      "дополнялся, а не переписывался | "
      f"{counts['журнал копии']:.1f} МиБ |")
    a("| `reports/` | `FINAL_REPORT.md` (этап выбора `w8`), `FINAL_REPORT_v2.md` (итоговый), "
      "`INTERIM_REPORT.md`, `CLEANUP_REPORT.md`, `DISCARDED_REVIEW.md` | 0,1 МиБ |")
    a(f"| `runs/variants/…` | манифесты ({n_manifests}), сводки `combined.json`, файлы сверок "
      f"`verify.json`, разборы сравнений, `results.jsonl` по всем сохранившимся прогонам ({n_results} файлов) | "
      f"{counts['варианты']:.1f} МиБ |")
    a("| `runs/baseline_*`, `runs/holdout6/` | базовые прогоны исходной модели (`dev`, `search`, "
      "`test`) и три прогона на удержанном наборе, включая огрызки неудачного запуска этапа `w8` | "
      f"{counts['базы и удержанный']:.1f} МиБ |")
    a("| `runs/analysis/` | сравнения пар, таблицы волн, зонды, сводные таблицы: `credit_table.txt`, "
      "`family_census.txt`, `recipe_index.txt`, `paired_survey.txt`, `test_format_split.txt`, "
      "`extraction_ceiling.txt`, `gold_in_tail.txt`, `holdout6_summary.txt`, `holdout_ceiling.txt`. "
      "Там же лежит код правила в отчёте §100.2 (извлечение описано в `scripts/eval_gsm8k.py`) | "
      f"{counts['анализ']:.1f} МиБ |")
    a("| `runs/logs/` | логи сборок и прогонов, журнал «чёрного окна» и запись восстановления, "
      "логи ошибок | 0,9 МиБ |")
    a("| `runs/splits/` | `splits.json` (индексы `dev`/`search`/`smoke`) и `holdout6.json` "
      "(400 индексов удержанного набора, хеш индексов, хеши вопросов) | 0,5 МиБ |")
    a("| `recipes/` | 90 рецептов правок (операции, слои, строки, множители, гипотеза, полоса) | "
      f"{counts['рецепты']:.1f} МиБ |")
    a("| `scripts/` | код: сборка весов, оценка, сравнения, прогонщики, сводные таблицы, "
      "самопроверки рецептов, упаковщик архива | 1,4 МиБ |")
    a("| `hashes/` | `weights.sha256.txt` (хеши весов: пересчёт для тех, что ещё на диске, и "
      "записанные из манифестов для остальных), `dataset_and_splits.sha256.txt` | малый |")
    a("| `environment/` | `versions.json` (python, torch, transformers, numpy, safetensors, "
      "настройки генерации), `pip_freeze.txt`, `gpu.txt`, `cpu.txt`, `как_воспроизвести.txt` | малый |")
    a("| `FILES.sha256`, `FILES.sizes.txt` | sha256 каждого файла архива в формате `sha256sum -c` "
      "и отдельно размеры (" + str(len(copied)) + " записей в каждом) | малый |")
    a("| `SECURITY_SCAN.txt` | результат проверки на секреты | малый |")
    a("")

    a("## 3. Что намеренно НЕ включено\n")
    a("* файлы весов всех моделей (`*.safetensors`) — по условию задания; их имена, размеры и "
      "sha256 перечислены в `hashes/weights.sha256.txt`;")
    a("* токенизаторы и прочие файлы моделей (копии в `runs/variants/*/model/`, по ~22 МиБ на "
      "вариант) — вне задания;")
    a("* датасет GSM8K целиком (parquet) — вместо него sha256 и размеры в "
      "`hashes/dataset_and_splits.sha256.txt`;")
    a("* виртуальное окружение `.venv` — вместо него `environment/pip_freeze.txt`.\n")

    a("## 4. Метрики и правила (расшифровка)\n")
    a("* **верно** — извлечённое из последней строки число совпало с эталоном; извлечение — "
      "строка `FINAL: <число>` (значение поля `n_final_lines` ≥ 1); **обрыв** — генерация "
      "дошла до лимита 1024 токенов без строны ответа;")
    a("* наборы: `dev` 100 и `search` 250 задач (вместе 350) — только отбор; официальный `test` "
      "1319 задач — только оценка; удержанный `holdout6` 400 задач `train` "
      "(хеш индексов `261affefc6c27b51…`) — одна финальная проверка;")
    a("* **кредит-гейт серии**: извлечение ≥ 336, математика ≥ 216, обрывы ≤ 14, математика "
      "`dev` ≥ 66 (на 350 задачах); ни один вариант не выполнил все четыре условия;")
    a("* **правило выбора победителя** (журнал §97.8, записано до подсчёта итоговой таблицы): "
      "сначала больше выполненных условий кредит-гейта, затем больше математика, затем меньше "
      "обрывов; публиковать в `runs/GATE_FINAL2` до прогона на удержанном наборе;")
    a("* точный двусторонний критерий `McNemar`: `p` = min(1, 2·Σ C(n,k) по k ≤ min(b,c)), где "
      "b, c — парные переходы; код: `scripts/wave6_table.py`, `scripts/paired_survey.py`;")
    a("* «качество на пересечении» — только задачи, где `FINAL` есть у обеих моделей "
      "(`scripts/quality_only.py`); «потолок только формата» — сколько задач стало бы верным "
      "при идеальном выборе уже написанного числа (`scripts/extraction_ceiling.py`, "
      "`scripts/holdout_ceiling.py`);")
    a("* условия генерации едины: `bfloat16`, `batch_size` 1, `max_new_tokens` 1024, жадная "
      "выборка, физическая видеокарта 1 (`RTX 3090`), 24 потока CPU; см. `environment/versions.json`.\n")

    a("## 5. Главные файлы\n")
    a("| вопрос аудита | файл |")
    a("|---|---|")
    a("| итоговые выводы | `runs/FINAL_REPORT_v2.md` |")
    a("| хронология и решения по ходу | `runs/JOURNAL.md` (блоки §90…§104 — финал серии) |")
    a("| публикация победителя | `runs/GATE_FINAL2` |")
    a("| удержанная проверка | `runs/holdout6/{orig,w8,w50_finalrow_rotation15_smallmargin}/results.jsonl`, "
      "`runs/analysis/holdout6_summary.txt` |")
    a("| попарный обзор всех 67 вариантов | `runs/analysis/paired_survey.txt` |")
    a("| разбор «формат против качества» | `runs/analysis/test_format_split.txt`, "
      "`runs/analysis/quality_only_*.txt` |")
    a("| перепись семейств и рецептов | `runs/analysis/family_census.txt`, `runs/analysis/recipe_index.txt` |")
    a("| счёт и кредит-гейт | `runs/analysis/credit_table.txt`, журнал §102.1 |")
    a("| сбои и исключённые результаты | `runs/logs/w82_blackout.log`, "
      "`runs/variants/w82_wave11_head_lever_on_w8/dev_failed_attempt/`, "
      "`runs/holdout6/w8_failed_attempt_zero_tasks/`, `runs/CLEANUP_REPORT.md`, журнал §88, §91…§93 |")
    a("")

    a("## 6. Противоречия и пробелы (не исправляются задним числом, описаны как есть)\n")
    a("1. **Два правила выбора дали разных победителей.** По прежнему правилу `pick_final2` "
      "(журнал §43.4: нужен +2 и против исходной модели, и против `w8`) победителем оставался "
      "`w8`; по действующему правилу §97.8 опубликован `w50`. Разница между ними на отборе — "
      "одна задача (217 против 216), а на удержанном наборе впереди `w8` (218 против 211, "
      "`p` = 0,3489). Оба факта приведены в `runs/GATE_FINAL2` и журнале §103.3.")
    a("2. **Дубли наборов веса.** `w75_z_flatten_l202122_on_w8` и `w82_wave11_head_lever_on_w8` "
      "имеют один и тот же sha256 `f1492c14…` и совпали ответами на всех 350 задачах; "
      "`w53_frozen10_dose003_mirrorpos` и `w79_finalrow_ten_rows_double_step_on_w8` — один хеш "
      "`cdca2052…`. Recipes разные, байты одинаковые. Из-за этого счёт «уникальных наборов» "
      "(81) меньше числа каталогов (83); вопрос учёта `w53`/`w79` остался спорным (журнал §84.5, §93.2).")
    a("3. **Счёт экспериментов зависит от методики.** Каталог с манифестом 83; с обоими "
      "отборочными разрезами 67; с одним лишь `dev` 15; только пробный 1; рецептов 90. "
      "В отчёте названы все четыре числа (журнал §102.1).")
    a("4. **Удалённые веса.** При разрешённой очистке 2026-11-18 удалено 58 файлов, в том числе "
      "наборы `w30_qknorm_sharp108`, `w36_z_late_extra094`, `w53_…`, `w77_…` (по решению "
      "пользователя не пересобраны). Их манифесты и результаты сохранены, но сверить байты "
      "с манифестом уже нельзя — в `hashes/weights.sha256.txt` для них стоит «нет на диске».")
    a("5. **Облегчённые манифесты.** Восемь крупных манифестов (до 1,2 ГиБ) скопированы без "
      "массивов `values_before/values_after`; в каждом файле есть блок `_upakovka` с размером и "
      "sha256 оригинала, поэтому копию можно сверить с оригиналом, если он понадобится.")
    a("6. **Ошибки процесса, зафиксированные в журнале:** нераспознанный ответ сервера в обёртке "
      "«чёрного окна» (§91), вложенная раскладка сборки, из-за которой `dev` не выполнил ни одной "
      "задачи (§92), путь к модели без `/model` на этапе `w8` удержанной проверки (§103.1, 0 задач), "
      "ложный вывод «отличий нет» из-за целочисленного усечения при проверке приращений (§98.1, "
      "проверка переделана побитово), а также проверка приращений в `w87` до пересчёта.")
    a("7. **Что нельзя проверить без весов:** побайтовое соответствие собранных файлов манифестам "
      "для удалённых наборов; детерминизм сборки (подтверждён косвенно: совпадение ответов 350/350 "
      "у `w75` и `w82`); точное совпадение токенизатора (файлы не включены); пересчёт метрик "
      "собственными силами — возможен по `results.jsonl`, потому что там лежат `question`, "
      "`expected`, `predicted`, `n_final_lines`, `truncated`, `output_tokens` по каждой задаче.")
    a("")
    a("## 7. Как проверить целостность архива\n")
    a("```bash")
    a("sha256sum <имя_архива>.tar.gz            # сверить с тем, что прислал собирающий")
    a("tar -tzf <имя_архива>.tar.gz | head      # состав")
    a("tar -xzf <имя_архива>.tar.gz -C /tmp/аудит")
    a("cd /tmp/аудит/weight-experiment-audit-2026-11-19 && sha256sum -c FILES.sha256")
    a("```")
    a("Метки `SECURITY_SCAN.txt` — результат поиска ключей и токенов по включённым файлам"
      + (f": замаскировано файлов: {secret_changed}." if secret_changed else ": совпадений нет."))
    if secret_report:
        a("")
        a("Найденные и замаскированные места:")
        for r in secret_report[:12]:
            a("  * " + r)
    text = "\n".join(lines) + "\n"
    (STAGE / TOP / "AUDIT_README.md").write_text(text, encoding="utf-8")


def collections_summary(copied):
    import collections
    tot = collections.Counter()
    bydate = collections.Counter()
    for rp, dst in copied:
        if not dst.exists():
            continue
        sz = os.path.getsize(dst)
        if rp.startswith("runs/journal_block_") or rp.startswith("runs/JOURNAL_backup_"):
            tot["журнал копии"] += sz
        elif rp.startswith("runs/variants/"):
            tot["варианты"] += sz
        elif rp.startswith("runs/baseline") or rp.startswith("runs/holdout6"):
            tot["базы и удержанный"] += sz
        elif rp.startswith("runs/analysis"):
            tot["анализ"] += sz
        elif rp.startswith("recipes"):
            tot["рецепты"] += sz
        if rp.endswith("manifest.json") or rp.endswith("manifest.slim.json"):
            bydate[datetime.fromtimestamp(dst.stat().st_mtime).strftime("%Y-%m-%d")] += 1
    tot = {k: v / 1048576 for k, v in tot.items()}
    tot["по датам"] = ", ".join(f"{d}: {n}" for d, n in sorted(bydate.items()))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="только список файлов, без упаковки")
    args = ap.parse_args()

    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    STAGE.mkdir(parents=True)
    copied = []
    for rel in collect_includes():
        src = ROOT / rel
        if not src.exists():
            print(f"пропущено (нет файла): {rel}")
            continue
        copied += copy_tree(rel)
    print(f"подготовлено файлов: {len(copied)}")
    if args.dry:
        for rp, _ in copied[:20]:
            print(" ", rp)
        return 0

    staged = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"объём подготовленного каталога: {staged/1048576:.1f} MiB")
    if staged > 800 * 1048576:
        print("ПРЕДУПРЕЖДЕНИЕ: стадия больше 800 МиБ — архив не будет компактным")

    write_env_files()
    write_hashes_files(copied)
    secret_changed, secret_report = scan_secrets(copied)
    write_readme(copied, secret_changed, secret_report)

    out_dir = ROOT / "runs/audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    archive = out_dir / f"wex_audit_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(STAGE / TOP, arcname=TOP)
    print("архив:", archive)
    print("размер байт:", archive.stat().st_size)
    print("sha256:", sha256_file(archive))
    return 0


if __name__ == "__main__":
    sys.exit(main())
