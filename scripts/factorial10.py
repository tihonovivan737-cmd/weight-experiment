#!/usr/bin/env python
"""Анализ волны 10 — форматная ветка (раздел 33 журнала).

Три предрегистрированных контраста (§33.5):
  (w41 − w8)     накопленная доза строк FINAL 1,2002 -> 1,3201
  (w42 − w41)     1,3201 -> 1,4403
  (w43 − w36)     1,2002 -> 1,4403 поверх поздней дозы z (0,9300 / 0,8742)
Ось считается реальной только если ≥3 контраста одного знака и |средний эффект| ≥ 4 задачи
(порог шума на 350 задачах — §22, правило §31.6). Вторичные признаки (не решающие):
обрывы снижаются хотя бы в 2 из 3 контрастов; FINAL не падает ни в одном контрасте.
Отдельно проверяется w44_mlp_branch_late105 — прямое усиление MLP-ветви 7 поздних блоков
(H33c: проверка механизма «относительная доля MLP», а не дозы ворот).

Только чтение results.jsonl; генерации не запускаются. Выход: runs/analysis/factorial10.txt
"""
import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
sys.path.insert(0, str(ROOT / "scripts"))
import wave6_table as W  # noqa: E402

VARIANTS = {
    "orig": None,
    "w8": "w8_z_gate_mild_format",
    "w36": "w36_z_late_extra094",
    "w41": "w41_format_embed132",
    "w42": "w42_format_embed144",
    "w43": "w43_format_embed144_latedose",
    "w44": "w44_mlp_branch_late105",
}
CONTRASTS = [("w41", "w8", "1,2002 -> 1,3201 (w41 − w8)"),
             ("w42", "w41", "1,3201 -> 1,4403 (w42 − w41)"),
             ("w43", "w36", "1,2002 -> 1,4403 поверх поздней дозы (w43 − w36)")]
ADDITIONAL = [("w42", "w8", "1,2002 -> 1,4403 (w42 − w8)"),
              ("w43", "w8", "w43 против контрольной точки (w43 − w8)"),
              ("w44", "w8", "MLP-ветвь LATE ×1,05 (w44 − w8)"),
              ("w44", "w36", "w44 против лучшей клетки математики (w44 − w36)")]


def combined(key):
    name = VARIANTS[key]
    if name is None:
        rows = W.load(str(ROOT / "runs/baseline_search/results.jsonl"))
        rows.update(W.load(str(ROOT / "runs/baseline_dev/results.jsonl")))
        return rows if len(rows) == 350 else None
    try:
        rows = W.load(W.V / name / "search/results.jsonl")
        rows.update(W.load(W.V / name / "dev/results.jsonl"))
    except FileNotFoundError:
        return None
    return rows if len(rows) == 350 else None


def contrast(rows_a, rows_b, label):
    c = W.cmp(rows_a, rows_b, "b")
    sa, sb = W.stats(rows_a), W.stats(rows_b)
    return {"label": label, "delta": c["delta_correct"], "p": c["mcnemar_p"],
            "delta_math": c["delta_math"], "p_math": c["mcnemar_p_math"],
            "d_final": sa["extracted"] - sb["extracted"],
            "d_tr": sa["truncated"] - sb["truncated"],
            "d_len": round(sa["mean_out_tokens"] - sb["mean_out_tokens"], 1),
            "a": sa, "b": sb}


def emit(lines, res, pairs, data, title):
    lines.append(title)
    for ka, kb, label in pairs:
        if ka not in data or kb not in data:
            lines.append(f"  {label:<52} — прогон ещё не завершён")
            continue
        c = contrast(data[ka], data[kb], label)
        res.append(c)
        lines.append(f"  {label:<52} Δверных {c['delta']:+4d} (p={c['p']:.5f})  "
                     f"Δмат {c['delta_math']:+4d} (p={c['p_math']:.5f})  "
                     f"ΔFINAL {c['d_final']:+3d}  Δобрывы {c['d_tr']:+3d}  "
                     f"Δдлины {c['d_len']:+6.1f}")
    return res


def axis_verdict(lines, res, name):
    if not res:
        lines.append(f"  {name}: нет измеренных контрастов")
        return
    mean = sum(c["delta"] for c in res) / len(res)
    meanm = sum(c["delta_math"] for c in res) / len(res)
    pos = sum(1 for c in res if c["delta"] > 0)
    neg = sum(1 for c in res if c["delta"] < 0)
    lines.append(f"  --> {name}: средний эффект {mean:+.2f} задачи по {len(res)} контрастам "
                 f"(плюс {pos} / минус {neg}); средний Δмат {meanm:+.2f}")
    lines.append(f"  --> критерий §31.6/§33.5 (≥3 контраста одного знака и |средний| ≥ 4): "
                 f"{'ось реальна' if (max(pos, neg) >= 3 and abs(mean) >= 4) else 'ось = шум (критерий не выполнен)'}")
    dtr = [c["d_tr"] for c in res]
    dfi = [c["d_final"] for c in res]
    lines.append(f"  --> вторичные признаки: обрывы снизились в {sum(1 for d in dtr if d < 0)} "
                 f"из {len(dtr)} контрастов; FINAL не упал в {sum(1 for d in dfi if d >= 0)} "
                 f"из {len(dfi)}")


def pred(lines, stat, key, label, text):
    if key not in stat:
        lines.append(f"  {label}: [{text}] — прогон ещё не завершён")
        return
    s = stat[key]
    lines.append(f"  {label}: верных {s['correct']}, FINAL {s['extracted']}, обрывов {s['truncated']}, "
                 f"верных+FINAL {s['correct_with_final']}, средняя длина {s['mean_out_tokens']} токена")
    lines.append(f"      предсказание: {text}")


def main():
    data = {k: rows for k, rows in ((k, combined(k)) for k in VARIANTS) if rows is not None}
    stat = {k: W.stats(rows) for k, rows in data.items()}
    lines = ["Волна 10: форматная лестница (строки embed_tokens 95429 'FINAL' и 64697 ' FINAL') "
             "и проверка механизма 'относительная доля MLP'",
             "Все числа — 350 задач отбора (250 search + 100 dev). Официальный test не использовался.",
             "Справка по дозам: контрольная точка w8 = 1,2002; w41 = 1,3201; w42 = w43 = 1,4403; "
             "w44 = 1,2002 плюс post_attention_layernorm LATE ×1,0499",
             ""]
    lines.append("Варианты (верных / FINAL / обрывы / верных+FINAL / средняя длина):")
    for k in ("orig", "w8", "w36", "w41", "w42", "w43", "w44"):
        if k not in data:
            lines.append(f"  {k:<5} — прогон ещё не завершён")
            continue
        s = W.stats(data[k])
        lines.append(f"  {k:<5} {VARIANTS[k] or 'исходная модель':<30} {s['correct']:>4} / "
                     f"{s['extracted']:>4} / {s['truncated']:>3} / {s['correct_with_final']:>4} / "
                     f"{s['mean_out_tokens']:>6.1f}")
    lines.append("")

    res = []
    emit(lines, res, CONTRASTS, data, "Предрегистрированные контрасты форматной оси (§33.3/§33.5):")
    axis_verdict(lines, res, "форматная ось")
    lines.append("")

    res2 = []
    emit(lines, res2, ADDITIONAL, data, "Дополнительные контрасты (описательные, вне критерия оси):")
    lines.append("")

    lines.append("Проверка предсказаний §33.4:")
    pred(lines, stat, "w41", "H33a (w41, доза 1,3201)",
         "≥218 верных, обрывы ≤17, FINAL ≥332")
    pred(lines, stat, "w42", "H33a (w42, доза 1,4403)",
         "≥220 верных, обрывы ≤16, FINAL ≥334, Δмат против w8 ≥ 0")
    pred(lines, stat, "w43", "H33b (w43, доза 1,4403 поверх z 0,93/0,87)",
         "≥219 верных, обрывы ≤18, FINAL ≥330; иначе компромисс 'поздняя математика <-> обрывы' неустраним")
    pred(lines, stat, "w44", "H33c (w44, MLP-ветвь LATE ×1,05)",
         "219..222 верных при обрывах 19..23 и FINAL 328..334; ≤216 — механизм доли MLP фальсифицирован")
    lines.append("")

    lines.append("Признак преждевременного FINAL (риск зафиксирован в §33.4):")
    for c in res + res2:
        shorter = c["d_len"] < -25
        worse = c["delta"] < 0
        lines.append(f"  {c['label']:<52} ΔFINAL {c['d_final']:+3d}, Δдлины {c['d_len']:+6.1f}, "
                     f"Δверных {c['delta']:+4d} -> "
                     f"{'преждевременный FINAL (рост формата при более коротком ответе и меньше верных)' if (c['d_final'] > 0 and shorter and worse) else 'признака нет'}")
    txt = "\n".join(lines) + "\n"
    (ROOT / "runs/analysis/factorial10.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
