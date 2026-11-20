#!/usr/bin/env python3
"""Потолок «только формата»: сколько задач стало бы верными, если бы модель всегда
выбирала правильное число из уже написанного хвоста.

Для каждой модели по dev (100) и search (250):
  верных сейчас;
  среди неверных с_FINAL — нужное число есть в последних 400 симвомах до строки FINAL
    (значит результат рассуждения верен, ошибён только выбор строки ответа);
  среди неверных без FINAL — нужное число есть в последних 400 символах текста
    (значит ответ получен, но строка FINAL не появилась);
  потолок = верных + обе группы; прирост = потолок − верных.

Только чтение существующих результатов.
Вывод: runs/analysis/extraction_ceiling.txt
"""
import json
from pathlib import Path

TAIL = 400
MODELS = {
    "исходная модель": "runs/baseline_{sp}/results.jsonl",
    "w8_z_gate_mild_format": "runs/variants/w8_z_gate_mild_format/{sp}/results.jsonl",
    "w50_finalrow_rotation15_smallmargin": "runs/variants/w50_finalrow_rotation15_smallmargin/{sp}/results.jsonl",
    "w79_finalrow_ten_rows_double_step_on_w8": "runs/variants/w79_finalrow_ten_rows_double_step_on_w8/{sp}/results.jsonl",
    "w81_finalrow_four_small_rows_step15_on_w8": "runs/variants/w81_finalrow_four_small_rows_step15_on_w8/{sp}/results.jsonl",
    "w82_wave11_head_lever_on_w8": "runs/variants/w82_wave11_head_lever_on_w8/{sp}/results.jsonl",
    "w83_mlp_layer_lever_on_w8": "runs/variants/w83_mlp_layer_lever_on_w8/{sp}/results.jsonl",
    "w84_mlp_layer_lever5_on_w8": "runs/variants/w84_mlp_layer_lever5_on_w8/{sp}/results.jsonl",
    "w85_mlp_layer_leversign_on_w8": "runs/variants/w85_mlp_layer_leversign_on_w8/{sp}/results.jsonl",
    "w86_attnshare_lever_on_w8": "runs/variants/w86_attnshare_lever_on_w8/{sp}/results.jsonl",
    "w87_stack_finalrow_on_w83": "runs/variants/w87_stack_finalrow_on_w83/{sp}/results.jsonl",
}
SPLITS = ("dev", "search")


def forms_of(expected):
    try:
        v = float(str(expected).replace(",", ""))
    except ValueError:
        return []
    if abs(v - round(v)) < 1e-9:
        i = int(round(v))
        return [str(i), f"{i:,}"]
    return [str(v)]


def appears(text, forms):
    import re
    for f in forms:
        for m in re.finditer(re.escape(f), text):
            a, b = m.start(), m.end()
            left = text[a - 1] if a else ""
            right = text[b] if b < len(text) else ""
            if left.isdigit() or left == ".":
                continue
            if right.isdigit():
                continue
            return True
    return False


def main():
    lines = [f"Хвост = последние {TAIL} символов. Склады: dev 100 + search 250 = 350 задач."]
    lines.append("")
    lines.append(f"{'модель':44s} {'верно':>6} {'с FINAL, нужное в хвосте':>27} "
                 f"{'без FINAL, нужное в хвосте':>30} {'потолок':>9} {'прирост':>9}")
    for label, tpl in MODELS.items():
        correct = with_final = no_final = n = 0
        missing = []
        for sp in SPLITS:
            p = Path(tpl.format(sp=sp))
            if not p.exists():
                missing.append(sp)
                continue
            for ln in p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                n += 1
                if r.get("correct"):
                    correct += 1
                    continue
                text = r.get("model_answer") or ""
                forms = forms_of(r.get("expected"))
                if not forms:
                    continue
                finals = [x for x in text.splitlines() if x.strip().upper().startswith("FINAL")]
                if finals:
                    body = text[:text.rfind(finals[-1])]
                    if appears(body[-TAIL:], forms):
                        with_final += 1
                elif appears(text[-TAIL:], forms):
                    no_final += 1
        head = f"{label:44s} {correct:>6} {with_final:>27} {no_final:>30} " \
               f"{correct + with_final + no_final:>9} {with_final + no_final:>9}"
        if missing:
            head += f"   (нет {'+'.join(missing)})"
        elif n:
            head += f"   из {n}"
        lines.append(head)
    Path("runs/analysis/extraction_ceiling.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
