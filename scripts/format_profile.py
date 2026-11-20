#!/usr/bin/env python3
"""Профиль формата финальной строки по завершённым search250-прогонам.

Считает по тексту ответа ПОСЛЕ начала assistant-реплики:
  - сколько ответов начинающих финальную строку с 'FINAL:' / 'Final:' / 'final:',
  - сколько ответов с двоением маркера ('Final: FINAL: N'),
  - сколько '**Final',
  - сколько вообще без финальной строки (extraction_error == no_final_line),
  - официальное число извлечённых ответов.

Скрипт только описывает уже полученные прогоны; никаких новых прогонов и никакого
подбора по тестовому набору. Использует то же правило извлечения, что и eval_gsm8k.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
Assistant_RE = re.compile(r"assistant\n", re.IGNORECASE)
UP_RE = re.compile(r"^\s*FINAL:")  # не используется: ниже отдельные регистровые шаблоны
Upper_RE = re.compile(r"^\s*FINAL:")
Mixed_RE = re.compile(r"^\s*Final:")
Lower_RE = re.compile(r"^\s*final:")
Doubled_RE = re.compile(r"(?i)^\s*final\s*:\s*\n?\s*final\b")
Bold_RE = re.compile(r"(?i)\*\*\s*final")
NoLine_RE = re.compile(r"(?im)^\s*FINAL:\s*(.+?)\s*$")

RUNS = [("baseline", ROOT / "runs/baseline_search/results.jsonl")]
RUNS += [(p.parent.parent.name, p)
         for p in sorted(ROOT.glob("runs/variants/*/search/results.jsonl"))]


def after_assistant(text: str) -> str:
    parts = Assistant_RE.split(text, maxsplit=1)
    return parts[1] if len(parts) > 1 else text


def profile(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    st = {
        "n": len(rows),
        "correct": sum(bool(r["correct"]) for r in rows),
        "extracted": sum(bool(r["answer_extracted"]) for r in rows),
        "truncated": sum(bool(r["truncated"]) for r in rows),
        "upper": 0,
        "mixed": 0,
        "lower": 0,
        "doubled": 0,
        "bold_final": 0,
        "no_final_line": 0,
        "n_final_lines_total": 0,
    }
    for r in rows:
        body = after_assistant(r["model_answer"])
        first_line = None
        for line in body.splitlines():
            if Upper_RE.match(line) or Mixed_RE.match(line) or Lower_RE.match(line):
                first_line = line
                break
        if first_line is not None:
            if Upper_RE.match(first_line):
                st["upper"] += 1
            elif Mixed_RE.match(first_line):
                st["mixed"] += 1
            else:
                st["lower"] += 1
        if Doubled_RE.match(body.lstrip()):
            st["doubled"] += 1
        if Bold_RE.search(body):
            st["bold_final"] += 1
        if r.get("extraction_error") == "no_final_line":
            st["no_final_line"] += 1
        st["n_final_lines_total"] += int(r.get("n_final_lines") or 0)
    return st


def main() -> int:
    out = []
    out.append(f"{'вариант':<24} {'верно':>6} {'извл':>6} 'FINAL:'>7")
    header = (f"{'вариант':<24} {'n':>4} {'верно':>6} {'извл':>6} {'обрыв':>6} "
              f"{'FINAL:':>7} {'Final:':>7} {'final:':>7} {'двоение':>8} {'**Final':>8} {'без строки':>11} {'строк FINAL':>12}")
    out = [header, "-" * len(header)]
    for name, path in RUNS:
        if not path.exists():
            out.append(f"{name:<24} НЕТ ФАЙЛА {path}")
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        if len(rows) != 250:
            out.append(f"{name:<24} НЕЗАВЕРШЁННЫЙ ПРОГОН: {len(rows)} строк")
            continue
        p = profile(path)
        out.append(f"{name:<24} {p['n']:>4} {p['correct']:>6} {p['extracted']:>6} {p['truncated']:>6} "
                   f"{p['upper']:>7} {p['mixed']:>7} {p['lower']:>7} {p['doubled']:>8} {p['bold_final']:>8} "
                   f"{p['no_final_line']:>11} {p['n_final_lines_total']:>12}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
