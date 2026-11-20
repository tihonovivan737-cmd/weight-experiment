#!/usr/bin/env python3
# Проверка журнала: ровно один маркер в конце файла, 0 CJK, латинские слова только из списка.
# Латинские слова в обратных кавычках (пути и имена тензоров) и идентификаторы вариантов проверяются отдельно.
import re
import sys
from pathlib import Path

ALLOW = {
    "FINAL", "GSM8K", "w8", "p", "JSON", "combined", "dev", "search", "sha256",
    "manifest", "results.jsonl", "finalize", "batch", "size", "dtype", "bfloat16",
    "GPU", "McNemar",
}
VARIANT = re.compile(r"^w\d+(_[A-Za-z0-9]+)*$")
CJK = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
MARKER = re.compile(r"<!--\s*конец журнал: (§\d+)")


def latin_tokens(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", text)
    out = set()
    for t in toks:
        t = t.rstrip(".,;:!?«»`")
        if t and t not in ALLOW and not VARIANT.match(t):
            out.add(t)
    return sorted(out)


def main():
    j = Path("runs/JOURNAL.md").read_text(encoding="utf-8")
    lines = j.splitlines()
    marks = [i for i, ln in enumerate(lines) if MARKER.search(ln)]
    print("строк:", len(lines))
    print("маркеров:", len(marks))
    if marks:
        last = lines[marks[-1]].strip()
        print("маркер последней строкой:", last == lines[-1].strip())
        print("метка:", MARKER.search(last).group(1))
    print("CJK:", len(CJK.findall(j)))
    if len(sys.argv) > 1:
        block = Path(sys.argv[1]).read_text(encoding="utf-8")
        bad = latin_tokens(block)
        print("латинские слова вне списка в новом блоке:", bad if bad else "нет")
    else:
        print("новый блок не передан — проверка только по маркеру и CJK")


main()
