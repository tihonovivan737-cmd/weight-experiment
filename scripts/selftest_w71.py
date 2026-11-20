#!/usr/bin/env python3
"""Самопроверка рецепта перед сборкой.

Проверяет, что поле `tensors` во всех операциях — список шаблонов, и подсчитывает,
сколько тензоров и значений затронет сборка. Отбор тензоров повторяет правило
matches() из scripts/modify_weights.py: без поля `layers` подходят все имена,
с полем `layers` — только имена с сегментом layers.N., где N в списке слоёв.
"""
import fnmatch
import json
import math
import re
import struct
import sys
from pathlib import Path

SHARD = Path("models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors")


def read_header(path: Path) -> dict:
    with path.open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n))


def matches(name: str, patterns: list, layers: list | None) -> bool:
    if not any(fnmatch.fnmatch(name, p) for p in patterns):
        return False
    if layers is None:
        return True
    m = re.search(r"layers\.(\d+)\.", name)
    return bool(m) and int(m.group(1)) in layers


def main() -> int:
    rec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "recipes/w71_finalrow_plus_qknorm108_on_w8.json")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    ops = rec["ops"]
    print("тип tensors:", "list" if all(isinstance(o["tensors"], list) for o in ops) else "НЕ list")
    header = read_header(SHARD)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    hit, vals, dtype = 0, 0, {}
    for op in ops:
        for name, info in tensors.items():
            if matches(name, op["tensors"], op.get("layers")):
                hit += 1
                vals += math.prod(info["shape"])
                dtype[info["dtype"]] = dtype.get(info["dtype"], 0) + 1
    print("совпало:", hit)
    print("значений:", vals)
    print("dtype:", dtype)
    rows = sum(len(op.get("rows", [])) for op in ops)
    print("строк:", rows)
    uniq = {k: v for k, v in tensors.items()
            if any(matches(k, op["tensors"], op.get("layers")) for op in ops)}
    changed = 0
    for op in ops:
        for name, info in tensors.items():
            if not matches(name, op["tensors"], op.get("layers")):
                continue
            dim = math.prod(info["shape"])
            if op.get("rows"):
                changed += len(op["rows"]) * info["shape"][-1]
            else:
                changed += dim
    print("уникальных тензоров:", len(uniq), "значений в них:",
          sum(math.prod(i["shape"]) for i in uniq.values()))
    print("планируемых изменённых значений:", changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
