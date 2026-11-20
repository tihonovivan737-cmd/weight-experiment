#!/usr/bin/env python3
"""Независимая (lineage-aware) проверка варианта волны 12.

Обычный verify_variant.py сравнивает байты варианта с хешами, которые манифест
записал по одной операции на диапазон. Когда один тензор трогается двумя операциями
по подряд (наследие w8: lam=0.93 по всем LIN-слоям, затем lam=0.94 по поздним слоям),
ожидание первого прохода устаревает — это особенность учёта, а не повреждение файла.

Скрипт независимо пересчитывает ожидаемое состояние каждого тензора, применяя
операции рецепта в порядке lineage, и сверяет с фактическими байтами файла варианта.
Только чтение: ничего не изменяет и никуда не пишет.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

ROOT = Path("/home/mpiscil/weight-experiment")


def read_header(fp):
    n = struct.unpack("<Q", fp.read(8))[0]
    return 8 + n, json.loads(fp.read(n).decode("utf-8"))  # данные начинаются после 8-байтного размера заголовка


DTYPE_MAP = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
            "F64": torch.float64, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool}


def apply_op(t: torch.Tensor, op: dict, rows: list[int]) -> torch.Tensor:
    """t — тензор в фактическом состоянии на момент операции, возвращает тот же dtype.

    Важно: modify_weights.py перечитывает строку из файла перед каждой операцией
    (patch_range работает с файловым handles 'r+b'), поэтому повторная встреча
    той же строки в списке rows применяется накопленно. Здесь то же самое: строка
    берётся из уже изменённого состояния, а не из snapshot всего тензора.
    """
    lam = op.get("lam")
    val = op.get("value")
    out = t.float().clone()
    idx = rows if rows else list(range(out.shape[0]))
    for r in idx:
        x = out[r]
        if op["op"] == "scale":
            y = x * lam
        elif op["op"] == "add":
            y = x + val
        elif op["op"] == "logscale":
            y = x + math.log(lam)
        elif op["op"] == "scale_offset":
            y = (1.0 + x) * lam - 1.0
        elif op["op"] == "set_row":
            y = x.clone()
            y[op["index"]] = val
        elif op["op"] == "tilt":
            # Построчные factors — единственный источник числа, который читает
            # modify_weights.py; ключ s в рецепте только происхождение.
            f = op.get("factors")
            if not f:
                raise SystemExit("tilt требует непустой список factors")
            coef = torch.tensor(f, dtype=torch.float32)
            if coef.shape[-1] != x.shape[-1]:
                raise SystemExit(f"tilt: factors {tuple(coef.shape)} не соответствуют "
                                 f"последней оси {tuple(x.shape[-1:])}")
            y = x * coef
            if op.get("renorm"):
                n0 = x.norm()
                n1 = y.norm().clamp_min(1e-12)
                y = y * (n0 / n1)
        else:
            raise SystemExit(f"неизвестная операция: {op['op']}")
        if op.get("clip_abs") is not None:
            c = op["clip_abs"]
            y = y.clamp(-c, c)
        if op.get("clip_min") is not None:
            y = y.clamp(min=op["clip_min"])
        if op.get("clip_max") is not None:
            y = y.clamp(max=op["clip_max"])
        # modify_weights.py пишет в файл значение уже округлённое до dtype,
        # поэтому повторный проход по той же строке работает с округлённым числом.
        out[r] = y.to(t.dtype).float()
    return out.to(t.dtype)


def main(recipe_path: str, variant_dir: str) -> int:
    recipe = json.loads((ROOT / recipe_path).read_text(encoding="utf-8"))
    vdir = ROOT / variant_dir
    man_path = vdir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    # Порядок доверия: рецепт (если в нём есть base_model_dir), затем манифест,
    # и обязательно сверка sha256 с записанной в манифесте.
    cands = [recipe.get("base_model_dir"), man.get("base_model_dir")]
    base_file = None
    for c in cands:
        if not c:
            continue
        p = ROOT / c
        shard = next((f for f in sorted(p.rglob("*.safetensors"))), None)
        if shard is None:
            continue
        if man.get("base_safetensors_sha256") and hashlib.sha256(shard.read_bytes()).hexdigest() != man["base_safetensors_sha256"]:
            continue
        base_file = shard
        break
    if base_file is None:
        raise SystemExit("не найдена базовая модель, совпадающая с манифестом (нужен args.base)")
    var_file = next(vdir.rglob("*.safetensors"))
    print(f"база: {base_file}")

    fb = open(base_file, "rb")
    fv = open(var_file, "rb")
    hn, hb = read_header(fb)
    hv, hd = read_header(fv)
    print(f"sha256 базы:     {hashlib.sha256(base_file.read_bytes()).hexdigest()}")
    print(f"sha256 варианта: {hashlib.sha256(var_file.read_bytes()).hexdigest()}")
    print(f"размер файла варианта: {var_file.stat().st_size} байт, заголовок {hv} байт")

    plan: dict[str, list[tuple[dict, list[int]]]] = {}
    for op in recipe["ops"]:
        globs = op.get("tensors") or [op["tensor"]]
        rows = [int(r) for r in op.get("rows", [])]
        layers = [int(l) for l in op.get("layers", [])]
        for name in hd:
            hit = False
            for g in globs:
                if not fnmatch.fnmatch(name, g):
                    continue
                if layers:
                    parts = name.split(".")
                    if len(parts) < 4 or not parts[3].isdigit() or int(parts[3]) not in layers:
                        continue
                hit = True
                break
            if hit:
                plan.setdefault(name, []).append((op, rows))

    n_ops = sum(len(v) for v in plan.values())
    diff_inside = 0
    diff_outside = 0
    problems = []
    for name, info in hd.items():
        off, end = info["data_offsets"]
        fb.seek(hn + off)
        fv.seek(hv + off)
        base_b = fb.read(end - off)
        var_b = fv.read(end - off)
        if name not in plan:
            if base_b != var_b:
                diff_outside += sum(1 for a, b in zip(base_b, var_b) if a != b)
            continue
        dt = DTYPE_MAP.get(info["dtype"], torch.bfloat16)
        cur = torch.frombuffer(bytearray(base_b), dtype=dt).reshape(info["shape"])
        for op, rows in plan[name]:
            cur = apply_op(cur, op, rows)
        actual = torch.frombuffer(bytearray(var_b), dtype=dt).reshape(info["shape"])
        nd = int((cur != actual).sum())
        diff_inside += nd
        if nd:
            problems.append(f"{name}: расхождение {nd} значений из {cur.numel()}")

    print(f"тензоров в плане: {len(plan)}, операций по плану: {n_ops}")
    print(f"расхождений внутри плановых тензоров: {diff_inside}")
    print(f"байтов вне плановых тензоров, отличающихся от базы: {diff_outside}")
    for p in problems[:20]:
        print("  ", p)
    ok = diff_inside == 0 and diff_outside == 0
    print("ИТОГ:", "ОК" if ok else "ПРОБЛЕМЫ")
    fb.close()
    fv.close()
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        raise SystemExit(main(sys.argv[1], sys.argv[2]))
    raise SystemExit(main("recipes/w49_z_late094_qknorm108.json", "runs/variants/w49_z_late094_qknorm108"))
