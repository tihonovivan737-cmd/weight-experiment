"""Программное изменение весов модели без обучения (weight surgery).

Принцип:
  1) берётся ИСХОДНАЯ копия модели (models/qwen3.5-0.8b, она не трогается),
  2) в runs/variants/<имя>/model/ создаётся полная копия файлов модели,
  3) safetensors-файл патчится на месте: форма/dtype/смещения тензоров не меняются,
     меняются только байты значений (вычисления в float32, округление до bf16),
  4) пишется manifest.json с гипотезой, списком операций, размерами, статистикой
     изменений, хешами диапазонов ДО/ПОСЛЕ и данными для точного откатывания.

Поддерживаемые операции (все — детерминированные, без данных и без градиентов):
  scale        : w <- w * lam
  add          : w <- w + v
  logscale     : w <- w + ln(lam)      # умножает exp(w) на lam (для A_log)
  scale_offset : w <- (1 + w) * lam - 1 # для zero-centered RMSNorm, у norm'ы множитель (1+w)
  set_row      : w[idx] <- value       (редко; для точечных правок)

Запуск:
  .venv/bin/python scripts/modify_weights.py --recipe recipes/h1_half_decay.json
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/mpiscil/weight-experiment")
BASE_MODEL = ROOT / "models/qwen3.5-0.8b"


# --------------------------------------------------------------- safetensors
def read_header(path: Path) -> tuple[dict, int]:
    """Возвращает (метаданные тензоров, длину заголовка) файла safetensors."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def sha256_range(path: Path, start: int, end: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = f.read(min(remaining, 1 << 20))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


# --------------------------------------------------------------- операции
def apply_op(name: str, tensor: torch.Tensor, op: dict) -> torch.Tensor:
    """Возвращает изменённый тензор того же dtype (вычисления в float32)."""
    lam = op.get("lam")
    val = op.get("value")
    x = tensor.float()
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
        # Наклон вдоль последней оси (например 4 тапа causal-conv1d).
        # renorm=true восстанавливает L2-норму каждого канала по последней оси,
        # т.е. меняется только временной профиль, а не масштаб канала.
        f = op.get("factors")
        if not f:
            raise ValueError("tilt требует непустой список factors")
        coef = torch.tensor(f, dtype=torch.float32, device=x.device)
        if coef.shape[-1] != x.shape[-1]:
            raise ValueError(f"tilt: factors {tuple(coef.shape)} не соответствуют "
                             f"последней оси {tuple(x.shape[-1:])}")
        y = x * coef
        if op.get("renorm"):
            n0 = x.norm(dim=-1, keepdim=True)
            n1 = y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            y = y * (n0 / n1)
    else:
        raise ValueError(f"Неизвестная операция: {op['op']}")
    if op.get("clip_abs") is not None:
        c = op["clip_abs"]
        y = y.clamp(-c, c)
    if op.get("clip_min") is not None:
        y = y.clamp(min=op["clip_min"])
    if op.get("clip_max") is not None:
        y = y.clamp(max=op["clip_max"])
    return y.to(tensor.dtype)


def matches(name: str, patterns: list[str], layers: list[int] | None) -> bool:
    if not any(fnmatch.fnmatch(name, p) for p in patterns):
        return False
    if layers is not None:
        m = re.search(r"layers\.(\d+)\.", name)
        if not m or int(m.group(1)) not in layers:
            return False
    return True


def patch_range(fh, op_i: int, op: dict, tensor_name: str, disp_name: str, shape: list[int],
                dtype, abs_s: int, abs_e: int, extra: dict | None = None) -> dict:
    """Читает байтовый диапазон, применяет операцию, пишет обратно, возвращает запись манифеста."""
    fh.seek(abs_s)
    raw_before = fh.read(abs_e - abs_s)
    t_before = torch.frombuffer(bytearray(raw_before), dtype=dtype)
    if len(shape) > 1:
        t_before = t_before.view(*shape)
    t_after = apply_op(disp_name, t_before, op)
    delta = (t_after.float() - t_before.float()).abs()
    changed_values = int((delta > 0).sum())
    raw_after = t_after.contiguous().view(torch.uint8).numpy().tobytes()
    fh.seek(abs_s)
    fh.write(raw_after)
    rec = {
        "op_index": op_i,
        "op": op["op"],
        "params": {k: v for k, v in op.items() if k not in ("tensors", "layers", "rows")},
        "tensor": tensor_name,
        "shape": list(t_before.shape),
        "dtype": {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16"}[dtype],
        "byte_offsets": [abs_s, abs_e],
        "numel": int(t_before.numel()),
        "values_actually_changed": changed_values,
        "mean_abs_delta": round(float(delta.mean()), 8),
        "max_abs_delta": round(float(delta.max()), 8),
        "sha256_before": hashlib.sha256(raw_before).hexdigest(),
        "sha256_after": hashlib.sha256(raw_after).hexdigest(),
        "note": op.get("note", ""),
    }
    if extra:
        rec.update(extra)
    if t_before.numel() <= 8192:
        rec["values_before"] = [round(float(v), 8) for v in t_before.float().reshape(-1)]
        rec["values_after"] = [round(float(v), 8) for v in t_after.float().reshape(-1)]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--base", default=str(BASE_MODEL))
    ap.add_argument("--out", default=None, help="по умолчанию runs/variants/<name>")
    args = ap.parse_args()

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    name = recipe["name"]
    out_dir = Path(args.out or (ROOT / "runs/variants" / name))
    model_dir = out_dir / "model"
    if model_dir.exists():
        raise SystemExit(f"Вариант уже существует: {model_dir} (удали или выбери другое имя)")
    model_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.base)
    base_st = next(iter(sorted(base_dir.glob("*.safetensors"))))

    # 1. копируем все файлы модели, кроме кэша; сам safetensors копируем побайтово
    for src in sorted(base_dir.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(base_dir)
        if ".cache" in rel.parts or rel.name == ".gitattributes":
            continue
        dst = model_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    new_st = model_dir / base_st.name
    base_sha = sha256_file(base_st)
    if sha256_file(new_st) != base_sha:
        raise SystemExit("Копия safetensors не совпала с исходной — прервано.")

    header, header_len = read_header(new_st)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    # 2. применяем операции
    changes: list[dict] = []
    skipped: list[dict] = []
    with new_st.open("r+b") as fh:
        for op_i, op in enumerate(recipe["ops"]):
            patterns = op["tensors"]
            layers = op.get("layers")
            hits = [n for n in tensors if matches(n, patterns, layers)]
            if not hits:
                skipped.append({"op_index": op_i, "patterns": patterns,
                                "layers": layers, "reason": "нет совпавших тензоров"})
                continue
            for tname in sorted(hits):
                meta = tensors[tname]
                dtype = DTYPE_MAP.get(meta["dtype"])
                if dtype is None or not dtype.is_floating_point:
                    skipped.append({"op_index": op_i, "tensor": tname,
                                    "reason": f"неподдерживаемый dtype {meta['dtype']}"})
                    continue
                s, e = meta["data_offsets"]
                abs_s, abs_e = header_len + s, header_len + e
                rows = op.get("rows")
                if rows:
                    if len(meta["shape"]) != 2:
                        skipped.append({"op_index": op_i, "tensor": tname,
                                        "reason": "rows поддержан только для 2D-тензоров"})
                        continue
                    item = torch.tensor(0, dtype=dtype).element_size()
                    row_bytes = meta["shape"][1] * item
                    for r in rows:
                        if not (0 <= r < meta["shape"][0]):
                            skipped.append({"op_index": op_i, "tensor": tname,
                                            "reason": f"строка {r} вне диапазона"})
                            continue
                        rs = abs_s + r * row_bytes
                        rec = patch_range(fh, op_i, op, tname, f"{tname}[{r}]",
                                          [meta["shape"][1]], dtype, rs, rs + row_bytes,
                                          extra={"row": r,
                                                 "row_token": op.get("row_tokens", {}).get(str(r), "")})
                        changes.append(rec)
                        if rec["values_actually_changed"] == 0:
                            skipped.append({"op_index": op_i, "tensor": tname, "row": r,
                                            "reason": "изменение меньше шага квантования"})
                    continue
                rec = patch_range(fh, op_i, op, tname, tname, meta["shape"], dtype, abs_s, abs_e)
                changes.append(rec)
                if rec["values_actually_changed"] == 0:
                    skipped.append({"op_index": op_i, "tensor": tname,
                                    "reason": "изменение меньше шага квантования bf16"})

    new_sha = sha256_file(new_st)

    # 3. манифест с данными для отката
    inverse = []
    for c in changes:
        p = c["params"]
        kind = c["op"]
        if kind == "scale":
            inv = {"op": "scale", "lam": 1.0 / p["lam"]}
        elif kind == "add":
            inv = {"op": "add", "value": -p["value"]}
        elif kind == "logscale":
            inv = {"op": "logscale", "lam": 1.0 / p["lam"]}
        elif kind == "scale_offset":
            inv = {"op": "scale_offset", "lam": 1.0 / p["lam"]}
        elif kind == "tilt":
            inv = {"op": "tilt",
                   "factors": [1.0 / float(f) for f in p["factors"]],
                   "renorm": bool(p.get("renorm", False))}
        else:
            inv = None
        if inv:
            inv.update({"tensor": c["tensor"], "byte_offsets": c["byte_offsets"],
                        "shape": c["shape"]})
            if "row" in c:
                inv["row"] = c["row"]
            inverse.append(inv)

    totals = {
        "tensors_changed": len({c["tensor"] for c in changes}),
        "touched_tensor_ops": len(changes),
        "values_actually_changed": sum(c["values_actually_changed"] for c in changes),
        "numel_touched": sum(c["numel"] for c in changes),
        "mean_abs_delta": round(sum(c["mean_abs_delta"] * c["numel"] for c in changes)
                                / max(1, sum(c["numel"] for c in changes)), 8),
        "max_abs_delta": round(max((c["max_abs_delta"] for c in changes), default=0.0), 8),
    }

    manifest = {
        "name": name,
        "hypothesis": recipe.get("hypothesis", ""),
        "expected_effect": recipe.get("expected_effect", ""),
        "base_model_dir": str(base_dir),
        "base_safetensors": base_st.name,
        "base_safetensors_sha256": base_sha,
        "new_model_dir": str(model_dir),
        "new_safetensors_sha256": new_sha,
        "header_length_bytes": header_len,
        "computation": "тензоры читались как bf16, операции выполнялись в float32, "
                       "записаны обратно как bf16",
        "recipe_ops": recipe["ops"],
        "changes": changes,
        "skipped": skipped,
        "totals": totals,
        "inverse_ops": inverse,
        "restore_note": "Откат = пересоздать копию базовой модели (её sha256 зафиксирован) "
                        "или применить inverse_ops к байтовым диапазонам; для небольших "
                        "тензоров сохранены values_before.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"Вариант: {name}")
    print(f"Модель: {model_dir}")
    print(f"SHA256 весов: было {base_sha[:16]}… стало {new_sha[:16]}…")
    print(f"Тензоров изменено: {totals['tensors_changed']}; "
          f"значений изменено: {totals['values_actually_changed']:,} "
          f"(затронуто {totals['numel_touched']:,})")
    print(f"Среднее |Δ|: {totals['mean_abs_delta']}  Макс |Δ|: {totals['max_abs_delta']}")
    if skipped:
        print(f"Пропущено операций/тензоров: {len(skipped)}")
        for s in skipped[:10]:
            print("  ", s)
    print(f"Манифест: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
