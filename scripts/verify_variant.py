"""Проверка целостности варианта весов (только чтение).

Сверяет runs/variants/<имя>/manifest.json с фактическими байтами:
  1) размер файла и заголовок safetensors идентичны базовому файлу;
  2) каждый записанный в манифест байтовый диапазон имеет sha256 == sha256_after,
     а его исходные байты в ВАРИАНТЕ отсутствуют (т.е. диапазон действительно изменён)
     и sha256 == sha256_before в БАЗОВОМ файле;
  3) внеunion изменённых диапазонов файлы побайтово идентичны.
Пишет runs/variants/<имя>/verify.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
CHUNK = 1 << 21


def sha256_range(path: Path, start: int, end: int) -> str:
    h = hashlib.sha256()
    remaining = end - start
    with path.open("rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(CHUNK, remaining))
            if not chunk:
                raise SystemExit(f"неожиданный конец файла {path}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="имя каталога в runs/variants")
    args = ap.parse_args()

    vdir = ROOT / "runs/variants" / args.variant
    man = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    base = Path(man["base_model_dir"]) / man["base_safetensors"]
    new = vdir / "model" / man["base_safetensors"]
    report: dict = {"variant": args.variant, "checks": {}, "problems": []}

    def check(name: str, ok: bool, detail=None):
        report["checks"][name] = "OK" if ok else "FAIL"
        if detail is not None:
            report["checks"][name + "_detail"] = detail
        if not ok:
            report["problems"].append(f"{name}: {detail}")

    # 0. хеш всего файла
    def fsha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        return h.hexdigest()

    base_sha, new_sha = fsha(base), fsha(new)
    check("базовый_файл_по_хешу", base_sha == man["base_safetensors_sha256"], base_sha[:16])
    check("файл_варианта_по_хешу", new_sha == man["new_safetensors_sha256"], new_sha[:16])
    check("размер_файла", base.stat().st_size == new.stat().st_size, new.stat().st_size)

    # 1. заголовок идентичен (ровно n байт JSON-заголовка, без захвата данных)
    with base.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head_base = f.read(n)
    with new.open("rb") as f:
        n2 = struct.unpack("<Q", f.read(8))[0]
        head_new = f.read(n2)
    check("заголовок_safetensors", head_base == head_new and n == n2, f"{n} байт")
    header_len = 8 + n

    # 2. каждый диапазон из манифеста
    ranges = []
    all_ok = True
    for c in man["changes"]:
        s, e = c["byte_offsets"]
        ranges.append((s, e))
        hb = sha256_range(base, s, e)
        hn = sha256_range(new, s, e)
        ok = hb == c["sha256_before"] and hn == c["sha256_after"] and hb != hn
        all_ok &= ok
        if not ok:
            report["problems"].append(
                f"диапазон {c['tensor']}{('@row' + str(c['row'])) if 'row' in c else ''}: "
                f"base {hb[:12]} (ожидалось {c['sha256_before'][:12]}), "
                f"variant {hn[:12]} (ожидалось {c['sha256_after'][:12]})")
    check("диапазоны_манифеста", all_ok, f"{len(ranges)} диапазонов, {sum(e - s for s, e in ranges):,} байт")

    # 3. всё остальное побайтово идентично
    merged: list[list[int]] = []
    for s, e in sorted(ranges):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    diff_outside = 0
    bytes_compared = 0
    size = base.stat().st_size

    def cmp_span(fb, fn, start: int, end: int) -> None:
        nonlocal diff_outside, bytes_compared
        p = start
        while p < end:
            n = min(CHUNK, end - p)
            fb.seek(p)
            fn.seek(p)
            a = fb.read(n)
            b = fn.read(n)
            if len(a) != n or len(b) != n:
                raise SystemExit(f"неожиданный конец файла на смещении {p}")
            bytes_compared += n
            if a != b:
                diff_outside += sum(1 for x, y in zip(a, b) if x != y)
            p += n

    with base.open("rb") as fb, new.open("rb") as fn:
        pos = 0
        for s, e in merged:          # отсортировано и без пересечений
            cmp_span(fb, fn, pos, s)
            pos = e
        cmp_span(fb, fn, pos, size)
    check("вне_диапазонов_идентично", diff_outside == 0,
          f"различающихся байт: {diff_outside} из {bytes_compared:,}")

    report["sizes"] = {"file_bytes": new.stat().st_size, "header_bytes": header_len,
                       "changed_ranges_bytes": sum(e - s for s, e in ranges)}
    out = vdir / "verify.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"=== {args.variant} ===")
    for k, v in report["checks"].items():
        print(f"  {k}: {v}")
    print("ИТОГ:", "OK" if not report["problems"] else f"ПРОБЛЕМЫ ({len(report['problems'])})")
    print("Отчёт:", out)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
