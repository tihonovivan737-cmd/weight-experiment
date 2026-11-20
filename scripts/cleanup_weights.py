#!/usr/bin/env python3
"""Удаление файлов весов старых неудачных вариантов (только runs/variants/<v>/model/*.safetensors).

Перед каждым удалением проверяются все условия пользователя:
  1) полный отборочный экран 350 выполнен (dev n=100 + search n=250);
  2) метрики строго не лучше w8 (верно / обрывы / извлечённый FINAL) на всех сплитах;
  3) сохранены манифест, рецепт, verify.json (все проверки OK, problems пусты),
     результаты dev/search и файлы сравнений;
  4) файл весов — обычный отдельный файл: 1 жёсткая связь, inode не совпадает
     ни с исходной моделью, ни с w8 (не общая жёсткая ссылка).
Ничего вне runs/variants не трогается. Каталоги, манифесты, рецепты и результаты
остаются на месте.

Запуск: python scripts/cleanup_weights.py [--dry-run]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "runs" / "variants"
RECIPES = ROOT / "recipes"
CKPT = "w8_z_gate_mild_format"
ORIG_DIR = ROOT / "models" / "qwen3.5-0.8b"

CANDIDATES = [
    "c0_final_norm_gain",
    "v2_Alog_decay_double",
    "v9_mlp_damp",
    "w12_z_gate_strong",
    "w15_final_case_suppress",
    "w16_final_rows12_only",
    "w17_z93_only",
    "w1_final_token_amp",
    "w38_z_corner_early10008_late08742",
    "w41_format_embed132",
    "w43_format_embed144_latedose",
    "w46_z_headsel_sharpen",
    "w7_format_boost_mild",
]


def stats(d: Path) -> dict | None:
    f = d / "results.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    return {"n": len(rows),
            "correct": sum(1 for r in rows if r.get("correct")),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "extracted": sum(1 for r in rows if r.get("answer_extracted"))}


def combined(res: dict) -> dict | None:
    d, s = res.get("dev"), res.get("search")
    if not d or not s or d["n"] != 100 or s["n"] != 250:
        return None
    return {k: d[k] + s[k] for k in ("correct", "truncated", "extracted")} | {"n": 350}


def df_free_g() -> int:
    out = subprocess.run(["df", "-BG", "--output=avail", str(ROOT)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[-1].strip().rstrip("G"))


def protected_inodes() -> set[tuple[int, int]]:
    s = set()
    for p in list(ORIG_DIR.rglob("*.safetensors")) + \
            list((VARIANTS / CKPT / "model").rglob("*.safetensors")):
        st = p.stat()
        s.add((st.st_ino, st.st_size))
    return s


def main() -> None:
    dry = "--dry-run" in sys.argv
    ref = {sp: stats(VARIANTS / CKPT / sp) for sp in ("dev", "search")}
    ref["combined"] = combined(ref)
    prot = protected_inodes()
    before = df_free_g()

    deleted, skipped = [], []
    for name in CANDIDATES:
        v = VARIANTS / name
        man_p, rec_p, ver_p = v / "manifest.json", RECIPES / f"{name}.json", v / "verify.json"
        res = {sp: stats(v / sp) for sp in ("dev", "search")}
        res["combined"] = combined(res)
        reports = sorted(str(p.relative_to(ROOT)) for p in v.glob("compare*.json"))
        checks = {}
        checks["каталог"] = v.is_dir()
        checks["манифест"] = man_p.exists()
        checks["рецепт"] = rec_p.exists()
        vj = json.loads(ver_p.read_text(encoding="utf-8")) if ver_p.exists() else {}
        checks["verify_OK"] = bool(vj) and not vj.get("problems") and all(
            x == "OK" for k, x in vj.get("checks", {}).items() if not k.endswith("_detail"))
        checks["результаты_dev100"] = bool(res["dev"]) and res["dev"]["n"] == 100
        checks["результаты_search250"] = bool(res["search"]) and res["search"]["n"] == 250
        checks["сравнения"] = len(reports) >= 2
        checks["экран_350"] = bool(res["combined"])
        better = []
        if res["combined"]:
            for sp in ("dev", "search", "combined"):
                a, b = res[sp], ref[sp]
                if a["correct"] > b["correct"]:
                    better.append(f"{sp}:верно")
                if a["truncated"] < b["truncated"]:
                    better.append(f"{sp}:обрывы")
                if a["extracted"] > b["extracted"]:
                    better.append(f"{sp}:FINAL")
        checks["не_лучше_w8"] = not better
        shards = sorted((v / "model").glob("*.safetensors")) if (v / "model").is_dir() else []
        checks["есть_файлы_весов"] = bool(shards)
        checks["отдельные_файлы"] = bool(shards) and all(
            p.stat().st_nlink == 1 and (p.stat().st_ino, p.stat().st_size) not in prot
            for p in shards)
        ok = all(checks.values())
        rec = {"variant": name, "checks": checks, "better_metrics": better,
               "recipe": str(rec_p.relative_to(ROOT)) if rec_p.exists() else None,
               "manifest": str(man_p.relative_to(ROOT)) if man_p.exists() else None,
               "verify": str(ver_p.relative_to(ROOT)) if ver_p.exists() else None,
               "reports": reports, "results": res,
               "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                          "links": p.stat().st_nlink} for p in shards]}
        if not ok:
            rec["SKIPPED"] = [k for k, x in checks.items() if not x] or ["метрики лучше w8"]
            skipped.append(rec)
            print(f"ПРОПУСК {name}: {rec['SKIPPED']}")
            continue
        for p in shards:
            if dry:
                print(f"[dry] удалил бы {p.relative_to(ROOT)} ({p.stat().st_size} B)")
            else:
                p.unlink()
        rec["deleted_bytes"] = sum(f["bytes"] for f in rec["files"])
        deleted.append(rec)
        verb = "[dry] удалил бы" if dry else "УДАЛЕНО "
        print(f"{verb} {name}: {rec['deleted_bytes']} B "
              f"({len(rec['files'])} файл(ов) весов)")

    after = df_free_g()
    log = {"before_free_G": before, "after_free_G": after,
           "freed_G": after - before, "deleted": deleted, "skipped": skipped,
           "reference_w8": ref}
    if not dry:
        (ROOT / "runs" / "analysis" / "cleanup_deleted.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nсвободно до: {before} G, после: {after} G, освобождено: {after - before} G")
    print(f"удалено вариантов: {len(deleted)}, пропущено: {len(skipped)}")
    if not deleted and not dry:
        sys.exit("ничего не удалено — проверьте условия")


if __name__ == "__main__":
    main()
