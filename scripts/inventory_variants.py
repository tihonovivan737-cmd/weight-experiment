#!/usr/bin/env python3
"""Инвентарь runs/variants для безопасной очистки весов (только чтение).

Референсы (w8) считаются из его собственных results.jsonl, а не вписаны руками.
Сравнение «лучше w8» идёт только по сопоставимым прогонам (dev n=100, search n=250,
они же combined n=350). Экран 350 = dev100 + search250.

Классификация (консервативная):
  KEEP_CKPT      — контрольная точка w8.
  KEEP_BETTER    — превзошёл w8 хотя бы по одной измеренной метрике
                   (верно / обрывы / извлечённый FINAL) на сопоставимом сплите.
  KEEP_UNTESTED  — полный экран 350 не завершён (перспективный / не проверенный).
  KEEP_UNKNOWN   — нельзя проверить: нет манифеста/рецепта/verify/сравнений,
                   либо файл весов имеет >1 связи / совпадает по inode с исходной
                   моделью или контрольной точкой.
  DROP_WEIGHTS   — полный экран 350, метрики строго не лучше w8, артефакты полные,
                   файл весов — обычный отдельный файл (1 связь).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "runs" / "variants"
RECIPES = ROOT / "recipes"
CKPT = "w8_z_gate_mild_format"


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


def main() -> None:
    ref = {sp: stats(VARIANTS / CKPT / sp) for sp in ("dev", "search", "test")}
    ref["combined"] = combined(ref)
    orig_shards = {}
    for p in (ROOT / "models" / "qwen3.5-0.8b").rglob("*.safetensors"):
        st = p.stat()
        orig_shards[(st.st_ino, st.st_size)] = str(p.relative_to(ROOT))
    ckpt_shards = {}
    for p in (VARIANTS / CKPT / "model").rglob("*.safetensors"):
        st = p.stat()
        ckpt_shards[(st.st_ino, st.st_size)] = str(p.relative_to(ROOT))

    out = []
    for v in sorted(p for p in VARIANTS.iterdir() if p.is_dir()):
        name = v.name
        man_p = v / "manifest.json"
        man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else None
        recipe = RECIPES / f"{name}.json"
        base = "?"
        if man:
            b = Path(man.get("base_model_dir") or "")
            n = b.name
            base = "orig" if (n if n != "model" else b.parent.name).startswith("qwen3.5") \
                else (n if n != "model" else b.parent.name)
        wfiles = []
        if (v / "model").is_dir():
            for f in sorted((v / "model").rglob("*.safetensors")):
                st = f.stat()
                wfiles.append({"path": str(f.relative_to(ROOT)), "size": st.st_size,
                               "links": st.st_nlink, "ino": st.st_ino})
        res = {sp: stats(v / sp) for sp in ("dev", "search", "test", "smoke")}
        res = {k: x for k, x in res.items() if x}
        res["combined"] = combined(res)
        reports = sorted(p.name for p in v.glob("compare*.json"))
        verify = v / "verify.json"
        vok = False
        if verify.exists():
            vj = json.loads(verify.read_text(encoding="utf-8"))
            vok = all(x == "OK" for k, x in vj.get("checks", {}).items()
                      if not k.endswith("_detail")) and not vj.get("problems")

        if name == CKPT:
            cls, why = "KEEP_CKPT", "контрольная точка серии"
        elif not wfiles:
            cls, why = "KEEP_UNKNOWN", "каталог без файлов весов (только результаты)"
        else:
            better = []
            for sp in ("dev", "search", "combined"):
                a, b = res.get(sp), ref.get(sp)
                if not a or not b or a["n"] != b["n"]:
                    continue
                if a["correct"] > b["correct"]:
                    better.append(f"{sp}: верно {a['correct']} > {b['correct']}")
                if a["truncated"] < b["truncated"]:
                    better.append(f"{sp}: обрывов {a['truncated']} < {b['truncated']}")
                if a["extracted"] > b["extracted"]:
                    better.append(f"{sp}: FINAL {a['extracted']} > {b['extracted']}")
            linked = [f for f in wfiles
                      if f["links"] > 1 or (f["ino"], f["size"]) in orig_shards
                      or (f["ino"], f["size"]) in ckpt_shards]
            if better:
                cls, why = "KEEP_BETTER", "; ".join(better)
            elif not res.get("combined"):
                cls, why = "KEEP_UNTESTED", "полный экран 350 не завершён"
            else:
                missing = [x for x, ok in (("манифест", bool(man)), ("рецепт", recipe.exists()),
                                           ("verify OK", vok), ("сравнения", bool(reports)))
                           if not ok]
                if missing:
                    cls, why = "KEEP_UNKNOWN", "не сохранено/не подтверждено: " + ", ".join(missing)
                elif linked:
                    cls, why = "KEEP_UNKNOWN", "файл весов имеет >1 связь или общий inode"
                else:
                    cls, why = "DROP_WEIGHTS", "строго не лучше w8 по всем метрикам экрана 350"
        out.append({"variant": name, "class": cls, "why": why, "base": base,
                    "recipe": f"recipes/{name}.json" if recipe.exists() else None,
                    "manifest": man_p.exists(), "verify": vok,
                    "reports": reports, "totals": (man or {}).get("totals"),
                    "results": res, "weight_files": wfiles,
                    "weight_bytes": sum(f["size"] for f in wfiles)})

    (ROOT / "runs" / "analysis" / "cleanup_inventory.json").write_text(
        json.dumps({"reference_w8": ref, "variants": out}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    lines = [f"референс w8: " + "  ".join(
        f"{sp} верно/обрыв/FINAL = {ref[sp]['correct']}/{ref[sp]['truncated']}/{ref[sp]['extracted']}"
        for sp in ("dev", "search", "combined", "test")), ""]
    tot = sum(o["weight_bytes"] for o in out)
    lines.append(f"вариантов в runs/variants: {len(out)}, весов: {tot / 1e9:.2f} GB")
    order = ["KEEP_CKPT", "KEEP_BETTER", "KEEP_UNTESTED", "KEEP_UNKNOWN", "DROP_WEIGHTS"]
    for cls in order:
        sel = [o for o in out if o["class"] == cls]
        lines.append(f"\n### {cls}: {len(sel)} вариантов, весов "
                     f"{sum(o['weight_bytes'] for o in sel) / 1e9:.2f} GB")
        for o in sorted(sel, key=lambda x: -x["weight_bytes"]):
            r = o["results"]
            fmt = lambda sp: ("нет" if not r.get(sp) else
                              f"{r[sp]['correct']}/{r[sp]['truncated']}/{r[sp]['extracted']}")
            lines.append(f"  {o['variant']:<34} {o['weight_bytes'] / 1e9:5.2f} GB  "
                         f"база {o['base']:<24} dev {fmt('dev'):<12} "
                         f"search {fmt('search'):<14} 350 {fmt('combined'):<14} "
                         f"| {o['why'][:80]}")
    text = "\n".join(lines) + "\n"
    (ROOT / "runs" / "analysis" / "cleanup_inventory.txt").write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
