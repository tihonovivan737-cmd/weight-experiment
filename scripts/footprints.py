#!/usr/bin/env python3
"""Авторитетная таблица следов изменения весов по manifest.json всех вариантов.

Пишет runs/analysis/footprints.txt и runs/analysis/footprints.json: для каждого варианта —
рецепт (операции), число затронутых тензоров, число реально изменённых значений,
mean/max |Δ|, хэши shard до/после, число проблем verify.json. Использовать как единственный
источник чисел в отчётах; расхождения с более ранними записями журнала разрешаются в пользу
этой таблицы (она пересчитана из манифестов).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
VD = ROOT / "runs/variants"


def main() -> int:
    rows = []
    for vd in sorted(VD.iterdir()):
        if not vd.is_dir():
            continue
        man = vd / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        v = vd / "verify.json"
        ver = json.loads(v.read_text(encoding="utf-8")) if v.exists() else {}
        t = m.get("totals", {})
        ops = []
        for op in m.get("recipe_ops", []):
            ten = ",".join(op.get("tensors", []))
            lam = op.get("lam", op.get("value", op.get("delta", "")))
            scope = ""
            if op.get("rows"):
                scope = " rows " + ",".join(str(r) for r in op["rows"])
            if op.get("indices"):
                scope = " idx " + ",".join(str(i) for i in op["indices"])
            ops.append(f"{op.get('op')}{lam and ' x' + str(lam) or ''} [{ten}{scope}]")
        rows.append({
            "variant": vd.name,
            "ops": len(m.get("recipe_ops", [])),
            "tensors_changed": t.get("tensors_changed"),
            "touched_tensor_ops": t.get("touched_tensor_ops"),
            "values_actually_changed": t.get("values_actually_changed"),
            "mean_abs_delta": t.get("mean_abs_delta"),
            "max_abs_delta": t.get("max_abs_delta"),
            "max_rel_change": t.get("max_rel_change"),
            "inverse_ops": len(m.get("inverse_ops", [])),
            "base_sha16": (m.get("base_safetensors_sha256") or "")[:16],
            "new_sha16": (m.get("new_safetensors_sha256") or "")[:16],
            "header_bytes": m.get("header_length_bytes"),
            "verify_problems": len(ver.get("problems", [])),
            "verify_checks": len(ver.get("checks", [])),
            "recipe": "; ".join(ops),
        })
    rows.sort(key=lambda r: -(r["values_actually_changed"] or 0))

    j = {"n_variants": len(rows), "rows": rows}
    (ROOT / "runs/analysis/footprints.json").write_text(
        json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = (f"{'variant':<26}{'опер':>6}{'тенз':>6}{'значений':>12}{'mean|Δ|':>11}{'max|Δ|':>10}"
           f"{'inv':>5}{'verif':>7}  sha(base→variant)")
    txt = ["Следы изменения весов по manifest.json (авторитетный источник чисел)",
           f"вариантов: {len(rows)}", "", hdr, "-" * len(hdr)]
    for r in rows:
        txt.append(
            f"{r['variant']:<26}{r['ops']:>6}{r['tensors_changed']:>6}"
            f"{r['values_actually_changed']:>12}{(r['mean_abs_delta'] or 0):>11.3e}"
            f"{(r['max_abs_delta'] or 0):>10.3e}{r['inverse_ops']:>5}"
            f"{r['verify_problems']:>7}  {r['base_sha16']}→{r['new_sha16']}")
    txt += ["", "Рецепты:", ""]
    for r in rows:
        txt.append(f"{r['variant']}: {r['recipe']}")
    (ROOT / "runs/analysis/footprints.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt[:len(rows) + 6]))
    print(f"\nСохранено: {ROOT/'runs/analysis/footprints.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
