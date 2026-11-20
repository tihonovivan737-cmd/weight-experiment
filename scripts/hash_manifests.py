"""Собирает манифест волны: рецепты, SHA-256 весов, факт изменений из manifest.json, результат verify.

Волна 3 уже зафиксирована в runs/wave3_manifest.json (вручную). Этот скрипт делает такие же
манифесты для волн 4 и 5 и может пересобрать любой волны по списку вариантов.

Запуск:
  .venv/bin/python scripts/hash_manifests.py wave4 w14_z93_final13 w15_final_case_suppress
  .venv/bin/python scripts/hash_manifests.py wave5 w17_z93_only w16_final_rows12_only
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
BASE_SHARD = ROOT / "models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors"


def sha(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            h.update(b)
            if limit is not None:
                limit -= len(b)
                if limit <= 0:
                    break
    return h.hexdigest()


def header_sha(p: Path) -> str:
    n = int.from_bytes(p.open("rb").read(8), "little")
    return hashlib.sha256(p.open("rb").read(8 + n)).hexdigest()


def main() -> None:
    wave = sys.argv[1]
    variants = sys.argv[2:]
    base_sha = sha(BASE_SHARD)
    out = {
        "wave": wave,
        "base_model": str(BASE_SHARD),
        "base_sha256": base_sha,
        "base_header_sha256": header_sha(BASE_SHARD),
        "variants": [],
    }
    for v in variants:
        vd = ROOT / "runs/variants" / v
        recipe = json.loads((ROOT / "recipes" / f"{v}.json").read_text(encoding="utf-8"))
        man = json.loads((vd / "manifest.json").read_text(encoding="utf-8"))
        ver = json.loads((vd / "verify.json").read_text(encoding="utf-8"))
        shard = vd / "model/model.safetensors-00001-of-00001.safetensors"
        rec = {
            "name": v,
            "recipe_file": f"recipes/{v}.json",
            "hypothesis": recipe.get("hypothesis"),
            "expected_effect": recipe.get("expected_effect"),
            "ops": recipe.get("ops"),
            "model_sha256": sha(shard),
            "model_bytes": shard.stat().st_size,
            "model_header_sha256": header_sha(shard),
            "build_log": f"runs/variants/{v}/build.log",
            "manifest_totals": man.get("totals"),
            "changed_tensors": [
                {"tensor": c["tensor"], "shape": c.get("shape"), "dtype": c.get("dtype"),
                 "byte_offsets": c.get("byte_offsets"), "numel": c.get("numel"),
                 "values_actually_changed": c.get("values_actually_changed"),
                 "mean_abs_delta": c.get("mean_abs_delta"), "max_abs_delta": c.get("max_abs_delta"),
                 "sha256_before": c.get("sha256_before"), "sha256_after": c.get("sha256_after")}
                for c in man.get("changes", [])
            ],
            "inverse_ops": man.get("inverse_ops"),
            "restore_note": man.get("restore_note"),
            "computation": man.get("computation"),
            "verify": {
                "problems": ver.get("problems", []),
                "checks": ver.get("checks"),
                "sizes": ver.get("sizes"),
                "restoration_data": bool(man.get("inverse_ops")),
            },
        }
        out["variants"].append(rec)
    dst = ROOT / f"runs/{wave}_manifest.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано: {dst}")
    for rec in out["variants"]:
        print(f"{rec['name']}: sha {rec['model_sha256'][:16]}…  tensors {len(rec['changed_tensors'])}  "
              f"values {rec['manifest_totals'].get('values_actually_changed')}  "
              f"problems {len(rec['verify']['problems'])}")


if __name__ == "__main__":
    main()
