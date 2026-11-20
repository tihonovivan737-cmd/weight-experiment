#!/usr/bin/env python3
"""Собирает сводную таблицу по всем вариантам в runs/variants/.

Источник — файлы на диске (проверка присутствия артефактов, а не память о запусках):
  manifest.json                — гипотеза и объём правки
  verify.json                  — целостность (итог OK/FAIL)
  compare_smoke.json           — smoke24
  compare_dev.json / dev/      — dev100
  compare_search.json / search/— search250
Пишет runs/analysis/variants_table.json и печатает таблицу.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
VAR = ROOT / "runs/variants"


def j(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def row(v: Path) -> dict:
    man = j(v / "manifest.json") or {}
    ver = j(v / "verify.json") or {}
    out = {
        "variant": v.name,
        "hypothesis": (man.get("hypothesis") or "")[:160],
        "values_changed": (man.get("totals") or {}).get("values_actually_changed"),
        "tensors_changed": (man.get("totals") or {}).get("tensors_changed"),
        "mean_abs_delta": (man.get("totals") or {}).get("mean_abs_delta"),
        "max_abs_delta": (man.get("totals") or {}).get("max_abs_delta"),
        "verify_ok": (len(ver.get("problems", ["?"])) == 0) if ver else None,
    }
    for split, key in (("smoke", "smoke"), ("dev", "dev"), ("search", "search")):
        cmp_ = j(v / f"compare_{split}.json")
        res = j(v / f"{key}/summary.json")
        n = None
        if cmp_:
            n = cmp_["variant_stats"]["n"]
            out[f"{split}_correct"] = cmp_["variant_stats"]["correct"]
            out[f"{split}_n"] = n
            out[f"{split}_delta"] = cmp_["delta_correct"]
            out[f"{split}_p"] = round(cmp_["mcnemar_p"], 4)
            out[f"{split}_extracted"] = cmp_["variant_stats"]["extracted"]
            out[f"{split}_truncated"] = cmp_["variant_stats"]["truncated"]
            out[f"{split}_tokens"] = cmp_["variant_stats"]["mean_out_tokens"]
        elif res:
            out[f"{split}_correct"] = res.get("correct")
            out[f"{split}_n"] = res.get("n")
        comb = j(v / "combined.json")
        if comb:
            c = comb["combined"]
            out["combined350_correct"] = c["variant_stats"]["correct"]
            out["combined350_delta"] = c["delta_correct"]
            out["combined350_p"] = round(c["mcnemar_p"], 4)
    return out


def main() -> None:
    rows = [row(v) for v in sorted(VAR.iterdir()) if v.is_dir()]
    rows.sort(key=lambda r: (r.get("search_delta") is None, -(r.get("search_delta") or -10**9)))
    (ROOT / "runs/analysis/variants_table.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = (f"{'variant':24} {'search':>10} {'Δ':>4} {'p':>6} {'dev':>10} {'Δ':>4} "
           f"{'вместе':>12} {'Δ':>4} {'p':>6} {'знач':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = f"{r.get('search_correct')}/{r.get('search_n')}" if r.get("search_correct") is not None else "—"
        d = f"{r.get('dev_correct')}/{r.get('dev_n')}" if r.get("dev_correct") is not None else "—"
        c = (f"{r.get('combined350_correct')}/350" if r.get("combined350_correct") is not None
             else ("350?" if r.get("search_correct") and r.get("dev_correct") else "—"))
        signif = "хуже" if (r.get("search_p") is not None and r["search_p"] < 0.05) else ""
        print(f"{r['variant']:24} {s:>10} {str(r.get('search_delta')):>4} "
              f"{str(r.get('search_p')):>6} {d:>10} {str(r.get('dev_delta')):>4} "
              f"{c:>12} {str(r.get('combined350_delta')):>4} {str(r.get('combined350_p')):>6} {signif:>5}")
    print(f"\nВсего вариантов: {len(rows)}")


if __name__ == "__main__":
    main()
