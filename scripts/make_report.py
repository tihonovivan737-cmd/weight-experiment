"""Сводная таблица по всем прогонам вариантов в runs/variants/*/*.

Собирает summary.json каждого прогона + compare_*.json (результаты compare_runs.py)
и печатает отсортированную таблицу: набор, accuracy, дельта, исправлено/сломано, p Макнемара,
обрывы, среднее число токенов. Также пишет runs/analysis/screening_table.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
BASELINES = {"dev": ROOT / "runs/baseline_dev/summary.json",
             "search": ROOT / "runs/baseline_search/summary.json",
             "smoke": ROOT / "runs/env_probe/smoke1024/summary.json",
             "test": ROOT / "runs/baseline_test/summary.json"}


def load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    rows = []
    for var_dir in sorted(glob.glob(str(ROOT / "runs/variants/*"))):
        name = Path(var_dir).name
        man = load(Path(var_dir) / "manifest.json")
        footprint = None
        if man:
            footprint = man.get("totals", {}).get("values_actually_changed")
        for split in ("smoke", "dev", "search", "test"):
            sub = Path(var_dir) / split
            s0 = load(sub / "summary.json")
            if not s0:
                continue
            s = s0.get("summary", s0)
            cmp_ = load(sub / "compare.json") or load(Path(var_dir) / f"compare_{split}.json")
            base_full = load(BASELINES[split])
            base = base_full.get("summary", base_full) if base_full else None
            row = {
                "variant": name, "split": split,
                "completed": s["completed"], "total": s["total"],
                "correct": s["correct"], "accuracy": s["accuracy"],
                "extracted": s["answer_extracted"], "truncated": s["truncated"],
                "mean_tokens": s["mean_output_tokens"], "mean_seconds": s["mean_seconds"],
                "values_changed": footprint,
                "weights_sha256_16": next(iter((s0.get("config", {})
                                                .get("model_info", {})
                                                .get("weights", {}) or {}).values()), {}).get("sha256", "")[:16],
                "base_accuracy": base["accuracy"] if base else None,
            }
            if cmp_:
                row.update({
                    "delta": cmp_.get("delta_correct"),
                    "fixed": len(cmp_.get("fixed", [])),
                    "broken": len(cmp_.get("broken", [])),
                    "p_mcnemar": cmp_.get("mcnemar_p"),
                    "answers_differ": cmp_.get("answers_differ"),
                })
            rows.append(row)

    # прокси
    proxy = {}
    for p in glob.glob(str(ROOT / "runs/proxy/*/proxy_summary.json")):
        key = Path(p).parent.name
        proxy[key] = load(Path(p))

    out = {"runs": rows, "proxy": proxy}
    Path(ROOT / "runs/analysis/screening_table.json").parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / "runs/analysis/screening_table.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    hdr = (f"{'variant':<24}{'split':<7}{'готово':>7}{'acc':>8}{'база':>8}{'Δ':>5}"
           f"{'+':>4}{'-':>4}{'p':>9}{'обрыв':>6}{'извл':>6}{'ток':>7}"
           f"{'значений':>11}{'sha256_16':>18}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["split"], -(x.get("delta") or -999), x["variant"])):
        p = r.get("p_mcnemar")
        print(f"{r['variant']:<24}{r['split']:<7}{r['completed']:>7}{r['accuracy']:>8}"
              f"{(r['base_accuracy'] if r['base_accuracy'] is not None else 0):>8}"
              f"{(r.get('delta') if r.get('delta') is not None else ''):>5}"
              f"{(r.get('fixed') if r.get('fixed') is not None else ''):>4}"
              f"{(r.get('broken') if r.get('broken') is not None else ''):>4}"
              f"{(f'{p:.3f}' if p is not None else ''):>9}"
              f"{r['truncated']:>6}{r['extracted']:>6}{r['mean_tokens']:>7}"
              f"{(r.get('values_changed') if r.get('values_changed') is not None else ''):>11}"
              f"{r.get('weights_sha256_16', ''):>18}")

    if proxy:
        print("\nПрокси (teacher-forced):")
        b = proxy.get("baseline_search100", {})
        print(f"{'variant':<24}{'nll_ref':>10}{'ans_prob':>10}{'top1_all':>9}")
        for k, v in sorted(proxy.items()):
            if not v:
                continue
            print(f"{k:<24}{v['mean_nll_ref']:>10}{v['mean_answer_prob']:>10}"
                  f"{v['answer_top1_all_match']:>9}")
            if b and k != "baseline_search100":
                dn = v["mean_nll_ref"] - b["mean_nll_ref"]
                dp = v["mean_answer_prob"] - b["mean_answer_prob"]
                print(f"{'  Δ к базе':<24}{dn:>10.5f}{dp:>10.5f}")
    print(f"\nСохранено: {ROOT/'runs/analysis/screening_table.json'}")


if __name__ == "__main__":
    main()
