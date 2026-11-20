#!/usr/bin/env python3
"""Семейная перепись серии: какие тензоры менялись, сколько вариантов и лучший исход.

Для каждого каталога в runs/variants читается manifest.json (список изменений) и, если есть,
dev/results.jsonl и search/results.jsonl. Вариант относится к семейству по набору изменённых
тензоров (имена слоёв схлопнуты). Для каждого семейства: число вариантов, число измеренных
на 350 задачах, лучший исход по математике вместе (и его формат/обрывы), а также медиана
математики по измеренным вариантам семейства.

Только чтение. Вывод: runs/analysis/family_census.txt
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARS = ROOT / "runs/variants"


def family_of(tensors):
    """Тензоры вида model.language_model.layers.N.<ветвь>.<имя> -> семейство."""
    keys = set()
    for t in tensors:
        parts = [x for x in t.split(".") if not x.isdigit()]
        tail = ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts)
        keys.add(tail.replace("model.language_model.", ""))
    return " + ".join(sorted(keys)) if keys else "нет изменений"


def read_results(path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    return dict(correct=sum(1 for r in rows if r.get("correct")),
                extracted=sum(1 for r in rows if (r.get("n_final_lines") or 0) >= 1),
                truncated=sum(1 for r in rows if r.get("truncated")),
                n=len(rows))


def main():
    fam = defaultdict(list)
    for m in sorted(VARS.glob("*/manifest.json")):
        name = m.parent.name
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        tensors = set()
        for ch in d.get("changes", []):
            if isinstance(ch, dict) and ch.get("tensor"):
                tensors.add(ch["tensor"])
        changed_values = (d.get("totals") or {}).get("values_actually_changed", 0)
        dev = read_results(VARS / name / "dev/results.jsonl")
        se = read_results(VARS / name / "search/results.jsonl")
        fam[family_of(tensors)].append({
            "name": name,
            "changed": changed_values,
            "dev": dev,
            "search": se,
        })

    lines = ["Перепись семейств изменённых тензоров (по манифестам и результатам).", ""]
    lines.append(f"{'семейство':56s} {'вариантов':>9} {'с 350':>6} {'медиана мат.':>13} {'лучший по математике':>34}")
    rows_out = []
    for f, items in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        full = []
        for it in items:
            if it["dev"] and it["search"]:
                comb = it["dev"]["correct"] + it["search"]["correct"]
                ext = it["dev"]["extracted"] + it["search"]["extracted"]
                tr = it["dev"]["truncated"] + it["search"]["truncated"]
                full.append((comb, ext, tr, it["name"]))
        med = ""
        best = ""
        if full:
            arr = sorted(x[0] for x in full)
            med = str(arr[len(arr) // 2])
            b = max(full)
            best = f"{b[3]}: {b[0]}/{b[1]}/{b[2]}"
        lines.append(f"{f[:56]:56s} {len(items):>9} {len(full):>6} {med:>13} {best:>34}")
        rows_out.append((f, len(items), len(full)))
    lines.append("")
    total = sum(n for _, n, _ in rows_out)
    lines.append(f"Всего каталогов с манифестом в переписи: {total}; семейств: {len(rows_out)}")
    text = "\n".join(lines)
    (ROOT / "runs/analysis/family_census.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
