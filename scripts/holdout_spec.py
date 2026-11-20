"""Печатает spec индексов удержанного пула из runs/splits/holdout6.json (никакого переролла)."""
import json, sys
from pathlib import Path
d = json.loads(Path("/home/mpiscil/weight-experiment/runs/splits/holdout6.json").read_text(encoding="utf-8"))
idx = sorted(int(i) for i in d["holdout_indices"])
spec = Path("/home/mpiscil/weight-experiment/runs/splits/holdout6.json").read_text(encoding="utf-8")
spec_str = d.get("indices_spec")
# сверка: spec из файла должен давать ровно те же индексы
def parse(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
if spec_str and parse(spec_str) != idx:
    print("НЕСОВПАДЕНИЕ indices_spec и holdout_indices", file=sys.stderr); sys.exit(1)
assert d["gold_answers_recorded"] is False, "эталонные ответы не должны быть записаны"
assert not d.get("overlap_with_used_train"), "пересечение с использованным train"
print(spec_str if spec_str else ",".join(map(str, idx)))
