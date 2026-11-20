#!/usr/bin/env python3
"""H31: связь поведения в ХВОСТЕ генерации (последние 30 шагов) с числом обрывов.

Только чтение уже существующих зондов прямых прогонов (без градиентов, без новых прогонов).
Сравниваются варианты с разным числом обрывов на одних и тех же 350 отборочных задачах:
  меньше обрывов: w29 (10), w24 (17), w8 (18)
  больше обрывов: w28 (27), w33 (72)
  база: 24
Метрики хвоста из probe_layers.py: *_tail = статистика по последним 30 шагам генерации
(steps[:, -30:]), n_steps_mean = средняя длина генерации в зонде.

Вывод: runs/analysis/h31_tail.txt
"""
import json
from pathlib import Path

FULL = [3, 7, 11, 15, 19, 23]
LIN = [l for l in range(24) if l not in FULL]

# поведение на фиксированных 350 задачах отбора (search250+dev100) берётся не «на память»,
# а суммированием из сводных таблиц волн: runs/analysis/wave6_table.json и wave8_table.json
TABLES = ["runs/analysis/wave6_table.json", "runs/analysis/wave8_table.json"]
PROBE_OF = {
    "base_qwen3.5-0.8b": "runs/analysis/probe_base_qwen3.5-0.8b.json",
    "w8_z_gate_mild_format": "runs/analysis/probe_w8_model.json",
    "w24_fullgate_open085": "runs/analysis/probe6_w24_fullgate_open085.json",
    "w28_fullgate_close115": "runs/analysis/probe6_w28_fullgate_close115.json",
    "w29_z_total087": "runs/analysis/probe6_w29_z_total087.json",
    "w33_fullgate_open_closed_heads": "runs/analysis/probe6_w33_fullgate_open_closed_heads.json",
    "w35_z_dominant_damp078": "runs/analysis/probe6_w35_z_dominant_damp078.json",
    "w36_z_late_extra094": "runs/analysis/probe6_w36_z_late_extra094.json",
    "w37_z_early_release1075": "runs/analysis/probe6_w37_z_early_release1075.json",
}

def load_behavior():
    """{variant: correct/truncated/extracted на 350 задачах} из сводных таблиц волн.

    Берётся готовая строка `combined` (search250+dev100), а не сложение вручную.
    Обе референсные модели (`orig`, `w8`) лежат внутри `vs.<ref>.base` любой записи
    таблицы по каждому сплиту — они суммируются по search+dev.
    """
    beh = {}
    entries = []
    for t in TABLES:
        d = json.loads(Path(t).read_text(encoding="utf-8"))
        for v, entry in d.items():
            c = entry.get("combined", {}).get("stats")
            if c:
                beh[v] = dict(correct=c["correct"], truncated=c["truncated"],
                              extracted=c["extracted"], n=c["n"], mean=c["mean_out_tokens"])
            entries.append(entry)
    for ref in ("orig", "w8"):
        parts = []
        for e in entries:
            p = [e.get(sp, {}).get("vs", {}).get(ref, {}).get("base") for sp in ("search", "dev")]
            p = [x for x in p if x]
            if len(p) == 2:
                parts = p
                break
        if not parts:
            continue
        n = sum(x["n"] for x in parts)
        beh[ref] = dict(correct=sum(x["correct"] for x in parts),
                        truncated=sum(x["truncated"] for x in parts),
                        extracted=sum(x["extracted"] for x in parts), n=n,
                        mean=sum(x["mean_out_tokens"] * x["n"] for x in parts) / n)
    return beh

def num(x):
    return None if x is None else float(x)

def summarize(p):
    L = p["layers"]
    g = lambda l, k: num(L[str(l)][k])
    out = {}
    out["n_steps"] = g(0, "n_steps_mean")
    for pre in ("", "_tail"):
        fa = sum(g(l, f"gain_attn{pre}") for l in FULL)
        la = sum(g(l, f"gain_attn{pre}") for l in LIN)
        ml = sum(g(l, f"gain_mlp{pre}") for l in range(24))
        out[f"full{pre}"], out[f"lin{pre}"], out[f"mlp{pre}"] = fa, la, ml
        out[f"attn_share{pre}"] = (fa + la) / (fa + la + ml)
    out["gate_sig_full"] = sum(g(l, "mean_gate_sigmoid") for l in FULL) / 6
    out["gate_sig_full_tail"] = None  # поле mean_gate_sigmoid_tail зондом не пишется (проверено по фактическому списку ключей)
    out["z_gate_lin"] = sum(g(l, "mean_z_gate") for l in LIN) / 18
    out["z_gate_lin_tail"] = sum(g(l, "mean_z_gate_tail") for l in LIN) / 18
    out["beta_lin"] = sum(g(l, "mean_beta") for l in LIN) / 18
    out["beta_lin_tail"] = None  # mean_beta_tail зондом не пишется; поголовно есть beta_head_tail
    out["beta_head_tail_max"] = max(max(L[str(l)]["beta_head_tail"]) for l in LIN)
    out["beta_head_tail_min"] = min(min(L[str(l)]["beta_head_tail"]) for l in LIN)
    out["rms_pre_tail_ratio"] = sum(g(l, "rms_residual_pre_tail") / g(l, "rms_residual_pre") for l in range(24)) / 24
    out["rms_pre_tail_last"] = sum(g(l, "rms_residual_pre_tail") for l in range(24)) / 24
    return out

def main():
    beh = load_behavior()
    if "orig" in beh:
        beh["base_qwen3.5-0.8b"] = beh.pop("orig")
    if "w8" in beh:
        beh["w8_z_gate_mild_format"] = beh.pop("w8")
    rows = {}
    for name, probe_file in PROBE_OF.items():
        path = Path(probe_file)
        if not path.exists():
            print(f"нет зонда: {path}")
            continue
        meta = beh.get(name)
        if meta is None:
            print(f"нет поведения в сводных таблицах: {name}")
            continue
        rows[name] = (meta, summarize(json.loads(path.read_text(encoding="utf-8"))))

    lines = []
    add = lines.append
    add("H31: хвост генерации (последние 30 шагов) против числа обрывов на 350 задачах")
    add("Метрика: gain = RMS ветки / RMS остаточного потока на входе блока; *_tail = последние 30 шагов.")
    add("")
    hdr = f"{'вариант':34s} {'обр.':>5} {'верн.':>6} {'FINAL':>6} {'ток.':>6} {'шаг.':>7} {'LIN полн':>9} {'LIN хвост':>10} {'FULL полн':>9} {'FULL хвост':>11} {'MLP полн':>9} {'MLP хвост':>10} {'attention-доля':>15}"
    add(hdr)
    for name, (meta, s) in sorted(rows.items(), key=lambda kv: kv[1][0]["truncated"]):
        share_all = (s["full"] + s["lin"]) / (s["full"] + s["lin"] + s["mlp"]) * 100
        share_tail = (s["full_tail"] + s["lin_tail"]) / (s["full_tail"] + s["lin_tail"] + s["mlp_tail"]) * 100
        add(f"{name:34s} {meta['truncated']:>5} {meta['correct']:>6} "
            f"{meta['extracted']:>6} {meta['mean']:>6.1f} {s['n_steps']:>7.1f} "
            f"{s['lin']:>9.4f} {s['lin_tail']:>10.4f} {s['full']:>9.4f} {s['full_tail']:>11.4f} "
            f"{s['mlp']:>9.4f} {s['mlp_tail']:>10.4f} {share_all:>7.1f}%->{share_tail:>5.1f}%")
    add("")
    add("Отношение хвост/полная генерация (>1 значит, что в конце генерации ветка влияет сильнее, чем в среднем):")
    for name, (meta, s) in rows.items():
        add(f"{name:34s} LIN {s['lin_tail']/s['lin']:>6.3f}  FULL {s['full_tail']/s['full']:>6.3f}  MLP {s['mlp_tail']/s['mlp']:>6.3f}")
    add("")
    add("Гейт FULL-внимания (среднее по 6 слоям; хвостового значения нет — зонд его не пишет),")
    add("гейт LIN mean_z_gate (18 слоёв, хвост есть) и beta (18 слоёв, хвост поголовно):")
    for name, (meta, s) in sorted(rows.items(), key=lambda kv: kv[1][0]["truncated"]):
        add(f"{name:34s} gate_sig {s['gate_sig_full']:>6.4f} (хвост: нет поля) | "
            f"z_gate {s['z_gate_lin']:>6.4f} -> хвост {s['z_gate_lin_tail']:>6.4f} "
            f"({s['z_gate_lin_tail']/s['z_gate_lin']:>5.3f}) | "
            f"beta {s['beta_lin']:>6.4f} | beta_head хвост min {s['beta_head_tail_min']:>6.4f} max {s['beta_head_tail_max']:>6.4f} | "
            f"RMS остатка хвост/полн {s['rms_pre_tail_ratio']:>5.3f}")
    add("")
    add("Примечание: зонды сделаны на 63 задачах search (шаг 4), max_new 200, bf16; обрывы в самой")
    add("генерации зонда ограничены 200 токенами, поэтому n_steps_mean здесь — длина в рамках зонда,")
    add("а реальные обрывы считаны по прогонам 350 задач с лимитом 1024 токена.")
    out = "runs/analysis/h31_tail.txt"
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nСохранено: {out}")

if __name__ == "__main__":
    main()
