#!/usr/bin/env python
"""Построчная геометрия волны 13 (CPU, только чтение, без генерации).

Для каждого рецепта волны 13 независимо пересчитывает предсказание по точному
тождеству механизма:
    новый logit строки = k * ((1 - s) * старый logit + s * logit_FINAL),
    k = L2_до / L2_после смешивания (renorm),
    новый запас = k * ((1 - s) * запас + s * L_FINAL).
Значки (запасы) берутся из измерения §43.2 (runs/analysis/final_logit_rows.json),
строки и нормы — из шарда контрольной точки w8, зеркалом bf16 через torch
(авторитет np.uint16 не используется: bf16 — не uint16).

Отдельно проверяются: норма строки после round-trip, фактический cos после
округления, число значений, реально изменившихся после округления до bf16,
число нулей в строке (tilt не может сдвинуть нулевой элемент) и максимальный
|factor| на строку.

Результат: runs/analysis/wave13_geometry.json
"""
import json
import struct
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W8_SHARD = ROOT / "runs/variants/w8_z_gate_mild_format/model"
MEAS = ROOT / "runs/analysis/final_logit_rows.json"
OUT = ROOT / "runs/analysis/wave13_geometry.json"
TENSOR = "model.language_model.embed_tokens.weight"
FINAL_ROWS = [95429, 64697]
RECIPES = ["w50_finalrow_rotation15_smallmargin",
           "w51_finalrow_rotation15_smallmargin_negsign"]


def load_shard(p):
    with open(p, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        h = json.loads(fp.read(n).decode("utf-8"))
        i = h[TENSOR]
        fp.seek(8 + n + i["data_offsets"][0])
        raw = fp.read(i["data_offsets"][1] - i["data_offsets"][0])
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(i["shape"])


def main():
    meas = json.loads(MEAS.read_text(encoding="utf-8"))
    L = sum(p["final_logit"] for p in meas["per_position"]) / len(meas["per_position"])
    shard = load_shard(sorted(W8_SHARD.rglob("*.safetensors"))[0])
    u = {r: shard[r].float() for r in FINAL_ROWS}
    rows = {r["row"]: r for r in meas["rows"]}
    res = {"model": str(W8_SHARD), "mean_final_logit": round(L, 3),
          "identity": "new_margin = k * ((1 - s) * margin + s * L_FINAL)",
          "recipes": {}}
    for name in RECIPES:
        rp = json.loads((ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
        entries = []
        acc = 0.0
        for op in rp["ops"]:
            r = op["rows"][0]
            s = float(op["s"])
            f = torch.tensor(op["factors"], dtype=torch.float32)
            x = shard[r].float()
            n0 = float(torch.linalg.norm(x))
            y = x * f
            n1 = float(torch.linalg.norm(y))
            k = n0 / max(n1, 1e-12)
            m = rows[r]["mean_margin"]
            # предсказание по тождеству, усреднённое по позициям этой строки
            pm = 0.0
            for p in meas["per_position"]:
                for tr in p["top"]:
                    # как в измерении §43.2: строка учитывается только там,
                    # где она действительно опережает FINAL (запас >= 0)
                    if tr[0] == r and tr[1] >= 0.0:
                        pm += k * ((1 - s) * tr[1] + s * p["final_logit"])
                        break
            pm /= rows[r]["n_positions"]
            yr = (y * (n0 / max(n1, 1e-12))).to(torch.bfloat16)
            ub = max(u, key=lambda rr: float(torch.dot(x, u[rr])))
            cos_a = float(torch.dot(x, u[ub]) / (n0 * float(torch.linalg.norm(u[ub]))))
            cos_b = float(torch.dot(yr.float(), u[ub]) / (float(torch.linalg.norm(yr.float())) * float(torch.linalg.norm(u[ub]))))
            neff = int((yr != shard[r]).sum())
            entries.append({"row": r, "token": op["row_tokens"][str(r)],
                            "n_positions": rows[r]["n_positions"],
                            "mean_margin": m, "s": s,
                            "kappa": round(k, 6),
                            "pred_new_margin": round(pm, 3),
                            "delta_predicted": round(pm - m, 3),
                            "cos_before": round(cos_a, 4), "cos_after": round(cos_b, 4),
                            "norm_before": round(n0, 4), "norm_after": round(n1, 4),
                            "max_abs_factor": round(float(f.abs().max()), 3),
                            "max_factor_step_pct": round(float(f.abs().max()) * float(x.abs().max()) / n0 * 100, 3),
                            "n_effective": neff, "zeros_in_row": int((x == 0).sum())})
            acc += pm * rows[r]["n_positions"]
        tot_n = sum(e["n_positions"] for e in entries)
        res["recipes"][name] = {"rows": sorted(entries, key=lambda e: -e["n_positions"]),
                                "weighted_pred_margin": round(acc / tot_n, 3),
                                "n_rows": len(entries)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, d in res["recipes"].items():
        print(f"=== {name}: {d['n_rows']} строк, среднее по позициям {d['weighted_pred_margin']:+.3f} ===")
        for e in d["rows"]:
            print(f"  {e['row']:>6} {e['token']:<7} n={e['n_positions']:>2} запас {e['mean_margin']:+.3f} -> {e['pred_new_margin']:+.3f} "
                  f"(Δ {e['delta_predicted']:+.3f}) |f|max {e['max_abs_factor']:6.1f} neff {e['n_effective']:>4} нули {e['zeros_in_row']}")


if __name__ == "__main__":
    main()
