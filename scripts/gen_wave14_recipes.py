#!/usr/bin/env python3
"""Генератор рецепта волны 14 (журнал §69.2, полосы П69.1).

Одиночное плечо: только расширение покрытия уже измеренного рычага формата с 10
строк до 12. 10 операций копируются из recipe w50 дословно (факторы — измеренные,
не переписываются вручную), 2 новые операции (строки 198 и 248046) считаются той же
формулой генератора волны 13:

    f_i = (1 - s) + s * (u_i / x_i)   =>   y_i = x_i * f_i = (1 - s) * x_i + s * u_i

где x — строка конкурента, u — ближайшая строка FINAL по косинусу, s = +0,015,
renorm = true. Правило отбора §43.2 (запас < 1) для этих двух строк не выполняется —
они берутся как измеренные строки с наибольшим запасом, какие рычаг формата никогда
не трогал. База сборки — контрольная точка w8_z_gate_mild_format, только на чтение.
"""
import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/variants/w8_z_gate_mild_format/model"
MEAS = ROOT / "runs/analysis/final_logit_rows.json"
W50 = ROOT / "recipes/w50_finalrow_rotation15_smallmargin.json"
FINAL_ROWS = [95429, 64697]
FINAL_TOKENS = {"95429": "FINAL", "64697": " FINAL"}
TENSOR = "model.language_model.embed_tokens.weight"
S_STEP = 0.015
NEW_ROWS = [198, 248046]
NAME = "w74_finalrow_plus_two_rows198_248046_on_w8"


def load_base_rows(path: Path):
    st = sorted(path.rglob("*.safetensors"))[0]
    with open(st, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(n).decode("utf-8"))
        info = header[TENSOR]
        data_start = 8 + n
        fp.seek(data_start + info["data_offsets"][0])
        raw = fp.read(info["data_offsets"][1] - info["data_offsets"][0])
    buf = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"])
    return st, buf


def factors_for_row(x, u, s):
    f = torch.empty_like(x)
    nz = x != 0
    f[nz] = (1.0 - s) + s * (u[nz] / x[nz])
    f[~nz] = 1.0 - s
    return f


def mean_L(meas):
    return sum(p["final_logit"] for p in meas["per_position"]) / len(meas["per_position"])


def main() -> int:
    meas = json.loads(MEAS.read_text(encoding="utf-8"))
    w50 = json.loads(W50.read_text(encoding="utf-8"))
    ops = [json.loads(json.dumps(o)) for o in w50["ops"]]
    print(f"операций перенесено из w50: {len(ops)}")
    st, buf = load_base_rows(BASE)
    L = mean_L(meas)
    print(f"база: {st}")

    by_row = {r["row"]: r for r in meas["rows"]}
    wsum_new, npos_new = 0.0, 0
    for row in NEW_ROWS:
        r = by_row[row]
        x = buf[row].float()
        best = max(FINAL_ROWS, key=lambda fr: float(torch.dot(x, buf[fr].float())))
        u = buf[best].float()
        n0 = float(torch.linalg.norm(x))
        cos = float(torch.dot(x, u) / (n0 * float(torch.linalg.norm(u))))
        f = factors_for_row(x, u, S_STEP)
        y = x * f
        n1 = float(torch.linalg.norm(y))
        kappa = n0 / max(n1, 1e-12)
        new_margin = kappa * (1 - S_STEP) * r["mean_margin"] + (kappa - 1) * L
        wsum_new += r["n_positions"] * new_margin
        npos_new += r["n_positions"]
        fmax = float(f.abs().max())
        fmin = float(f.abs().min())
        step_pct = float((y - x).abs().max()) / n0 * 100.0
        neff = int(((y.to(torch.bfloat16).float() - x) != 0).sum())
        guard = int((x == 0).sum())
        print(f"строка {row} ({r['token']}): запас {r['mean_margin']}, "
              f"cos к {best} = {cos:.4f}, ожидаемый запас после поворота "
              f"{new_margin:+.3f}, |f|max {fmax:.4f}, |f|min {fmin:.4f}, "
              f"изменённых значений {neff} из 1024, нулей {guard}")
        ops.append({
            "tensors": [TENSOR],
            "op": "tilt",
            "renorm": True,
            "s": S_STEP,
            "rows": [row],
            "factors": [float(v) for v in f.tolist()],
            "row_tokens": {str(row): r["token"]},
            "note": (f"строка {row} ({r['token']}), n={r['n_positions']}, "
                     f"средний запас {r['mean_margin']:.3f}, cos к строке {best} "
                     f"({FINAL_TOKENS[str(best)]}) = {cos:.3f}, ожидаемый запас после "
                     f"поворота {new_margin:+.3f}; |f|max={fmax:.3f}, шаг max {step_pct:.2f} % "
                     f"нормы строки, реально изменённых значений после округления до bf16 "
                     f"{neff} из 1024, guard для нулей {guard}; правило §43.2 не выполнено, "
                     f"строка взята как измеренная с наибольшим запасом"),
        })

    recipe = {
        "name": NAME,
        "hypothesis": (
            "H52 (волна 14, журнал §69.2): один рычаг — только расширение покрытия "
            "измеренного рычага формата с 10 строк до 12: те же 10 операций w50 "
            "плюс строки 198 и 248046. Тот же оператор tilt, тот же шаг s = +0,015, "
            "то же восстановление нормы строки, та же база w8, та же формула факторов. "
            "Факторы для новых строк взяты из генератора волны 13, не оценены вручную. "
            "Никакого головного рычага и никаких других тензоров."
        ),
        "expected_effect": (
            f"Средний ожидаемый запас после поворота по двум новым строкам: "
            f"{wsum_new / npos_new:+.3f} логита против {L:.3f} у FINAL (взвешенно по "
            f"{npos_new} позициям); по 10 перенесённым строкам оценка не меняется, она "
            "приведена в журнале §43. Отбор сделан по пробе на 63 задачах опоры search, "
            "поэтому search читается с осторожностью, решение по качеству — по dev. "
            "Полосы П69.1: dev математика 58-64, извлечение 94-98, обрывы 1-6; search "
            "математика 148-156, извлечение 236-244, обрывы 8-15; вместе математика "
            "206-218, извлечение 330-342, обрывы 9-21."
        ),
        "band": "П69.1",
        "ops": ops,
    }
    out = ROOT / "recipes" / f"{NAME}.json"
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записан {out.relative_to(ROOT)} ({len(ops)} операций, "
          f"{sum(len(o['factors']) for o in ops)} факторов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
