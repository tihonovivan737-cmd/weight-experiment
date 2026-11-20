#!/usr/bin/env python
"""Генератор рецепта волны 27 (журнал §72.2, полосы П72.1).

Одиночное плечо — только рычаг формата: тот же оператор `tilt` с восстановлением
L2-нормы строки, тот же шаг s = +0,015, та же база w8, та же формула факторов,
что и в генераторе волны 13 (журнал §43):

    f_i = (1 - s) + s * (u_i / x_i)   =>   y_i = x_i * f_i = (1 - s) * x_i + s * u_i

где x — строка конкурента, u — ближайшая строка FINAL по скалярному произведению.

Отбор (§72.2): только 8 строк со средним запасом к FINAL не меньше 0,25 на позициях
фактической записи FINAL — строки 198, 248046, 2972, 271, 1088, 328, 498, 1510
по измеренному журналу §71.2 (условие 2 не сработало, ряд строк не закрыт).
Четыре строки с запасом меньше 0,25 (12650, 361, 695, 332) в это плечо не входят.

Факторы берутся из измеренного файла runs/analysis/final_logit_rows.json, вручную
не оцениваются. Рецепт: recipes/w77_finalrow_plus_8rows_on_w8.json
База сборки — контрольная точка w8_z_gate_mild_format, читается только на чтение.
"""
import json
import struct
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/variants/w8_z_gate_mild_format/model"
MEAS = ROOT / "runs/analysis/final_logit_rows.json"
FINAL_ROWS = [95429, 64697]
FINAL_TOKENS = {"95429": "FINAL", "64697": " FINAL"}
TENSOR = "model.language_model.embed_tokens.weight"
S_STEP = 0.015
MARGIN_RULE = 0.25  # §72.2: строки со средним запасом не меньше 0,25
BAND = "П72.1"
NAME = "w77_finalrow_plus_8rows_on_w8"
PLANNED_BAND = "формат 329–341, обрывы вместе 8–22, математика вместе 205–218"


def load_base_rows(path: Path):
    """Строки logit-весов из shard базы (bf16, authoritative mirror)."""
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
    """f_i = (1 - s) + s * (u_i / x_i); guard для x_i == 0: f_i = 1 - s."""
    f = torch.empty_like(x)
    nz = x != 0
    f[nz] = (1.0 - s) + s * (u[nz] / x[nz])
    f[~nz] = 1.0 - s
    return f


def mean_L(meas):
    return sum(p["final_logit"] for p in meas["per_position"]) / len(meas["per_position"])


def main():
    meas = json.loads(MEAS.read_text(encoding="utf-8"))
    st, buf = load_base_rows(BASE)
    L = mean_L(meas)
    print(f"база: {st}")

    rows = [r for r in meas["rows"] if r["row"] not in FINAL_ROWS and r["mean_margin"] >= MARGIN_RULE]
    rows.sort(key=lambda r: -r["mean_margin"])
    print(f"отбор §72.2: {len(rows)} строк с запасом >= {MARGIN_RULE}")
    print("строки:", [r["row"] for r in rows])

    ops = []
    wsum = 0.0
    npos = 0
    for r in rows:
        row = r["row"]
        x = buf[row].float()
        best = max(FINAL_ROWS, key=lambda fr: float(torch.dot(x, buf[fr].float())))
        u = buf[best].float()
        n0 = float(torch.linalg.norm(x))
        cos = float(torch.dot(x, u) / (n0 * float(torch.linalg.norm(u))))
        s = S_STEP
        f = factors_for_row(x, u, s)
        y = x * f
        n1 = float(torch.linalg.norm(y))
        kappa = n0 / max(n1, 1e-12)
        new_margin = kappa * (1 - s) * r["mean_margin"] + (kappa - 1) * L
        wsum += r["n_positions"] * new_margin
        npos += r["n_positions"]
        fmax = float(f.abs().max())
        step_pct = float((y - x).abs().max()) / n0 * 100.0
        neff = int(((y.to(torch.bfloat16).float() - x) != 0).sum())
        guard = int((x == 0).sum())
        ops.append({
            "tensors": [TENSOR],
            "op": "tilt",
            "renorm": True,
            "s": s,
            "rows": [row],
            "factors": [float(v) for v in f.tolist()],
            "row_tokens": {str(row): r["token"]},
            "note": (f"строка {row} ({r['token']}), n={r['n_positions']}, "
                     f"средний запас {r['mean_margin']:.3f}, cos к строке {best} "
                     f"({FINAL_TOKENS[str(best)]}) = {cos:.3f}, ожидаемый запас после "
                     f"поворота {new_margin:+.3f}; |f|max={fmax:.3f}, шаг max {step_pct:.2f} % "
                     f"нормы строки, реально изменённых значений после округления до bf16 "
                     f"{neff} из 1024, guard для нулей {guard}"),
        })

    avg_new = wsum / npos
    recipe = {
        "name": NAME,
        "hypothesis": (
            "H53 (волна 27, журнал §72.2): один рычаг — только тот же измеренный рычаг формата "
            "(поворот logit-строк в плоскости (строка, строка FINAL) с восстановлением L2-нормы), "
            "тот же оператор `tilt`, тот же шаг s = +0,015, та же база w8, та же формула факторов. "
            "Меняется только состав набора: 8 строк со средним запасом к FINAL не меньше 0,25 "
            "(198, 248046, 2972, 271, 1088, 328, 498, 1510), четыре строки w50 с запасом меньше "
            "0,25 (12650, 361, 695, 332) исключены. Проверяемая гипотеза: вклад строки в формат "
            "определяется её измеренным запасом, а строки с запасом меньше 0,25 дают только "
            "стоимость по качеству решения. Никакого головного рычага и никаких других тензоров."
        ),
        "expected_effect": (
            f"Средний ожидаемый запас после поворота по 8 строкам (взвешенно по {npos} позициям): "
            f"{avg_new:+.3f} логита против {L:.3f} у FINAL. Сдвиг по-прежнему мал, поэтому "
            f"ожидаемый эффект на 250 строках `search` близок к нулю; решающим каналом остаётся "
            f"состав набора. Полосы П72.1 записаны до сборки: {PLANNED_BAND}. "
            "Решение о качестве принимается только по `dev` (строки отобраны на 63 задачах `search`)."
        ),
        "band": BAND,
        "ops": ops,
    }
    out = ROOT / "recipes" / f"{NAME}.json"
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"рецепт записан: {out}")
    print("операций:", len(ops), "строк:", sum(len(o["rows"]) for o in ops),
          "планируемых изменённых значений:", len(ops) * 1024)


main()
