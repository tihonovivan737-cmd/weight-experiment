#!/usr/bin/env python
"""Генератор рецепта волны 28 (журнал §73.4, полосы П73.1).

Одиночное плечо — только рычаг формата: тот же оператор `tilt` с восстановлением
L2-нормы строки, тот же шаг s = +0,015, та же база w8, та же измеренная формула
факторов (журнал §43):

    f_i = (1 - s) + s * (u_i / x_i)   =>   y_i = x_i * f_i = (1 - s) * x_i + s * u_i

где x — строка конкурента, u — ближайшая строка FINAL по скалярному произведению.

Отбор (§73.4, гипотеза H54): шесть строк со средним запасом к FINAL от 0,25 до 1,0
по измеренному журналу §71.2 — 2972, 271, 1088, 1510, 498, 328. Строки 198 и 248046
(наибольший измеренный запас) и четыре строки с запасом меньше 0,25 (12650, 361, 695,
332) исключены. Это недостающая клетка: сравнение с w50 изолирует вклад четырёх строк
ниже порога, сравнение с w77 изолирует вклад строк 198 и 248046.

Факторы берутся из измеренного файла runs/analysis/final_logit_rows.json, вручную не
оцениваются. Рецепт: recipes/w78_finalrow_six_mid_rows_on_w8.json
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
MARGIN_LO, MARGIN_HI = 0.25, 1.0  # §73.4: полоса запаса для набора
BAND = "П73.1"
NAME = "w78_finalrow_six_mid_rows_on_w8"
PLANNED_BAND = "формат 327–340, обрывы вместе 9–21, математика вместе 209–218"


def load_base_rows(path: Path):
    """Строки logit-весов из shard базы (bfloat16, authoritative mirror)."""
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

    rows = [r for r in meas["rows"] if r["row"] not in FINAL_ROWS
            and MARGIN_LO <= r["mean_margin"] <= MARGIN_HI]
    rows.sort(key=lambda r: -r["mean_margin"])
    print(f"отбор §73.4: {len(rows)} строк с запасом {MARGIN_LO}–{MARGIN_HI}")
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
                     f"нормы строки, реально изменённых значений после округления до bfloat16 "
                     f"{neff} из 1024, guard для нулей {guard}"),
        })

    avg_new = wsum / npos
    recipe = {
        "name": NAME,
        "hypothesis": (
            "H54 (волна 28, журнал §73.4): один рычаг — только тот же измеренный рычаг формата "
            "(поворот logit-строк в плоскости (строка, строка FINAL) с восстановлением нормы "
            "строки), тот же оператор `tilt`, тот же шаг s = +0,015, та же база w8, та же формула "
            "факторов. Меняется только состав набора: шесть строк со средним запасом от 0,25 до 1,0 "
            "(2972, 271, 1088, 1510, 498, 328); строки 198 и 248046 с наибольшим запасом и четыре "
            "строки с запасом меньше 0,25 исключены. Проверяемая гипотеза: стоимость по качеству "
            "решения держится за строками 198 и 248046, а шесть строк со средним запасом дают "
            "извлечение без этой стоимости. Никакого головного рычага и никаких других тензоров."
        ),
        "expected_effect": (
            f"Средний ожидаемый запас после поворота по {len(rows)} строкам (взвешенно по {npos} "
            f"позициям): {avg_new:+.3f} логита против {L:.3f} у FINAL. Ожидание по математике "
            f"вместе — возврат к уровню w8, по извлечению — между w8 и w50. Полосы П73.1 записаны "
            f"до сборки: {PLANNED_BAND}. Решение о качестве принимается только по `dev` "
            f"(строки отобраны на 63 задачах `search`)."
        ),
        "band": BAND,
        "ops": ops,
    }
    out = ROOT / "recipes" / f"{NAME}.json"
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"рецепт: {out}, ops {len(ops)}, планируемых изменённых значений {len(ops) * 1024}")
    print("полоса рецепта:", recipe["band"])
    print("значение `band` в рецепте совпадает с полосой П73.1 из журнала.")


main()
