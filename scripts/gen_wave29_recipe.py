#!/usr/bin/env python
"""Генератор рецепта волны 29 (журнал §77.5–§77.7, полосы П74.1).

Одиночное изменение — только шаг того же измеренного рычага формата: тот же оператор
`tilt` с восстановлением L2-нормы строки, та же база w8, тот же тензор, та же формула
факторов (журнал §43):

    f_i = (1 - s) + s * (u_i / x_i)   =>   y_i = x_i * f_i = (1 - s) * x_i + s * u_i

Набор строк — ровно десять строк w50 со средним запасом меньше 1,0 к строке FINAL
по измеренному файлу runs/analysis/final_logit_rows.json: 2972, 271, 1088, 1510, 498,
328, 12650, 361, 695, 332. Строки 198 и 248046 (запас выше 1,0) не входят.

Проверяемое утверждение (§77.5): канал качества держится за четырьмя строками с малым
запасом (12650, 361, 695, 332), а не за контрольными токенами; при удвоении шага
(плюс 0,030 против плюс 0,015) ожидается рост извлечения FINAL при сохранении качества.

Рецепт: recipes/w79_finalrow_ten_rows_double_step_on_w8.json
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
S_STEP = 0.030  # §77.7: единственное изменение — шаг
BAND = "П74.1"
NAME = "w79_finalrow_ten_rows_double_step_on_w8"
PLANNED_BAND = ("формат вместе 320–348, обрывы вместе 9–23, математика вместе 206–220; "
                "dev 59–66 / 90–98 / 0–6; search 147–157 / 228–244 / 9–17")


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

    rows = [r for r in meas["rows"] if r["row"] not in FINAL_ROWS and r["mean_margin"] < 1.0]
    rows.sort(key=lambda r: -r["mean_margin"])
    print(f"отбор §43.2 (тот же набор w50): {len(rows)} строк с запасом меньше 1,0")
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
                     f"поворота {new_margin:+.3f}; шаг удвоен до {s:.3f}; |f|max={fmax:.3f}, "
                     f"шаг max {step_pct:.2f} % нормы строки, реально изменённых значений "
                     f"после округления до bfloat16 {neff} из 1024, guard для нулей {guard}"),
        })

    avg_new = wsum / npos
    recipe = {
        "name": NAME,
        "hypothesis": (
            "H55 (волна 29, журнал §77.5–§77.7): один рычаг — тот же измеренный рычаг формата "
            "(поворот logit-строк в плоскости (строка, строка FINAL) с восстановлением нормы "
            "строки), тот же оператор `tilt`, та же база w8, тот же тензор, та же формула "
            "факторов, тот же набор из десяти строк w50 (2972, 271, 1088, 1510, 498, 328, 12650, "
            "361, 695, 332). Меняется только шаг: он равен плюс 0,030 вместо плюс 0,015. Строки "
            "198 и 248046 по-прежнему исключены. Проверяемая гипотеза: канал качества держится за "
            "четырьмя строками с малым запасом (12650, 361, 695, 332), поэтому удвоение шага "
            "поднимает извлечение FINAL выше w50, не разрушая качество; сравнение с w50 остаётся "
            "одно переменным по шагу. Никакого головного рычага и никаких других тензоров."
        ),
        "expected_effect": (
            f"Средний ожидаемый запас после поворота по {len(rows)} строкам (взвешенно по {npos} "
            f"позициям): {avg_new:+.3f} логита против {L:.3f} у FINAL — вдвое выше, чем при шаге "
            f"плюс 0,015 (по w50 это было плюс 0,720). Ожидание по извлечению FINAL вместе "
            f"320–348, по математике вместе 206–220, по обрывам вместе 9–23. Полосы П74.1 записаны "
            f"до сборки (§77.7): {PLANNED_BAND}. Решение о качестве принимается на dev и search; "
            f"официальный набор из 1319 задач используется только для оценки."
        ),
        "band": BAND,
        "ops": ops,
    }
    out = ROOT / "recipes" / f"{NAME}.json"
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"рецепт: {out}, ops {len(ops)}, планируемых изменённых значений {len(ops) * 1024}")
    print("полоса рецепта:", recipe["band"])
    print("значение `band` в рецепте совпадает с полосой П74.1 из журнала §77.7.")


main()
