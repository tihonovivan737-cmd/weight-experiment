#!/usr/bin/env python
"""Генератор рецептов волны 13 (§43, §44): поворот строк logit-весов в плоскости
(строка, строка FINAL) с сохранением L2-нормы строки.

Только `tilt` + `renorm`. `tilt` реализуется мультипликативно:
    f_i = (1 - s) + s * (u_i / x_i)   =>   y_i = x_i * f_i = (1 - s) * x_i + s * u_i
где x — строка конкурента, u — ближайшая строка FINAL (по косинусу), s — шаг поворота.
s > 0 — поворот к FINAL (пререгистрация §43.1), s < 0 — тот же механизм с обратным знаком:
измерение §44.2 показало, что при s > 0 запас конкурента к FINAL растёт, вниз конкурента
двигает обратный знак.

Факторы задаются на уровне операции, поэтому на каждую строку — отдельная операция.
Списки rows — только уникальные (§42.6). Мультипликативный tilt не может переместить
нулевой элемент строки: для него f_i = 1 - s (строк с нулями в отборе нет, проверка сохраняется).

Рецепты:
  w50_finalrow_rotation15_smallmargin           s = +0.015 (пререгистрированный знак)
  w51_finalrow_rotation15_smallmargin_negsign   s = -0.015 (знак, исправленный измерением)
База сборки — контрольная точка w8_z_gate_mild_format (§43.1), читается только на чтение.
"""
import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/variants/w8_z_gate_mild_format/model"
MEAS = ROOT / "runs/analysis/final_logit_rows.json"
FINAL_ROWS = [95429, 64697]
FINAL_TOKENS = {"95429": "FINAL", "64697": " FINAL"}
TENSOR = "model.language_model.embed_tokens.weight"
S_STEP = 0.015
MARGIN_RULE = 1.0  # §43.2: конкуренты с запасом < 1 на позициях фактической записи FINAL


def load_base_rows(path: Path):
    """Строки embed_tokens из shard базы (bf16, authoritative mirror)."""
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

    rows = [r for r in meas["rows"]
            if r["row"] not in FINAL_ROWS and r["mean_margin"] < MARGIN_RULE]
    rows.sort(key=lambda r: -r["n_positions"])
    print(f"отбор §43.2: {len(rows)} строк с запасом < {MARGIN_RULE}")

    arms = [(+S_STEP, "w50_finalrow_rotation15_smallmargin"),
            (-S_STEP, "w51_finalrow_rotation15_smallmargin_negsign")]

    for s, name in arms:
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
            "name": name,
            "hypothesis": (
                "H43 (волна 13): один рычаг — поворот logit-строк в плоскости (строка, FINAL) "
                "шагом 1,5 % нормы строки с восстановлением L2-нормы, только на строках с запасом "
                "менее 1 к FINAL на позициях фактической записи FINAL (10 строк, §43.2). "
                f"Знак s = {s:+.3f}: "
                + ("пререгистрированный (поворот к FINAL)" if s > 0 else
                   "исправленный измерением §44.2 (тот же механизм в обратную сторону): "
                   "при s > 0 запас конкурента растёт, вниз конкурента двигает обратный знак")
                + ". Масштаб каналов не трогается, второй рычаг на той же цепи отсутствует (H39). "
                "Оценка по коридорам §43.3: формат 336-342, обрывы <= 14, математика 216-221."
            ),
            "expected_effect": (
                f"Средний ожидаемый запас после поворота по 350-позиционной выборке: "
                f"{avg_new:+.3f} логита против {L:.3f} у FINAL (взвешенно по числу позиций); "
                "сдвиг меньше большинства измеренных запасов, поэтому ожидаемый эффект на 350 "
                "строках близок к нулю — решающим каналом остаётся сокращение обрывов (§43.4). "
                "Гейт pick_final2 (Δмат >= +2 против обоих референсов) заранее оговорён "
                "как маловероятный (§43.4)."
            ),
            "ops": ops,
        }
        out = ROOT / "recipes" / f"{name}.json"
        out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"записан {out.relative_to(ROOT)} ({len(ops)} операций, "
              f"{sum(len(o['factors']) for o in ops)} факторов, ожидаемый средний запас "
              f"{avg_new:+.3f})")


if __name__ == "__main__":
    sys.exit(main())
