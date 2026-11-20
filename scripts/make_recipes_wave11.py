#!/usr/bin/env python3.13
"""Генератор рецептов волны 11 (§38, предрегистрировано до первых сборок).

Два новых механизма, оба послойные и оба не повторают закрытые оси:

A. Селективность по головам внутри `in_proj_z` [2048, 1024] (16 голов по 128 строк).
   Заранее (до прогонов) по сохранённому зонду `runs/analysis/z_gate_heads_w8.json`
   для каждого из 18 LIN-слоёв берутся 4 самые «громкие» головы (максимальное среднее
   |silu(z)|) и 4 самые «тихие» (минимальное). Дозировка сбалансирована точно:
   4 головы ×0,94 и 12 голов ×1,0208, где 1,0208 = 0,94^(-1/3), поэтому
   4·ln(0,94) + 12·ln(1,0208) = 0 — средний лог-усилитель каждого слоя и всего
   тензора не меняется относительно базы w8, меняется только его распределение.
   Относительно исходной модели: демпфированные головы 0,93·0,94 = 0,8742,
   поднятые 0,93·1,0208 = 0,9494 — обе величины внутри измеренной благополучной
   зоны 0,88–0,95 (§30, §35).
     w45_z_headsel_flatten — демпфируются 4 самые громкие головы каждого слоя
       (усиливаются 12 остальных): распределение выравнивается;
     w46_z_headsel_sharpen — демпфируются 4 самые тихие головы (усиливаются
       4 самых громких + 8 средних): распределение заостряется.
   multiset факторов одинаков, различается только назначение голов.

B. Наклон окна `linear_attn.conv1d.weight` [6144, 1, 4] (глубинное, 3 левых нуля,
   отсчёт тапов: индекс 3 = текущий токен, индекс 0 = токен 3 шага назад —
   `causal_conv1d_fn` с padding=3 и срезом [:seq_len], `causal_conv1d_update`
   через conv_state; оба пути дают out[t] = Σ_k w[k]·x[t+k-3]).
     w47_conv_tap_recency — факторы [0,90, 0,95, 1,05, 1,10]: ближние тапы усиливаются;
     w48_conv_tap_context — зеркальные [1,10, 1,05, 0,95, 0,90]: дальние тапы.
   `renorm: true` — каждая строка окна после наклона возвращает свою исходную
   норму L2 (в bf16 — точно), поэтому величина сигнала на выходе свёртки не
   меняется, меняется только форма окна.

Дозы выбирались так, чтобы шаг был минимально различимым в bf16 (шаг 2^-8 ≈ 0,39 %):
все использованные факторы отличаются от 1 больше чем на один bf16-шаг.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "recipes"
ANALYSIS = ROOT / "runs" / "analysis"

HEAD_DIM = 128
N_HEADS = 16
HIDDEN = 1024
DAMP = 0.94
RELEASE = 0.94 ** (-1.0 / 3.0)     # = 1,020787…; в рецепт кладётся 1,0208
RELEASE_R = round(RELEASE, 4)
TAP_RECENCY = [0.90, 0.95, 1.05, 1.10]
TAP_CONTEXT = [1.10, 1.05, 0.95, 0.90]
LIN = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
Z_TENSOR = "model.language_model.layers.*.linear_attn.in_proj_z.weight"
CONV_TENSOR = "model.language_model.layers.*.linear_attn.conv1d.weight"


def load_head_means():
    d = json.loads((ANALYSIS / "z_gate_heads_w8.json").read_text(encoding="utf-8"))
    hm = d["z_gate_head"]
    assert sorted(int(k) for k in hm) == LIN, sorted(int(k) for k in hm)
    for L, v in hm.items():
        assert len(v) == N_HEADS and all(isinstance(x, float) for x in v), (L, v)
    return {int(k): v for k, v in hm.items()}


def head_sets(layer_heads):
    """(loud, quiet) — по 4 головы на слой, ранжирование по среднему |silu(z)|."""
    loud, quiet = {}, {}
    lines = []
    for L in LIN:
        means = layer_heads[L]
        order = sorted(range(N_HEADS), key=lambda h: -means[h])
        loud[L] = sorted(order[:4])
        quiet[L] = sorted(order[-4:])
        spread = means[max(range(N_HEADS), key=lambda h: means[h])] / \
                 means[min(range(N_HEADS), key=lambda h: means[h])]
        lines.append(f"слой {L:>2}: громкие {loud[L]} "
                     f"({', '.join(f'{means[h]:.4f}' for h in loud[L])}) | "
                     f"тихие {quiet[L]} "
                     f"({', '.join(f'{means[h]:.4f}' for h in quiet[L])}) | "
                     f"разброс {spread:.2f}×")
    return loud, quiet, lines


def head_ops(heads, lam, lam_rest):
    """По две операции scale на слой: явные диапазоны строк выбранных голов."""
    ops = []
    for L in LIN:
        rows_sel = sorted(r for h in heads[L] for r in range(h * HEAD_DIM, (h + 1) * HEAD_DIM))
        rows_rest = sorted(r for h in range(N_HEADS) if h not in heads[L]
                           for r in range(h * HEAD_DIM, (h + 1) * HEAD_DIM))
        assert len(rows_sel) == 4 * HEAD_DIM and len(rows_rest) == 12 * HEAD_DIM
        ops.append({"tensors": [Z_TENSOR], "op": "scale", "lam": lam,
                    "layers": [L], "rows": rows_sel,
                    "note": f"L{L}: 4 выбраные головы по 128 строк"})
        ops.append({"tensors": [Z_TENSOR], "op": "scale", "lam": lam_rest,
                    "layers": [L], "rows": rows_rest,
                    "note": f"L{L}: 12 остальных голов"})
    return ops


RECIPES_FILES = {
    "w45_z_headsel_flatten": {
        "hypothesis": "Волна 11 / рычаг A1 (§38). В базе w8 все головы in_proj_z "
                      "умножены на 0,93 равномерно, но средний |silu(z)| различается "
                      "между головами одного слоя в 2,2–8,5 раза. Демпфирование 4 самых "
                      "громких голов каждого LIN-слоя до 0,94 с одновременным подъёмом "
                      "12 остальных до 1,0208 сохраняет средний лог-усилитель слоя ровно "
                      "на уровне w8 (4·ln0,94 + 12·ln1,0208 = 0), то есть проверяет "
                      "распределение сигнала изолированно от дозы. Ожидание: если "
                      "вклад w8 дают именно громкие головы, их подавление ухудшит "
                      "математику; если вклад даёт разнообразие голов, выравнивание "
                      "добавит 2–5 верных ответов.",
        "expected_effect": "C (350) ≥ 218 при T ≤ 18, Δмат ≥ +2 против оригинала и "
                           "против w8. Побочный ожидаемый признак: средний |silu(z)| "
                           "по головам слоя станет ровнее (меньше CV).",
    },
    "w46_z_headsel_sharpen": {
        "hypothesis": "Волна 11 / рычаг A2 (§38). Тот же multiset факторов, что у w45 "
                      "(4 × 0,94 и 12 × 1,0208, средний лог-усилитель слоя не меняется), "
                      "но демпфируются 4 самые тихие головы каждого LIN-слоя, а 4 самые "
                      "громкие получают 1,0208. Проверяется обратная гипотеза: выигрыш "
                      "w8 обеспечивается доминированием громких голов, и его стоит "
                      "усилить, а не сгладить.",
        "expected_effect": "C (350) ≥ 218 при T ≤ 18, Δмат ≥ +2 против оригинала и "
                           "против w8. Побочный ожидаемый признак: CV среднего |silu(z)| "
                           "по головам слоя вырастет.",
    },
    "w47_conv_tap_recency": {
        "hypothesis": "Волна 11 / рычаг B1 (§38). Окно depthwise-свёртки qkv (ядро 4, "
                      "3 левых нуля) усредняет текущий токен (тап 3) и три предыдущих "
                      "(тапы 0–2) без выраженного предпочтения. Наклон [0,90, 0,95, 1,05, "
                      "1,10] с ренормировкой каждой строки к исходной норме L2 усиливает "
                      "ближние тапы и ослабляет дальние, не меняя ни величины сигнала, "
                      "ни дозы других рычагов. Ожидание: более «локальный» q/k/v должен "
                      "снизить долю обрывов и сократить рассуждения; риск — потеря "
                      "учёта контекста на многословных задачах (падение C).",
        "expected_effect": "Рост доли FINAL и снижение T относительно w8 (T ≤ 18 при "
                           "C ≥ 218); Δмат ≥ +2 против оригинала и против w8.",
    },
    "w48_conv_tap_context": {
        "hypothesis": "Волна 11 / рычаг B2 (§38). Зеркальный наклон [1,10, 1,05, 0,95, "
                      "0,90] с той же ренормировкой норм L2: усиливаются дальние тапы, "
                      "ослабляется текущий токен, то есть окно становится более "
                      "сглаживающим (шире эффективное окно). Проверяет обратную гипотезу "
                      "к B1 и одновременно служит контролем порядка факторов: если оба "
                      "наклона одинаково нейтральны, механизм окна в этой модели не "
                      "задействован.",
        "expected_effect": "Рост доли FINAL и снижение T относительно w8 (T ≤ 18 при "
                           "C ≥ 218); Δмат ≥ +2 против оригинала и против w8.",
    },
}


def main():
    layer_heads = load_head_means()
    loud, quiet, hlines = head_sets(layer_heads)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "wave11_head_sets.txt").write_text(
        "Наборы голов in_proj_z для волны 11 (по runs/analysis/z_gate_heads_w8.json, "
        "среднее |silu(z)| на голову, база w8, greedy dev100).\n"
        "Меньший номер головы = меньший индекс; демпфируются по 4 головы на слой.\n"
        + "\n".join(hlines) + "\n", encoding="utf-8")

    built = []
    for name, meta in RECIPES_FILES.items():
        if name == "w45_z_headsel_flatten":
            ops = head_ops(loud, DAMP, RELEASE_R)
            exp_vals = 16 * HEAD_DIM * HIDDEN * len(LIN)   # значения: вся in_proj_z LIN-слоёв
        elif name == "w46_z_headsel_sharpen":
            ops = head_ops(quiet, DAMP, RELEASE_R)
            exp_vals = 16 * HEAD_DIM * HIDDEN * len(LIN)
        elif name == "w47_conv_tap_recency":
            ops = [{"tensors": [CONV_TENSOR], "op": "tilt", "factors": TAP_RECENCY,
                    "renorm": True, "layers": LIN,
                    "note": "наклон окна по тапам 0..3 (0 = 3 токена назад, 3 = текущий), "
                            "ренормировка нормы L2 каждой строки"}]
            exp_vals = 6144 * 4 * len(LIN)
        else:
            ops = [{"tensors": [CONV_TENSOR], "op": "tilt", "factors": TAP_CONTEXT,
                    "renorm": True, "layers": LIN,
                    "note": "зеркальный наклон окна по тапам 0..3, "
                            "ренормировка нормы L2 каждой строки"}]
            exp_vals = 6144 * 4 * len(LIN)

        recipe = {"name": name, "hypothesis": meta["hypothesis"],
                  "expected_effect": meta["expected_effect"], "ops": ops}
        p = RECIPES / f"{name}.json"
        p.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        built.append((name, len(ops), exp_vals))

    import math
    bal = 4 * math.log(DAMP) + 12 * math.log(RELEASE_R)
    print("записано рецептов:", len(built))
    for name, n_ops, vals in built:
        print(f"  {name:<26} ops={n_ops:<4} ожидаемые изменённые значения={vals:,}".replace(",", " "))
    print(f"  баланс A: 4·ln({DAMP}) + 12·ln({RELEASE_R}) = {bal:+.3e}  "
          f"→ средний лог-усилитель слоя не изменён относительно w8")
    print(f"  A: демпф. головы 0,93·{DAMP} = {0.93 * DAMP:.4f} от исходного; "
          f"поднятые головы 0,93·{RELEASE_R} = {0.93 * RELEASE_R:.4f} от исходного")
    print(f"  B: сумма ln факторов окна = {sum(math.log(f) for f in TAP_RECENCY):+.6f}, "
          f"произведение = {TAP_RECENCY[0] * TAP_RECENCY[1] * TAP_RECENCY[2] * TAP_RECENCY[3]:.6f}; "
          f"норма L2 каждой строки восстановлена (юнит-тест dev = 0)")
    touched_rows = N_HEADS * HEAD_DIM * len(LIN)
    print(f"  A: строк in_proj_z в LIN-слоях {touched_rows} "
          f"(16 голов × 128 строк × 18 слоёв), из них 4×128×18 = {4 * HEAD_DIM * len(LIN)} "
          f"демпфируются и 12×128×18 = {12 * HEAD_DIM * len(LIN)} поднимаются")
    print(f"  B: изменённых значений 18 × 6144 × 4 = {18 * 6144 * 4} (меньший след серии)")


if __name__ == "__main__":
    main()
