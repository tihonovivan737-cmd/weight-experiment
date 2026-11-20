#!/usr/bin/env python3
"""Рецепты волны 7. Гипотезы выводятся ТОЛЬКО из замера 250 search + 100 dev
и из послойных зондов (runs/analysis/probe_*.json); официальный test не
использовался. Числовые пороги отбора зафиксированы раньше (scripts/pick_final2.py).

Замеры волны 6 (search 250 + dev 100):
  orig 202 / FINAL 310 / обр 24 ; w8 216 / 332 / 18
  w25 (29 доминирующих z-голов x0.88)  213 / 320 / 27  дельта только-математика +10 к orig, 0 к w8
  w30 (q_norm/k_norm (1+w)*1.08-1)      215 / 337 / 9   дельта только-математика -1 к orig, -2 к w8
  w29 (кумулятив 0.874 по всем головам) 203 / 340 / 10  дельта только-математика -11 / -17
  w24 (гейт полного внимания x0.85)     210 / 332 / 18  дельта только-математика 0 / -5
  w28 (гейт x1.15)                      204 / 332 / 16  дельта только-математика -8 / -8
  w27 (in_proj_b x1.30, 7 слоёв)        209 / 326 / 23
  w26 (att.norm x0.93, амплитуда)       200 / 314 / 28
Вывод: дальше сильнее приглушать ветвь нельзя (w29), амплитудная подмена не работает (w26),
гейт полного внимания целиком в обе стороны не работает (w24/w28). Работают два
независимых рычага: селективное приглушение доминирующих z-голов (математика) и
температура логитов q/k-норм (формат). Волна 7 пробует вернуть доминирующим головам
часть амплитуды, уменьшить дозу qk-рычага и открыть только самые закрытые головы
полного внимания.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = json.loads((ROOT / "runs/analysis/probe_w8_model.json").read_text(encoding="utf-8"))["layers"]
LIN = [l for l in range(24) if l not in (3, 7, 11, 15, 19, 23)]
FULLS = [3, 7, 11, 15, 19, 23]
Z_TENSOR = "model.language_model.layers.*.linear_attn.in_proj_z.weight"
QP = "model.language_model.layers.*.self_attn.q_proj.weight"
QN = "model.language_model.layers.*.self_attn.q_norm.weight"
KN = "model.language_model.layers.*.self_attn.k_norm.weight"
EMB = "model.language_model.embed_tokens.weight"
W8DIR = "runs/variants/w8_z_gate_mild_format/model"
BASE = "models/qwen3.5-0.8b"

# доминирующие головы по зонду w8: |silu(z)| > 0.8 (29 голов в 13 слоях)
Z_HEADS = {l: [h for h, v in enumerate(PROBE[str(l)]["z_gate_head"]) if v > 0.8] for l in LIN}
NHEADS = sum(len(v) for v in Z_HEADS.values())

# самые закрытые головы полного внимания по зонду w8: 4 нижние по sigmoid на слой
CLOSE_HEADS = {l: sorted(range(8), key=lambda h: PROBE[str(l)]["gate_sigmoid_head"][h])[:4] for l in FULLS}
CLOSE_ROWS = [r for l in FULLS for h in CLOSE_HEADS[l] for r in range(h * 512 + 256, h * 512 + 512)]


def z_restore_ops(lam: float):
    ops = []
    for l in LIN:
        heads = Z_HEADS[l]
        if not heads:
            continue
        rows = [r for h in heads for r in range(h * 128, h * 128 + 128)]
        ops.append({"tensors": [Z_TENSOR], "op": "scale", "lam": lam, "layers": [l], "rows": rows,
                    "note": f"слои {l} головы {heads}: |silu(z)|>0.8 в зонде w8, "
                            f"кумулятив 0.93*{lam}={0.93 * lam:.4f}"})
    return ops


def qk_ops(lam: float):
    return [{"tensors": [QN], "op": "scale_offset", "lam": lam, "layers": FULLS,
             "note": "RMSNorm-вес хранит смещённую норму: (1+w)*lam-1"},
            {"tensors": [KN], "op": "scale_offset", "lam": lam, "layers": FULLS,
             "note": "то же для ключей"}]


def gate_open_ops(lam: float):
    return [{"tensors": [QP], "op": "scale", "lam": lam, "rows": CLOSE_ROWS,
             "note": f"открываются 4 самые закрытые головы в каждом из 6 полных слоёв "
                     f"(всего {len(CLOSE_ROWS)} строк по 1024 значения)"}]


RECIPES = [
    {"name": "w31_z_dominant_restore106", "base": W8DIR, "ops": z_restore_ops(1.06),
     "hypothesis": "H23 (возврат доминирующих голов). Зонд w8: 29 из 288 голов линейного "
                   "внимания дают |silu(z)|>0.8 (L10 5 голов, L14 5, L16 4, L13 4, L22 3); "
                   "w25 дополнительно приглушил именно их (кумулятив 0.809) и потерял 12 "
                   "задач формата, сохранив математику на уровне w8. Значит эти головы "
                   "несут и форматную, и вычислительную часть ветви. Возврат их к "
                   "кумулятиву 0.986 (остальные 259 голов остаются на 0.93) должен "
                   "поднять gain_attn поздних слоёв на 2-5 % без отката формата.",
     "expected_effect": "Δ только-математика к w8 >= +3, обрывов <= 20, FINAL >= 325"},
    {"name": "w32_qknorm104_on_w8", "base": W8DIR, "ops": qk_ops(1.04),
     "hypothesis": "H24 (уменьшенная доза qk-рычага). w30 (q_norm/k_norm (1+w)*1.08-1) дал "
                   "лучший формат за всю историю (FINAL 337/350, обрывов 9) и ухудшил "
                   "только-математику на 2. Доза 1.04 (вдвое меньше по логиту) должна "
                   "оставить большую часть выигрыша формата, убрав математическую цену.",
     "expected_effect": "FINAL >= 334, обрывов <= 14, Δ только-математика к w8 >= -1"},
    {"name": "w33_fullgate_open_closed_heads", "base": W8DIR, "ops": gate_open_ops(0.85),
     "hypothesis": "H25 (селективное открытие полного внимания). Зонд w8: sigmoid гейта по "
                   "головам L11 0.118..0.368, L23 0.169..0.411, L3 0.064..0.165 - разброс в "
                   "2-3 раза внутри одного слоя. w24 одинаково приоткрыл все 48 голов "
                   "(дельта только-математики 0) и закрыл их (w28, -8): однородное "
                   "воздействие не работает, разброс по головам игнорировался. Открытие "
                   "только 24 самых закрытых голов (4 на слой) действует ровно там, где "
                   "ветвь сейчас подавлена сильнее всего.",
     "expected_effect": "Δ только-математика к w8 >= +2, FINAL >= 330, обрывов <= 18"},
    {"name": "w34_restore_plus_qknorm104", "base": W8DIR, "ops": z_restore_ops(1.06) + qk_ops(1.04),
     "hypothesis": "H26 (комбинация двух независимых рычагов). Математический рычаг "
                   "(возврат 29 доминирующих z-голов) и форматный рычаг (температура "
                   "q/k-норм) действуют на разные тензоры и разные оси провала w8 "
                   "(85 wrong_with_final и 13 обрывов в search, 32 и 4 в dev). Вместе они "
                   "должны закрыть обе оси сразу - это единственный измеренный путь к "
                   "выполнению всех четырёх условий отбора.",
     "expected_effect": "C >= 218 из 350, Δ только-математика >= +2 к обеим точкам, "
                        "обрывов <= 18"},
]

for r in RECIPES:
    name = r.pop("base")
    doc = {"name": r["name"], "hypothesis": r["hypothesis"], "expected_effect": r["expected_effect"],
           "base_model_dir": name, "ops": r["ops"],
           "selection_note": "выбрано по замерам 350 отборочных задач и послойным зондам; "
                             "official test (1319) для выбора не использовался"}
    p = ROOT / "recipes" / f"{r['name']}.json"
    assert not p.exists(), f"{p} уже существует"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{p.relative_to(ROOT)}  ops={len(doc['ops'])}")

print("\nдоминирующие z-головы (всего %d):" % NHEADS, json.dumps({k: v for k, v in Z_HEADS.items() if v}))
print("самые закрытые головы полного внимания:", json.dumps(CLOSE_HEADS))
print("строк гейта в w33:", len(CLOSE_ROWS), "= значений", len(CLOSE_ROWS) * 1024)
