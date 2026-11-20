"""
Рецепты волны 6b: только направленные изменения, основанные на activation-зонде
(runs/analysis/probe_base_*.json и probe_w8_model.json, 63 задачи из search-набора)
и на уже измеренных эффектах волн 1-5. Никаких результатов test1319 здесь нет.

Общий вывод зонда (base vs w8, 63 задач, среднее и хвост генерации):
  * silu(z) (вентиль Gated DeltaNet) снижен на 3-6 % во всех 18 линейных слоях;
  * gainAttn линейных слоёв упал сильнее всего в поздних слоях 16..22 (-5..-7 %),
    а gainMLP вырос на +1..+6 % -> w8 перебалансирует блок в пользу MLP;
  * гейт output-гейта полных self_attn слоёв ОТРИЦАТЕЛЬНЫЙ (логит -1.19..-2.68,
    sigmoid 0.11..0.26), и у w8 он стал ещё закрытее на 2-8 %: scale lam>1
    над строками гейта УСИЛИВАЕТ закрытие, lam<1 — открывает гейт;
  * beta по 16 головам одного слоя разнородна: 0.04..0.95 (L9: h7=0.04, h11=0.68);
    |silu(z)| по головам: 0.12..1.8, в L10/L14 по 5 голов >0.8, в 10 слоях ни одной.
"""
import json
from pathlib import Path

ROOT = Path("/home/mpiscil/weight-experiment")
REC = ROOT / "recipes"
PROBE = json.loads((ROOT / "runs/analysis/probe_w8_model.json").read_text(encoding="utf-8"))["layers"]
LINS = [int(k) for k in sorted(PROBE, key=int) if PROBE[k]["type"] == "linear_attention"]
FULLS = [int(k) for k in sorted(PROBE, key=int) if PROBE[k]["type"] != "linear_attention"]
LATE = [l for l in LINS if l >= 12]
Z_TENSOR = "model.language_model.layers.*.linear_attn.in_proj_z.weight"
B_TENSOR = "model.language_model.layers.*.linear_attn.in_proj_b.weight"
INLN = "model.language_model.layers.*.input_layernorm.weight"
QP = "model.language_model.layers.*.self_attn.q_proj.weight"

recipes = []

# --- w24: открыть output-гейт полных self_attn слоёв (единственный незакрытый
#     механизм attention, не тронутый ни одной предыдущей волной).
#     Логит гейта отрицательен (по головам -3.02..-1.52), поэтому lam=0.85
#     делает логит менее отрицательным: sigmoid 0.113 -> ~0.14, 0.267 -> ~0.31,
#     т.е. вклад softmax-attention растёт на 25-40 %. Строки гейта — вторые
#     половины каждых 512 строк q_proj (query/gate чередуются по головам).
gate_rows = [r for h in range(8) for r in range(h * 512 + 256, h * 512 + 512)]
recipes.append({
    "name": "w24_fullgate_open085",
    "hypothesis": "Output-гейт Qwen3_5Attention (attn_output * sigmoid(gate), "
                  "modeling_qwen3_5.py:805) в измеренных слоях закрыт: средний логит "
                  "-1.19..-2.68, sigmoid 0.113..0.267. У w8 он закрыт ещё сильнее (-2..-8 %) "
                  "как побочный эффект. Логиты отрицательны, значит scale lam<1 над 2048 "
                  "строками гейта q_proj (2-я половина каждых 512 строк, по 256 на голову) "
                  "приближает логит к нулю и ОТКРЫВАЕТ гейт: sigmoid вырастет на 25-40 %, "
                  "вклад softmax-attention в остаточный поток увеличится. Тест на то, "
                  "усиливает ли более сильный глобальный поиск по контексту качество "
                  "многошаговых арифметических рассуждений.",
    "expected_effect": "Рост rms_attn в слоях 3,7,11,15,19,23 на 20-35 %; ответы могут стать "
                       "длиннее. Направленный тест против готового w22_fullgate_amp115 "
                       "(lam=1.15 — дополнительное закрытие того же гейта).",
    "ops": [
        {"tensors": [QP], "op": "scale", "lam": 0.85, "layers": FULLS, "rows": gate_rows,
         "note": "строки гейта (256 на голову), lam<1 при отрицательных логитах = открыть гейт"},
    ],
})

# --- w25: head-selective z damp — concentrация единственного воспроизведённого рычага
#     (in_proj_z x0.93, поиск +9) на тех value-головах, которые реально несут амплитуду.
z_ops, z_heads = [], {}
for l in LINS:
    heads = [h for h in range(16) if PROBE[str(l)]["z_gate_head"][h] > 0.8]
    if not heads:
        continue
    z_heads[l] = heads
    rows = [r for h in heads for r in range(h * 128, h * 128 + 128)]
    z_ops.append({"tensors": [Z_TENSOR], "op": "scale", "lam": 0.88, "layers": [l], "rows": rows,
                  "note": f"голова {heads}: |silu(z)|>0.8 в зонде w8"})
recipes.append({
    "name": "w25_z_headsel_damp088",
    "hypothesis": f"Зонд w8 по {len(LINS)} линейным слоям (16 value-голов, 128 строк на голову): "
                  f"среднее |silu(z)| по голове 0.12..1.81. Амплитуда сосредоточена в немногих "
                  f"головах: L10 [0,2,4,6,7], L13 [2,7,9,14], L14 [4,5,6,14,15], L16 [6,7,8,11], "
                  f"L22 [0,7,12]; в 8 слоях (0,1,9,17,18) ни одна голова не превышает 0.8. "
                  "Сильно приглушать все 288 голов одним множителем (w8: 0.93) — грубо; "
                  "приглушение x0.88 только 28 доминирующих голов (128 строк на голову) даёт "
                  "схожий по величине суммарный эффект (топ-голова silu~1.2 -> -12 %) приfootprint "
                  "в 10 раз меньшем и без подавления 260 уже слабых голов.",
    "expected_effect": "Локальное снижение rms_attn в слоях 10,13,14,16,22 при неизменных "
                       "остальных слоях; проверка, что выигрыш w8 сосредоточен в доминирующих "
                       "головах, а не в равномерном усилении всего стека.",
    "ops": z_ops,
})

# --- w26: механизменная абляция на ИСХОДНОЙ модели: тот же масштаб ветви, но без
#     изменения формы вентиля (norm.weight умножается ПОСЛЕ RMS-нормы и silu-гейта).
recipes.append({
    "name": "w26_norm_amp093",
    "hypothesis": "Qwen3_5RMSNormGated (modeling_qwen3_5.py:204) считает RMS-норму по 128 "
                  "каналам, умножает на weight[128] (общий для всех 16 value-голов) и только "
                  "затем на silu(z). Поэтому scale in_proj_z (w8, lam 0.93) меняет ФОРМУ "
                  "вентиля (силуоидную точку) и величину, а scale linear_attn.norm.weight "
                  "меняет ТОЛЬКО величину whole linear-ветви на те же -7 %. Если w26 повторит "
                  "прирост w8, механизм выигрыша — амплитуда linear-ветви (перебалансировка "
                  "attn/MLP, измеренная как gainMLP +1..+6 %), а не форма вентиля. Если нет — "
                  "важна именно нелинейность silu(gate). Это ключевой развилочный эксперимент; "
                  "строится от ИСХОДНОЙ модели, чтобы рычаг был один.",
    "expected_effect": "Чистое снижение rms_attn линейных слоёв ~7 % без изменения beta/dt/z; "
                       "сравнение с w17_z93_only (search +8, dev -1).",
    "ops": [
        {"tensors": ["model.language_model.layers.*.linear_attn.norm.weight"],
         "op": "scale", "lam": 0.93, "layers": LINS,
         "note": "F32-вес RMSNormGated [128], общий для 16 голов"},
    ],
})

# --- w27: температура решений beta только в поздних линейных слоях.
#     scale in_proj_b lam>1 делает логит beta более крутым: головы с beta>0.5
#     уходят к 1 (запись состояния каждый шаг), головы с beta<0.5 — к 0 (долгая память).
#     В поздних слоях измерен самый большой разброс: L22 0.04..0.95, L20 0.22..0.99.
recipes.append({
    "name": "w27_beta_sharpen_late130",
    "hypothesis": "beta = sigmoid(in_proj_b(h)) задаёт скорость записи состояния Gated DeltaNet. "
                  "Зонд w8: в поздних линейных слоях разброс по головам максимален "
                  "(L22: 0.04..0.95, L20: 0.22..0.99, L18: 0.22..0.91), в ранних минимален "
                  "(L0 0.49 ср., L1 0.44). Глобальное усиление beta на всех слоях (v4_beta_amplify, "
                  "in_proj_b x1.5) дало dev 0, смягчение (v5 x0.6) +1 — то есть эффект есть, "
                  "но он размазан по всем 18 слоям. Усиление x1.30 только по 6 поздним слоям "
                  "12,13,14,16,17,18,20,21,22 (по факту слоям >=12) проверяет, работает ли "
                  "бинаризация 'перезаписать состояние / сохранить память' именно там, где "
                  "модель держит итоговое значение ответа.",
    "expected_effect": "beta в поздних слоях уходит к краям (0.04->0.02, 0.95->0.98); "
                       "никаких изменений в z-вентиле, гейтах и MLP.",
    "ops": [
        {"tensors": [B_TENSOR], "op": "scale", "lam": 1.30, "layers": LATE},
    ],
})

REC.mkdir(parents=True, exist_ok=True)
for r in recipes:
    p = REC / f"{r['name']}.json"
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{p.name:<28} ops={len(r['ops'])}")
print("слои LIN:", LINS)
print("слои FULL:", FULLS, " строки гейта:", gate_rows[:6], "...", len(gate_rows))
print("z-головы (|silu|>0.8):", json.dumps(z_heads))
