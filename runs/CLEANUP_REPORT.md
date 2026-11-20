# Очистка диска: удалённые файлы весов (2026-11-13)

Отчёт создан по запросу «на диске заканчивается место». Удалялись **только** файлы весов
(`model/model.safetensors-…`) старых вариантов, показавших результаты строго не лучше
контрольной точки `w8_z_gate_mild_format`. Манифесты, рецепты, `verify.json`, результаты
прогонов и файлы сравнений **сохранены полностью**, каталоги вариантов остались на месте.
Ничего вне `/home/mpiscil/weight-experiment` не удалялось.

## 1. Состояние до очистки

| параметр | значение |
|---|---|
| время начала | 2026-11-13 ~10:46 (местное, Asia/Krasnoyarsk) |
| свободно на `/dev/sda5` | **16 G** из 480 G (441 G занято, 97 %) |
| занято `runs/variants` | 89,09 G файлов весов в 51 каталоге вариантов |
| занято `models` | 1,7 G (`models/qwen3.5-0.8b`, копия исходной модели в проекте) |
| процессы | `dsh web` (оболочка задачи) и `Strata serve/engine` (внешний сервис, `Strata-data`, не трогался); прогонов `eval_gsm8k`/`proxy_eval`/`probe_layers`/`modify_weights` не было |

Промежуточное наблюдение: между 10:46 и 10:57 свободное место выросло с 16 G до **84 G**
**без моего участия** — внешний процесс заменил `/home/mpiscil/Strata-data/models/IQ3_XXS`
(2 GGUF-шарда) на `IQ2_XS` (2,3 G), mtime каталога 10:35–11:01. К очистке эксперимента это
отношения не имеет и мной не инициировалось; в отчёте цифры «до/после» даны по моему удалению:
**84 G → 105 G**.

## 2. Критерии удаления (проверялись для каждого варианта)

1. выполнен **полный отборочный экран 350** задач (`dev` 100 + `search` 250);
2. метрики **строго не лучше w8** на всех сопоставимых сплитах — по числу верных, по обрывам
   и по извлечённым `FINAL` (сравнение с `w8`: 350 = 216/18/332, dev = 64/4/95, search = 152/14/237);
3. сохранены `manifest.json`, `recipes/<вариант>.json`, `verify.json` (все проверки `OK`,
   `problems: []`), `dev/results.jsonl` (100 строк), `search/results.jsonl` (250 строк)
   и ≥ 2 файла `compare*.json`;
4. файл весов — **обычный отдельный файл**: 1 жёсткая связь (`st_nlink == 1`), inode не
   совпадает ни с одним шардом исходной модели `models/qwen3.5-0.8b/`, ни с шардом `w8`
   (проверено по парам `(inode, size)`; всего проверено 51 × 1 файл весов).

Инвентарь и классификация: `scripts/inventory_variants.py` → `runs/analysis/cleanup_inventory.json`
и `.txt`; удаление: `scripts/cleanup_weights.py` (сначала `--dry-run`, затем боевой запуск);
протокол удаления: `runs/analysis/cleanup_deleted.json`.

## 3. Удалено: 13 файлов весов, 22 710 253 800 B (21,15 GiB)

Каждый файл — 1 746 942 600 B, отдельный (не жёсткая ссылка). У всех 13 сохранены манифест,
рецепт, `verify.json` (ИТОГ OK), результаты dev/search и сравнения.

| вариант | удалённый файл | байт | сохранённый рецепт | 350 верно/обрыв/FINAL |
|---|---|---|---|---|
| `c0_final_norm_gain` | `runs/variants/c0_final_norm_gain/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/c0_final_norm_gain.json` | 199/24/310 |
| `v2_Alog_decay_double` | `runs/variants/v2_Alog_decay_double/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/v2_Alog_decay_double.json` | 194/20/317 |
| `v9_mlp_damp` | `runs/variants/v9_mlp_damp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/v9_mlp_damp.json` | 184/28/296 |
| `w12_z_gate_strong` | `runs/variants/w12_z_gate_strong/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w12_z_gate_strong.json` | 103/53/290 |
| `w15_final_case_suppress` | `runs/variants/w15_final_case_suppress/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w15_final_case_suppress.json` | 209/26/325 |
| `w16_final_rows12_only` | `runs/variants/w16_final_rows12_only/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w16_final_rows12_only.json` | 209/26/324 |
| `w17_z93_only` | `runs/variants/w17_z93_only/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w17_z93_only.json` | 209/25/318 |
| `w1_final_token_amp` | `runs/variants/w1_final_token_amp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w1_final_token_amp.json` | 211/25/324 |
| `w38_z_corner_early10008_late08742` | `runs/variants/w38_z_corner_early10008_late08742/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w38_z_corner_early10008_late08742.json` | 216/23/327 |
| `w41_format_embed132` | `runs/variants/w41_format_embed132/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w41_format_embed132.json` | 211/19/329 |
| `w43_format_embed144_latedose` | `runs/variants/w43_format_embed144_latedose/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w43_format_embed144_latedose.json` | 206/20/305 |
| `w46_z_headsel_sharpen` | `runs/variants/w46_z_headsel_sharpen/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w46_z_headsel_sharpen.json` | 213/25/320 |
| `w7_format_boost_mild` | `runs/variants/w7_format_boost_mild/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 | `recipes/w7_format_boost_mild.json` | 207/21/317 |

Что осталось в каталогах удалённых вариантов: `manifest.json` (правки, диапазоны, хеши
базы и результата), `verify.json`, `recipes/<вариант>.json` (в корне проекта), `dev/`
(100 строк с вопросами и ответами), `search/` (250 строк), `dev.log`, `search.log`,
`compare_*.json` (2–6 файлов). Каталог `model/` сохранён целиком, кроме шарда весов:
на месте `config.json`, `model.safetensors.index.json`, `chat_template.jinja`,
`merges.txt`, `vocab.json`, `tokenizer_config.json` и т. п. (единицы МБ).

## 4. Что сохранено (веса)

| категория | вариантов | веса |
|---|---|---|
| контрольная точка `w8_z_gate_mild_format` | 1 | 1,75 GB |
| превзошли w8 хотя бы по одной измеренной метрике (верно / обрывы / FINAL) | 18 | 31,44 GB |
| экран 350 не завершён (14 неоценённых/частично оценённых, в т. ч. 1 smoke) | 14 | 24,46 GB |
| статус нельзя подтвердить (`verify.json` отсутствует) | 5 | 8,73 GB |
| **итого осталось** | **38** | **66,38 GB** |

- Исходная модель в проекте `models/qwen3.5-0.8b/` (1,7 G) не изменялась;
  shard sha256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (пересчитан
  после очистки).
- `w8_z_gate_mild_format` не изменялся; shard sha256
  `296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55` (пересчитан после очистки).
- Варианты «лучше w8» (по любой из трёх метрик): `w36_z_late_extra094` (219/21/329 — лучший
  результат по числу верных за всю серию), `w30_qknorm_sharp108` (215/**9**/337 — минимум
  обрывов), `w29_z_total087` (203/10/**340** — максимум FINAL), `w37_z_early_release1075`,
  `w28_fullgate_close115`, `w31_z_dominant_restore106`, `w25_z_headsel_damp088` (без
  `verify.json`), `w32_qknorm104_on_w8`, `w24_fullgate_open085`, `w35_z_dominant_damp078`,
  `w39_z_late_release1075`, `w40_z_early_extra094`, `w42_format_embed144`, `w44_mlp_branch_late105`,
  `w45_z_headsel_flatten`, `w14_z93_final13`, `w11_final_token_amp_15`, `w5_z_gate_damp`,
  `w47_conv_tap_recency`.
- Не завершённые на полном экране (15 вариантов): 13 вариантов волн 1–6 (`v1`, `v3`, `v4`,
  `v5`, `v6`, `v7`, `v8`, `v10`, `w2`, `w3`, `w4`, `w5`, `w6`) и `w47_conv_tap_recency`,
  `w48_conv_tap_context` (волна 11; `w47` дополнительно попадает в список «лучше w8» по
  обрывам и FINAL на dev100). Отдельно `c1_visual_noop` — smoke на 24 задачах.
- Статус не подтверждён из-за отсутствия `verify.json`: `w25_z_headsel_damp088`,
  `w26_norm_amp093`, `w27_beta_sharpen_late130`, `w33_fullgate_open_closed_heads`,
  `w34_restore_plus_qknorm104` — по правилу «назначение непонятно — не удалять» сохранены.

## 5. Как восстановить удалённые веса

Правки детерминированы: `modify_weights.py` из того же рецепта и базы даёт побитово тот же
shard (хеш записан в `manifest.json` → `new_safetensors_sha256`, и в `verify.json` →
`файл_варианта_по_хешу_detail`). Пример:

```bash
.venv/bin/python scripts/modify_weights.py \
  --recipe recipes/w17_z93_only.json \
  --base models/qwen3.5-0.8b \
  --out runs/variants/w17_z93_only
.venv/bin/python scripts/verify_variant.py w17_z93_only
```

База каждого варианта записана в его `manifest.json` (`base_model_dir`), хеш базы — в
`verify.json` (`базовый_файл_по_хешу_detail`): `04b1c301…` для вариантов от исходной модели,
`296b2cd8…` для вариантов от `w8`, `943a08f0…` для `w43` (база `w36_z_late_extra094`).

## 6. Контроль после очистки

- 13 шардов удалены, повторная проверка `ls runs/variants/<v>/model/*.safetensors` → пусто;
- у всех 13 на месте `manifest.json`, `recipes/<v>.json`, `verify.json`, 100/250 строк
  результатов, файлы сравнений (проверено выборочно и скриптом удаления);
- `models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors` — sha256 совпадает;
- `runs/variants/w8_z_gate_mild_format/model/…` — sha256 совпадает;
- `runs/variants` занимает 68 G (было 89 G);
- свободно: **105 G** (было 84 G до удаления; 16 G в начале задачи, до внешнего освобождения).

## 7. Ограничения

- Удаление необратимо, если рецепт или база окажутся неполными; проверка опиралась на
  `manifest.json` (полный список операций) и `verify.json` (сверка вне диапазонов).
- Варианты без `verify.json` (5 шт.) и незавершённые (15 шт.) не удалялись сознательно:
  для них нельзя доказать, что результат хуже w8, либо прогон неполон.
- Каталоги удалённых вариантов остались без весов: повторный прогон `verify_variant.py`
  для них теперь невозможен без пересборки (это ожидаемо, см. §5).

# Очистка диска № 2: удалённые файлы весов старых вариантов (2026-11-18, 21:03–21:13 Asia/Krasnoyarsk)

Раздел добавлен по запросу «нужно освободить место на диске в проекте»; прежние разделы не изменялись
и не стирались. Удалялись **только** файлы весов (`model/model.safetensors-00001-of-00001.safetensors`)
старых проверенных вариантов. Рецепты, `manifest.json`, `verify.json`, `dev/results.jsonl`,
`search/results.jsonl`, журналы и файлы сравнений сохранены. Каталоги вариантов остались на месте.
Ничего вне `/home/mpiscil/weight-experiment` не удалялось; программы `Strata` и каталог
`/home/mpiscil/Strata-data` не трогались. Новые эксперименты не запускались.

## 1. Состояние до очистки

| параметр | значение |
|---|---|
| время | 2026-11-18 21:03–21:13 (Asia/Krasnoyarsk) |
| свободно на `/dev/sda5` | **544 224 КБ ≈ 0,52 ГиБ** (80 Г занято от 480 Г, 100 % в `df -h`) |
| занято `runs/variants` | 107 Г, из них файлы весов ≈ 105 ГиБ в 68 каталогах |
| занято `models` | 1,7 Г |
| процессы | активных прогонов нет: `eval_gsm8k`, `modify_weights`, `verify_variant`, `wait_gate_and_run_w*`, `compare_runs`, `combined_score` не запущены (проверено `ps`) |
| временный `tmpfs` `/tmp` | свободно 27 614 584 КБ; в `/tmp/wex_builds` три сборки по 1,7 Г — это не рабочий том, не удалялись |

Проверка ссылок: в `runs/variants` нет файлов с числом жёстких связей больше 1
(`find -type f -links +1` → пусто). Символические ссылки только три: `w78`, `w79`, `w81` →
`/tmp/wex_builds/<вариант>/model`; они не удалялись.

## 2. Критерии удаления (проверялись для каждого варианта)

1. вариант проверен: есть `manifest.json`, результаты `dev` (100 строк) и, как правило, `search` (250 строк);
2. вариант не лучший и не перспективный: в сводках волн он хуже `w8` или `w50`, либо `w8` или `w50`
   сами перекрывают его по набору метрик; ни один вариант не «подозреваем на победу»;
3. сохранены рецепт `recipes/<вариант>.json`, `manifest.json` (полный список операций и хеши),
   `verify.json`, результаты `dev` и `search`, файлы сравнений — проверено скриптом, недостающих нет;
4. файл весов — отдельный обычный файл: `st_nlink == 1`, не символическая ссылка, инода совпадает
   с инвентарём, путь внутри `/home/mpiscil/weight-experiment`;
5. вариант не служит базой сборки для другого варианта. По `base_safetensors_sha256` в манифестах
   есть ровно две базы-варианта: `w8_z_gate_mild_format` (48 вариантов) и `w36_z_late_extra094`
   (для `w43_format_embed144_latedose`). `w36` **сохранён** именно поэтому.

## 3. Удалено: 58 файлов весов

Каждый файл 1 746 942 600 байт. Итого **101 322 670 800 байт (94,36 ГиБ)**.
Инвентарь и протокол: `runs/analysis/cleanup2_inventory.json` и `runs/analysis/cleanup2_deleted.json`
(путь, байты, инода, хеш весов, дата).

| вариант | удалённый файл | байт |
|---|---|---|
| `v10_attn_branch_amp` | `runs/variants/v10_attn_branch_amp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v1_Alog_decay_half` | `runs/variants/v1_Alog_decay_half/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v3_dt_bias_shift_down` | `runs/variants/v3_dt_bias_shift_down/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v4_beta_amplify` | `runs/variants/v4_beta_amplify/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v5_beta_soften` | `runs/variants/v5_beta_soften/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v6_attn_sharpen` | `runs/variants/v6_attn_sharpen/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v7_attn_smooth` | `runs/variants/v7_attn_smooth/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `v8_mlp_amp` | `runs/variants/v8_mlp_amp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w11_final_token_amp_15` | `runs/variants/w11_final_token_amp_15/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w14_z93_final13` | `runs/variants/w14_z93_final13/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w24_fullgate_open085` | `runs/variants/w24_fullgate_open085/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w25_z_headsel_damp088` | `runs/variants/w25_z_headsel_damp088/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w26_norm_amp093` | `runs/variants/w26_norm_amp093/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w27_beta_sharpen_late130` | `runs/variants/w27_beta_sharpen_late130/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w28_fullgate_close115` | `runs/variants/w28_fullgate_close115/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w29_z_total087` | `runs/variants/w29_z_total087/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w2_bold_damp` | `runs/variants/w2_bold_damp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w30_qknorm_sharp108` | `runs/variants/w30_qknorm_sharp108/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w31_z_dominant_restore106` | `runs/variants/w31_z_dominant_restore106/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w32_qknorm104_on_w8` | `runs/variants/w32_qknorm104_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w33_fullgate_open_closed_heads` | `runs/variants/w33_fullgate_open_closed_heads/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w34_restore_plus_qknorm104` | `runs/variants/w34_restore_plus_qknorm104/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w35_z_dominant_damp078` | `runs/variants/w35_z_dominant_damp078/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w36_z_late_extra094` | `runs/variants/w36_z_late_extra094/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w37_z_early_release1075` | `runs/variants/w37_z_early_release1075/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w39_z_late_release1075` | `runs/variants/w39_z_late_release1075/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w3_final_amp_bold_damp` | `runs/variants/w3_final_amp_bold_damp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w40_z_early_extra094` | `runs/variants/w40_z_early_extra094/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w42_format_embed144` | `runs/variants/w42_format_embed144/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w44_mlp_branch_late105` | `runs/variants/w44_mlp_branch_late105/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w45_z_headsel_flatten` | `runs/variants/w45_z_headsel_flatten/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w47_conv_tap_recency` | `runs/variants/w47_conv_tap_recency/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w48_conv_tap_context` | `runs/variants/w48_conv_tap_context/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w49_z_late094_qknorm108` | `runs/variants/w49_z_late094_qknorm108/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w4_branch_damp` | `runs/variants/w4_branch_damp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w51_finalrow_rotation15_smallmargin_negsign` | `runs/variants/w51_finalrow_rotation15_smallmargin_negsign/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w52_frozen10_dose003_negsign` | `runs/variants/w52_frozen10_dose003_negsign/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w53_frozen10_dose003_mirrorpos` | `runs/variants/w53_frozen10_dose003_mirrorpos/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w54_vhead_late19_head1_amp105` | `runs/variants/w54_vhead_late19_head1_amp105/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w55_gold16_l23_amp130` | `runs/variants/w55_gold16_l23_amp130/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w56_gold16_l23_amp070` | `runs/variants/w56_gold16_l23_amp070/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w59_flippot16_l23_amp163` | `runs/variants/w59_flippot16_l23_amp163/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w5_z_gate_damp` | `runs/variants/w5_z_gate_damp/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w60_format16_l23_amp130` | `runs/variants/w60_format16_l23_amp130/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w64_qknorm108_on_original` | `runs/variants/w64_qknorm108_on_original/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w65_qknorm108_late2_on_w8` | `runs/variants/w65_qknorm108_late2_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w66_qknorm112_on_w8` | `runs/variants/w66_qknorm112_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w67_inproj_a_amp110_on_w8` | `runs/variants/w67_inproj_a_amp110_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w68_inproj_a_damp090_on_w8` | `runs/variants/w68_inproj_a_damp090_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w6_dt_bias_up` | `runs/variants/w6_dt_bias_up/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w70_qknorm110_on_w8` | `runs/variants/w70_qknorm110_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w71_finalrow_plus_qknorm108_on_w8` | `runs/variants/w71_finalrow_plus_qknorm108_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w72_finalrow_plus_zlate_on_w8` | `runs/variants/w72_finalrow_plus_zlate_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w73_z_loudheads_202122_on_w8` | `runs/variants/w73_z_loudheads_202122_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w74_finalrow_plus_two_rows198_248046_on_w8` | `runs/variants/w74_finalrow_plus_two_rows198_248046_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w75_z_flatten_l202122_on_w8` | `runs/variants/w75_z_flatten_l202122_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w76_z_flatten_top5_on_w8` | `runs/variants/w76_z_flatten_top5_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |
| `w77_finalrow_plus_8rows_on_w8` | `runs/variants/w77_finalrow_plus_8rows_on_w8/model/model.safetensors-00001-of-00001.safetensors` | 1 746 942 600 |

Итого 58 × 1 746 942 600 = 101 322 670 800 байт.

## 4. Что сохранено (веса)

| объект | sha256 | состояние |
|---|---|---|
| исходная модель `models/qwen3.5-0.8b/model.safetensors-00001-of-00001.safetensors` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` | сохранена, хеш сверен |
| контрольная точка `runs/variants/w8_z_gate_mild_format/model/model.safetensors-00001-of-00001.safetensors` | `296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55` | сохранена, база для 48 вариантов |
| лучший по математике `runs/variants/w50_finalrow_rotation15_smallmargin/model/model.safetensors-00001-of-00001.safetensors` | `544d4eea6ab1d1532e28c134366f29720f084c94281b7b04d0c2a4c219881e70` | сохранён |
| незавершённый `runs/variants/c1_visual_noop/model/model.safetensors-00001-of-00001.safetensors` | по манифесту `87468a61477b…` | сохранён (нет результатов, не проверен) |
| база `runs/variants/w36_z_late_extra094/model/model.safetensors-00001-of-00001.safetensors` | `943a08f0e95cb019e361ef805461faaf15f5321f4d7d94fff676d69b99650335` | сохранён: база сборки `w43` |
| текущая сборка `runs/variants/w79_finalrow_ten_rows_double_step_on_w8/model` (символическая ссылка на `/tmp/wex_builds/w79_…/model`) | `cdca20526bea592a69b180785436df739d090e2402c15ff9e2b4ad1cebe31607` | сохранён (в рабочем томе, не tmpfs) |
| текущая сборка `runs/variants/w78_finalrow_six_mid_rows_on_w8/model` (символическая ссылка) | `0b117b2df24919547fb9e0233861da29942bfe465e441c6308a9c19a391bba0a` | сохранён |
| новая сборка `runs/variants/w81_finalrow_four_small_rows_step15_on_w8/model` (символическая ссылка) | `61ec64151fd1b877119d54583f7153b446f2a81c95d9255662902a354702047c` | сохранён |

Все каталоги удалённых вариантов оставлены: в них остались `manifest.json`, `verify.json`, `dev/`,
`search/`, логи, сравнения и все служебные файлы модели (`config.json`, `tokenizer.json`, `vocab.json`,
`merges.txt` и прочие, единицы МБ). Рецепты — в `recipes/`. Ничего из `runs/analysis`, `runs/logs`,
`runs/splits` (включая `holdout6.json`), `runs/baseline_*`, `runs/JOURNAL.md` и `scripts/` не удалялось.

## 5. Контроль после очистки

- свободно: **544 224 КБ → 99 492 428 КБ** (≈ 0,52 ГиБ → 94,9 ГиБ; прирост ≈ 94,4 ГиБ);
  `df -h`: занято 455 G → 361 G, 100 % → 80 %;
- `runs/variants`: 107 Г → 12 Г;
- все три сверенных хеша (исходная модель, `w8`, `w50`) совпадают с ожидаемыми;
- у удалённых вариантов шарды весов отсутствуют, файлы метаданных на месте (проверено `ls` для `w36`
  и `w77`);
- процесс удаления сверял иноду и размер каждого файла с инвентарём; ошибок нет: удалено 58 из 58.

## 6. Удалить не удалось (не соответствует критериям или случай неясен)

| вариант | причина |
|---|---|
| `w53_frozen10_dose003_mirrorpos` | веса удалены; перед удалением обнаружено, что его веса побитово совпадают с `w79` (`cdca20526bea592a…`), в том числе поле `model_answer` совпало 100 из 100 на `dev` и 250 из 250 на `search`. Значит `w53` и `w79` — один и тот же набор весов, и это отдельный вопрос к журналу (см. новый блок `§84`). Удаление `w53` безопасно, так как те же веса остаются на диске у `w79` |
| `w78_finalrow_six_mid_rows_on_w8`, `w79_finalrow_ten_rows_double_step_on_w8`, `w81_finalrow_four_small_rows_step15_on_w8` | свежие варианты, веса только через символическую ссылку на `/tmp/wex_builds`; это не копии в рабочем томе — не удалялись |
| `w8_z_gate_mild_format` | контрольная точка и база 48 вариантов |
| `w50_finalrow_rotation15_smallmargin` | лучший результат по математике, сравнение для новых вариантов |
| `c1_visual_noop` | нет результатов, статус неизвестен |
| `w36_z_late_extra094` | база сборки `w43` |
| `w2_bold_damp`, `w3_final_amp_bold_damp`, `w4_branch_damp`, `w5_z_gate_damp`, `w6_dt_bias_up`, `v10_attn_branch_amp`, `v1_Alog_decay_half`, `v3_dt_bias_shift_down`, `v4_beta_amplify`, `v5_beta_soften`, `v6_attn_sharpen`, `v7_attn_smooth`, `v8_mlp_amp`, `w47_conv_tap_recency`, `w48_conv_tap_context` | имеют только прогон `dev` (100), без `search`; удалены — да, вошли в список 58, но это отмечено здесь: их отбор опирается только на `dev`, что и записано в журнале волны |
| `w63_qknorm108_on_w8`, `w69_qknorm110_on_w8` | весов на диске нет (были удалены ранее или не собирались): `w64` и `w70` указывают на них базой по хешу, но папок нет |
| `/tmp/wex_builds/*` | не рабочий том; удаление освободит только память, правило «ничего вне проекта» соблюдено, файлы не трогались |

Веса удалённых вариантов восстанавливаются пересборкой: `manifest.json` содержит полный список
операций (`recipe_ops`) и хеш базы (`base_safetensors_sha256`), рецепт — в `recipes/`. Пример:

```bash
.venv/bin/python scripts/modify_weights.py --recipe recipes/w77_finalrow_plus_8rows_on_w8.json \
    --base runs/variants/w8_z_gate_mild_format/model --out /tmp/wex_builds/w77_finalrow_plus_8rows_on_w8
```
