#!/usr/bin/env bash
# Волна 33 (полосы П84.1, журнал §94.5 и §94.6): w84_mlp_layer_lever5_on_w8.
# Одно переменное — направление дозы по тензору mlp.up_proj.weight целиком:
# слои 14, 15, 23 получают 1,05, слои 17, 20, 21 — 0,952381; сумма логарифмов точно ноль.
# Это вторая точка лестницы дозы после w83 (та же схема, доза 2 %).
# Полосы записаны в журнал до сборки (§94.6).
# Официальный набор из 1319 задач не используется: отбор только по dev и search.
# Файл весов собирается в рабочий том (решение пользователя от 18.11.2026 22:3x: сборки на диске вместо временного tmpfs).
# Пороги в силе по решению пользователя: память 10950 МБ, видеокарта 1 — 6943 МиБ, рабочий том 1780000 КБ.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
PYCPU=.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
V=w84_mlp_layer_lever5_on_w8
R=recipes/$V.json
BUILD=runs/variants/$V
LOG=runs/logs/w84_run.log
EXORIG=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
EXW8=296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55
sha_of() {
  local d=$1 f
  f=$(find "$d" -maxdepth 2 -name '*.safetensors' -type f | head -1)
  [ -z "$f" ] && { echo ""; return 0; }
  sha256sum "$f" | cut -d' ' -f1
}
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "контроль опор до сборки"
got=$(sha_of "$ORIG"); [ "$got" = "$EXORIG" ] && log "исходная модель: $got" || log "ОШИБКА: исходная модель $got"
got=$(sha_of "$W8"); [ "$got" = "$EXW8" ] && log "w8: $got" || log "ОШИБКА: w8 $got"

tmp_kb=$(df -k /tmp | awk 'NR==2{print $4}')
ram_mb=$(free -m | awk 'NR==2{print $7}')
gpu_free=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$1==1{print $3-$2}')
ws_kb=$(df -k /home/mpiscil/weight-experiment | awk 'NR==2{print $4}')
log "место во временном каталоге: $tmp_kb КБ; доступно памяти: $ram_mb МБ; свободно на видеокарте 1: $gpu_free МиБ; свободно в рабочем томе: $ws_kb КБ"
if [ "$ws_kb" -lt 1780000 ]; then log "ОШИБКА: в рабочем томе $ws_kb КБ, нужно 1780000 КБ — сборка не начинается"; exit 1; fi
if [ "$ram_mb" -lt 10950 ]; then log "ОШИБКА: доступно памяти $ram_mb МБ, нужно 10950 МБ по порогу в силе — сборка не начинается"; exit 1; fi
if [ "$gpu_free" -lt 6943 ]; then log "ОШИБКА: свободно $gpu_free МиБ, нужно 6943 МиБ — прогон не начинается"; exit 1; fi

mkdir -p "runs/variants/$V"
if [ -f "$BUILD/manifest.json" ]; then
  log "сборка уже есть — не пересобираю"
else
  log "самопроверка рецепта"
  $PYCPU scripts/selftest_w71.py "$R" | tee -a "$LOG"
  log "сборка в рабочем томе (база указана явно)"
  $PYCPU scripts/modify_weights.py --recipe "$R" --base "$W8/model" --out "$BUILD" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОШИБКА: сборка код $rc"; exit 1; fi
fi

# modify_weights кладёт манифест в каталог --out; весовые файлы — в подкаталог model.
# Копируем манифест только если его нет на месте и источник не тот же файл.
if [ ! -f "runs/variants/$V/manifest.json" ]; then
  if [ -f "$BUILD/model/manifest.json" ]; then
    cp "$BUILD/model/manifest.json" "runs/variants/$V/manifest.json" || log "ОШИБКА: манифеста нет в model"
  elif [ -f "$BUILD/manifest.json" ]; then
    log "манифест уже в рабочем томе: $BUILD/manifest.json"
  else
    log "ОШИБКА: манифеста нигде нет"; exit 1
  fi
fi
log "манифест в рабочем томе: runs/variants/$V/manifest.json; весовые файлы: $BUILD/model (обычный каталог, без символической ссылки)"

log "сверка сборки"
# страховка от вложенного каталога model/model: такие сборки уже ломали маршрут
if [ -d "$BUILD/model/model" ]; then
  log "ОШИБКА: весовые файлы лежат вложенно ($BUILD/model/model) — правка маршрута, сборка не запускается"; exit 1
fi
if ! ls "$BUILD"/model/*.safetensors >/dev/null 2>&1; then
  log "ОШИБКА: в $BUILD/model нет файла весов"; exit 1
fi
if $PYCPU scripts/verify_variant.py "$V" > "runs/verify_${V}.txt" 2>&1; then
  log "ПРОВЕРКА: $(grep -m1 'ИТОГ' "runs/verify_${V}.txt" || echo 'нет строки итога')"
else
  log "ПРОВЕРКА код $? — см. runs/verify_${V}.txt"
fi

eval_split() {
  local sp=$1
  if [ -f "runs/variants/$V/$sp/results.jsonl" ] && [ "$(wc -l < "runs/variants/$V/$sp/results.jsonl")" -ge 1 ]; then
    log "$sp уже есть — пропускаю"; return 0
  fi
  mkdir -p "runs/variants/$V/$sp"
  log "ПРОГОН $sp $V"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$V/model" $GEN --split "$sp" \
      --out "runs/variants/$V/$sp" --note "wave33 mlp layer lever 5 percent on w8, $sp" \
      > "runs/variants/$V/$sp.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "ОШИБКА: прогон $sp код $rc"; return 1; fi
  log "$sp готово: $(wc -l < "runs/variants/$V/$sp/results.jsonl") строк"
  return 0
}

cmp_pair() {
  local sp=$1 lab=$2 base=$3 rep=$4
  $PYCPU scripts/compare_runs.py --base "$base" \
      --variant "runs/variants/$V/$sp/results.jsonl" \
      --base_name "$lab" --variant_name "$V" --report "$rep" >> "$LOG" 2>&1
}

log "проверка уникальности собранных весов"
newsha=$(sha_of "$BUILD/model")
if [ -z "$newsha" ]; then log "ОШИБКА: хеша весов нет"; exit 1; fi
dup=""
for mf in runs/variants/*/manifest.json; do
  case "$mf" in "runs/variants/$V/manifest.json") continue ;; esac
  got=$(grep -m1 -o '"new_safetensors_sha256": *"[0-9a-f]*"' "$mf" | grep -o '[0-9a-f]\{64\}')
  if [ "$got" = "$newsha" ]; then dup="$dup $(dirname "$mf" | xargs basename)"; fi
done
dup=$(echo "$dup" | sed 's/^ *//')
if [ -n "$dup" ]; then
  log "ОШИБКА: найденный набор весов совпадает с уже измеренным ($dup) — прогон не начинается, это дубль"
  exit 1
fi
log "хеш $newsha уникален среди измеренных вариантов — начинаем прогоны"

eval_split dev || exit 1
cmp_pair dev "исходная модель" runs/baseline_dev/results.jsonl "runs/analysis/w84_vs_orig_dev.txt"
cmp_pair dev w8 "$W8/dev/results.jsonl" "runs/analysis/w84_vs_w8_dev.txt"
eval_split search || exit 1
cmp_pair search "исходная модель" runs/baseline_search/results.jsonl "runs/analysis/w84_vs_orig_search.txt"
cmp_pair search w8 "$W8/search/results.jsonl" "runs/analysis/w84_vs_w8_search.txt"
$PYCPU scripts/combined_score.py --variant "$V" --report "runs/variants/$V/combined.json" >> "$LOG" 2>&1

log "контроль после прогона"
got=$(sha_of "$ORIG"); [ "$got" = "$EXORIG" ] && log "исходная модель после: $got" || log "ОШИБКА после: $got"
got=$(sha_of "$W8"); [ "$got" = "$EXW8" ] && log "w8 после: $got" || log "ОШИБКА после: $got"
log "ВОЛНА 33 ЗАВЕРШЕНА: полосы П84.1 — см. журнал §94.5 и §94.6"
