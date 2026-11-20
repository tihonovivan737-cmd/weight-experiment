#!/usr/bin/env bash
# Волна 30 (полосы П81.1, журнал §81.6): w81_finalrow_four_small_rows_step15_on_w8.
# Один изменённый признак против w50: состав строк (четыре малые строки, шесть средних
# строк исключены), шаг остаётся плюс 0,015. Полосы записаны в журнал до сборки.
# Официальный набор из 1319 задач не используется: отбор только по dev и search.
# Файл весов собирается во временном tmpfs, манифест, результаты и отчёты — в рабочем томе.
# Пороги не понижены: tmpfs 2000000 КБ, память 10950 МБ, видеокарта 1 — 6943 МиБ.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
PYCPU=.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
V=w81_finalrow_four_small_rows_step15_on_w8
R=recipes/$V.json
BUILD=/tmp/wex_builds/$V
LOG=runs/logs/w81_run.log
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
if [ "$tmp_kb" -lt 2000000 ]; then log "ОШИБКА: во временном каталоге $tmp_kb КБ, нужно 2000000 КБ — сборка не начинается"; exit 1; fi
if [ "$ram_mb" -lt 10950 ]; then log "ОШИБКА: доступно памяти $ram_mb МБ, нужно 10950 МБ по порогу в силе — сборка не начинается"; exit 1; fi
if [ "$gpu_free" -lt 6943 ]; then log "ОШИБКА: свободно $gpu_free МиБ, нужно 6943 МиБ — прогон не начинается"; exit 1; fi

mkdir -p "$BUILD" "runs/variants/$V"
if [ -f "$BUILD/manifest.json" ]; then
  log "сборка уже есть — не пересобираю"
else
  log "самопроверка рецепта"
  $PYCPU scripts/selftest_w71.py "$R" | tee -a "$LOG"
  log "сборка во временном каталоге (база указана явно)"
  $PYCPU scripts/modify_weights.py --recipe "$R" --base "$W8/model" --out "$BUILD" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОШИБКА: сборка код $rc"; exit 1; fi
fi

cp "$BUILD/manifest.json" "runs/variants/$V/manifest.json" 2>/dev/null || log "ОШИБКА: манифеста нет во временном каталоге"
[ -e "runs/variants/$V/model" ] || ln -s "$BUILD/model" "runs/variants/$V/model"
log "манифест в рабочем томе: runs/variants/$V/manifest.json; файл весов: $BUILD/model (символическая ссылка)"

log "сверка сборки"
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
      --out "runs/variants/$V/$sp" --note "wave30 four small rows step 0.015 on w8, $sp" \
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

eval_split dev || exit 1
cmp_pair dev "исходная модель" runs/baseline_dev/results.jsonl "runs/analysis/w81_vs_orig_dev.txt"
cmp_pair dev w8 "$W8/dev/results.jsonl" "runs/analysis/w81_vs_w8_dev.txt"
eval_split search || exit 1
cmp_pair search "исходная модель" runs/baseline_search/results.jsonl "runs/analysis/w81_vs_orig_search.txt"
cmp_pair search w8 "$W8/search/results.jsonl" "runs/analysis/w81_vs_w8_search.txt"
$PYCPU scripts/combined_score.py --variant "$V" --report "runs/variants/$V/combined.json" >> "$LOG" 2>&1

log "контроль после прогона"
got=$(sha_of "$ORIG"); [ "$got" = "$EXORIG" ] && log "исходная модель после: $got" || log "ОШИБКА после: $got"
got=$(sha_of "$W8"); [ "$got" = "$EXW8" ] && log "w8 после: $got" || log "ОШИБКА после: $got"
log "ВОЛНА 30 ЗАВЕРШЕНА: полосы П81.1 — см. журнал §81.6"
