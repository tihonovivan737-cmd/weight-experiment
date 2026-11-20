#!/usr/bin/env bash
# Волна 27 (§72.2, полосы П72.1): w77_finalrow_plus_8rows_on_w8 — тот же рычаг формата
# (поворот строк к строке FINAL, шаг 0,015, база w8), состав набора — 8 строк
# со средним запасом не ниже 0,25; четыре строки w50_finalrow_rotation15_smallmargin исключены.
# Полосы П72.1 записаны в журнал до сборки (§72.2). Официальный набор из 1319 задач не используется.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
PYCPU=.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
V=w77_finalrow_plus_8rows_on_w8
R=recipes/$V.json
LOG=runs/logs/w77_run.log
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
free=$(df -h . | awk 'NR==2{print $4}')
log "свободное место: $free"
need_kb=1780000
avail_kb=$(df -k . | awk 'NR==2{print $4}')
if [ "$avail_kb" -lt "$need_kb" ]; then log "ОШИБКА: свободно $avail_kb КБ, нужно $need_kb КБ — сборка не начинается"; exit 1; fi

if [ -d "runs/variants/$V/model" ]; then
  log "сборка уже есть — не пересобираю"
else
  log "самопроверка рецепта"
  $PYCPU scripts/selftest_w71.py "$R" | tee -a "$LOG"
  log "сборка (база указана явно)"
  $PYCPU scripts/modify_weights.py --recipe "$R" --base "$W8/model" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОШИБКА: сборка код $rc"; exit 1; fi
fi

log "сверка сборки"
if $PYCPU scripts/verify_variant.py "$V" > "runs/verify_${V}.txt" 2>&1; then
  log "VERIFY: $(grep -m1 'ИТОГ' "runs/verify_${V}.txt" || echo 'нет строки ИТОГ')"
else
  log "VERIFY код $? — см. runs/verify_${V}.txt"
fi

eval_split() {
  local sp=$1
  if [ -f "runs/variants/$V/$sp/results.jsonl" ] && [ "$(wc -l < "runs/variants/$V/$sp/results.jsonl")" -ge 1 ]; then
    log "$sp уже есть — пропускаю"; return 0
  fi
  log "EVAL $sp $V"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$V/model" $GEN --split "$sp" \
      --out "runs/variants/$V/$sp" --note "wave27 format rows margin >= 0.25 on w8, $sp" \
      > "runs/variants/$V/$sp.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "ОШИБКА: eval $sp код $rc"; return 1; fi
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
cmp_pair dev orig runs/baseline_dev/results.jsonl "runs/analysis/w77_vs_orig_dev.txt"
cmp_pair dev w8 "$W8/dev/results.jsonl" "runs/analysis/w77_vs_w8_dev.txt"
eval_split search || exit 1
cmp_pair search orig runs/baseline_search/results.jsonl "runs/analysis/w77_vs_orig_search.txt"
cmp_pair search w8 "$W8/search/results.jsonl" "runs/analysis/w77_vs_w8_search.txt"
$PYCPU scripts/combined_score.py --variant "$V" --report "runs/variants/$V/combined.json" >> "$LOG" 2>&1

log "контроль после прогона"
got=$(sha_of "$ORIG"); [ "$got" = "$EXORIG" ] && log "исходная модель после: $got" || log "ОШИБКА после: $got"
got=$(sha_of "$W8"); [ "$got" = "$EXW8" ] && log "w8 после: $got" || log "ОШИБКА после: $got"
log "ВОЛНА 27 ЗАВЕРШЕНА: П72.1 — см. журнал §72.2"
