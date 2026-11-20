#!/usr/bin/env bash
# Волна 7: четыре направленных варианта поверх контрольной точки w8.
# Гипотезы взяты только из замеров 250 search + 100 dev и послойных зондов
# (scripts/make_recipes_wave7.py). Официальный test (1319) не используется.
# Для каждого варианта: search250 + dev100, сравнение НЕЗАВИСИМО с исходной
# моделью (runs/baseline_*) и с контрольной точкой w8.
# Настройки идентичны всем предыдущим прогонам: bfloat16, batch 1, max_new_tokens 1024.
# Исходная модель и w8 не перезаписываются: они читаются как основа и как референс.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
W8=runs/variants/w8_z_gate_mild_format
W8MODEL=$W8/model
LOG=runs/wave7_run.log
REPORT=runs/wave7_report.txt
MIN_FREE_MIB=4000
VARIANTS=(
  w31_z_dominant_restore106
  w32_qknorm104_on_w8
  w33_fullgate_open_closed_heads
  w34_restore_plus_qknorm104
)

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }

: > "$REPORT"
log "волна 7, варианты: ${VARIANTS[*]}"

# 1. сборка вариантов (CPU/диск, GPU не трогает) — исходник читается только на чтение
for v in "${VARIANTS[@]}"; do
  if [ -f "runs/variants/$v/manifest.json" ]; then log "сборка $v уже есть — пропускаю"; continue; fi
  if [ ! -d "$W8MODEL" ]; then log "ОСТАНОВКА: нет $W8MODEL"; exit 1; fi
  log "BUILD $v (основа — контрольная точка, не исходная модель)"
  $PY scripts/modify_weights.py --recipe "recipes/$v.json" --base "$W8MODEL" \
      --out "runs/variants/$v" >> runs/build_wave7.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: build $v код $rc (runs/build_wave7.log)"; exit 1; fi
  log "собран $v"
done

# 2. ожидание завершения чужих прогонов (один прогон за раз, GPU 1).
# Сторож заякорен на абсолютный путь python: случайная оболочка, в которой просто
# упомянуто имя скрипта, его больше не вызывает (раньше само-блокировала цикл).
busy() { pgrep -f "^$PY .*(eval_gsm8k|proxy_eval|probe_layers)\.py" >/dev/null; }
while busy || pgrep -f "^bash scripts/run_wave6.sh$" >/dev/null; do
  log "другой прогон/сборка работает — ждём 60 с"; sleep 60
done

eval_split() {  # variant  split  min_lines  note
  local v=$1 sp=$2 min=$3 note=$4 f
  if [ -f "runs/variants/$v/$sp/results.jsonl" ] && [ "$(wc -l < "runs/variants/$v/$sp/results.jsonl")" -ge "$min" ]; then
    log "$sp для $v уже есть ($(wc -l < "runs/variants/$v/$sp/results.jsonl") строк) — пропускаю прогон"; return 0
  fi
  f=$(free1)
  if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then
    log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB})"; return 1
  fi
  log "EVAL $sp $v ; свободно VRAM на GPU 1: ${f} MiB"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split "$sp" \
      --out "runs/variants/$v/$sp" --note "$note" > "runs/variants/$v/$sp.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval $sp $v код $rc (runs/variants/$v/$sp.log)"; return 1; fi
}

cmp_pair() {  # variant  split  base_label  base_results  report
  local v=$1 sp=$2 lab=$3 base=$4 rep=$5
  $PY scripts/compare_runs.py --base "$base" \
      --variant "runs/variants/$v/$sp/results.jsonl" \
      --base_name "$lab" --variant_name "$v" --report "$rep" | tee -a "$REPORT"
}

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет модели"; continue; fi
  eval_split "$v" search 250 "wave7 lever on w8, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8      "$W8/search/results.jsonl"          "runs/variants/$v/compare_search_vs_w8.json"
  eval_split "$v" dev 100 "wave7 lever on w8, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8      "$W8/dev/results.jsonl"          "runs/variants/$v/compare_dev_vs_w8.json"
  $PY scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

log "ВОЛНА 7 ЗАВЕРШЕНА"
