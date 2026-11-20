#!/usr/bin/env bash
# Волна 8: три направленных варианта ПОВЕРХ контрольной точки w8_z_gate_mild_format.
# Гипотезы предзарегистрированы в runs/JOURNAL.md (раздел 27) и в recipes/*.json
# до начала прогонов; они выведены только из 250 search + 100 dev и послойных
# зондов активаций. Официальный test (1319 задач) не используется.
# Исходная модель и w8 не перезаписываются: читаются только как основа и референс.
# Настройки идентичны всем предыдущим прогонам: bfloat16, batch 1, max_new_tokens 1024.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
W8=runs/variants/w8_z_gate_mild_format
W8MODEL=$W8/model
LOG=runs/wave8_run.log
REPORT=runs/wave8_report.txt
MIN_FREE_MIB=4000
VARIANTS=(
  w35_z_dominant_damp078
  w36_z_late_extra094
  w37_z_early_release1075
)

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }

: > "$REPORT"
log "волна 8, варианты: ${VARIANTS[*]}"

# 1. сборка вариантов (CPU/диск, GPU не трогает) — основа читается только на чтение
: > runs/build_wave8.log
for v in "${VARIANTS[@]}"; do
  if [ -f "runs/variants/$v/manifest.json" ]; then log "сборка $v уже есть — пропускаю"; continue; fi
  if [ ! -d "$W8MODEL" ]; then log "ОСТАНОВКА: нет $W8MODEL"; exit 1; fi
  log "BUILD $v (основа — контрольная точка w8, не исходная модель)"
  $PY scripts/modify_weights.py --recipe "recipes/$v.json" --base "$W8MODEL" \
      --out "runs/variants/$v" >> runs/build_wave8.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: build $v код $rc (runs/build_wave8.log)"; exit 1; fi
  log "собран $v"
done

# 2. ожидание завершения чужих прогонов (один прогон за раз, GPU 1).
# Сторож заякорен на абсолютный путь python: оболочка, в которой просто упомянуто
# имя скрипта (в том числе эта), цикл не блокирует.
busy() { pgrep -f "^$PY .*(eval_gsm8k|proxy_eval|probe_layers)\.py" >/dev/null; }
other_wave() {  # чужие волновые launcher'ы; собственный PID ($$) исключён — иначе цикл ждёт сам себя
  local p
  for p in $(pgrep -f "^bash scripts/run_wave[0-9]+\.sh$"); do
    [ "$p" != "$$" ] && return 0
  done
  return 1
}
other_probe() {  # probe_wave6.sh со всеми своими аргументами; собственный PID тоже исключён
  local p
  for p in $(pgrep -f "scripts/probe_wave6\.sh"); do
    [ "$p" != "$$" ] && return 0
  done
  return 1
}
while busy || other_wave || other_probe; do
  log "другой прогон работает — ждём 60 с"; sleep 60
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
  eval_split "$v" search 250 "wave8 lever on w8, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8      "$W8/search/results.jsonl"          "runs/variants/$v/compare_search_vs_w8.json"
  eval_split "$v" dev 100 "wave8 lever on w8, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8      "$W8/dev/results.jsonl"          "runs/variants/$v/compare_dev_vs_w8.json"
  $PY scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

if $PY scripts/wave8_table.py > runs/wave8_table.stdout 2>&1; then
  log "сводная таблица: runs/wave8_table.txt"
else
  log "ОШИБКА wave8_table.py — см. runs/wave8_table.stdout"; exit 1
fi
$PY scripts/pick_final2.py --table runs/analysis/wave8_table.json \
    --out runs/selection_wave8.txt | tee -a "$REPORT"
log "ВОЛНА 8 ЗАВЕРШЕНА"
