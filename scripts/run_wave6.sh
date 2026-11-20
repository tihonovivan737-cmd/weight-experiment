#!/usr/bin/env bash
# Волна 6: направленные рычаги поверх контрольной точки w8 (плюс одна абляция от исходной модели).
# Порядок = приоритет: сначала самые информативные измерением рычаги.
# Для каждого варианта: search250 + dev100, сравнение НЕЗАВИСИМО с исходной моделью
# (runs/baseline_*) и с контрольной точкой w8 (runs/variants/w8_z_gate_mild_format/*).
# settings идентичны всем предыдущим прогонам: bfloat16, batch 1, max_new_tokens 1024, тот же суффикс промпта.
# Исходная модель и w8 не перезаписываются: прогоны только читают их weights и результаты.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
W8=runs/variants/w8_z_gate_mild_format
VARIANTS=(
  w25_z_headsel_damp088
  w24_fullgate_open085
  w28_fullgate_close115
  w29_z_total087
  w26_norm_amp093
  w27_beta_sharpen_late130
  w30_qknorm_sharp108
)
REPORT=runs/wave6_report.txt
MIN_FREE_MIB=4000

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }

log "волна 6, варианты: ${VARIANTS[*]}"
while pgrep -f "build_wave6b.sh" >/dev/null; do log "сборка вариантов ещё идёт — ждём 30 с"; sleep 30; done
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
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
  eval_split "$v" search 250 "wave6 lever on w8, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8      "$W8/search/results.jsonl"          "runs/variants/$v/compare_search_vs_w8.json"
  eval_split "$v" dev 100 "wave6 lever on w8, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8      "$W8/dev/results.jsonl"          "runs/variants/$v/compare_dev_vs_w8.json"
  $PY scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

log "ВОЛНА 6 ЗАВЕРШЕНА"
