#!/usr/bin/env bash
# Волна 5 (абляции квадрата 2x2 вокруг лучшего варианта w8):
#   w16_final_rows12_only — только эмбеддинг FINAL/' FINAL' x1.2 (форматная часть w8)
#   w17_z93_only          — только мягкий z-гейт in_proj_z x0.93 (вторая часть w8)
# search250 для обоих, затем dev100 -> суммарная метрика по 350 train-задачам.
# Только физическая GPU 1 (CUDA_VISIBLE_DEVICES=1 -> cuda:0), одна GPU-задача за раз.
# При нехватке VRAM или ошибке прогона — остановиться и сообщить.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
VARIANTS=(w17_z93_only w16_final_rows12_only)
REPORT=runs/wave5_report.txt
MIN_FREE_MIB=4000

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }

log "волна 5, варианты: ${VARIANTS[*]}"
# Сначала дожидаемся окончания волны 4 (её dev100-прогоны стартуют позже её search-прогонов,
# поэтому одного проверки "нет eval_gsm8k.py" недостаточно — иначе будет два GPU-процесса сразу).
while pgrep -f "run_wave4.sh" >/dev/null; do
  log "волна 4 ещё работает — ждём 60 с"; sleep 60
done
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
done

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет модели"; continue; fi
  if [ -f "runs/variants/$v/search/results.jsonl" ] && [ "$(wc -l < "runs/variants/$v/search/results.jsonl")" -ge 250 ]; then
    log "search250 для $v уже есть ($(wc -l < "runs/variants/$v/search/results.jsonl") строк) — пропускаю прогон"
  else
    f=$(free1)
    if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then
      log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB}) — не запускаю прогон"
      exit 1
    fi
    log "EVAL search250 $v ; свободно VRAM на GPU 1: ${f} MiB"
    $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split search \
        --out "runs/variants/$v/search" --note "wave5 ablation, search250" \
        > "runs/variants/$v/search.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval search250 $v код $rc (runs/variants/$v/search.log)"; exit 1; fi
  fi
  $PY scripts/compare_runs.py --base runs/baseline_search/results.jsonl \
      --variant "runs/variants/$v/search/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_search.json" | tee -a "$REPORT"
done

for v in "${VARIANTS[@]}"; do
  if [ ! -f "runs/variants/$v/search/results.jsonl" ]; then log "ПРОПУСК dev для $v: нет search"; continue; fi
  if [ -f "runs/variants/$v/dev/results.jsonl" ] && [ "$(wc -l < "runs/variants/$v/dev/results.jsonl")" -ge 100 ]; then
    log "dev100 для $v уже есть — пропускаю прогон"
  else
    f=$(free1)
    if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then
      log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB}) — не запускаю прогон"
      exit 1
    fi
    log "EVAL dev100 $v ; свободно VRAM на GPU 1: ${f} MiB"
    $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split dev \
        --out "runs/variants/$v/dev" --note "wave5 ablation dev100, вторая половина 350-задачной метрики" \
        > "runs/variants/$v/dev.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval dev100 $v код $rc (runs/variants/$v/dev.log)"; exit 1; fi
  fi
  $PY scripts/compare_runs.py --base runs/baseline_dev/results.jsonl \
      --variant "runs/variants/$v/dev/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_dev.json" | tee -a "$REPORT"
  $PY scripts/combined_score.py --variant "$v" \
      --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
done

log "ВОЛНА 5 ЗАВЕРШЕНА"
