#!/usr/bin/env bash
# Волна 4: w14_z93_final13 (два измеренных положительных механизма вместе) и
# w15_final_case_suppress (супрессия конкурирующего маркера 'Final').
# Сначала search250 для обоих (решающий промежуточный набор), затем dev100 для обоих
# -> суммарная метрика по 350 нерасходящимся train-задачам (правила 14/14.1 журнала).
# Только физическая GPU 1 (CUDA_VISIBLE_DEVICES=1 -> cuda:0). Одна GPU-задача за раз.
# Не трогать GPU 0 и процесс Strata. При нехватке VRAM или ошибке — остановиться и сообщить.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
VARIANTS=(w14_z93_final13 w15_final_case_suppress)
REPORT=runs/wave4_report.txt

log() { echo "=== $(date -Is) $* ==="; }
freevram() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; }

log "волна 4, варианты: ${VARIANTS[*]}"
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
done

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет модели"; continue; fi
  free_now=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  if [ "${free_now:-0}" -lt 4000 ]; then
    log "ОСТАНОВКА: на GPU 1 свободно только ${free_now} MiB (< 4000)"; exit 1
  fi
  log "EVAL search250 $v ; свободно VRAM: $(freevram)"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split search \
      --out "runs/variants/$v/search" --note "wave4 confirmation, search250" \
      > "runs/variants/$v/search.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval search250 $v завершился с кодом $rc (смотри runs/variants/$v/search.log)"; exit 1; fi
  $PY scripts/compare_runs.py --base runs/baseline_search/results.jsonl \
      --variant "runs/variants/$v/search/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_search.json" | tee -a "$REPORT"
done

for v in "${VARIANTS[@]}"; do
  if [ ! -f "runs/variants/$v/search/results.jsonl" ]; then log "ПРОПУСК dev для $v: нет search-результатов"; continue; fi
  log "EVAL dev100 $v ; свободно VRAM: $(freevram)"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split dev \
      --out "runs/variants/$v/dev" --note "wave4 dev100, вторая половина 350-задачной метрики" \
      > "runs/variants/$v/dev.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval dev100 $v завершился с кодом $rc (смотри runs/variants/$v/dev.log)"; exit 1; fi
  $PY scripts/compare_runs.py --base runs/baseline_dev/results.jsonl \
      --variant "runs/variants/$v/dev/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_dev.json" | tee -a "$REPORT"
  $PY scripts/combined_score.py --variant "$v" \
      --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
done

log "ВОЛНА 4 ЗАВЕРШЕНА"
