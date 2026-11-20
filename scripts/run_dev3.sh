#!/usr/bin/env bash
# Волна 3 (вторая половина): прогон вариантов из runs/selection_wave3.txt на dev100.
# Нужен для суммарной метрики по 350 нерасходящимся train-задачам (правило 14/14.1 журнала).
# Запуск: ждёт файл runs/GATE_DEV3. Только физическая GPU 1 (CUDA_VISIBLE_DEVICES=1 -> cuda:0).
# Одна GPU-задача за раз; не трогать GPU 0 и процесс Strata.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
SELECTION=runs/selection_wave3.txt
REPORT=runs/dev3_report.txt

log() { echo "=== $(date -Is) $* ==="; }
free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; }

mapfile -t VARIANTS < <(sed 's/#.*//' "$SELECTION" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -vE '^$')
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  echo "Список вариантов пуст: заполни $SELECTION"; exit 1
fi
log "dev100 для волны 3, варианты: ${VARIANTS[*]}"

while [ ! -f runs/GATE_DEV3 ]; do log "ждём файл runs/GATE_DEV3"; sleep 30; done
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
done

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет runs/variants/$v/model"; continue; fi
  log "EVAL dev100 $v ; свободно VRAM: $(free)"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split dev \
      --out "runs/variants/$v/dev" --note "wave3 dev100, вторая половина 350-задачной метрики" \
      > "runs/variants/$v/dev.log" 2>&1
  $PY scripts/compare_runs.py --base runs/baseline_dev/results.jsonl \
      --variant "runs/variants/$v/dev/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_dev.json" | tee -a "$REPORT"
  $PY scripts/combined_score.py --variant "$v" \
      --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
done

log "DEV100 ДЛЯ ВОЛНЫ 3 ЗАВЕРШЁН"
