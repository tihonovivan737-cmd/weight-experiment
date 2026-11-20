#!/usr/bin/env bash
# Волна 2: подтверждение отобранных вариантов на search250 (парное сравнение с baseline_search).
# Список вариантов берётся из runs/selection_wave2.txt (по одному имени в строке; строки с # игнорируются).
# Запуск по разрешению: ждёт файл runs/GATE_WAVE2 (создаётся вручную после разбора волну 1/1b).
# Только физическая GPU 1 (CUDA_VISIBLE_DEVICES=1 -> cuda:0 внутри процесса). Одна GPU-задача за раз.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
SELECTION=runs/selection_wave2.txt
REPORT=runs/wave2_report.txt

log() { echo "=== $(date -Is) $* ==="; }
free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; }

mapfile -t VARIANTS < <(sed 's/#.*//' "$SELECTION" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -vE '^$')
if [ "${#VARIANTS[@]}" -eq 0 ]; then
  echo "Список вариантов пуст: заполни $SELECTION"; exit 1
fi
log "волна 2, варианты: ${VARIANTS[*]}"

# ---- ожидание разрешения (gate-файл) и свободной GPU ---------------------------------------
# Gate создаётся вручную после разбора результатов волны 1/1b: `touch runs/GATE_WAVE2`.
while true; do
  if [ -f runs/GATE_WAVE2 ]; then log "gate runs/GATE_WAVE2 создан"; break; fi
  if ! pgrep -f "run_wave1.sh" >/dev/null && ! pgrep -f "run_wave1b.sh" >/dev/null; then
    log "предыдущих очередей нет; ждём файл runs/GATE_WAVE2"
  fi
  sleep 30
done
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
done

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then
    log "ПРОПУСК $v: нет runs/variants/$v/model"; continue
  fi
  log "EVAL search250 $v ; свободно VRAM: $(free)"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split search \
      --out "runs/variants/$v/search" --note "wave2 confirmation, search250" \
      > "runs/variants/$v/search.log" 2>&1
  $PY scripts/compare_runs.py --base runs/baseline_search/results.jsonl \
      --variant "runs/variants/$v/search/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_search.json" | tee -a "$REPORT"
done

log "ВОЛНА 2 ЗАВЕРШЕНА"
