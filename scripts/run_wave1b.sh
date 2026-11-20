#!/usr/bin/env bash
# Волна 1b: промежуточный скрининг новых вариантов (гипотезы из замеренных отказов baseline)
# на dev100 + парное сравнение с runs/baseline_dev.
# Список: runs/screen_wave1b.txt (по имени в строке, # — комментарий).
# Ждёт маркер "ВОЛНА 1 ЗАВЕРШЕНА" в runs/wave1_run.log. GPU: строго одна задача за раз.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
LIST=runs/screen_wave1b.txt
REPORT=runs/wave1b_report.txt

log() { echo "=== $(date -Is) $* ==="; }
free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; }

mapfile -t VARIANTS < <(sed 's/#.*//' "$LIST" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -vE '^$')
if [ "${#VARIANTS[@]}" -eq 0 ]; then echo "Список пуст: $LIST"; exit 1; fi
log "волна 1b, варианты: ${VARIANTS[*]}"

while true; do
  if grep -q "ВОЛНА 1 ЗАВЕРШЕНА" runs/wave1_run.log 2>/dev/null; then
    log "волна 1 завершена"; break
  fi
  if ! pgrep -f "run_wave1.sh" >/dev/null; then
    log "процесс волны 1 не найден (продолжаем; проверь runs/wave1_run.log)"; break
  fi
  sleep 30
done

# дополнительная страховка от двух GPU-процессов одновременно
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон ещё работает — ждём 60 с"; sleep 60
done

# ---- нулевой контроль c1 (визуальные веса): ожидается побайтовая идентичность ------------
if [ -d runs/variants/c1_visual_noop/model ]; then
  log "CONTROL c1_visual_noop (smoke24): ожидается ИДЕНТИЧНЫЙ вывод baseline"
  $PY scripts/eval_gsm8k.py --model runs/variants/c1_visual_noop/model $GEN --split smoke \
      --out runs/variants/c1_visual_noop/smoke \
      --note "no-op control: масштабирование неиспользуемых визуальных весов" \
      > runs/variants/c1_visual_noop/smoke.log 2>&1
  $PY scripts/compare_runs.py --base runs/env_probe/smoke1024/results.jsonl \
      --variant runs/variants/c1_visual_noop/smoke/results.jsonl \
      --base_name baseline_smoke --variant_name c1_visual_noop \
      --report runs/variants/c1_visual_noop/compare_smoke.json | tee -a "$REPORT"
fi

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет runs/variants/$v/model"; continue; fi
  log "EVAL dev100 $v ; свободно VRAM: $(free)"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split dev \
      --out "runs/variants/$v/dev" --note "wave1b screening (гипотезы формата/дливы), dev100" \
      > "runs/variants/$v/dev.log" 2>&1
  $PY scripts/compare_runs.py --base runs/baseline_dev/results.jsonl \
      --variant "runs/variants/$v/dev/results.jsonl" \
      --base_name baseline --variant_name "$v" \
      --report "runs/variants/$v/compare_dev.json" | tee -a "$REPORT"
  $PY - <<PYEOF | tee -a "$REPORT"
import json
b=json.load(open('runs/baseline_dev/summary.json'))['summary']
v=json.load(open('runs/variants/$v/dev/summary.json'))['summary']
print('  $v: acc %.1f%% (база %.1f%%), извлечено %d/%d (база %d/%d), обрывов %d (база %d), токенов %.0f (база %.0f)'
      % (100*v['accuracy'], 100*b['accuracy'], v['answer_extracted'], v['completed'],
         b['answer_extracted'], b['completed'], v['truncated'], b['truncated'],
         v['mean_output_tokens'], b['mean_output_tokens']))
PYEOF
done

log "ВОЛНА 1B ЗАВЕРШЕНА"
