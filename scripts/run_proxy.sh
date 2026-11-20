#!/usr/bin/env bash
# Прокси-метрика (только диагностика/фильтр, НЕ решающая): teacher-forced NLL эталонного
# решения и вероятность финального числа. Список вариантов: runs/proxy_list.txt (пустая строка/`#` — игнор;
# строка "BASELINE" означает исходную модель).
# Запуск по разрешению: ждёт файл runs/GATE_PROXY; перед стартом проверяет, что GPU-прогонов нет.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
BASE=/home/mpiscil/weight-experiment/models/qwen3.5-0.8b
LIST=runs/proxy_list.txt
REPORT=runs/proxy_report.txt
mkdir -p runs/proxy

log() { echo "=== $(date -Is) $* ==="; }
free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; }

while [ ! -f runs/GATE_PROXY ]; do sleep 30; done
while pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
done

mapfile -t ITEMS < <(grep -vE '^\s*(#|$)' "$LIST")
log "прокси, объекты: ${ITEMS[*]}"

for v in "${ITEMS[@]}"; do
  if [ "$v" = "BASELINE" ]; then DIR="$BASE"; TAG=baseline; else DIR="runs/variants/$v/model"; TAG=$v; fi
  if [ ! -d "$DIR" ]; then log "ПРОПУСК $v: нет $DIR"; continue; fi
  log "PROXY $TAG (search100) ; свободно VRAM: $(free)"
  $PY scripts/proxy_eval.py --model "$DIR" --split search --limit 100 --max-ref-tokens 320 \
      --out "runs/proxy/${TAG}_search100" > "runs/proxy/${TAG}_search100.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ПРОКСИ $TAG завершился с rc=$rc (см. runs/proxy/${TAG}_search100.log)"; continue; fi
  $PY - <<PYEOF | tee -a "$REPORT"
import json
a=json.load(open('runs/proxy/baseline_search100/proxy_summary.json'))
b=json.load(open('runs/proxy/${TAG}_search100/proxy_summary.json'))
print('  ${TAG}: nll %.5f (база %.5f) ans_prob %.5f (база %.5f) top1all %d (база %d)' % (
  b['mean_nll_ref'], a['mean_nll_ref'], b['mean_answer_prob'], a['mean_answer_prob'],
  b['answer_top1_all_match'], a['answer_top1_all_match']))
PYEOF
done

log "ПРОКСИ ЗАВЕРШЁН"
