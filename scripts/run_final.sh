#!/usr/bin/env bash
# Финальный прогон: выбранный вариант на официальном test (1319 задач) и сравнение с runs/baseline_test.
# Вариант берётся из runs/selection_final.txt (первая строка — имя каталога в runs/variants/).
# Запуск по разрешению: ждёт файл runs/GATE_FINAL (создаётся scripts/pick_final.py --commit).
# Только физическая GPU 1 (CUDA_VISIBLE_DEVICES=1 -> cuda:0 внутри процесса), одна задача за раз.
# При нехватке VRAM или ошибке прогона — остановиться и сообщить.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
REPORT=runs/final_report.txt
MIN_FREE_MIB=4000

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }

V=$(sed 's/#.*//' runs/selection_final.txt | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -vE '^$' | head -1)
if [ -z "$V" ]; then log "ОСТАНОВКА: runs/selection_final.txt пуст"; exit 1; fi
if [ ! -d "runs/variants/$V/model" ]; then log "ОСТАНОВКА: нет runs/variants/$V/model"; exit 1; fi
log "финальный вариант: $V"
log "SHA256 весов варианта: $(sha256sum runs/variants/$V/model/model.safetensors-00001-of-00001.safetensors | cut -c1-16)…"

# Gate создаётся scripts/pick_final.py --commit (или вручную `touch runs/GATE_FINAL`).
while true; do
  if [ -f runs/GATE_FINAL ]; then log "gate runs/GATE_FINAL создан"; break; fi
  sleep 30
done
while pgrep -f "run_wave[0-9]+.sh" >/dev/null || pgrep -f "eval_gsm8k.py" >/dev/null || pgrep -f "proxy_eval.py" >/dev/null; do
  log "другой GPU-прогон работает — ждём 60 с"; sleep 60
done

f=$(free1)
if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then
  log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB}) — не запускаю финальный прогон"
  exit 1
fi
log "EVAL test1319 $V ; свободно VRAM на GPU 1: ${f} MiB"
$PY scripts/eval_gsm8k.py --model "runs/variants/$V/model" $GEN --split test \
    --out "runs/variants/$V/test" --note "final variant on official test 1319" \
    > "runs/variants/$V/test.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval test1319 $V код $rc (runs/variants/$V/test.log)"; exit 1; fi
n=$( [ -f "runs/variants/$V/test/results.jsonl" ] && wc -l < "runs/variants/$V/test/results.jsonl" || echo 0 )
log "строк результата: $n (ожидалось 1319)"

$PY scripts/compare_runs.py --base runs/baseline_test/results.jsonl \
    --variant "runs/variants/$V/test/results.jsonl" \
    --base_name baseline --variant_name "$V" \
    --report "runs/variants/$V/compare_test.json" | tee -a "$REPORT"
$PY scripts/analyze_errors.py "runs/variants/$V/test/results.jsonl" --show 15 \
    > "runs/variants/$V/errors.txt" 2>&1
$PY scripts/flip_breakdown.py "$V" > "runs/variants/$V/flip_breakdown_test.txt" 2>&1 || true
$PY scripts/make_report.py | tee -a "$REPORT"

log "ФИНАЛЬНЫЙ ПРОГОН ЗАВЕРШЁН ($V)"
