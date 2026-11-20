#!/usr/bin/env bash
# Зонды активаций новых вариантов (только forward-hooks, без градиентов).
# Запускается после завершения прогонов оценки: сериализация по pgrep, GPU 1.
# ВАЖНО: probe_layers.py по умолчанию пишет runs/analysis/<tag>_<имя папки модели>.json,
# а папка модели у всех вариантов называется "model" -> все зонды перезаписывали бы друг
# друга. Поэтому путь выхода задаётся явно через --out.
set -uo pipefail
cd /home/mpiscil/weight-experiment

PY=.venv/bin/python
LOG=runs/probe_wave6.log
: > "$LOG"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_idle() {
  # якорные шаблоны: сопоставляем полную командную строку, чтобы pgrep не находил
  # ни себя, ни родительский shell, который запустил этот скрипт
  while pgrep -f "(^|/)$PY .*(eval_gsm8k|proxy_eval|probe_layers)\.py" >/dev/null \
        || pgrep -f "^bash scripts/run_wave[0-9]+\.sh$" >/dev/null; do
    sleep 60
  done
}

free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }

probe() {
  local v="$1"
  local out="runs/analysis/probe6_${v}.json"
  if [ -f "$out" ]; then echo "skip $v (уже есть $out)" | tee -a "$LOG"; return 0; fi
  local f; f=$(free1)
  if [ "${f:-0}" -lt 4000 ]; then echo "МАЛО VRAM на GPU 1: $f MiB — пропуск $v" | tee -a "$LOG"; return 0; fi
  echo "=== $(date -Is) PROBE $v (свободно VRAM: $f MiB) ===" | tee -a "$LOG"
  $PY scripts/probe_layers.py --model "runs/variants/$v/model" --out "$out" \
      --n 63 --stride 4 --max-new 200 --device cuda:0 --dtype bfloat16 --tag "probe6" >> "$LOG" 2>&1
  echo "probe rc=$? $v -> $out" | tee -a "$LOG"
}

wait_idle
for v in "$@"; do
  probe "$v"
done
echo "DONE $(date -Is)" | tee -a "$LOG"
