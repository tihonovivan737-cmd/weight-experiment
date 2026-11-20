#!/usr/bin/env bash
# Пробные прогоны волны 10 (механизм форматной дозы).
# 1) послойные пробы вариантов w41..w44 (scripts/probe_layers.py, tag probe6);
# 2) проба логитов строки FINAL (scripts/final_logit_probe.py): один и тот же текст ответа
#    прогоняется через каждое модельное состояние;
# 3) сводный отчёт scripts/wave10_probe_report.py.
# Скрипт ждёт, пока прогон оценок освободит GPU 1, дубликаты не запускает: уже готовые
# runs/analysis/probe6_<вариант>.json пропускаются.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
PY=/home/mpiscil/weight-experiment/.venv/bin/python
LOG=runs/wave10_probe.log
MIN_FREE_MIB=4000
PROBE_MAX_NEW="${PROBE_MAX_NEW:-200}"
VARIANTS=(w41_format_embed132 w42_format_embed144 w43_format_embed144_latedose w44_mlp_branch_late105)

log() { echo "=== $(date -Is) $* ==="; }

free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1; }

busy() { pgrep -f "^$PY .*(eval_gsm8k|proxy_eval|probe_layers|final_logit_probe)\.py" >/dev/null; }

other_wave() {
  local p
  for p in $(pgrep -f "^bash scripts/run_wave[0-9]+\.sh$"); do [ "$p" != "$$" ] && return 0; done
  return 1
}

other_probe() {
  local p
  for p in $(pgrep -f "scripts/probe_wave[0-9]+\.sh"); do [ "$p" != "$$" ] && return 0; done
  for p in $(pgrep -f "^bash scripts/run_wave[0-9]+_probe\.sh$"); do [ "$p" != "$$" ] && return 0; done
  return 1
}

wait_gpu() {
  while busy || other_wave || other_probe; do
    log "другой прогон работает — ждём 60 с"
    sleep 60
  done
  local f
  f=$(free1)
  if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then
    log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB})"
    exit 1
  fi
}

for v in "${VARIANTS[@]}"; do
  out="runs/analysis/probe6_${v}.json"
  if [ -f "$out" ]; then
    log "PROBE ${v}: уже есть ${out} — пропускаю"
    continue
  fi
  wait_gpu
  log "PROBE ${v} ; свободно VRAM на GPU 1: $(free1) MiB"
  "$PY" scripts/probe_layers.py --model "runs/variants/${v}/model" --n 63 --stride 4 \
      --max-new "$PROBE_MAX_NEW" --tag probe6 --out "runs/analysis/probe6_${v}.json" \
      --device cuda:0 --dtype bfloat16 \
      > "runs/probe_${v}.log" 2>&1
  rc=$?
  log "PROBE ${v} код ${rc} -> ${out}"
  if [ "$rc" -ne 0 ]; then
    log "ОСТАНОВКА: проба ${v} завершилась ошибкой (см. runs/probe_${v}.log)"
    exit 1
  fi
done

if [ ! -f runs/analysis/final_logit_probe.txt ]; then
  wait_gpu
  log "FINAL LOGIT PROBE ; свободно VRAM на GPU 1: $(free1) MiB"
  "$PY" scripts/final_logit_probe.py > runs/final_logit_probe.stdout 2>&1
  rc=$?
  log "FINAL LOGIT PROBE код ${rc}"
  if [ "$rc" -ne 0 ]; then
    log "ОСТАНОВКА: проба логитов завершилась ошибкой (см. runs/final_logit_probe.stdout)"
    exit 1
  fi
else
  log "проба логитов уже есть: runs/analysis/final_logit_probe.txt"
fi

"$PY" scripts/wave10_probe_report.py > runs/wave10_probe_report.stdout 2>&1
log "PROBE REPORT код $? -> runs/analysis/wave10_probe_report.txt"
log "ПРОБЫ ВОЛНЫ 10 ЗАВЕРШЕНЫ"
