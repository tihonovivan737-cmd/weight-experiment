#!/usr/bin/env bash
# Последовательная финальная проверка на удержанном наборе из 400 задач.
# Порядок: сначала выбранный вариант, затем исходная модель, затем w8, затем сравнения.
# Одновременные прогоны запрещены, повторный прогон по удержанному набору невозможен
# (проверка внутри scripts/run_holdout6.sh).
set -uo pipefail
cd /home/mpiscil/weight-experiment
LOG=runs/logs/holdout6.log
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $*" | tee -a "$LOG"; }

exec 220>runs/logs/holdout6_all.lock
flock -n 220 || { log "другая копия финальной проверки уже работает — выход"; exit 1; }

if [ ! -f runs/GATE_FINAL2 ]; then
  log "ОШИБКА: runs/GATE_FINAL2 нет — публикация победителя обязательна до прогона"
  exit 1
fi

for target in winner orig w8; do
  log "этап $target начинается"
  bash scripts/run_holdout6.sh "$target"
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ОШИБКА: этап $target завершился с кодом $rc — дальнейшие этапы не запускаются"
    exit $rc
  fi
  log "этап $target завершён"
done

log "сводные сравнения"
bash scripts/run_holdout6.sh compare || { log "ОШИБКА: сравнение не удалось"; exit 1; }
log "финальная проверка на удержанном наборе завершена"
