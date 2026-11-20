#!/usr/bin/env bash
# Наблюдатель за порогами для w81 (решение пользователя от 14:43: оставить порог 10950 МБ и ждать).
# Пороги не понижает, ничего не удаляет. Запускает единственный прогон только когда все три
# порога пройдены: доступно памяти 10950 МБ, временный каталог 2000000 КБ, видеокарта 1 — 6943 МиБ.
# Официальный набор из 1319 задач не используется. Полосы П81.1 записаны в журнал §81.6 до сборки.
set -uo pipefail
cd /home/mpiscil/weight-experiment
V=w81_finalrow_four_small_rows_step15_on_w8
LOG=runs/logs/w81_gate_wait.log
INTERVAL=300
MAX_CHECKS=240
mkdir -p runs/logs

if [ -f "runs/variants/$V/dev/results.jsonl" ] && [ "$(wc -l < "runs/variants/$V/dev/results.jsonl")" -ge 1 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] результаты уже есть — наблюдатель не нужен" | tee -a "$LOG"
  exit 0
fi

for i in $(seq 1 $MAX_CHECKS); do
  ram_mb=$(free -m | awk 'NR==2{print $7}')
  tmp_kb=$(df -k /tmp | awk 'NR==2{print $4}')
  gpu_free=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$1==1{print $3-$2}')
  line="[$(date '+%Y-%m-%d %H:%M:%S')] проверка $i: память $ram_mb МБ (нужно 10950); временный каталог $tmp_kb КБ (нужно 2000000); видеокарта 1 — $gpu_free МиБ (нужно 6943)"
  echo "$line" | tee -a "$LOG"
  if [ "$ram_mb" -ge 10950 ] && [ "$tmp_kb" -ge 2000000 ] && [ "$gpu_free" -ge 6943 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] пороги пройдены — запускается единственный прогон w81" | tee -a "$LOG"
    bash scripts/run_w81_tmpfs.sh | tee -a "$LOG"
    exit 0
  fi
  sleep $INTERVAL
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] окно наблюдения исчерпано — прогон не запущен, пороги не понижены" | tee -a "$LOG"
exit 0
