#!/usr/bin/env bash
# Чёрное окно для прогона w82_wave11_head_lever_on_w8.
# Решение пользователя от 19.11.2026 (11:50):
#  - выгрузка только штатная: POST /unload с application/json;
#  - принудительное завершение движка НЕ используется как автоматический запасной путь:
#    при ошибке выгрузки сохранить точный ответ сервера и остановиться, прогон не начинается;
#  - перед прогоном проверить, что выгрузка действительно завершилась, пороги памяти
#    выполнены и другие прогоны не работают;
#  - во время прогона не допускается ни одного запроса к модели: любой запрос загрузил бы
#    большую модель на ту же видеокарту;
#  - после завершения или обычной ошибки попробовать восстановить Strata и записать результат;
#  - если восстановить не удалось — сохранить точную ошибку и команду ручного восстановления;
#  - исходная модель, w8 и runs/discarded не изменяются; официальный набор из 1319 задач
#    не используется, прогонщик пишет только dev и search.
# Диалог во время чёрного окна невозможен: эта сессия отвечает через Strata.
set -uo pipefail
cd /home/mpiscil/weight-experiment

URL=http://127.0.0.1:8080
CONF=/home/mpiscil/Strata/strata-iq2_xs.json
MODEL_NAME=qwen3.8-flash-next-iq2_xs
SNAP=runs/logs/strata_snapshot_w82.txt
LOG=runs/logs/w82_blackout.log
MANUAL=runs/logs/strata_manual_restore_w82.txt
RUN_TIMEOUT=4200
GATE_RAM=10950
GATE_GPU1=6943
GATE_WS=1780000
INITIAL_DELAY=120   # ждём, пока закончится ход диалога, иначе выгрузка отвечает 409 busy

SERVER_PID=
ENGINE_PID=
SERVER_CMD=
ENGINE_CMD=

log() { echo "$(date '+%F %Z %T') $*" | tee -a "$LOG"; }

save_fail() {
  # точная ошибка и команда ручного восстановления
  {
    echo "Чёрное окно w82 остановлено на шаге: $1"
    echo "Точный ответ сервера: $2"
    echo "Сервер: PID $SERVER_PID, командная строка: $SERVER_CMD"
    echo "Движок: PID ${ENGINE_PID:-нет}, командная строка: ${ENGINE_CMD:-нет}"
    echo "Конфиг: $CONF, модель: $MODEL_NAME, порт 8080"
    echo "Ручное восстановление (сначала убедиться, что прогон завершён и карта 1 свободна):"
    echo "  curl -s -X POST -H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8080' -d '{}' '$URL/load'"
    echo "Если сервер не отвечает, поднять его:"
    echo "  setsid nohup $SERVER_CMD >> runs/logs/strata_restart.log 2>&1 < /dev/null &"
    echo "Если движок не поднимается по /load, любой запрос к нему тоже его запускает:"
    echo "  curl -s -X POST -H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8080' -d '{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}' '$URL/v1/chat/completions'"
  } | tee -a "$LOG" | tee -a "$MANUAL"
}

json_post() {
  curl -s -m 90 -X POST -H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8080' -d '{}' "$URL$1"
}

detect_pids() {
  SERVER_PID=$(pgrep -f 'serve/server.py --engine strata' | head -1)
  ENGINE_PID=$(pgrep -f 'engine/strata --serve' | head -1)
}

cmdline() {
  local p=$1
  [ -n "$p" ] && [ -r "/proc/$p/cmdline" ] && tr '\0' ' ' < "/proc/$p/cmdline"
}

gate_check() {
  local ram gpu ws
  ram=$(free -m | awk 'NR==2{print $7}')
  ws=$(df -k /home/mpiscil/weight-experiment | awk 'NR==2{print $4}')
  gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$1==1{print $3-$2}')
  log "пороги: память $ram МБ (нужно $GATE_RAM), карта 1 свободно $gpu МиБ (нужно $GATE_GPU1), рабочий том $ws КБ (нужно $GATE_WS)"
  [ "$ram" -ge "$GATE_RAM" ] && [ "$gpu" -ge "$GATE_GPU1" ] && [ "$ws" -ge "$GATE_WS" ]
}

restore() {
  log "восстановление Strata: попытка загрузки"
  detect_pids
  if ! curl -s -m 30 "$URL/health" >/dev/null; then
    log "сервер не отвечает: поднимаем его по записанной командной строке"
    setsid nohup $SERVER_CMD >> runs/logs/strata_restart.log 2>&1 < /dev/null &
    sleep 20
    detect_pids
  fi
  local i r
  for i in 1 2 3 4 5 6; do
    r=$(json_post "/load")
    log "ответ /load: ${r:-пусто}"
    case "$r" in
      *loaded*) log "Strata загружена, диалог возобновляется"; return 0 ;;
    esac
    save_fail "восстановление, попытка $i" "${r:-пусто}"
    sleep 15
  done
  log "по /load не получилось: последний запрос, который сам запускает движок"
  r=$(curl -s -m 180 -X POST -H 'Content-Type: application/json' -H 'Origin: http://127.0.0.1:8080' -d \
    "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" \
    "$URL/v1/chat/completions")
  log "ответ на крошечный запрос: ${r:-пусто}"
  case "$r" in
    *"content"*|*"error"* ) save_fail "восстановление через запрос" "${r:-пусто}" ;;
  esac
  log "проверьте http://127.0.0.1:3080: модель поднимается на следующем запросе"
  return 0
}

trap 'restore' EXIT
trap 'log "получен сигнал, завершаем и возвращаем Strata"; exit 1' INT TERM

exec 200>runs/logs/w82_blackout.lock
flock -n 200 || { log "другая копия уже работает, прогон не начинается"; exit 1; }

detect_pids
SERVER_CMD=$(cmdline "$SERVER_PID")
ENGINE_CMD=$(cmdline "$ENGINE_PID")
log "старт чёрного окна: сервер PID ${SERVER_PID:-нет}, движок PID ${ENGINE_PID:-нет}; ждём $INITIAL_DELAY с, чтобы ход диалога закончился"
sleep "$INITIAL_DELAY"
detect_pids
SERVER_CMD=$(cmdline "$SERVER_PID")
ENGINE_CMD=$(cmdline "$ENGINE_PID")

# снимок состояния Strata до выгрузки
{
  echo "снимок для w82: $(date '+%F %Z %T')"
  echo "сервер: PID $SERVER_PID | $SERVER_CMD"
  echo "движок: PID $ENGINE_PID | $ENGINE_CMD"
  echo "конфиг: $CONF, модель: $MODEL_NAME, порт 8080"
  echo "память процесса движка: $(awk '/VmRSS/{print $2, $3}' "/proc/$ENGINE_PID/status" 2>/dev/null || echo нет)"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
} >> "$SNAP"
log "снимок записан в $SNAP"

h=$(curl -s -m 30 "$URL/health")
if [ -z "$h" ]; then save_fail "проверка состояния сервера" "пусто"; exit 1; fi
log "состояние до выгрузки: $h"
case "$h" in
  *'"loaded":false'*) log "модель уже выгружена, выгрузку пропускаем" ;;
  *'"loaded":true'*)  ;;
  *)                  log "состояние не распознано, выгрузку пропускаем" ;;
esac

r=$(json_post "/unload?wait=1")
log "ответ на выгрузку: ${r:-пусто}"
case "$r" in
  *unloaded*|*"not loaded"*) ;;
  *) save_fail "выгрузка" "${r:-пусто}"; log "выгрузку не приняли: прогон не начинается, ничего не убиваем"; exit 1 ;;
esac

# проверка, что штатная выгрузка действительно завершилась
left=60
while [ "$left" -gt 0 ] && [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; do
  sleep 3; left=$((left - 3))
done
if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
  save_fail "ожидание завершения движка" "движок PID $ENGINE_PID всё ещё работает"
  log "выгрузка не завершилась: прогон не начинается, ничего не убиваем"
  exit 1
fi
h=$(curl -s -m 30 "$URL/health")
log "состояние после выгрузки: ${h:-пусто}"
case "$h" in
  *'"loaded":false'*) ;;
  *) save_fail "проверка состояния после выгрузки" "${h:-пусто}"; exit 1 ;;
esac
gate_check || { log "после выгрузки пороги не выполнены: прогон не начинается"; exit 1; }

if pgrep -f 'modify_weights.py|eval_split.py|run_w82_disk.sh' >/dev/null; then
  log "найден другой прогон: прогон не начинается"; exit 1
fi
log "других прогонов нет, пороги выполнены, движок выгружен — начинаю прогон w82"

# наблюдатель: если кто-то запросом поднимет большую модель во время прогона, записываем это
RUNMARK=runs/logs/w82_running.marker
touch "$RUNMARK"
(
  while [ -f "$RUNMARK" ]; do
    p=$(pgrep -f 'engine/strata --serve' | head -1)
    if [ -n "$p" ]; then
      log "ВНИМАНИЕ: во время прогона поднялся движок Strata PID $p — этого быть не должно, убивать не будем, но факт записан"
    fi
    sleep 60
  done
) &
WATCH=$!

timeout "$RUN_TIMEOUT" bash scripts/run_w82_disk.sh >> "$LOG" 2>&1
code=$?
log "прогон завершился с кодом $code"
rm -f "$RUNMARK"
wait "$WATCH" 2>/dev/null

restore
exit "$code"
