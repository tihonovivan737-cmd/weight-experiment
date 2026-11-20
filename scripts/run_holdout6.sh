#!/usr/bin/env bash
# Финальная проверка на удержанном наборе из 400 задач (runs/splits/holdout6.json).
# Набор используется ОДИН раз: только для финальной проверки заранее опубликованного
# победителя против исходной модели и w8 (правила журнала §76.7 и план §97.8).
#
# Запуск возможен ТОЛЬКО после публикации победителя в runs/GATE_FINAL2: без этого файла
# скрипт останавливается и ничего не считает.
#
# Вызов:
#   bash scripts/run_holdout6.sh orig     — исходная модель (контрольный прогон)
#   bash scripts/run_holdout6.sh w8       — контрольная точка w8 (контрольный прогон)
#   bash scripts/run_holdout6.sh winner   — вариант, опубликованный в runs/GATE_FINAL2
#   bash scripts/run_holdout6.sh compare  — сводные сравнения трёх прогонов
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
W8DIR=runs/variants/w8_z_gate_mild_format   # каталог варианта; веса лежат в подкаталоге model
GATE=runs/GATE_FINAL2
OUT=runs/holdout6
LOG=runs/logs/holdout6.log
EXORIG=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
EXW8=296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
sha_of() {
  local d=$1 f
  f=$(find "$d" -maxdepth 2 -name '*.safetensors' -type f | head -1)
  [ -z "$f" ] && { echo ""; return 0; }
  sha256sum "$f" | cut -d' ' -f1
}

mode=${1:-}
[ -z "$mode" ] && { log "ОШИБКА: не указан режим (orig|w8|winner|compare)"; exit 1; }

# ---------- контроль публикации победителя ----------
if [ ! -f "$GATE" ]; then
  log "ОШИБКА: файла $GATE нет — победитель не опубликован, прогон на удержанном наборе запрещён"
  exit 1
fi
WINNER=$(grep -m1 -E '^[[:space:]]*выбранный_вариант:' "$GATE" | sed 's/.*:[[:space:]]*//')
WINNER_SHA=$(grep -m1 -E '^[[:space:]]*хеш_весов:' "$GATE" | sed 's/.*:[[:space:]]*//')
[ -z "$WINNER" ] || [ -z "$WINNER_SHA" ] && { log "ОШИБКА: в $GATE нет строк выбранный_вариант: или хеш_весов:"; exit 1; }
log "победитель из $GATE: $WINNER (хеш ${WINNER_SHA:0:16}…)"

# ---------- контроль целостности набора ----------
if ! spec=$($PY scripts/holdout_indices.py); then
  log "ОШИБКА: удержанный набор не прошёл проверку целостности — прогон не начинается"
  exit 1
fi
n_idx=$(echo "$spec" | $PY -c 'import sys
spec = sys.stdin.read().strip()
out = []
for part in spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        a, b = part.split("-")
        out.extend(range(int(a), int(b) + 1))
    else:
        out.append(int(part))
print(len(sorted(set(out))))')
log "набор прошёл проверку целостности, индексов: $n_idx"
[ "$n_idx" != "400" ] && { log "ОШИБКА: индексов $n_idx, ожидалось 400 — прогон не начинается"; exit 1; }

# ---------- режим сравнения ----------
if [ "$mode" = "compare" ]; then
  for t in orig w8 "$WINNER"; do
    f="$OUT/$t/results.jsonl"
    [ -s "$f" ] || { log "ОШИБКА: нет файлов $f — сравнение невозможно"; exit 1; }
    log "$t строк: $(wc -l < "$f")"
  done
  $PY scripts/compare_runs.py --base "$OUT/orig/results.jsonl" --variant "$OUT/$WINNER/results.jsonl" \
      --base_name "исходная модель" --variant_name "$WINNER" \
      --report "runs/analysis/holdout6_winner_vs_orig.txt" | tee -a "$LOG"
  $PY scripts/compare_runs.py --base "$OUT/w8/results.jsonl" --variant "$OUT/$WINNER/results.jsonl" \
      --base_name "w8" --variant_name "$WINNER" \
      --report "runs/analysis/holdout6_winner_vs_w8.txt" | tee -a "$LOG"
  $PY scripts/compare_runs.py --base "$OUT/orig/results.jsonl" --variant "$OUT/w8/results.jsonl" \
      --base_name "исходная модель" --variant_name "w8" \
      --report "runs/analysis/holdout6_w8_vs_orig.txt" | tee -a "$LOG"
  log "сравнения записаны в runs/analysis/holdout6_*.txt"
  exit 0
fi

# ---------- выбор модели ----------
case "$mode" in
  orig) MODEL="$ORIG"; LABEL="orig"
        h=$(sha_of "$ORIG"); [ "$h" = "$EXORIG" ] || { log "ОШИБКА: исходная модель $h"; exit 1; } ;;
  w8)   MODEL="$W8DIR/model"; LABEL="w8"
        h=$(sha_of "$W8"); [ "$h" = "$EXW8" ] || { log "ОШИБКА: w8 $h"; exit 1; } ;;
  winner) MODEL="runs/variants/$WINNER/model"; LABEL="$WINNER"
        h=$(sha_of "$MODEL")
        [ "$h" = "$WINNER_SHA" ] || { log "ОШИБКА: хеш весов $h не совпал с опубликованным $WINNER_SHA"; exit 1; }
        log "вариант сверен по хешу: $h" ;;
  *) log "ОШИБКА: неизвестный режим $mode"; exit 1 ;;
esac

res="$OUT/$LABEL/results.jsonl"
if [ -s "$res" ]; then
  log "результаты $LABEL уже есть ($(wc -l < "$res") строк) — повторный прогон на удержанном наборе не делается"
  exit 0
fi

ram_mb=$(free -m | awk 'NR==2{print $7}')
gpu_free=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$1==1{print $3-$2}')
ws_kb=$(df -k /home/mpiscil/weight-experiment | awk 'NR==2{print $4}')
log "пороги: память $ram_mb МБ (нужно 10950), карта 1 свободно $gpu_free МиБ (нужно 6943), рабочий том $ws_kb КБ (нужно 1780000)"
[ "$ram_mb" -lt 10950 ] && { log "ОШИБКА: память ниже порога — прогон не начинается"; exit 1; }
[ "$gpu_free" -lt 6943 ] && { log "ОШИБКА: видеокарта 1 ниже порога — прогон не начинается"; exit 1; }
[ "$ws_kb" -lt 1780000 ] && { log "ОШИБКА: рабочий том ниже порога — прогон не начинается"; exit 1; }
if pgrep -f '[e]val_gsm8k' >/dev/null; then
  log "ОШИБКА: уже идёт другой прогон — одновременные прогоны запрещены"; exit 1
fi

mkdir -p "$OUT/$LABEL"
log "ПРОГОН удержанного набора: $LABEL ($MODEL)"
$PY scripts/eval_gsm8k.py --model "$MODEL" $GEN --split custom --indices "$spec" \
    --out "$OUT/$LABEL" --note "holdout6 final check, $LABEL" >> "$OUT/$LABEL/eval.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "ОШИБКА: прогон $LABEL код $rc — набор по этой модели не израсходован (0 задач)"; exit $rc; fi
log "$LABEL готово: строк $(wc -l < "$res")"
h=$(sha_of "$ORIG"); [ "$h" = "$EXORIG" ] && log "исходная модель после: $h" || log "ОШИБКА после: $h"
h=$(sha_of "$W8"); [ "$h" = "$EXW8" ] && log "w8 после: $h" || log "ОШИБКА после: $h"
log "проверка $LABEL завершена; сводку делать режимом compare после трёх прогонов"
