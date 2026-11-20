#!/usr/bin/env bash
# Волна 12 (пред-регистрация — runs/JOURNAL.md раздел 40, записан ДО сборки и до прогонов).
# Один вариант: w49_z_late094_qknorm108 = объединение операций w8 (in_proj_z x0,93 по 18 LIN-слоям
# + строки embed 95429/64697 x1,2) + w36 (in_proj_z x0,94 по поздним слоям 14,16,17,18,20,21,22)
# + w30 (q_norm/k_norm x1,08 по 6 FULL-слоям). Порядок ops в рецепте сохраняет побайтовое
# совпадение со стеком w8 -> w36 -> w30.
# Основа сборки — ИСХОДНАЯ модель (models/qwen3.5-0.8b). Исходная модель и w8 читаются только
# как основа и референсы, никогда не перезаписываются. Контроль w50 из §39.8 побайтово равен w49
# (§40.4), поэтому отдельного прогона нет; компоненты w30 и w36 уже измерены на тех же 350 строках.
# Отбор: только 250 search + 100 dev (dev100 первым и служит экраном затрат).
# Официальный test (1319 задач) не запускается; финальная проверка победителя — только
# независимый заранее выделенный набор runs/splits/holdout6.json.
# Настройки идентичны предыдущим прогонам: bfloat16, batch 1, max_new_tokens 1024, GPU 1.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
LOG=runs/wave12_run.log
REPORT=runs/wave12_report.txt
MIN_FREE_MIB=4000
MIN_FREE_DISK_G=9
MIN_FREE_RAM_G=6      # §24 п.10: при нехватке RAM запуск не начинается
DEV_SCREEN_LOSS=4   # §40.5: потеря больше 4 верных на dev100 против w8 — search250 не запускается
VARIANTS=(w49_z_late094_qknorm108)

base_of() { echo "$ORIG"; }   # исправление против run_wave11.sh: основа волны 12 — исходная модель

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }
freeg() { df -BG --output=avail . | tail -1 | tr -d ' ' | tr -d 'G'; }
freeram() { free -g | awk 'NR==2{print "Mem: total="$2" used="$3" free="$4" available="$7" (GiB)"}'; }

: > "$REPORT"
log "волна 12, вариант: ${VARIANTS[*]}"

# 0. ресурсы перед сборкой (одна сборка ~1,75 ГБ)
g=$(freeg)
log "диск: свободно ${g}G"
if [ "${g:-0}" -lt "$MIN_FREE_DISK_G" ]; then
  log "ОСТАНОВКА: свободно ${g}G (< ${MIN_FREE_DISK_G}G) — сборки не начинаются"; exit 1
fi
log "RAM: $(freeram)"
ram_avail=$(awk '/^Mem:/{print $7}' <<< "$(free -g)")
if [ "${ram_avail:-0}" -lt "$MIN_FREE_RAM_G" ]; then
  log "ОСТАНОВКА: доступно RAM ${ram_avail} Gi (< ${MIN_FREE_RAM_G} Gi) — прогон не начинается"; exit 1
fi
f0=$(free1)
log "VRAM GPU 1: свободно ${f0} MiB ; GPU 0 не используется"
if [ "${f0:-0}" -lt "$MIN_FREE_MIB" ]; then
  log "ОСТАНОВКА: на GPU 1 свободно ${f0} MiB (< ${MIN_FREE_MIB})"; exit 1
fi

# 1. сборка (CPU/диск, GPU не трогает); основа читается только на чтение
: > runs/build_wave12.log
for v in "${VARIANTS[@]}"; do
  b=$(base_of "$v")
  if [ -f "runs/variants/$v/manifest.json" ]; then log "сборка $v уже есть — пропускаю"; continue; fi
  if [ ! -d "$b" ]; then log "ОСТАНОВКА: нет основы $b для $v"; exit 1; fi
  log "BUILD $v (основа: $b)"
  $PY scripts/modify_weights.py --recipe "recipes/$v.json" --base "$b" \
      --out "runs/variants/$v" >> runs/build_wave12.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: build $v код $rc (runs/build_wave12.log)"; exit 1; fi
  log "собран $v"
done

# 1b. побайтовая сверка сборки. Диапазоны считаются по экземплярам операций:
#     18 (in_proj_z, op0) + 7 (in_proj_z, op2) + 2 строки embed_tokens + 6 q_norm + 6 k_norm
#     = 39 диапазонов; различные затронутые тензоры = 18 + 1 + 6 + 6 = 31.
#     Byte-сумма по экземплярам операций: 25 x 4194304 + 2 x 2048 + 12 x 512 = 104867840 —
#     для 7 многопроходовых тензоров она считает диапазон дважды, отсюда расхождение
#     с "уникальными" байтами (52433920 значений x 2 = 104867840). Ожидание verify_variant
#     для многопроходовых диапазонов не выполнимо, поэтому решающая сверка — lineage.
for v in "${VARIANTS[@]}"; do
  if $PY scripts/verify_variant.py "$v" > "runs/verify_${v}.txt" 2>&1; then
    log "VERIFY $v: $(grep -m1 'ИТОГ' "runs/verify_${v}.txt" || echo 'нет строки ИТОГ')"
  else
    # Обычная сверка сравнивает байты с хешами по одной операции на диапазон.
    # Здесь 7 тензоров in_proj_z тронуты двумя операциями подряд (наследие w8
    # lam=0.93 по всем LIN-слоям, затем lam=0.94 по поздним слоям), поэтому
    # ожидание первого прохода устаревает: это особенность учёта, не повреждение.
    # Решающая сверка — lineage: пересчёт ожидаемых байтов по порядку операций.
    log "VERIFY $v: $(grep -m1 'ИТОГ' "runs/verify_${v}.txt" || echo 'нет строки ИТОГ') — переходу к lineage-сверке"
    $PY scripts/verify_wave12.py "recipes/${v}.json" "runs/variants/$v" > "runs/verify_${v}_lineage.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      log "ОСТАНОВКА: lineage-сверка $v код $rc — см. runs/verify_${v}_lineage.txt"; exit 1
    fi
    log "LINEAGE $v: $(grep -m1 'ИТОГ' "runs/verify_${v}_lineage.txt") — многопроходовые тензоры сверены по порядку операций"
  fi
done

# 2. ожидание чужих прогонов (один прогон за раз, GPU 1). Собственный PID исключён.
busy() { pgrep -f "^$PY .*(eval_gsm8k|proxy_eval|probe_layers)\.py" >/dev/null; }
other_wave() {
  local p
  for p in $(pgrep -f "^bash scripts/run_wave[0-9]+\.sh$"); do
    [ "$p" != "$$" ] && return 0
  done
  return 1
}
other_probe() {
  local p
  for p in $(pgrep -f "scripts/probe_wave6\.sh"); do
    [ "$p" != "$$" ] && return 0
  done
  return 1
}
while busy || other_wave || other_probe; do
  log "другой прогон работает — ждём 60 с"; sleep 60
done

eval_split() {  # variant  split  min_lines  note
  local v=$1 sp=$2 min=$3 note=$4 f
  if [ -f "runs/variants/$v/$sp/results.jsonl" ] && [ "$(wc -l < "runs/variants/$v/$sp/results.jsonl")" -ge "$min" ]; then
    log "$sp для $v уже есть ($(wc -l < "runs/variants/$v/$sp/results.jsonl") строк) — пропускаю прогон"; return 0
  fi
  f=$(free1)
  if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then
    log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB})"; return 1
  fi
  g=$(freeg)
  if [ "${g:-0}" -lt "$MIN_FREE_DISK_G" ]; then
    log "ОСТАНОВКА: диск ${g}G (< ${MIN_FREE_DISK_G}G) — прогон не начинается"; return 1
  fi
  log "EVAL $sp $v ; VRAM GPU 1: ${f} MiB ; диск ${g}G"
  $PY scripts/eval_gsm8k.py --model "runs/variants/$v/model" $GEN --split "$sp" \
      --out "runs/variants/$v/$sp" --note "$note" > "runs/variants/$v/$sp.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: eval $sp $v код $rc (runs/variants/$v/$sp.log)"; return 1; fi
}

cmp_pair() {  # variant  split  base_label  base_results  report
  local v=$1 sp=$2 lab=$3 base=$4 rep=$5
  $PY scripts/compare_runs.py --base "$base" \
      --variant "runs/variants/$v/$sp/results.jsonl" \
      --base_name "$lab" --variant_name "$v" --report "$rep" | tee -a "$REPORT"
}

dev_delta() {
  $PY - "$1" <<'PYX'
import json, sys
from pathlib import Path
v = sys.argv[1]
def cnt(p):
    rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    return sum(1 for r in rows if r.get("correct"))
d = cnt(f"runs/variants/{v}/dev/results.jsonl") - cnt("runs/variants/w8_z_gate_mild_format/dev/results.jsonl")
print(d)
PYX
}

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет модели"; continue; fi
  eval_split "$v" dev 100 "wave12 w36+q_norm108 composition from original, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8       "$W8/dev/results.jsonl"         "runs/variants/$v/compare_dev_vs_w8.json"
  cmp_pair "$v" dev w30      "runs/variants/w30_qknorm_sharp108/dev/results.jsonl"  "runs/variants/$v/compare_dev_vs_w30.json"
  cmp_pair "$v" dev w36      "runs/variants/w36_z_late_extra094/dev/results.jsonl"  "runs/variants/$v/compare_dev_vs_w36.json"
  dd=$(dev_delta "$v")
  log "экран dev100 $v: Δверных против w8 = ${dd}"
  if [ "${dd:-0}" -le -$((DEV_SCREEN_LOSS + 1)) ]; then
    log "ОТСЕВ $v: потеря ${dd} верных на dev100 (порог ${DEV_SCREEN_LOSS}) — search250 не запускается"
    continue
  fi
  eval_split "$v" search 250 "wave12 w36+q_norm108 composition from original, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8       "$W8/search/results.jsonl"         "runs/variants/$v/compare_search_vs_w8.json"
  cmp_pair "$v" search w30      "runs/variants/w30_qknorm_sharp108/search/results.jsonl" "runs/variants/$v/compare_search_vs_w30.json"
  cmp_pair "$v" search w36      "runs/variants/w36_z_late_extra094/search/results.jsonl" "runs/variants/$v/compare_search_vs_w36.json"
  $PY scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

if $PY scripts/wave12_table.py > runs/wave12_table.stdout 2>&1; then
  log "сводная таблица: runs/wave12_table.txt и runs/analysis/wave12_table.json"
else
  log "ОШИБКА wave12_table.py — см. runs/wave12_table.stdout"; exit 1
fi
$PY scripts/pick_final2.py --table runs/analysis/wave12_table.json \
    --out runs/selection_wave12.txt | tee -a "$REPORT"
log "ВОЛНА 12 ЗАВЕРШЕНА"
