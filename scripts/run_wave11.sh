#!/usr/bin/env bash
# Волна 11: головное перераспределение гейта in_proj_z с сохранением лог-бюджета слоя
# (w45 flatten / w46 sharpen) и перераспределение taps окна causal-свёртки (w47 recency /
# w48 context). Гипотезы, дозы и пороги отбора предзарегистрированы в runs/JOURNAL.md
# (раздел 38) ДО сборок и до прогонов.
# Основа каждой сборки — сохранённая контрольная точка w8_z_gate_mild_format; исходная
# модель и w8 читаются только как основа и референс, никогда не перезаписываются.
# Отбор: только 250 search + 100 dev (dev100 идёт первым и служит экраном затрат).
# Официальный test (1319 задач) не запускается.
# Настройки идентичны предыдущим прогонам: bfloat16, batch 1, max_new_tokens 1024, GPU 1.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
W8=runs/variants/w8_z_gate_mild_format
LOG=runs/wave11_run.log
REPORT=runs/wave11_report.txt
MIN_FREE_MIB=4000
MIN_FREE_DISK_G=9
DEV_SCREEN_LOSS=4   # §38.4 п.1: потеря больше 4 верных на dev100 против w8 — search не запускается
VARIANTS=(
  w45_z_headsel_flatten
  w46_z_headsel_sharpen
  w47_conv_tap_recency
  w48_conv_tap_context
)
# основа сборки (§38.2): во всех четырёх вариантах — контрольная точка w8
base_of() { echo "$W8/model"; }

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }
freeg() { df -BG --output=avail . | tail -1 | tr -d ' ' | tr -d 'G'; }

: > "$REPORT"
log "волна 11, варианты: ${VARIANTS[*]}"

# 0. диск: каждая сборка ~1,75 ГБ, всего ~7 ГБ
g=$(freeg)
log "свободно на разделе: ${g}G"
if [ "${g:-0}" -lt "$MIN_FREE_DISK_G" ]; then
  log "ОСТАНОВКА: свободно ${g}G (< ${MIN_FREE_DISK_G}G) — сборки не начинаются"; exit 1
fi

# 1. сборки (CPU/диск, GPU не трогает); основа читается только на чтение
: > runs/build_wave11.log
for v in "${VARIANTS[@]}"; do
  b=$(base_of "$v")
  if [ -f "runs/variants/$v/manifest.json" ]; then log "сборка $v уже есть — пропускаю"; continue; fi
  if [ ! -d "$b" ]; then log "ОСТАНОВКА: нет основы $b для $v"; exit 1; fi
  log "BUILD $v (основа: $b)"
  $PY scripts/modify_weights.py --recipe "recipes/$v.json" --base "$b" \
      --out "runs/variants/$v" >> runs/build_wave11.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: build $v код $rc (runs/build_wave11.log)"; exit 1; fi
  log "собран $v"
done

# 1b. побайтовая сверка каждой сборки (ожидаемые диапазоны: 36 для w45/w46, 18 для w47/w48)
for v in "${VARIANTS[@]}"; do
  if $PY scripts/verify_variant.py "$v" > "runs/verify_${v}.txt" 2>&1; then
    log "VERIFY $v: $(grep -m1 'ИТОГ' "runs/verify_${v}.txt" || echo 'нет строки ИТОГ')"
  else
    log "ОСТАНОВКА: verify_variant $v код $? — см. runs/verify_${v}.txt"; exit 1
  fi
done

# 2. ожидание завершения чужих прогонов (один прогон за раз, GPU 1). Собственный PID исключён.
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
  log "EVAL $sp $v ; свободно VRAM на GPU 1: ${f} MiB"
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

# экранный признак §38.4 п.1: дельта верных ответов на dev100 против контрольной точки
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
  b=$(base_of "$v")
  eval_split "$v" dev 100 "wave11 head/conv redistribution on w8, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8       "$W8/dev/results.jsonl"         "runs/variants/$v/compare_dev_vs_w8.json"
  cmp_pair "$v" dev w8_z_gate_mild_format "$W8/dev/results.jsonl" "runs/variants/$v/compare_dev_vs_base.json"
  dd=$(dev_delta "$v")
  log "экран dev100 $v: Δверных против w8 = ${dd}"
  if [ "${dd:-0}" -le -$((DEV_SCREEN_LOSS + 1)) ]; then
    log "ОТСЕВ $v: потеря ${dd} верных на dev100 (порог ${DEV_SCREEN_LOSS}) — search250 не запускается"
    continue
  fi
  eval_split "$v" search 250 "wave11 head/conv redistribution on w8, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8       "$W8/search/results.jsonl"         "runs/variants/$v/compare_search_vs_w8.json"
  cmp_pair "$v" search w8_z_gate_mild_format "$W8/search/results.jsonl" "runs/variants/$v/compare_search_vs_base.json"
  $PY scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

if $PY scripts/wave11_table.py > runs/wave11_table.stdout 2>&1; then
  log "сводная таблица: runs/wave11_table.txt и runs/analysis/wave11_table.json"
else
  log "ОШИБКА wave11_table.py — см. runs/wave11_table.stdout"; exit 1
fi
$PY scripts/pick_final2.py --table runs/analysis/wave11_table.json \
    --out runs/selection_wave11.txt | tee -a "$REPORT"
log "ВОЛНА 11 ЗАВЕРШЕНА"
