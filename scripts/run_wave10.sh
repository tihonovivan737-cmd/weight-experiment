#!/usr/bin/env bash
# Волна 10: форматная ветка (усиление строк FINAL в embed_tokens) плюс проверка механизма
# относительной доли MLP в поздних блоках. Гипотезы и пороги предзарегистрированы в
# runs/JOURNAL.md (раздел 33) и в recipes/*.json ДО сборок и до прогонов.
# Основа каждой сборки — сохранённый вариант (w8_z_gate_mild_format или w36_z_late_extra094);
# исходная модель и w8 не перезаписываются, читаются только как основа и референс.
# Отбор: только 250 search + 100 dev. Официальный test (1319 задач) не запускается.
# Настройки идентичны предыдущим прогонам: bfloat16, batch 1, max_new_tokens 1024, GPU 1.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
W8=runs/variants/w8_z_gate_mild_format
W36=runs/variants/w36_z_late_extra094
LOG=runs/wave10_run.log
REPORT=runs/wave10_report.txt
MIN_FREE_MIB=4000
MIN_FREE_DISK_G=9
VARIANTS=(
  w41_format_embed132
  w42_format_embed144
  w43_format_embed144_latedose
  w44_mlp_branch_late105
)
# основа сборки для каждого варианта (§33.3)
base_of() {
  case "$1" in
    w43_format_embed144_latedose) echo "$W36/model" ;;
    *) echo "$W8/model" ;;
  esac
}

log() { echo "=== $(date -Is) $* ==="; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }
freeg() { df -BG --output=avail . | tail -1 | tr -d ' ' | tr -d 'G'; }

: > "$REPORT"
log "волна 10, варианты: ${VARIANTS[*]}"

# 0. диск: каждая сборка ~1,7 ГБ
g=$(freeg)
log "свободно на разделе: ${g}G"
if [ "${g:-0}" -lt "$MIN_FREE_DISK_G" ]; then
  log "ОСТАНОВКА: свободно ${g}G (< ${MIN_FREE_DISK_G}G) — сборки не начинаются"; exit 1
fi

# 1. сборки (CPU/диск, GPU не трогает); основа читается только на чтение
: > runs/build_wave10.log
for v in "${VARIANTS[@]}"; do
  b=$(base_of "$v")
  if [ -f "runs/variants/$v/manifest.json" ]; then log "сборка $v уже есть — пропускаю"; continue; fi
  if [ ! -d "$b" ]; then log "ОСТАНОВКА: нет основы $b для $v"; exit 1; fi
  log "BUILD $v (основа: $b)"
  $PY scripts/modify_weights.py --recipe "recipes/$v.json" --base "$b" \
      --out "runs/variants/$v" >> runs/build_wave10.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: build $v код $rc (runs/build_wave10.log)"; exit 1; fi
  log "собран $v"
done

# 1b. побайтовая сверка каждой сборки и замер накопленных доз по файлам
for v in "${VARIANTS[@]}"; do
  if $PY scripts/verify_variant.py "$v" > "runs/verify_${v}.txt" 2>&1; then
    log "VERIFY $v: $(grep -m1 'ИТОГ' "runs/verify_${v}.txt" || echo 'нет строки ИТОГ')"
  else
    log "ОСТАНОВКА: verify_variant $v код $? — см. runs/verify_${v}.txt"; exit 1
  fi
done
if $PY scripts/wave10_dose_ledger.py > runs/wave10_dose_ledger.stdout 2>&1; then
  log "дозовая ведомость: runs/analysis/dose_ledger_wave10.txt"
else
  log "ОСТАНОВКА: wave10_dose_ledger.py код $? — см. runs/wave10_dose_ledger.stdout"; exit 1
fi

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

for v in "${VARIANTS[@]}"; do
  if [ ! -d "runs/variants/$v/model" ]; then log "ПРОПУСК $v: нет модели"; continue; fi
  b=$(base_of "$v")
  blab=$(basename "$(dirname "$(dirname "$b")")")
  eval_split "$v" search 250 "wave10 format lever on $blab, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8       "$W8/search/results.jsonl"         "runs/variants/$v/compare_search_vs_w8.json"
  cmp_pair "$v" search "$blab"  "$b/../search/results.jsonl"       "runs/variants/$v/compare_search_vs_base.json"
  eval_split "$v" dev 100 "wave10 format lever on $blab, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8       "$W8/dev/results.jsonl"         "runs/variants/$v/compare_dev_vs_w8.json"
  cmp_pair "$v" dev "$blab"  "$b/../dev/results.jsonl"       "runs/variants/$v/compare_dev_vs_base.json"
  $PY scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

if $PY scripts/wave10_table.py > runs/wave10_table.stdout 2>&1; then
  log "сводная таблица: runs/wave10_table.txt"
else
  log "ОШИБКА wave10_table.py — см. runs/wave10_table.stdout"; exit 1
fi
$PY scripts/pick_final2.py --table runs/analysis/wave10_table.json \
    --out runs/selection_wave10.txt | tee -a "$REPORT"
log "ВОЛНА 10 ЗАВЕРШЕНА"
