#!/usr/bin/env bash
# Волна 13 (§43, §44): поворот logit-строк конкурентов в плоскости (строка, FINAL)
# с сохранением L2-нормы. Один рычаг, только tilt+renorm, шаг 1,5 % нормы строки.
# Два плеча знаком: w50 (s=+0,015, пререгистрированный) и w51 (s=-0,015, исправленный
# измерением §44.2). Основа — контрольная точка w8, не перезаписывается.
# Оценка только на разрешённых наборах search250 + dev100; официальный test не участвует.
# Решают заранее зарегистрированные коридоры §43.3 — см. scripts/wave13_bands.py.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
LOG=runs/wave13_run.log
REPORT=runs/wave13_report.txt
MIN_FREE_MIB=4000
MIN_FREE_DISK_G=9
MIN_FREE_RAM_G=6
DEV_SCREEN_LOSS=4
VARIANTS=(w51_finalrow_rotation15_smallmargin_negsign w50_finalrow_rotation15_smallmargin)
PY=python3          # системный python3: transformers (для прогонов); .venv его не имеет
PYCPU=.venv/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }
# Экран диска: df -G в этой системе не поддерживался (invalid option -- 'G'),
# поэтому считаем MiB доступных через df -m и переводим в GiB.
freeg() {
  local m
  m=$(df -m . | tail -1 | awk '{print $4}')
  [ -n "$m" ] && echo $((m / 1024)) || echo 0
}
freeram() { free -g | awk 'NR==2{print $7}'; }

base_of() { echo "$W8/model"; }   # конвенция run_wave11.sh: основа — подкаталог model

# 1. экраны ресурсов: GPU 1 (физическая карта), диск, RAM. При нехватке — остановка.
f=$(free1); g=$(freeg); r=$(freeram)
log "ресурсы: GPU 1 свободно ${f} MiB; диск ${g}G; RAM ${r}Gi"
if [ "${f:-0}" -lt "$MIN_FREE_MIB" ]; then log "ОСТАНОВКА: на GPU 1 свободно ${f} MiB (< ${MIN_FREE_MIB})"; exit 1; fi
if [ "${g:-0}" -lt "$MIN_FREE_DISK_G" ]; then log "ОСТАНОВКА: диск ${g}G (< ${MIN_FREE_DISK_G}G)"; exit 1; fi
if [ "${r:-0}" -lt "$MIN_FREE_RAM_G" ]; then log "ОСТАНОВКА: доступно RAM ${r} Gi (< ${MIN_FREE_RAM_G} Gi)"; exit 1; fi

# 1a. контроль целостности контрольной точки и исходной модели (только чтение)
# Авторитетный источник — хеш самого файла шарда. Ключи манифеста:
# base_safetensors_sha256 и new_safetensors_sha256 (не "sha256"), поэтому
# сравниваем оба: фактический хеш файла и значение манифеста.
sha_of() {
  local d=$1 f
  f=$(find "$d" -maxdepth 2 -name '*.safetensors' -type f | head -1)
  if [ -z "$f" ]; then echo ""; return 0; fi
  sha256sum "$f" | cut -d' ' -f1
}
EXORIG=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
EXW8=296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55
got=$(sha_of "$ORIG"); [ "$got" = "$EXORIG" ] && log "SHA исходной модели подтверждён ($got)" || { log "ОСТАНОВКА: SHA исходной модели $got (ожидалось $EXORIG)"; exit 1; }
got=$(sha_of "$W8"); [ "$got" = "$EXW8" ] && log "SHA контрольной точки w8 подтверждён ($got)" || { log "ОСТАНОВКА: SHA w8 $got (ожидалось $EXW8)"; exit 1; }

# 1b. сборка (детерминированные байтовые правки, Recipes построчные)
for v in "${VARIANTS[@]}"; do
  b=$(base_of "$v")
  if [ -f "runs/variants/$v/manifest.json" ]; then log "сборка $v уже есть — пропускаю"; continue; fi
  if [ ! -d "$b" ]; then log "ОСТАНОВКА: нет основы $b для $v"; exit 1; fi
  log "BUILD $v (основа: $b)"
  $PYCPU scripts/modify_weights.py --recipe "recipes/$v.json" --base "$b" \
      --out "runs/variants/$v" >> runs/build_wave13.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "ОСТАНОВКА: build $v код $rc (runs/build_wave13.log)"; exit 1; fi
  log "собран $v"
done

# 1c. побайтовая сверка. Рецепты волны 13 — по одной операции на уникальную строку,
#     многопроходовых тензоров нет, поэтому обычная сверка корректна.
for v in "${VARIANTS[@]}"; do
  if [ -f "runs/variants/$v/manifest.json" ]; then
    if $PYCPU scripts/verify_variant.py "$v" > "runs/verify_${v}.txt" 2>&1; then
      log "VERIFY $v: $(grep -m1 'ИТОГ' "runs/verify_${v}.txt" || echo 'нет строки ИТОГ')"
    else
      log "VERIFY $v: $(grep -m1 'ИТОГ' "runs/verify_${v}.txt" || echo 'нет строки ИТОГ') — переходу к lineage-сверке"
      $PYCPU scripts/verify_wave12.py "recipes/${v}.json" "runs/variants/$v" > "runs/verify_${v}_lineage.txt" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        log "ОСТАНОВКА: lineage-сверка $v код $rc — см. runs/verify_${v}_lineage.txt"; exit 1
      fi
      log "LINEAGE $v: $(grep -m1 'ИТОГ' "runs/verify_${v}_lineage.txt")"
    fi
  fi
done

# 2. ожидание чужих прогонов (один прогон за раз, GPU 1). Собственный PID исключён.
busy() { pgrep -f "(eval_gsm8k|proxy_eval|probe_layers|final_logit_probe|final_logit_rows)\.py" >/dev/null; }
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
  local v=$1 sp=$2 min=$3 note=$4 f g
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
  $PYCPU scripts/compare_runs.py --base "$base" \
      --variant "runs/variants/$v/$sp/results.jsonl" \
      --base_name "$lab" --variant_name "$v" --report "$rep" | tee -a "$REPORT"
}

dev_delta() {
  $PYCPU - "$1" <<'PYX'
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
  eval_split "$v" dev 100 "wave13 final-row rotation, sign-corrected arm, dev100" || exit 1
  cmp_pair "$v" dev baseline runs/baseline_dev/results.jsonl "runs/variants/$v/compare_dev.json"
  cmp_pair "$v" dev w8       "$W8/dev/results.jsonl"         "runs/variants/$v/compare_dev_vs_w8.json"
  dd=$(dev_delta "$v")
  log "экран dev100 $v: Δверных против w8 = ${dd}"
  if [ "${dd:-0}" -le -$((DEV_SCREEN_LOSS + 1)) ]; then
    log "ОТСЕВ $v: потеря ${dd} верных на dev100 (порог ${DEV_SCREEN_LOSS}) — search250 не запускается"
    continue
  fi
  eval_split "$v" search 250 "wave13 final-row rotation, sign-corrected arm, search250" || exit 1
  cmp_pair "$v" search baseline runs/baseline_search/results.jsonl "runs/variants/$v/compare_search.json"
  cmp_pair "$v" search w8       "$W8/search/results.jsonl"         "runs/variants/$v/compare_search_vs_w8.json"
  $PYCPU scripts/combined_score.py --variant "$v" --report "runs/variants/$v/combined.json" | tee -a "$REPORT"
  log "вариант $v оценён"
done

if $PYCPU scripts/wave13_table.py > runs/wave13_table.stdout 2>&1; then
  log "сводная таблица: runs/wave13_table.txt и runs/analysis/wave13_table.json"
else
  log "ОШИБКА wave13_table.py — см. runs/wave13_table.stdout"; exit 1; fi
# Решающая проверка — заранее зарегистрированные коридоры §43.3 (формат, обрывы,
# математика, преждевременный FINAL), они оцениваются раздельно.
if $PYCPU scripts/wave13_bands.py > runs/wave13_bands.stdout 2>&1; then
  log "коридоры §43.3: runs/wave13_bands.txt"
  cat runs/wave13_bands.txt | tee -a "$REPORT"
else
  log "ОШИБКА wave13_bands.py — см. runs/wave13_bands.stdout"; exit 1
fi
# Гейт применяется справочно: §43.4 заранее оговаривает, что рычаг формата
# с высокой вероятностью не пройдёт требование Δмат >= +2.
$PYCPU scripts/pick_final2.py --table runs/analysis/wave13_table.json \
    --out runs/selection_wave13.txt | tee -a "$REPORT"
log "ВОЛНА 13 ЗАВЕРШЕНА"
