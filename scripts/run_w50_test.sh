#!/usr/bin/env bash
# w50_finalrow_rotation15_smallmargin — официальный набор GSM8K из 1319 задач.
# Запрос пользователя: оценка уже измеренного варианта, отбор новых вариантов не производится.
# Условия прогона — те же, что у baseline и w8: тот же набор индексов, тот же промпт,
# greedy (do_sample false), bfloat16, batch size 1, max_new_tokens 1024.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
PYCPU=.venv/bin/python
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
ORIG=models/qwen3.5-0.8b
W8=runs/variants/w8_z_gate_mild_format
V=w50_finalrow_rotation15_smallmargin
OUT=runs/variants/$V/test
LOG=runs/logs/w50_test_run.log
EXORIG=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
EXW8=296b2cd888606d549446f93341eda44a88e438a72ddb59f990dc782289ba6c55
EXW50=544d4eea6ab1d1532e28c134366f29720f084c94281b7b04d0c2a4c219881e70

sha_of() {
  local d=$1 f
  f=$(find "$d" -maxdepth 2 -name '*.safetensors' -type f | head -1)
  [ -z "$f" ] && { echo ""; return 0; }
  sha256sum "$f" | cut -d' ' -f1
}
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "проверка опор до прогона"
got=$(sha_of "$ORIG")
if [ "$got" = "$EXORIG" ]; then log "исходная модель: $got"; else log "ОШИБКА: исходная модель $got"; exit 1; fi
got=$(sha_of "$W8")
if [ "$got" = "$EXW8" ]; then log "w8: $got"; else log "ОШИБКА: w8 $got"; exit 1; fi
got=$(sha_of "runs/variants/$V/model")
if [ "$got" = "$EXW50" ]; then log "w50 (совпадает с manifest): $got"; else log "ОШИБКА: w50 $got не совпадает с manifest"; exit 1; fi

avail_kb=$(df -k . | awk 'NR==2{print $4}')
need_kb=150000
log "свободное место: $avail_kb КБ, нужно $need_kb КБ (только запись результатов)"
if [ "$avail_kb" -lt "$need_kb" ]; then log "ОШИБКА: свободно $avail_kb КБ — прогон не начинается, ничего не удаляется"; exit 1; fi
ram_mb=$(free -m | awk 'NR==2{print $7}')
log "доступно оперативной памяти: $ram_mb МБ"
g1free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==1{print $2}')
log "свободно памяти видеокарты на видеокарте №1: $g1free МиБ"
if [ "$g1free" -lt 4096 ]; then log "ОШИБКА: на видеокарте №1 свободно $g1free МиБ, правило требует не менее 4096 МиБ"; exit 1; fi

if [ -f "$OUT/results.jsonl" ]; then
  log "файл результатов уже есть, строк: $(wc -l < "$OUT/results.jsonl") — продолжается с того места, где остановился"
fi
mkdir -p "$OUT"

log "ПРОГОН официального набора из 1319 задач для w50"
$PY scripts/eval_gsm8k.py --model "runs/variants/$V/model" $GEN --split test \
    --out "$OUT" --note "w50 official GSM8K test 1319, evaluation only, GPU 1" \
    > "$OUT/run.log" 2>&1
rc=$?
log "код eval: $rc"
[ -f "$OUT/results.jsonl" ] && log "строк результатов: $(wc -l < "$OUT/results.jsonl")"
if [ $rc -ne 0 ]; then log "ОШИБКА: прогон прерван, см. $OUT/run.log"; exit 1; fi

log "сравнение с исходной моделью"
$PYCPU scripts/compare_runs.py --base runs/baseline_test/results.jsonl \
    --variant "$OUT/results.jsonl" --base_name orig --variant_name "$V" \
    --report "runs/analysis/w50_test_vs_orig.txt" >> "$LOG" 2>&1
log "сравнение с w8"
$PYCPU scripts/compare_runs.py --base "$W8/test/results.jsonl" \
    --variant "$OUT/results.jsonl" --base_name w8 --variant_name "$V" \
    --report "runs/analysis/w50_test_vs_w8.txt" >> "$LOG" 2>&1

log "контроль после прогона"
got=$(sha_of "$ORIG")
[ "$got" = "$EXORIG" ] && log "исходная модель после: $got" || log "ОШИБКА после: $got"
got=$(sha_of "$W8")
[ "$got" = "$EXW8" ] && log "w8 после: $got" || log "ОШИБКА после: $got"
got=$(sha_of "runs/variants/$V/model")
[ "$got" = "$EXW50" ] && log "w50 после: $got" || log "ОШИБКА после: $got"
log "ПРОГОН ЗАВЕРШЁН — см. журнал §75"
