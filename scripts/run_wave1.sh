#!/usr/bin/env bash
# Волна 1: проверка конвейера + дешёвый прокси + промежуточная оценка вариантов на dev100.
# Запускается только после завершения базовых прогонов (ожидает marker в runs/baseline_run.log).
# Только физическая GPU 1 (CUDA_VISIBLE_DEVICES=1 -> cuda:0 внутри процесса).
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
BASE=/home/mpiscil/weight-experiment/models/qwen3.5-0.8b
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024"
VARIANTS=(c0_final_norm_gain v1_Alog_decay_half v2_Alog_decay_double v3_dt_bias_shift_down \
          v4_beta_amplify v5_beta_soften v6_attn_sharpen v7_attn_smooth v8_mlp_amp v9_mlp_damp \
          v10_attn_branch_amp)

log() { echo "=== $(date -Is) $* ==="; }

# ---- ожидание окончания базового прогона -------------------------------------------------
while true; do
  if grep -q "ВСЕ БАЗОВЫЕ ПРОГОНЫ ЗАВЕРШЕНЫ" runs/baseline_run.log 2>/dev/null; then
    log "базовые прогоны завершены"; break
  fi
  if ! pgrep -f "run_baseline.sh" >/dev/null && ! pgrep -f "eval_gsm8k.py --model .*qwen3.5-0.8b --split" >/dev/null; then
    log "базовый процесс не найден (продолжаем, но проверь runs/baseline_*)"; break
  fi
  sleep 30
done

free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '; }

# ---- 1. контроль C0: должно дать ИДЕНТИЧНЫЕ ответы на smoke24 ----------------------------
log "BASELINE smoke24 (max_new_tokens=1024) — эталон для контроля"
$PY scripts/eval_gsm8k.py --model "$BASE" $GEN --split smoke \
    --out runs/env_probe/smoke1024 --note "baseline original model, smoke24 (не входит в search/dev)" \
    > runs/env_probe/smoke1024.log 2>&1
log "CONTROL c0 (smoke24, ожидаем идентичные ответы)"
$PY scripts/eval_gsm8k.py --model runs/variants/c0_final_norm_gain/model $GEN --split smoke \
    --out runs/variants/c0_final_norm_gain/smoke --note "control: final norm gain, ожидание = идентичный вывод" \
    > runs/variants/c0_final_norm_gain/smoke.log 2>&1
$PY scripts/compare_runs.py --base runs/env_probe/smoke1024/results.jsonl \
    --variant runs/variants/c0_final_norm_gain/smoke/results.jsonl \
    --base_name baseline_smoke --variant_name c0_control \
    --report runs/variants/c0_final_norm_gain/compare_smoke.json | tee -a runs/wave1_report.txt

# ---- 2. прокси-метрика: база и все варианты на первых 100 индексах search ---------------
log "PROXY baseline (search100)"
$PY scripts/proxy_eval.py --model "$BASE" --split search --limit 100 --max-ref-tokens 320 \
    --out runs/proxy/baseline_search100 > runs/proxy/baseline_search100.log 2>&1
for v in "${VARIANTS[@]}"; do
  log "PROXY $v (search100) ; свободно VRAM: $(free)"
  $PY scripts/proxy_eval.py --model runs/variants/$v/model --split search --limit 100 --max-ref-tokens 320 \
      --out runs/proxy/${v}_search100 > runs/proxy/${v}_search100.log 2>&1
  $PY -c "
import json,sys
a=json.load(open('runs/proxy/baseline_search100/proxy_summary.json'))
b=json.load(open('runs/proxy/${v}_search100/proxy_summary.json'))
print('  $v: nll %.5f (база %.5f) ans_prob %.5f (база %.5f) top1all %d (база %d)' % (
  b['mean_nll_ref'],a['mean_nll_ref'],b['mean_answer_prob'],a['mean_answer_prob'],
  b['answer_top1_all_match'],a['answer_top1_all_match']))
" | tee -a runs/wave1_report.txt
done

# ---- 3. генерация на dev100 для всех вариантов, кроме контроля --------------------------
for v in "${VARIANTS[@]}"; do
  [ "$v" = "c0_final_norm_gain" ] && continue
  log "EVAL dev100 $v ; свободно VRAM: $(free)"
  $PY scripts/eval_gsm8k.py --model runs/variants/$v/model $GEN --split dev \
      --out runs/variants/$v/dev --note "wave1 screening, dev100" \
      > runs/variants/$v/dev.log 2>&1
  $PY scripts/compare_runs.py --base runs/baseline_dev/results.jsonl \
      --variant runs/variants/$v/dev/results.jsonl \
      --base_name baseline --variant_name "$v" \
      --report runs/variants/$v/compare_dev.json | tee -a runs/wave1_report.txt
done

log "ВОЛНА 1 ЗАВЕРШЕНА"
