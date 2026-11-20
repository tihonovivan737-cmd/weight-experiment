#!/usr/bin/env bash
# Начальные прогоны ИСХОДНОЙ модели (до любых изменений весов).
# Только физическая GPU 1: CUDA_VISIBLE_DEVICES=1 -> внутри процесса это cuda:0.
# Strata не затрагивается; перед каждым этапом печатается свободная память.
set -uo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mpiscil/weight-experiment/.venv/bin/python
M=/home/mpiscil/weight-experiment/models/qwen3.5-0.8b
GEN="--device cuda:0 --dtype bfloat16 --batch-size 1 --max-new-tokens 1024 --split"

free_mib() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader; }

for stage in "test:runs/baseline_test:baseline original model, GSM8K test 1319" \
             "search:runs/baseline_search:baseline original model, GSM8K train search 250" \
             "dev:runs/baseline_dev:baseline original model, GSM8K train dev 100"; do
  IFS=: read -r split out note <<< "$stage"
  echo "=== $(date -Is) старт $split -> $out ; свободно VRAM: $(free_mib) ==="
  "$PY" scripts/eval_gsm8k.py --model "$M" $GEN "$split" --out "$out" --note "$note"
  rc=$?
  echo "=== $(date -Is) конец $split rc=$rc ; свободно VRAM: $(free_mib) ==="
  if [ $rc -ne 0 ]; then echo "ЭТАП $split ЗАВЕРШИЛСЯ ОШИБКОЙ rc=$rc"; exit $rc; fi
done
echo "ВСЕ БАЗОВЫЕ ПРОГОНЫ ЗАВЕРШЕНЫ"
