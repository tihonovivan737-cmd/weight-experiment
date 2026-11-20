#!/usr/bin/env bash
set -euo pipefail
cd /home/mpiscil/weight-experiment
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
run() {
  local model="$1" tag="$2"
  if [ -f "runs/analysis/${tag}.json" ]; then echo "skip ${tag} (уже есть)"; return 0; fi
  echo "=== probe ${tag} (${model}) ==="
  $PY scripts/probe_layers.py --model "$model" --n 64 --stride 4 --max-new 256 --tag "$tag" \
    > "runs/analysis/${tag}.txt" 2> "runs/analysis/${tag}.err"
}
run models/qwen3.5-0.8b probe_base
run runs/variants/w8_z_gate_mild_format/model probe_w8
echo DONE
