#!/usr/bin/env bash
set -uo pipefail
cd /home/mpiscil/weight-experiment
PY=.venv/bin/python
W8=runs/variants/w8_z_gate_mild_format/model
LOG=runs/build_wave6b.log
: > "$LOG"
build() {  # name  base
  local name=$1 base=$2
  if [ -f "runs/variants/$name/manifest.json" ]; then echo "skip $name (уже построен)" | tee -a "$LOG"; return 0; fi
  $PY scripts/modify_weights.py --recipe "recipes/$name.json" --base "$base" --out "runs/variants/$name" >> "$LOG" 2>&1
  echo "build rc=$? $name" | tee -a "$LOG"
}
build w24_fullgate_open085   "$W8"
build w25_z_headsel_damp088  "$W8"
build w26_norm_amp093        models/qwen3.5-0.8b
build w27_beta_sharpen_late130 "$W8"
echo DONE | tee -a "$LOG"
