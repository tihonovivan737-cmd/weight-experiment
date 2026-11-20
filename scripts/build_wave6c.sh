#!/usr/bin/env bash
set -uo pipefail
cd /home/mpiscil/weight-experiment
PY=.venv/bin/python; W8=runs/variants/w8_z_gate_mild_format/model; LOG=runs/build_wave6c.log; : > "$LOG"
for pair in "w28_fullgate_close115:$W8" "w29_z_total087:$W8" "w30_qknorm_sharp108:$W8"; do
  name=${pair%%:*}; base=${pair#*:}
  if [ -f "runs/variants/$name/manifest.json" ]; then echo "skip $name" | tee -a "$LOG"; continue; fi
  $PY scripts/modify_weights.py --recipe "recipes/$name.json" --base "$base" --out "runs/variants/$name" >> "$LOG" 2>&1
  echo "build rc=$? $name" | tee -a "$LOG"
done
echo DONE | tee -a "$LOG"
