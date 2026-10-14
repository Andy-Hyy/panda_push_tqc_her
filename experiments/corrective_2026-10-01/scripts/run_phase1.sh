#!/bin/bash
# Phase 1 driver. Worker logs stay in logs/. This script prints only DONE or FAILED.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/partial"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
fail=0
pids=()
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/phase1_physical_2x2.py" --model-seed "$seed" --episodes 600 --partial-dir "$OUT/partial" > "$OUT/logs/phase1_seed${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo FAILED
  exit 1
fi
if ! "$PY" "$OUT/scripts/phase1_physical_2x2.py" --summarize --episodes 600 --partial-dir "$OUT/partial" > "$OUT/logs/summarize.log" 2>&1; then
  echo FAILED
  exit 1
fi
echo DONE
