#!/bin/bash
# Phase C pilot: 50 frozen scenes x writers A/B/C x mass-only and joint physical.
# Delay is true zero. This is an implementation probe, not a paper rate.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_round3_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
status="$OUT/logs/phaseC_pilot.status"
echo running > "$status"
pids=()
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/phaseC_mass_implementation.py" --stage pilot --seed "$seed" \
    > "$OUT/logs/phaseC_pilot_seed${seed}.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  echo failed > "$status"
  echo FAILED
  exit 1
fi
echo done > "$status"
echo DONE
