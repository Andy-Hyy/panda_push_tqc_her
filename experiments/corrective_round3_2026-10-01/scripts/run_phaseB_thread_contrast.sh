#!/bin/bash
# One-variable contrast: historical reset, fresh case, phase-1-like threading.
# OMP and MKL stay at 1. OpenBLAS, vecLib, and torch intra-op threads stay at process defaults.
# Does not overwrite the threads=1 traces.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_round3_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/state_action_traces_thread_phase1like"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset OPENBLAS_NUM_THREADS
unset VECLIB_MAXIMUM_THREADS
export PYTHONUNBUFFERED=1
status="$OUT/logs/phaseB_thread_contrast.status"
echo running > "$status"
pids=()
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/phaseB_diagnose.py" --seed "$seed" --repeats 3 --threads -1 \
    --modes fresh \
    --trace-dir "$OUT/state_action_traces_thread_phase1like" \
    > "$OUT/logs/phaseB_thread_seed${seed}.log" 2>&1 &
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
