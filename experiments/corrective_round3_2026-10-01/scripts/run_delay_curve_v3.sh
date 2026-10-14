#!/bin/bash
# Full delay 0..9 curve. Refuses to run beside the formal P0-P3 job.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_round3_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/tmp/phaseE_py${PYTHONPATH:+:$PYTHONPATH}"
phase_e="$OUT/logs/phaseE.status"
if [ -f "$phase_e" ] && [ "$(tr -d '[:space:]' < "$phase_e")" = "running" ]; then
  echo "phase E is still running"
  echo FAILED
  exit 1
fi
status="$OUT/logs/delay_curve_v3.status"
echo running > "$status"
pids=()
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/phaseE_delay_curve.py" --seed "$seed" \
    > "$OUT/logs/delay_curve_v3_seed${seed}.log" 2>&1 &
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
