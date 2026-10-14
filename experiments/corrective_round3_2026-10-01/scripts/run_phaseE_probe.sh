#!/bin/bash
# Two-process resume probe. Does not write P*_v3 files.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_round3_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/phaseE_resume_probe"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/tmp/phaseE_py${PYTHONPATH:+:$PYTHONPATH}"

run_mode() {
  local mode="$1"
  local pids=()
  local fail=0
  for seed in 0 42 123; do
    "$PY" "$OUT/scripts/phaseE_formal.py" --seed "$seed" --mode "$mode" \
      > "$OUT/logs/phaseE_${mode}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "FAILED $mode"
    exit 1
  fi
  echo "PASSED $mode"
}

run_mode resume-write
run_mode resume-verify
echo DONE
