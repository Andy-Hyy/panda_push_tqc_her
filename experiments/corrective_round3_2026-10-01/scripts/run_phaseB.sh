#!/bin/bash
# Phase B diagnostic. Prints only DONE or FAILED.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_round3_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/state_action_traces"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
status="$OUT/logs/phaseB.status"
echo running > "$status"
pids=()
seeds=()
cleanup() {
  local pid
  for pid in "${pids[@]-}"; do
    if [ -n "${pid:-}" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/phaseB_diagnose.py" --seed "$seed" --repeats 3 --threads 1 \
    --modes fresh,reused_a,reused_b,clean,clean_after_a \
    > "$OUT/logs/phaseB_seed${seed}.log" 2>&1 &
  pids+=("$!")
  seeds+=("$seed")
done
fail=0
while [ "${#pids[@]}" -gt 0 ]; do
  next_pids=()
  next_seeds=()
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    seed="${seeds[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      now=$(date +%s)
      mtime=$(stat -f %m "$OUT/logs/phaseB_seed${seed}.log")
      if [ $((now - mtime)) -gt 1500 ]; then
        echo failed > "$status"
        echo FAILED
        exit 1
      fi
      next_pids+=("$pid")
      next_seeds+=("$seed")
    else
      if ! wait "$pid"; then
        fail=1
      fi
    fi
  done
  pids=("${next_pids[@]-}")
  seeds=("${next_seeds[@]-}")
  if [ "$fail" -ne 0 ]; then
    echo failed > "$status"
    echo FAILED
    exit 1
  fi
  if [ "${#pids[@]}" -eq 0 ]; then
    break
  fi
  sleep 20
done
echo done > "$status"
trap - EXIT
echo DONE
