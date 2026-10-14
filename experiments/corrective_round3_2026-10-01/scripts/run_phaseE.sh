#!/bin/bash
# Formal P0-v3..P3-v3. Requires the resume-verify markers from run_phaseE_probe.sh.
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
status="$OUT/logs/phaseE.status"
if [ ! -f "$OUT/phaseE_resume_probe/verified_seed0.json" ] \
  || [ ! -f "$OUT/phaseE_resume_probe/verified_seed42.json" ] \
  || [ ! -f "$OUT/phaseE_resume_probe/verified_seed123.json" ]; then
  echo failed > "$status"
  echo FAILED
  exit 1
fi
echo running > "$status"
echo FORMAL_STARTED > "$OUT/logs/phaseE_gate.txt"
pids=()
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/phaseE_formal.py" --seed "$seed" --mode formal \
    > "$OUT/logs/phaseE_seed${seed}.log" 2>&1 &
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
