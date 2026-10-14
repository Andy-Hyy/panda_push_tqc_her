#!/bin/bash
# Clean-reset spatial matrix. Equivalence must pass before the 14400 trajectories.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/spatial_clean_2026-10-02"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
SCRIPT="$OUT/scripts/spatial_formal.py"
LOG="$OUT/logs"
mkdir -p "$LOG"
unset PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
status="$LOG/spatial.status"

fail() {
  echo failed > "$status"
  echo FAILED
  exit 1
}

echo $$ > "$LOG/spatial.owner.pid"
echo running > "$status"
echo self-check > "$LOG/spatial.stage"
"$PY" "$SCRIPT" --mode self-check > "$LOG/self_check.log" 2>&1 || fail
echo freeze > "$LOG/spatial.stage"
"$PY" "$SCRIPT" --mode freeze > "$LOG/freeze.log" 2>&1 || fail

echo equivalence > "$LOG/spatial.stage"
pids=()
for seed in 0 42 123; do
  "$PY" "$SCRIPT" --seed "$seed" --mode equivalence > "$LOG/equivalence_seed${seed}.log" 2>&1 &
  pids+=("$!")
done
eq_fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    eq_fail=1
  fi
done
if [ "$eq_fail" -ne 0 ]; then
  fail
fi
for seed in 0 42 123; do
  "$PY" -c "import json; m=json.load(open('$OUT/equivalence_seed${seed}.json')); assert m['passed']" \
    >> "$LOG/equivalence_gate.log" 2>&1 || fail
done

echo formal > "$LOG/spatial.stage"
pids=()
for seed in 0 42 123; do
  "$PY" "$SCRIPT" --seed "$seed" --mode formal > "$LOG/formal_seed${seed}.log" 2>&1 &
  pids+=("$!")
done
formal_fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    formal_fail=1
  fi
done
if [ "$formal_fail" -ne 0 ]; then
  fail
fi

"$PY" - "$OUT" >> "$LOG/row_count.log" 2>&1 << 'PY' || fail
import csv
import sys
from pathlib import Path
out = Path(sys.argv[1])
for seed in (0, 42, 123):
    path = out / f"spatial_seed{seed}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4800:
        raise SystemExit(f"{path.name} has {len(rows)} rows")
print("4800 rows x 3")
PY

echo done > "$status"
echo DONE
