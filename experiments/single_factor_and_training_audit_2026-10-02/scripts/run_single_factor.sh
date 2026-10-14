#!/bin/bash
# Formal 29-cell grid. Gate and readback are required to already have passed.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/single_factor_and_training_audit_2026-10-02"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
LOG="$OUT/logs"
mkdir -p "$LOG"
echo $$ > "$LOG/single_factor.owner.pid"
rm -f "$LOG/single_factor.status"
unset MUJOCO_GL
export PYTHONPATH="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if ! "$PY" -c 'import json; assert json.load(open("'"$OUT"'/gate_result.json"))["passed"]'; then
  echo failed > "$LOG/single_factor.status"
  exit 1
fi
pids=()
for seed in 0 42 123; do
  "$PY" "$OUT/scripts/single_factor_formal.py" formal --seed "$seed" > "$LOG/formal_seed${seed}.log" 2>&1 &
  pids+=("$!")
done
echo "${pids[*]}" > "$LOG/single_factor.worker.pids"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
"$PY" - <<'PY'
import csv
from pathlib import Path
out = Path("/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/single_factor_and_training_audit_2026-10-02")
bad = []
for seed in (0, 42, 123):
    path = out / f"single_factor_seed{seed}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["factor"], row["value"], row["episode_id"]) for row in rows}
    if len(rows) != 17400 or len(keys) != 17400:
        bad.append(f"seed {seed} rows {len(rows)} unique {len(keys)}")
if bad:
    raise SystemExit("; ".join(bad))
print("17400 rows x 3")
PY
check=$?
if [ "$fail" -ne 0 ] || [ "$check" -ne 0 ]; then
  echo failed > "$LOG/single_factor.status"
  exit 1
fi
echo done > "$LOG/single_factor.status"
exit 0
