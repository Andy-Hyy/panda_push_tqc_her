#!/bin/bash
# Sequential formal training, then shared-distribution evaluation.
# A completed model is skipped. An incomplete one is renamed and started over.
set -euo pipefail

OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/spatial_training_controlled_2026-10-02"
SCRIPTS="$OUT/scripts"
PANDA="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final"
PY="$PANDA/.venv/bin/python"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
unset MUJOCO_GL
export PYTHONPATH="$SCRIPTS:$PANDA"
cd "$SCRIPTS"

"$PY" - <<'PY'
import json
from pathlib import Path
import controlled_common as common
gate = json.loads((common.OUT / "gate_result.json").read_text())
if not gate.get("passed"):
    raise SystemExit("preflight gate is not passed")
common.assert_protocol_code()
PY

STATUS="$LOGS/formal.status"
echo "RUNNING $(date +%Y-%m-%dT%H:%M:%S)" > "$STATUS"
trap 'echo "FAILED exit $? $(date +%Y-%m-%dT%H:%M:%S)" >> "$STATUS"' ERR

for method in spatial_dr fixed_large; do
  for seed in 0 42 123; do
    echo "RUNNING train ${method} ${seed} $(date +%Y-%m-%dT%H:%M:%S)" >> "$STATUS"
    caffeinate -i "$PY" -c "import train_controlled; train_controlled.run_training('${method}', ${seed})" \
      > "$LOGS/train_${method}_seed${seed}.log" 2>&1
    echo "DONE train ${method} ${seed} $(date +%Y-%m-%dT%H:%M:%S)" >> "$STATUS"
  done
done

for phase in validation final; do
  for method in spatial_dr fixed_large; do
    for seed in 0 42 123; do
      echo "RUNNING eval ${phase} ${method} ${seed} $(date +%Y-%m-%dT%H:%M:%S)" >> "$STATUS"
      caffeinate -i "$PY" -c "import eval_controlled; eval_controlled.run_phase('${method}', ${seed}, '${phase}')" \
        > "$LOGS/eval_${phase}_${method}_seed${seed}.log" 2>&1
      echo "DONE eval ${phase} ${method} ${seed} $(date +%Y-%m-%dT%H:%M:%S)" >> "$STATUS"
    done
  done
done

echo "RUNNING audit $(date +%Y-%m-%dT%H:%M:%S)" >> "$STATUS"
"$PY" -c "import eval_controlled; print(eval_controlled.audit())" > "$LOGS/audit.log" 2>&1
echo "DONE all $(date +%Y-%m-%dT%H:%M:%S)" > "$STATUS"
