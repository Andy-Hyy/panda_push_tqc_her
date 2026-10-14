#!/bin/bash
# Phase 3 only. Phase 2 and Phase 4 partials are already complete.
set -u
OUT="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/data_and_analysis/corrective_2026-10-01"
PY="/Users/hejin/Documents/文稿 - 何进的Mac mini/Panda_Mujoco_Gym/panda_final/.venv/bin/python"
mkdir -p "$OUT/logs" "$OUT/partial/round2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

diag="$OUT/logs/driver_phase3_diag.log"
status="$OUT/logs/phase3.status"
: > "$diag"
echo running > "$status"

pids=()
names=()
starts=()
states=()

cleanup() {
  local pid
  if [ "${#pids[@]}" -eq 0 ]; then
    return
  fi
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_job() {
  local name="$1"
  shift
  "$PY" "$@" > "$OUT/logs/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  starts+=("$(date +%s)")
  states+=("running")
  echo "$(date +%s) start $name pid $!" >> "$diag"
}

wait_jobs() {
  local fail=0
  local n=${#pids[@]}
  local i pid name now mtime age uptime j
  while true; do
    local alive=0
    i=0
    while [ "$i" -lt "$n" ]; do
      if [ "${states[$i]}" = "running" ]; then
        pid="${pids[$i]}"
        name="${names[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
          alive=1
          now=$(date +%s)
          mtime=$(stat -f %m "$OUT/logs/${name}.log")
          age=$((now - mtime))
          uptime=$((now - ${starts[$i]}))
          if [ "$uptime" -gt 180 ] && [ "$age" -gt 1500 ]; then
            echo "$(date +%s) stall $name log_age=$age" >> "$diag"
            kill "$pid" 2>/dev/null || true
            fail=1
          fi
        else
          if wait "$pid"; then
            states[$i]="ok"
            echo "$(date +%s) ok $name" >> "$diag"
          else
            states[$i]="bad"
            fail=1
            echo "$(date +%s) bad $name" >> "$diag"
            j=0
            while [ "$j" -lt "$n" ]; do
              if [ "$j" -ne "$i" ] && [ "${states[$j]}" = "running" ]; then
                kill "${pids[$j]}" 2>/dev/null || true
              fi
              j=$((j + 1))
            done
          fi
        fi
      fi
      i=$((i + 1))
    done
    if [ "$alive" -eq 0 ]; then
      break
    fi
    sleep 30
  done
  return "$fail"
}

for seed in 0 42 123; do
  start_job "phase3_seed${seed}" "$OUT/scripts/phase3_physical_ablation.py" --model-seed "$seed" --episodes 600
done
if ! wait_jobs; then
  echo failed > "$status"
  echo FAILED
  exit 1
fi
pids=()
if ! "$PY" "$OUT/scripts/summarize_round2.py" --which ablation --episodes 600 > "$OUT/logs/summarize_ablation.log" 2>&1; then
  echo "$(date +%s) summarize ablation failed" >> "$diag"
  echo failed > "$status"
  echo FAILED
  exit 1
fi
echo "$(date +%s) summarize ablation ok" >> "$diag"
echo done > "$status"
echo DONE
