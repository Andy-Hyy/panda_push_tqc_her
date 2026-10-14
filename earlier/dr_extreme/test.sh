#!/bin/bash
cd "$(dirname "$0")"
python evaluate_dr_randomization.py \
  --model-path logs/baseline_model.zip \
  --mass-range 0.2,3.0 \
  --friction-range 0.05,2.5 \
  --damping-range 0.01,0.3 \
  --delay-range 0,10 \
  --goal-range "-0.15,-0.15,0.0;0.15,0.15,0.0" \
  --episodes 100
