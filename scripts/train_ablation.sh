#!/usr/bin/env bash
# Ablation (Table 4): train every row M0..M8 exactly like LMS-TRD (seed 42)
# and evaluate on Rain100H. M0 is NeRD-Rain; it needs more memory, so it uses
# gradient accumulation (same effective batch of 16; set ACCUM_M0 to 2/4/8 to fit memory).
#   TRAIN_ROOT=Datasets/train/Rain13K TEST_ROOT=Datasets/test bash scripts/train_ablation.sh
set -euo pipefail
TRAIN_ROOT=${TRAIN_ROOT:-Datasets/train/Rain13K}
TEST_ROOT=${TEST_ROOT:-Datasets/test}
ROWS=${ROWS:-"M0 M1 M2 M3 M4 M5 M6 M7 M8"}
python tools/profile_complexity.py
for m in $ROWS; do
  EXTRA=""
  if [ "$m" = "M0" ] || [ "$m" = "M1" ]; then EXTRA="--accum-steps ${ACCUM_M0:-4}"; fi
  python train.py --config "configs/ablation/$m.yaml" --train-root "$TRAIN_ROOT" --seed 42 \
      --run-dir "runs/ablation_$m" $EXTRA
  python test.py --checkpoint "runs/ablation_$m/last.pth" --test-root "$TEST_ROOT" \
      --datasets Rain100H --json "results/eval_ablation_$m.json"
done
python tools/aggregate_results.py ablation --complexity results/complexity.json results/eval_ablation_M*.json
