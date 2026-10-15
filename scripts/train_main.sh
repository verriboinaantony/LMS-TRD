#!/usr/bin/env bash
# Main results (Table 2 and the multi-seed statistic in Sec. 4.3):
# train LMS-TRD with seeds 42, 123 and 456 on Rain13K and evaluate the final
# (last-epoch) checkpoint of each run on the five test sets.
#   TRAIN_ROOT=Datasets/train/Rain13K TEST_ROOT=Datasets/test bash scripts/train_main.sh
set -euo pipefail
TRAIN_ROOT=${TRAIN_ROOT:-Datasets/train/Rain13K}
TEST_ROOT=${TEST_ROOT:-Datasets/test}
SEEDS=${SEEDS:-"42 123 456"}
for s in $SEEDS; do
  python train.py --config configs/lms_trd.yaml --train-root "$TRAIN_ROOT" --seed "$s" --run-dir "runs/lms_trd_s$s"
  python test.py --checkpoint "runs/lms_trd_s$s/last.pth" --test-root "$TEST_ROOT" \
      --save-dir "results/images/lms_trd_s$s" --json "results/eval_lms_trd_s$s.json"
done
python tools/aggregate_results.py main results/eval_lms_trd_s*.json
