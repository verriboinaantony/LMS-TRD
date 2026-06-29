#!/usr/bin/env bash
# Convenience wrapper: train then evaluate LMS-TRD.
set -e
CONFIG=${CONFIG:-configs/lms_trd.yaml}
TRAIN_ROOT=${TRAIN_ROOT:-data/train}
VAL_ROOT=${VAL_ROOT:-data/val}
TEST_ROOT=${TEST_ROOT:-data/test/Rain100H}
DEVICE=${DEVICE:-cuda}

echo ">> sanity check (build + measured complexity)"
python smoke_test.py

echo ">> training"
python train.py --config "$CONFIG" --train-root "$TRAIN_ROOT" --val-root "$VAL_ROOT" --device "$DEVICE"

echo ">> evaluation (+ measured complexity)"
python test.py --config "$CONFIG" --checkpoint checkpoints/lms_trd_best.pth \
    --test-root "$TEST_ROOT" --report-complexity --device "$DEVICE"
