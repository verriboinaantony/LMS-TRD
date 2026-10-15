#!/usr/bin/env bash
# Complexity (Tables 3-4) and latency (Sec. 4.4) of LMS-TRD and NeRD-Rain.
set -euo pipefail
python tools/profile_complexity.py                      # params / MACs, all rows, 256x256
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  for p in M8 M0; do
    python tools/benchmark_latency.py --device cuda --preset "$p" --json "results/latency_cuda_$p.json"
  done
fi
for p in M8 M0; do
  python tools/benchmark_latency.py --device cpu --preset "$p" --json "results/latency_cpu_$p.json"
done
