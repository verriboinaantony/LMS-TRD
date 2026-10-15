# On-device latency: measurement protocol

This protocol states everything needed to reproduce a latency figure. Report a
number only together with all of the items in Section 3.

## 1. Model to deploy

```bash
python tools/export_onnx.py --checkpoint runs/lms_trd_s42/last.pth --size 256 256 \
    --out deploy/lms_trd_256.onnx
```
* fixed input 1 × 3 × 256 × 256 (batch 1), float32 graph, derained output clipped to [0, 1];
* the Lite-INR positional encoding is evaluated once and stored as a constant;
* the script checks the ONNX graph against PyTorch (max |diff| ≈ 1e-6).

## 2. Running on a phone

**Option A — Qualcomm AI Hub (no phone needed).** Physical devices hosted by
Qualcomm, e.g. Samsung Galaxy S23 family = Snapdragon 8 Gen 2:
```bash
pip install qai-hub && qai-hub configure --api_token <TOKEN>
python tools/aihub_profile.py --onnx deploy/lms_trd_256.onnx \
    --device "Samsung Galaxy S23 (Family)" --runtime tflite --compute-unit gpu \
    --json results/aihub_s23_gpu.json
```
Repeat with `--compute-unit npu` (the compiler then chooses the precision the NPU
supports; record it) and for NeRD-Rain if it compiles. Keep the AI Hub job IDs:
they are a permanent record of the measurement.

**Option B — own phone.** Convert the ONNX model for the target runtime (TFLite
/ LiteRT GPU delegate, QNN, or ONNX Runtime Mobile) and use the runtime's own
benchmark tool (e.g. TFLite `benchmark_model --graph=... --use_gpu=true
--num_runs=100 --warmup_runs=20`).

## 3. What to report

| Item | Example |
|---|---|
| Device and SoC | Samsung Galaxy S23, Snapdragon 8 Gen 2 |
| OS / runtime and version | Android 14, TFLite 2.16 (via AI Hub) |
| Compute unit and precision | GPU, FP16 (or NPU, INT8 + calibration method) |
| Input | 1 × 3 × 256 × 256 |
| Warm-up / timed runs | 20 / 100 (AI Hub: service default after load) |
| Statistic | median (and 90th percentile) per frame |
| Scope | network only (AI Hub) or end-to-end incl. pre/post-processing |
| Thermal state | device idle and cool before the run, no other load |
| Accuracy after conversion | PSNR of the converted model on Rain100H (FP16/INT8 may change results) |

## 4. Desktop GPU / CPU

```bash
python tools/benchmark_latency.py --device cuda --preset M8 --json results/latency_cuda_M8.json
python tools/benchmark_latency.py --device cuda --preset M0 --json results/latency_cuda_M0.json
python tools/benchmark_latency.py --device cpu  --threads 8 --preset M8
```
20 warm-up and 100 timed runs, synchronised timing, median/mean/std/p90 for the
network alone and end-to-end; GPU model, driver/CUDA, PyTorch version and
thread count are written to the JSON.
