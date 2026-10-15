#!/usr/bin/env python3
"""Measure on-device latency on a real Snapdragon phone through Qualcomm AI Hub.

Qualcomm AI Hub (https://aihub.qualcomm.com) compiles a model for a target
runtime and runs it on physical devices hosted by Qualcomm, e.g. the Samsung
Galaxy S23 family (Snapdragon 8 Gen 2). A free account and API token are needed:

    pip install qai-hub
    qai-hub configure --api_token <YOUR_TOKEN>

Then:
    python tools/export_onnx.py --checkpoint runs/lms_trd_s42/last.pth --size 256 256 \
        --out deploy/lms_trd_256.onnx
    python tools/aihub_profile.py --onnx deploy/lms_trd_256.onnx \
        --device "Samsung Galaxy S23 (Family)" --runtime tflite --compute-unit gpu \
        --json results/aihub_s23_gpu.json

AI Hub runs the compiled model `--iterations` times (default 100) after loading
and reports per-iteration timings; the median and the full summary returned by
the service are stored in the JSON together with every setting. Pre- and
post-processing are NOT included in AI Hub timings (network execution only).

Check the qai-hub documentation for the option names supported by your client
version; this script prints the raw profile summary so that nothing is lost.
"""

import argparse
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--device', default='Samsung Galaxy S23 (Family)')
    ap.add_argument('--runtime', default='tflite', help='tflite | qnn_context_binary | onnx')
    ap.add_argument('--compute-unit', default='gpu', help='npu | gpu | cpu')
    ap.add_argument('--size', type=int, nargs=2, default=[256, 256])
    ap.add_argument('--iterations', type=int, default=100)
    ap.add_argument('--json', default='results/aihub_profile.json')
    args = ap.parse_args()

    import qai_hub as hub
    device = hub.Device(args.device)
    h, w = args.size
    compile_job = hub.submit_compile_job(
        model=args.onnx, device=device, input_specs={'rainy': (1, 3, h, w)},
        options=f'--target_runtime {args.runtime} --compute_unit {args.compute_unit}')
    target = compile_job.get_target_model()
    profile_job = hub.submit_profile_job(
        model=target, device=device,
        options=f'--compute_unit {args.compute_unit} --max_profiler_iterations {args.iterations}')
    profile = profile_job.download_profile()

    summary = profile.get('execution_summary', {})
    times_us = summary.get('all_inference_times') or []
    report = {
        'device': args.device, 'runtime': args.runtime, 'compute_unit': args.compute_unit,
        'input': [1, 3, h, w], 'iterations': args.iterations,
        'compile_job': getattr(compile_job, 'job_id', None),
        'profile_job': getattr(profile_job, 'job_id', None),
        'estimated_inference_time_ms': (summary.get('estimated_inference_time') or 0) / 1000.0,
        'median_ms': statistics.median(times_us) / 1000.0 if times_us else None,
        'peak_memory_bytes': summary.get('estimated_inference_peak_memory'),
        'execution_summary': summary,
    }
    print(json.dumps({k: v for k, v in report.items() if k != 'execution_summary'}, indent=2))
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == '__main__':
    main()
