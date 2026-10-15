#!/usr/bin/env python3
"""Export the derained-image output of LMS-TRD to ONNX for on-device deployment
(e.g. ONNX Runtime Mobile, or conversion for Qualcomm AI Hub / TFLite), and
check the exported graph against PyTorch with ONNX Runtime.

    python tools/export_onnx.py --checkpoint runs/lms_trd_s42/last.pth --size 256 256 \
        --out deploy/lms_trd_256.onnx

The export uses a fixed input size (batch 1). Because the Lite-INR positional
encoding depends only on the feature-map size, it is evaluated once and stored
as a constant in the exported graph.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model, LiteINR  # noqa: E402


class DerainOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0].clamp(0, 1)


def freeze_positional_encodings(model, h, w):
    """Replace each Lite-INR hash-grid lookup by its constant output for the export size."""
    with torch.no_grad():
        sizes = {'S': (h // 4, w // 4), 'M': (h // 2, w // 2)}
        for lv, inr in model.inr.items():
            if isinstance(inr, LiteINR):
                hh, ww = sizes[lv]
                from models.inr import make_coord
                pos = inr.hashgrid(make_coord((hh, ww))).t().reshape(1, -1, hh, ww)
                inr.register_buffer('frozen_pos', pos.clone())

                def fwd(feat, _inr=inr):
                    x = torch.relu(_inr.fc1_feat(feat) + _inr.fc1_pos(_inr.frozen_pos))
                    return _inr.fc3(torch.relu(_inr.fc2(x)))
                inr.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--preset', default='M8')
    ap.add_argument('--size', type=int, nargs=2, default=[256, 256])
    ap.add_argument('--out', default='deploy/lms_trd_256.onnx')
    args = ap.parse_args()
    h, w = args.size
    assert h % 16 == 0 and w % 16 == 0, 'export size must be a multiple of 16'

    torch.manual_seed(0)
    model = build_model({'preset': args.preset}).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        ref = model(x)[0].clamp(0, 1).numpy()
    freeze_positional_encodings(model, h, w)
    net = DerainOnly(model).eval()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    # torch.export-based exporter (PyTorch >= 2.5); shared weights are stored once
    torch.onnx.export(net, (x,), args.out, input_names=['rainy'], output_names=['derained'],
                      dynamo=True, optimize=True)
    size_mb = os.path.getsize(args.out) / 2 ** 20
    print(f'exported {args.out} ({size_mb:.1f} MiB, input 1x3x{h}x{w})')
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'rainy': x.numpy()})[0]
        print(f'ONNX Runtime vs PyTorch: max |diff| = {np.abs(out - ref).max():.2e}')
    except ImportError:
        print('onnxruntime not installed; skipped numerical check')


if __name__ == '__main__':
    main()
