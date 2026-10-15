#!/usr/bin/env python3
"""Quick self-check of the released implementation (CPU, ~2 minutes).

  1. every ablation preset M0-M8 builds and runs;
  2. LMS-TRD (M8): the composite loss and backward pass run, and EVERY parameter
     receives a gradient (no unused modules);
  3. cross-scale sharing is real: the shared stage-1 U-Net is executed by all
     three pyramid levels, stage 2 by two, stage 3 by one;
  4. activation checkpointing leaves the gradients unchanged;
  5. measured parameters / MACs of LMS-TRD at 256x256.
"""

import torch

from models import build_model, PRESETS
from training import LMSTRDLoss
from utils import complexity_report, set_seed


def main():
    set_seed(0)
    x = torch.rand(1, 3, 64, 64)

    # 1. all presets
    for key in PRESETS:
        out = build_model({'preset': key}).eval()(x)
        assert out[0].shape == x.shape and out[3].shape[-1] == 32 and out[4].shape[-1] == 16
    print(f'[ok] presets {", ".join(PRESETS)} build and run')

    # 2. loss + backward, all parameters used
    model = build_model().train()
    gt = torch.rand_like(x)
    loss, logs = LMSTRDLoss()(model(x), gt)
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not unused, f'parameters without gradient: {unused[:5]}'
    print(f'[ok] loss={logs["loss"]:.4f}; all {sum(1 for _ in model.parameters())} parameter tensors receive gradients')

    # 3. sharing: count executions of each shared stage
    calls = {k: 0 for k in model.stages}
    hooks = [model.stages[k].enc1.register_forward_hook(lambda m, i, o, k=k: calls.__setitem__(k, calls[k] + 1))
             for k in model.stages]
    with torch.no_grad():
        model.eval()(x)
    for h in hooks:
        h.remove()
    assert calls == {'1': 3, '2': 2, '3': 1}, calls
    print(f'[ok] shared stages executed per forward pass: {calls} (stage 1 by S, M, X; stage 2 by M, X)')

    # 4. checkpointing does not change gradients
    def grads(ckpt):
        set_seed(1)
        m = build_model().train().set_grad_checkpointing(ckpt)
        m.load_state_dict(model.state_dict())
        l, _ = LMSTRDLoss()(m(x), gt)
        l.backward()
        return torch.cat([p.grad.flatten() for p in m.parameters()])
    g0, g1 = grads(False), grads(True)
    rel = ((g0 - g1).norm() / g0.norm()).item()
    assert rel < 1e-5, rel
    print(f'[ok] activation checkpointing: relative gradient difference {rel:.1e}')

    # 5. complexity
    rep = complexity_report(build_model(), (1, 3, 256, 256))
    print(f'[measured] LMS-TRD: {rep["params_M"]:.3f} M parameters, {rep["gmacs"]:.2f} GMACs '
          f'({rep["gflops"]:.2f} GFLOPs) at 256x256')


if __name__ == '__main__':
    main()
