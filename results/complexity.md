# Measured complexity (256x256, batch 1)

Counter: torch.utils.flop_counter.FlopCounterMode (FLOPs = 2 x MACs); PyTorch 2.14.1+cu130.

| ID | Configuration | Params (M) | MACs (G) | FLOPs (G) |
|---|---|---|---|---|
| M0 | NeRD-Rain (baseline) | 22.893 | 156.32 | 312.63 |
| M1 | + channel reduction 48->32 | 10.530 | 79.28 | 158.55 |
| M2 | + depth pruning [2,2,2] | 8.644 | 69.40 | 138.79 |
| M3 | + DS-FFN | 8.609 | 68.90 | 137.79 |
| M4 | + DW-MSA linear attention | 8.255 | 64.54 | 129.08 |
| M5 | + Lite-INR (hash grid) | 7.896 | 47.75 | 95.50 |
| M6 | + eBFU (gated additive) | 6.984 | 44.95 | 89.90 |
| M7 | + cross-scale sharing (SST) | 3.910 | 44.95 | 89.90 |
| M8 | + RRB (LMS-TRD) | 3.912 | 45.09 | 90.17 |

LMS-TRD vs NeRD-Rain: parameters -82.9%, MACs -71.2%.

Cross-check with ptflops (convolution and linear layers only, the profiler of the official NeRD-Rain code): NeRD-Rain 148.41 GMACs, LMS-TRD 43.76 GMACs.

## Per-module breakdown

| Module | NeRD-Rain MACs (G) | LMS-TRD MACs (G) | NeRD-Rain params (M) | LMS-TRD params (M) |
|---|---|---|---|---|
| Transformer U-Net stages | 124.81 | 42.21 | 18.634 | 3.075 |
| Embeddings, context and output convs | 3.28 | 1.51 | 0.070 | 0.033 |
| INR branch (latent upsampling + INR) | 21.98 | 1.23 | 2.156 | 0.801 |
| Cross-scale fusion (BFPU / eBFU) | 6.24 | 0.00 | 2.033 | 0.001 |
| Rain Refinement Block | 0.00 | 0.14 | 0.000 | 0.002 |
