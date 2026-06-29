# Paper reference values

These are the numbers **reported in the manuscript**. They are reproduced here
verbatim as the targets a reviewer would check against. **None of them is
embedded in the evaluation code** — `test.py` and the profiler compute their own
numbers from the model's actual outputs. Where a measured number from this
reconstruction differs, that is documented in the main README (sections 7–10).

## Table 1 — dataset splits (Sec. 4.1)

| Dataset | Train | Test | Test name |
|---|---|---|---|
| Rain14000 [3] | 11200 | 2800 | Test2800 |
| Rain12 [18] | 12 | – | – |
| Rain1800 [1] | 1800 | – | – |
| Rain100L [1] | 200 | 100 | Rain100L |
| Rain100H [1] | 1800 | 100 | Rain100H |
| Rain1200 [4] | – | 1200 | Test1200 |
| Rain800 [19] | 700 | 100 | Test100 |

The text states **13,712** pooled training pairs over the seven datasets and five
evaluation benchmarks (Test100, Rain100L, Rain100H, Test1200, Test2800). (Note:
the per-row training counts above sum to a larger figure depending on how
overlapping Yang et al. [1] subsets are de-duplicated; the loader is dataset-
agnostic and simply reads whatever paired files are present, so reconcile the
exact pooled count with the co-authors when finalising Sec. 4.1.)

## Table 2 — quantitative results, LMS-TRD row (Sec. 4.3)

| Benchmark | PSNR (dB) | SSIM |
|---|---|---|
| Test100 | 31.66 | 0.897 |
| Rain100L | 39.86 | 0.976 |
| Rain100H | 31.66 | 0.895 |
| Test1200 | 33.41 | 0.921 |
| Test2800 | 34.67 | 0.948 |
| **Average** | **34.25** | **0.927** |

NeRD-Rain baseline average: **34.48 / 0.933** (gap 0.23 dB / 0.006).
Multi-seed (42/123/456) on Rain100H: PSNR 31.66 ± 0.04, SSIM 0.895 ± 0.001.

## Table 3 — aggregate complexity at 256×256 (Sec. 4.4)

| Method | FLOPs (G) | Params (M) | fps (Snapdragon 8 Gen 2) |
|---|---|---|---|
| NeRD-Rain | 148.41 | 22.89 | < 5 |
| **LMS-TRD** | **64.22** | **7.87** | **18** |

Claimed reductions: −57% FLOPs, −67% params. Speed: 53 fps on RTX 3090.

## Table 4 — per-module FLOPs, "estimated analytically" (Sec. 4.4)

| Module | NeRD-Rain (G) | LMS-TRD (G) |
|---|---|---|
| SST / Transformer backbone | 103.4 | 43.2 |
| Fusion unit (BFPU / eBFU) | 15.3 | 4.8 |
| INR branch | 29.7 | 14.4 |
| Rain Refinement Block | – | 1.82 |
| **Total** | **148.41** | **64.22** |

The paper explicitly notes these are *estimated analytically* from the channel
dimensions and Eq. (2), not profiled. (Also: the LMS-TRD column sums to
43.2 + 4.8 + 14.4 + 1.82 = **64.22**, and the paper separately states the SST
backbone alone is **7.4 M** parameters — both of which a reviewer can check.)

## Table 5 — ablation on Rain100H, M0→M8 (Sec. 4.5)

| ID | Configuration | Params (M) | FLOPs (G) | PSNR | SSIM |
|---|---|---|---|---|---|
| M0 | NeRD-Rain (baseline) | 22.89 | 148.41 | 31.98 | 0.901 |
| M1 | + channel reduction (48→32) | 13.62 | 98.5 | 31.76 | 0.897 |
| M2 | + depth pruning [2,2,2] | 10.80 | 83.4 | 31.65 | 0.894 |
| M3 | + DS-FFN | 10.50 | 71.9 | 31.62 | 0.893 |
| M4 | + linear attention (DW-MSA) | 10.20 | 68.4 | 31.57 | 0.892 |
| M5 | + Lite-INR (hash-grid) | 9.70 | 67.8 | 31.58 | 0.892 |
| M6 | + eBFU (gated additive) | 9.30 | 67.2 | 31.57 | 0.892 |
| M7 | + weight sharing (SST) | 7.80 | 62.4 | 31.46 | 0.890 |
| M8 | + RRB (LMS-TRD full) | 7.87 | 64.22 | 31.66 | 0.895 |

The ablation is constructed by scaling **down from NeRD-Rain** one change at a
time — relevant to the complexity discussion (README §8): if the absolute
params/FLOPs were obtained by applying reduction ratios to M0's 22.89 M /
148.41 G rather than by profiling each lightweight variant, they inherit the
baseline's scale.
