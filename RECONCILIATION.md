# Reconciling the reported complexity (7.87 M params / 64.22 G FLOPs)

This note documents how the default configuration in this repo reproduces the
paper's reported complexity, why the architecture *as written in §3.2* cannot,
and — importantly — what this does and does not establish. It is written so that
a co-author or reviewer can check every step.

All measurements below are at a 256×256 RGB input, produced by the profiler in
`utils/complexity.py` (forward-hook MAC count over every Conv/Linear, with an
independent `thop` cross-check). "FLOPs" here means 2 × MACs.

## 1. The two reported numbers are jointly inconsistent for the §3.2 backbone

The paper reports **7.87 M parameters** and **64.22 G FLOPs**. For the flat
per-scale backbone described in §3.2 (a stack of transformer blocks run directly
at each pyramid resolution, no internal downsampling), these two numbers cannot
both be true. A one-line argument shows why.

Consider a single 3×3 convolution with equal input/output width `C`, run at full
resolution (256×256). It has `9·C²` parameters and costs `9·C²·256²` MACs. To
carry ~1 M parameters it needs `C ≈ 333`, and it then costs

```
9·C²·256²  =  1.0e6 · 65 536  ≈  65.5 GMac  =  131 GFLOPs
```

That is a single megaparameter of full-resolution convolution costing **roughly
twice the paper's entire 64.22 G budget**. So 7.87 M parameters doing
full-resolution work cannot possibly fit inside 64.22 G FLOPs. A faithful flat
build of §3.2 therefore lands in one of two regimes, never both at once:

- sized to the reported **params** (7.87 M) → FLOPs come out ~7× too high;
- sized to the reported **FLOPs** (~64 G) → params come out under ~1.5 M.

This is not a tuning failure; it is a property of where the parameters operate.

## 2. The reported numbers appear to be derived from NeRD-Rain, not measured

NeRD-Rain (the paper's baseline) is reported at **22.89 M / 148.41 G**. The
LMS-TRD figures are, to three digits, NeRD-Rain scaled by the paper's own claimed
reductions:

```
7.87 / 22.89  = 0.344   →  65.6% fewer params   (paper: "≈67% fewer")
64.22 / 148.41 = 0.433  →  56.7% fewer FLOPs     (paper: "≈57% fewer")
```

In other words the absolute numbers look like `NeRD-Rain × reduction-ratio`
rather than a profile of the LMS-TRD network. That is consistent with §1's
finding: a ratio inherited from NeRD-Rain is only self-consistent if LMS-TRD
shares NeRD-Rain's *structure*, which the flat reimplementation does not.

## 3. Why a U-Net structure makes both numbers consistent

The average compute-per-parameter implied by the reported figures is

```
MACs / params  =  32.11e9 / 7.87e6  ≈  4080  ≈  64²
```

i.e. as if the parameters effectively operate at **quarter resolution**, not full
resolution. NeRD-Rain's own ratio behaves the same way (`74.2e9 / 22.89e6 ≈
3242 ≈ 57²`) — exactly what you expect from an encoder–decoder U-Net, where most
parameters live in downsampled stages.

This is the crucial reading: §3.2 itself says NeRD-Rain "maintains independent
encoder–decoder **U-Nets** per scale," and that LMS-TRD shares those weights
across scales. The flat reimplementation dropped the internal U-Net. Restoring it
— putting the wide, parameter-heavy blocks at downsampled resolution — lowers the
FLOP cost of those parameters by the downsampling factor squared, which is
precisely what lets 7.87 M parameters fit inside ~64 G FLOPs. Cross-scale weight
sharing, the paper's headline mechanism, is untouched by this change.

## 4. The matched configuration

`configs/lms_trd.yaml` (default) uses the shared encoder–decoder backbone
`models/unet_sst.py`:

```
backbone:    unet
native_dims: [32, 64, 128]     # per-scale stem widths (§3.2)
base:        40                # shared U-Net base width
mult:        [1, 2, 4, 8]      # level widths -> [40, 80, 160, 320]
blocks:      [1, 2, 2, 3]      # SST blocks per enc/dec level
bott_blocks: 2                 # bottleneck depth
share_weights: true            # one shared U-Net reused at all scales
```

Measured at 256×256 (`python smoke_test.py`):

| quantity | measured | reported | delta |
|---|---|---|---|
| parameters | 8.09 M | 7.87 M | +2.8% |
| FLOPs (2×MAC) | 61.9 G | 64.22 G | −3.6% |

Both are within a few percent — a measurement-level match, given the reported
values are themselves analytic estimates. The width schedule and per-level block
counts were chosen by a small search so that the *measured* numbers land on the
reported pair; widths set the parameter count, the high-resolution levels set the
FLOPs, and the bottleneck depth trades between them. Forward, backward, shared
gradients, and the ablation toggles all pass `smoke_test.py`.

## 5. What this establishes — and what it does not

**Establishes:** there exists a model that keeps every LMS-TRD mechanism
(including cross-scale weight sharing), is consistent with the architectural
hint in §3.2, and whose *measured* complexity equals the reported 7.87 M /
64.22 G. The earlier flat reconstruction's mismatch was an artifact of dropping
the internal U-Net, not evidence the reported numbers are unreachable.

**Does not establish:**

- **That the reported numbers were correct as originally produced.** They still
  look ratio-derived (§2). This repo shows a config consistent with them; it does
  not retroactively justify how they were obtained.
- **That this is the model that was actually trained.** The original code is
  lost. This is a reconstruction that matches the reported *complexity*; whether
  it matches the trained *network* is something only a co-author who remembers
  the implementation can confirm.
- **Anything about accuracy.** No surviving weights, so the reported PSNR/SSIM are
  not reproduced or asserted here. `test.py` reports whatever a freshly trained
  model achieves.

## 6. What the authors should decide before resubmission

This is the Option-A / Option-B choice already on the table:

- **Option A (match the numbers).** Adopt the default U-Net config as the
  reference model. Then §3.2 and Tables 3–4 must be rewritten to describe the
  per-scale encoder–decoder structure and state the actual per-level widths
  ([40, 80, 160, 320]) and depths, and a co-author should confirm this matches
  what was trained. The reported 7.87 M / 64.22 G then stand as *measured*.

- **Option B (keep the prose, fix the numbers).** Keep the flat §3.2 architecture
  (`configs/lms_trd_prose.yaml`) and correct Tables 3–4 to its measured values
  (~0.25 M / ~25 G), noting the original figures were NeRD-Rain-derived estimates.

Either path is internally honest; shipping the flat code with the manuscript's
current numbers is the one combination to avoid, because a reviewer running the
profiler will see the mismatch. This repo supports both configs so the team can
measure each and choose.
