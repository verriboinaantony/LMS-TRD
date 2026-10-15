# LMS-TRD: Lightweight Multi-Scale Transformer with Rain-Refinement for Single Image Deraining

PyTorch implementation of **LMS-TRD**.

> Antony Verriboina, Debesh Kumar Shandilya, Tekumudi Divya, Navjot Singh
> Vision Intelligence Lab, Department of Information Technology, IIIT Allahabad

LMS-TRD is built from NeRD-Rain [Chen, Pan & Dong, CVPR 2024] by a sequence of
eight modifications. The repository implements **every row of the ablation study**
(M0 = NeRD-Rain … M8 = LMS-TRD) as a configuration of a single model class, so each
row can be built, profiled, trained and evaluated with the same code.

---

## 1. What is implemented (manuscript ↔ code)

| Manuscript | Code |
|---|---|
| Sec. 3.1 pipeline (3-level pyramid, cascaded U-Net stages, coarse-to-fine INR injection, fine-to-coarse feedback) | `models/lms_trd.py` → `LMSTRD.forward` |
| Sec. 3.2 Shared-Split Transformer: cross-scale weight sharing + scale-alignment LayerNorm | `LMSTRD.__init__` (`share_weights`), `LMSTRD.stage`, `LMSTRD.aligned` |
| Sec. 3.2 DW-MSA (linear attention, ℓ2-normalised feature map, low-rank Q/K) | `models/attention.py` → `DWMSA` |
| Sec. 3.2 DS-FFN (r = 4) | `models/ffn.py` → `DSFFN` |
| Sec. 3.3 Lite-INR (two-level hash grid + 3-layer MLP, width 64) | `models/inr.py` → `HashGrid2D`, `LiteINR` |
| Sec. 3.4 eBFU (gated additive fusion, grouped 1×1 gate) | `models/fusion.py` → `EBFU` |
| Sec. 3.5 Rain Refinement Block (DW3×3 → GELU → DW5×5 dil. 2 → ECA → 1×1, residual) | `models/rrb.py` → `RRB`, `ECA` |
| Sec. 3.6 objective, Eq. (5)–(8) | `training/losses.py` |
| Sec. 4.2 training recipe | `configs/lms_trd.yaml`, `training/trainer.py` |
| Sec. 4.1 metrics (Y channel, Rain13K protocol) | `utils/metrics.py`, `test.py` |
| Sec. 4.4 complexity and latency | `tools/profile_complexity.py`, `tools/benchmark_latency.py` |
| Sec. 4.5 ablation rows M0–M8 | `PRESETS` in `models/lms_trd.py`, `configs/ablation/M*.yaml` |

### LMS-TRD at a glance (preset `M8`)

* Pyramid levels S (1/4), M (1/2), X (1/1), nearest-neighbour resizing (as NeRD-Rain).
* Three-level Transformer U-Nets with widths **(32, 64, 128)**, **2 blocks per level ([2, 2, 2])**,
  heads (1, 2, 4). Level S runs stage 1, level M stages 1–2, level X stages 1–3.
* **Cross-scale sharing:** the U-Net of stage k is shared by all levels that run it
  (stage 1: S, M, X; stage 2: M, X; stage 3: X) — 3 U-Nets instead of NeRD-Rain's 6.
  Level-specific ("split"): patch embeddings, context/output convolutions, INR branch,
  fusions, and a LayerNorm before every shared stage.
* Block = LN → DW-MSA → residual, LN → DS-FFN → residual.
* DW-MSA: 1×1 Q/K/V projections + depthwise 3×3; per head d_v = C/h, d_k = d_v/2;
  φ(u) = elu(u/‖u‖₂) + 1; linear attention over all pixels; 1×1 output projection.
* DS-FFN: 1×1 (C→4C) → depthwise 3×3 → GELU → 1×1 (4C→C).
* Lite-INR: f(x) = 3×3-unfolded feature, H(x) = 2-level hash grid (resolutions 16, 64;
  table 2¹²; 2 features/level), MLP (9C+4 → 64 → 64 → 3).
* eBFU: z = σ(GConv1×1([F_a; F_b])) with C groups, F_out = F_a + z ⊙ F_b.
* RRB on the final full-resolution decoder feature (32 channels), ECA kernel k = 3.

---

## 2. Measured complexity (256 × 256 input, batch 1)

Measured with `python tools/profile_complexity.py` (PyTorch operator counter;
convolutions, linear layers and attention products; FLOPs = 2 × MACs).
Full output: `results/complexity.md`, `results/complexity.json`.

| ID | Configuration | Params (M) | MACs (G) |
|---|---|---|---|
| M0 | NeRD-Rain (baseline) | 22.893 | 156.32 |
| M1 | + channel reduction 48→32 | 10.530 | 79.28 |
| M2 | + depth pruning [2,2,2] | 8.644 | 69.40 |
| M3 | + DS-FFN | 8.609 | 68.90 |
| M4 | + DW-MSA linear attention | 8.255 | 64.54 |
| M5 | + Lite-INR (hash grid) | 7.896 | 47.75 |
| M6 | + eBFU (gated additive) | 6.984 | 44.95 |
| M7 | + cross-scale sharing (SST) | 3.910 | 44.95 |
| **M8** | **+ RRB = LMS-TRD** | **3.912** | **45.09** |

The M0 and M1 values equal the published NeRD-Rain (22.89 M, 156.3 G) and
NeRD-Rain-S (10.53 M, 79.2 G) figures. With ptflops (convolution/linear layers
only, the profiler used by the official NeRD-Rain code) NeRD-Rain measures
148.41 GMACs and LMS-TRD 43.76 GMACs.

### Equivalence with the official NeRD-Rain

```bash
git clone https://github.com/cschenxiang/NeRD-Rain.git ../NeRD-Rain
python tools/verify_nerd_rain.py --nerd-rain ../NeRD-Rain
```
loads the official weights into M0 / M1 and checks that parameter counts and all
outputs are identical (maximum difference 0). It also confirms that the official
INR evaluates its MLP four times on identical inputs whose relative-coordinate
part is zero (the four local-ensemble queries coincide because the INR is queried
on its own feature grid).

---

## 3. Installation

```bash
pip install -r requirements.txt      # Python >= 3.9, PyTorch >= 2.1
python smoke_test.py                 # ~2 min on CPU
```
`smoke_test.py` builds all presets, runs the loss and backward pass, checks that
every parameter receives a gradient, that the shared stages are executed by the
expected pyramid levels, and that activation checkpointing does not change the
gradients, then prints the measured complexity.

---

## 4. Data (Rain13K protocol)

```
Datasets/
├── train/Rain13K/{input,target}/     13,712 pairs: Rain14000 (11,200) + Rain1800 (1,800)
│                                      + Rain800 (700) + Rain12 (12)
└── test/
    ├── Test100/{input,target}/       100  (Rain800 test split)
    ├── Rain100H/{input,target}/      100
    ├── Rain100L/{input,target}/      100
    ├── Test2800/{input,target}/      2,800 (Rain14000 test split)
    └── Test1200/{input,target}/      1,200 (Rain1200)
```
This is the split distributed with MPRNet (Zamir et al., CVPR 2021). Pairs are
matched by file name (`rainy/` and `clean/` folder names are also accepted).

---

## 5. Training

```bash
python train.py --config configs/lms_trd.yaml --train-root Datasets/train/Rain13K \
    --seed 42 --run-dir runs/lms_trd_s42
```
Recipe (Sec. 4.2): Adam (β = 0.9, 0.999), learning rate 2×10⁻⁴ halved every
50 epochs, 200 epochs, batch 16, random 256 × 256 crops, horizontal flips and
90° rotations, float32.

*Memory.* One 256 × 256 training sample stores ≈5.6 GiB of activations without
checkpointing. Activation checkpointing of the Transformer blocks is therefore
enabled by default (≈0.5 GiB per sample); it changes memory and time only, not the
gradients. If a GPU still runs out of memory, use `--accum-steps 2` (or 4): the
model has no batch-dependent layers, so accumulation is equivalent to batch 16.

Each run writes `run.json` (configuration + software/hardware environment),
`log.csv` (per-epoch losses, learning rate, time, optional validation metrics)
and `last.pth` / `best.pth`. Test sets are never used for model selection; report
the `last.pth` checkpoint.

All three seeds and the evaluation:
```bash
TRAIN_ROOT=Datasets/train/Rain13K TEST_ROOT=Datasets/test bash scripts/train_main.sh
```
Ablation rows M0–M8 (each trained like LMS-TRD, evaluated on Rain100H):
```bash
TRAIN_ROOT=Datasets/train/Rain13K TEST_ROOT=Datasets/test bash scripts/train_ablation.sh
```

---

## 6. Evaluation

```bash
python test.py --checkpoint runs/lms_trd_s42/last.pth --test-root Datasets/test \
    --save-dir results/images/lms_trd_s42 --json results/eval_lms_trd_s42.json
python tools/aggregate_results.py main results/eval_lms_trd_s*.json   # mean ± std, LaTeX row
```
Protocol: full-resolution images, reflect-padded to a multiple of 16 and cropped
back; outputs clipped and quantised to 8 bits; PSNR and SSIM on the Y channel
(ITU-R BT.601), SSIM with an 11 × 11 Gaussian window (σ = 1.5) over the valid
region — the protocol of the Rain13K benchmark used for all compared methods.
The saved images can also be scored with MPRNet's MATLAB script.

---

## 7. Complexity and latency

```bash
python tools/profile_complexity.py                                   # Tables 3-4
python tools/benchmark_latency.py --device cuda --preset M8          # GPU latency
python tools/benchmark_latency.py --device cpu --threads 8 --preset M8
```
Latency protocol: batch 1, fixed input size (default 256 × 256), 20 warm-up runs,
100 timed runs, device synchronisation around each run; median (primary), mean,
standard deviation and 90th percentile are reported, for the network alone and
end-to-end (including uint8 → float conversion, padding, host↔device copies,
cropping and quantisation). Every setting and the hardware/software environment
are written to the JSON output.

### Export for on-device inference

```bash
python tools/export_onnx.py --checkpoint runs/lms_trd_s42/last.pth --size 256 256 \
    --out deploy/lms_trd_256.onnx
```
Exports the derained output at a fixed size (the Lite-INR positional encoding is
stored as a constant) and checks the ONNX graph against PyTorch with ONNX Runtime
(maximum absolute difference ≈1e-6 at 256 × 256). See `docs/on_device_protocol.md`
for measuring on a phone.

---

## 8. Repository structure

```
models/      lms_trd.py (model, presets M0-M8), attention.py, ffn.py, inr.py,
             fusion.py, rrb.py, common.py, io.py
training/    losses.py (Eq. 5-8), trainer.py
datasets/    derain_dataset.py
utils/       metrics.py, complexity.py, misc.py
tools/       profile_complexity.py, benchmark_latency.py, export_onnx.py,
             verify_nerd_rain.py, aggregate_results.py
configs/     lms_trd.yaml, ablation/M0.yaml ... M8.yaml
scripts/     train_main.sh, train_ablation.sh, profile.sh
results/     complexity.md / complexity.json (measured)
docs/        on_device_protocol.md
```

## Acknowledgements

This work was supported by CST-UP, India, under sanction no. CST/D-1396 (dated
20.10.2023) for the project "Road Object Detection during Rainy Nights for
Autonomous Vehicles". The NeRD-Rain components (M0) follow the official
implementation at https://github.com/cschenxiang/NeRD-Rain.
