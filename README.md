# LMS-TRD: Lightweight Multi-Scale Transformer with Rain-Refinement for Single Image Deraining

PyTorch reference implementation of **LMS-TRD**, a lightweight multi-scale Transformer for single-image deraining.

> **Antony Verriboina, Debesh Kumar Shandilya, Tekumudi Divya, Navjot Singh**
> Vision Intelligence Lab, Department of Information Technology,
> Indian Institute of Information Technology Allahabad (IIIT-Allahabad)

This repository contains a complete, runnable implementation of every module in the paper — the SST backbone, Lite-INR branch, eBFU fusion, Rain Refinement Block, the composite loss, the training pipeline, PSNR/SSIM evaluation, and a parameter/MAC/FLOP profiler.

---

## Abstract

Image deraining aims to restore clean images from single rainy inputs, where rain streaks degrade visual quality and impair downstream tasks such as object detection and semantic segmentation. While recent Transformer-based deraining models achieve strong results, their high computational cost makes them unsuitable for real-time deployment on resource-constrained devices.

LMS-TRD is a lightweight multi-scale Transformer that reduces parameters by ~67% (7.87 M) and FLOPs by ~57% (64.22 G) relative to the NeRD-Rain baseline, while maintaining competitive restoration quality (34.25 dB average PSNR across five benchmarks, versus 34.48 dB for the heavier baseline). Four co-designed mechanisms drive these savings: a **Shared-Split Transformer (SST)** backbone with cross-scale weight sharing and linear-complexity O(n) attention, a **Lite-INR** hash-grid implicit branch, a gated **Efficient Bidirectional Feedback Unit (eBFU)**, and a task-specific **Rain Refinement Block (RRB)**. The model runs at 53 fps on an RTX 3090 and 18 fps on a Snapdragon 8 Gen 2, enabling real-time deployment on mobile and embedded platforms.

---

## Architecture

A rainy image is decomposed into a three-level pyramid (quarter, half, and full resolution). Features at each scale are extracted by a single **shared** SST backbone, fused across scales by the eBFU, refined by the RRB, and combined with a rain-free proxy produced by the Lite-INR branch.

```
Rainy image
    │
    ▼
Shallow feature extraction (3×3 conv)
    │
    ├──────────────────────────────► Lite-INR branch (2-level hash-grid)
    │                                   │  rain-free RGB proxy + feature injection
    ▼                                   │
Three-level pyramid  {S1, S2, S3}       │
    │     (1/4)   (1/2)   (full)        │
    ▼                                   │
Shared-Split Transformer (SST) ◄────────┘   one shared backbone reused at all scales
    │   • DW-MSA   linear O(n) attention
    │   • DS-FFN   depthwise-separable feed-forward
    ▼
eBFU  (gated-additive bidirectional cross-scale fusion)
    │
    ▼
Rain Refinement Block (RRB)
    │   DWConv 3×3 → dilated DWConv 5×5 (d=2) → ECA channel attention → 1×1 proj
    ▼
Reconstruction (conv) + shallow skip
    │
    ▼
Derained image
```

### Components

| Module           | File                                       | Description                                                                                       |
| ---------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **SST backbone** | `models/unet_sst.py`, `models/sst_flat.py` | Shared transformer reused across all three pyramid scales (cross-scale weight sharing).           |
| **DW-MSA**       | `models/blocks.py`                         | Linear-complexity O(n) self-attention with ℓ2 feature normalization and low-rank Q/K projections. |
| **DS-FFN**       | `models/blocks.py`                         | Depthwise-separable feed-forward network (3×3 depthwise spatial mixing).                          |
| **Lite-INR**     | `models/lite_inr.py`                       | Two-level hash-grid implicit branch producing a rain-free RGB proxy (Eq. 3).                      |
| **eBFU**         | `models/ebfu.py`                           | Gated-additive bidirectional feature fusion across adjacent scales (Eq. 4).                       |
| **RRB**          | `models/rrb.py`                            | Residual artifact corrector: depthwise 3×3 → dilated depthwise 5×5 → ECA → 1×1.                   |
| **Loss**         | `training/losses.py`                       | Charbonnier + edge + frequency + INR terms (Eq. 5–8).                                             |
| **Full model**   | `models/lms_trd.py`                        | Assembles all modules; exposes ablation toggles.                                                  |

---

## Two configurations

This repo ships two backbone configurations. Both keep cross-scale weight sharing and every other mechanism; they differ only in whether the shared backbone has an internal encoder–decoder (U-Net) structure.

| Config                           | Backbone                      | Measured params | Measured FLOPs (2×MAC) | Purpose                                                |
| -------------------------------- | ----------------------------- | --------------- | ---------------------- | ------------------------------------------------------ |
| `configs/lms_trd.yaml` (default) | encoder–decoder U-Net SST     | **8.09 M**      | **61.9 G**             | Reproduces the reported 7.87 M / 64.22 G (within ~3%). |
| `configs/lms_trd_prose.yaml`     | flat SST (literal §3.2 prose) | ~0.25 M         | ~25 G                  | Faithful to the method exactly as written in the text. |

> All complexity figures above are **measured** by the profiler in `utils/complexity.py` at 256×256 input. Run `python smoke_test.py` to reproduce them. See `RECONCILIATION.md` for the architectural rationale behind the default configuration.

---

## Installation

Requires Python 3.8+ and PyTorch.

```bash
git clone <your-repo-url>
cd LMS-TRD

pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `torch>=1.10`, `numpy>=1.21`, `Pillow>=8.0`, `PyYAML>=5.4`. The `thop` package is optional and used only as an independent cross-check in the complexity profiler.

### Quick sanity check

Build the model, run a forward + backward pass, verify shared-gradient flow, and print measured complexity — all in one shot:

```bash
python smoke_test.py
```

---

## Dataset layout

Training and test data follow a paired `rainy` / `clean` folder structure; pairs are matched by filename.

```
data/
├── train/
│   ├── rainy/    *.png
│   └── clean/    *.png
├── val/
│   ├── rainy/
│   └── clean/
└── test/
    ├── Rain100L/{rainy,clean}/
    ├── Rain100H/{rainy,clean}/
    ├── Test100/{rainy,clean}/
    ├── Test1200/{rainy,clean}/
    └── Test2800/{rainy,clean}/
```

The five evaluation benchmarks are Test100, Rain100L, Rain100H, Test1200, and Test2800. Training uses 256×256 random crops with horizontal-flip and 90°-rotation augmentation.

---

## Training

```bash
python train.py \
    --config configs/lms_trd.yaml \
    --train-root data/train \
    --val-root   data/val \
    --device     cuda
```

Optional flags: `--epochs`, `--batch-size`, `--lr`, `--seed`, and `--resume <checkpoint.pth>`. Any value passed on the command line overrides the config.

**Default training recipe** (from the paper, set in the config): Adam (β1=0.9, β2=0.999), initial learning rate 2×10⁻⁴ halved every 50 epochs, batch size 16, 256×256 crops, 200 epochs. Multi-run stability is reported with seeds 42 / 123 / 456 (set `train.seed`). Checkpoints are written to `checkpoints/`, with the best model saved as `checkpoints/lms_trd_best.pth`.

---

## Testing / Evaluation

```bash
python test.py \
    --config       configs/lms_trd.yaml \
    --checkpoint   checkpoints/lms_trd_best.pth \
    --test-root    data/test/Rain100H \
    --report-complexity \
    --device       cuda
```

This reports **measured** PSNR (dB) and SSIM on the chosen benchmark. With `--report-complexity` it also prints measured parameters and MACs/FLOPs. Use `--rainy-dir` / `--clean-dir` if your subfolders are named differently from `rainy` / `clean`.

---

## Inference on your own images

```bash
# single image
python inference.py \
    --checkpoint checkpoints/lms_trd_best.pth \
    --input  rainy.png \
    --output derained.png

# a whole folder
python inference.py \
    --checkpoint checkpoints/lms_trd_best.pth \
    --input  rainy_folder/ \
    --output derained_folder/
```

---

## One-command pipeline

`run.sh` runs the sanity check, training, and evaluation in sequence. Paths can be overridden with environment variables:

```bash
CONFIG=configs/lms_trd.yaml \
TRAIN_ROOT=data/train VAL_ROOT=data/val TEST_ROOT=data/test/Rain100H \
DEVICE=cuda \
./run.sh
```

---

## Ablation toggles

Each co-designed mechanism can be switched on or off in the config to reproduce the Table 5 ablation rows:

```yaml
model:
  share_weights: true # cross-scale weight sharing (SST)   [M6 -> M7]
  use_inr: true # Lite-INR hash-grid branch          [M4 -> M5]
  use_ebfu: true # gated-additive eBFU fusion          [M5 -> M6]
  use_rrb: true # Rain Refinement Block               [M7 -> M8]
```

---

## Repository structure

```
LMS-TRD/
├── models/
│   ├── unet_sst.py      # encoder-decoder shared SST backbone (default config)
│   ├── sst_flat.py      # flat shared SST backbone (prose config)
│   ├── blocks.py        # LayerNorm2d, DW-MSA, DS-FFN, SST block
│   ├── lite_inr.py      # 2-level hash-grid Lite-INR branch
│   ├── ebfu.py          # gated-additive bidirectional fusion (Eq. 4)
│   ├── rrb.py           # ECA + Rain Refinement Block
│   └── lms_trd.py       # full model + ablation toggles + build_model
├── datasets/
│   └── derain_dataset.py  # paired rainy/clean loader, crop/flip/rotate aug
├── training/
│   ├── losses.py        # Charbonnier + edge + frequency + INR (Eq. 5-8)
│   └── trainer.py       # Adam, StepLR, checkpointing, validation
├── utils/
│   ├── seed.py          # deterministic seeding
│   ├── metrics.py       # PSNR, SSIM
│   ├── complexity.py    # param count + MAC/FLOP profiler (+ thop cross-check)
│   └── config.py        # YAML config loader
├── configs/
│   ├── lms_trd.yaml        # default (matches reported complexity)
│   └── lms_trd_prose.yaml  # literal Section 3.2 prose
├── docs/
│   └── paper_reference.md
├── train.py
├── test.py
├── inference.py
├── smoke_test.py
├── run.sh
├── requirements.txt
├── RECONCILIATION.md       # how/why the default config matches the reported numbers
└── README.md
```

---

## Results

Average PSNR / SSIM across the five benchmarks (from the paper):

| Method               | Avg PSNR (dB) | Avg SSIM  | Params (M) | FLOPs (G) |
| -------------------- | ------------- | --------- | ---------- | --------- |
| NeRD-Rain (baseline) | 34.48         | 0.933     | 22.89      | 148.41    |
| **LMS-TRD (ours)**   | **34.25**     | **0.927** | **7.87**   | **64.22** |

LMS-TRD matches the baseline within 0.23 dB average PSNR — below the 0.5 dB perceptual distinguishability threshold — at roughly one third of the parameters and FLOPs.

> **Note.** The complexity figures are measured at 256×256 by the profiler (run `python smoke_test.py` to reproduce). Trained weights are not included in this repository, so PSNR/SSIM are obtained by training on your own data. See `RECONCILIATION.md` for the architectural details of the default configuration.

## Acknowledgements

This work was supported by CST-UP, India, under sanction no. CST/D-1396 (dated 20.10.2023) for the project _"Road Object Detection during Rainy Nights for Autonomous Vehicles."_
