<div align="center">

# Support-Conditioned Vision

**Support-conditioned industrial image classification with frozen DINOv2 features and a hard-margin linear classifier.**

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2--S%2F14-1877F2)](https://github.com/facebookresearch/dinov2)
[![GitHub stars](https://img.shields.io/github/stars/meamaturinlove221/support-conditioned-vision?style=flat)](https://github.com/meamaturinlove221/support-conditioned-vision/stargazers)

[Overview](#overview) · [Method](#method) · [Quick start](#quick-start) · [Input and output](#input-and-output) · [Scope](#release-scope)

<img src="assets/pipeline.svg" alt="Support-Conditioned Vision pipeline" width="100%"/>

</div>

## Overview

Support-Conditioned Vision performs **task-conditioned A/B classification**. A pair of labelled Support sets defines the classification task; the resulting classifier is then applied to a separate Query set.

This repository is the **generic inference and validation source edition**. It contains the DINOv2 feature path, Support statistics, the FP64 hard-margin linear classifier, registration-profile handling, and the command-line inference and validation entry points.

It does not contain the desktop or web applications, the LoRA training workflow, customer datasets, or the cylinder quantity research branch.

## Results

We achieved 98.12% accuracy on a real industrial image test set.
## Method

### 1. Frozen visual encoder

The encoder is Meta's DINOv2 ViT-S/14 with registers.

| Setting | Value |
|---|---:|
| Input | RGB, 448 × 448 |
| Patch size | 14 × 14 |
| Patch grid | 32 × 32 |
| Hidden dimension | 384 |
| Transformer blocks | 12 |
| Register tokens | 4 |
| Global feature | CLS, 384 dimensions |
| Local features | 1,024 ordered Patch tokens, 384 dimensions each |

The backbone is frozen. Feature extraction uses FP32, with TF32 disabled in the inference engine.

### 2. Class-balanced Support statistics

Let the two Support classes contain K_A and K_B images. Each A image receives weight `1 / (2 K_A)` and each B image receives weight `1 / (2 K_B)`. Therefore, each class contributes a total weight of 0.5, regardless of the number of registered images.

CLS and Patch features are normalized separately using the weighted Support center and scale. Query features reuse these same statistics; they do not estimate a new center or scale.

### 3. Hard-margin linear classifier

The normalized CLS and Patch features are concatenated into `z_i`. Support labels `y_i` are encoded as -1 or +1. Registration solves the hard-margin objective below:

<p align="center"><img src="assets/hard-margin-classifier.svg" alt="Class-balanced Support weights and hard-margin linear classification objective and margin constraint" width="100%"/></p>

The intercept `b` is unregularized. The implementation solves the dual problem with SciPy SLShard-margin linear classifier and an active-set correction based on KKT conditions. It checks the final margin of every Support image before accepting the new registration.

The registered `w`, `b`, center, and scale are reused for the Query batch. Optimization and classification use FP64.
### 4. A/B swap verification

The optional `--verify-swap` check reverses the two Support classes and verifies that the output columns reverse while the physical predictions remain consistent. This verifies class ordering and output mapping.

## Quick start

### Requirements

- Python 3.10 or later
- PyTorch 2.4 or later
- Pillow, NumPy, SciPy
- The official DINOv2 source and ViT-S/14 Register checkpoint

Install PyTorch using the [official selector](https://pytorch.org/get-started/locally/) for your CUDA or CPU environment, then install the remaining dependencies:

```bash
git clone https://github.com/meamaturinlove221/support-conditioned-vision.git
cd support-conditioned-vision

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

python -m pip install -r requirements.txt
```

### Prepare DINOv2

Clone the official [DINOv2 repository](https://github.com/facebookresearch/dinov2) and obtain the ViT-S/14 Register checkpoint:

- Model: `dinov2_vits14_reg`
- Checkpoint: [dinov2_vits14_reg4_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth)

Copy the public configuration template:

```bash
# Windows PowerShell
Copy-Item configs/models.example.json configs/models.local.json

# Linux / macOS
# cp configs/models.example.json configs/models.local.json
```

Set the local paths in `configs/models.local.json`:

```json
{
  "default": {
    "checkpoint": "C:/models/dinov2_vits14_reg4_pretrain.pth",
    "official_source": "C:/repos/dinov2"
  },
  "custom": {
    "checkpoint": "C:/models/dinov2_vits14_reg4_pretrain.pth",
    "official_source": "C:/repos/dinov2",
    "adapter_checkpoint": "C:/models/adapter.pt"
  }
}
```

The custom adapter is optional. No backbone or adapter weights are distributed in this repository.

## Input and output

### Support manifest

A Support manifest is a JSON object with an `entries` array. Each entry contains an image path and its class role.

```json
{
  "entries": [
    {"path": "support/A/001.png", "role": "NG"},
    {"path": "support/A/002.png", "role": "NG"},
    {"path": "support/B/001.png", "role": "OK"},
    {"path": "support/B/002.png", "role": "OK"}
  ]
}
```

Supported roles include `NG` / `OK` and `A` / `B`. Relative paths are resolved against the manifest directory.

### Run inference

The public routing table uses generic sample folder identifiers. Place the Query images under the matching `Difficult/<sample>` folder, or explicitly select a registered sample product.

```powershell
python infer.py `
  --model-version default `
  --support-manifest "C:/data/support.json" `
  --query-dir "C:/data/Difficult/sample_generic/test" `
  --output-dir runs/sample_inference `
  --device cuda `
  --verify-swap
```

### Run validation

For validation, arrange labelled Query images in class subfolders:

```text
query/
├── NG/
│   ├── image_001.png
│   └── image_002.png
└── OK/
    ├── image_003.png
    └── image_004.png
```

Then run:

```powershell
python validate.py `
  --model-version default `
  --support-manifest "C:/data/support.json" `
  --query-dir "C:/data/query" `
  --output-dir runs/sample_validation `
  --verify-swap
```

### Outputs

Each run writes:

| File | Contents |
|---|---|
| `predictions.csv` | Per-image A/B scores and predictions |
| `summary.json` | Model identity, route, Support counts, and swap verification |
| `metrics.json` | Validation confusion matrix, accuracy, per-class recall, and balanced accuracy (validation only) |

## Product routing

The public source uses example product identifiers to demonstrate directory-based routing:

| Query path | Route |
|---|---|
| `Difficult/sample_generic` | Generic DINOv2 + hard-margin linear classifier |
| `Difficult/sample_part_a` | Generic DINOv2 + hard-margin linear classifier |
| `Difficult/sample_part_b` | Generic DINOv2 + hard-margin linear classifier |
| Unknown or conflicting product folders | `REVIEW` |

These identifiers are sample folder names. Routing uses the folder name or an explicit `--product` selection.

## Repository layout

```text
configs/
  models.example.json       Public model-path template
model/
  README.md                 Local model input requirements
src/
  confirmed_support/        Support statistics and FP64 hard-margin linear classifier
  frozen_dino/              Frozen DINOv2 encoder and adapter loader
  inference/                Inference engine and registration profiles
  task_selection/            Example directory-based routing
infer.py                     Query inference
validate.py                  Labelled Query validation
```

## Release scope and limitations

- This repository contains the generic A/B inference and validation source only.
- Desktop UI, web UI, LoRA training workflow, customer-specific product routing, and cylinder quantity research are not included in this public source edition.
- The public routing identifiers are generic examples.
- Prepare the official DINOv2 backbone checkpoint locally; provide an optional LoRA adapter from an authorized source.
- Industrial images, Support/Query datasets, registration archives, and experiment outputs remain external to the public source.
- Cylinder quantity experiments are maintained separately from this generic classification release.

## Related project

Generate synthetic A/B image groups and manifests with [Industrial Image Synthesis](https://github.com/meamaturinlove221/industrial-image-synthesis). Organize the generated images into Support and Query sets, then use this repository to register the classifier and evaluate Query images.

## References

- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- Darcet et al., [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).
- [Official DINOv2 implementation and pretrained models](https://github.com/facebookresearch/dinov2).

## License

This repository currently has no `LICENSE` file. Public visibility does not by itself grant permission to use, modify, or redistribute this project's code.

The upstream [DINOv2 repository](https://github.com/facebookresearch/dinov2) lists its code and pretrained model weights under Apache 2.0. That upstream license applies to the DINOv2 materials, not automatically to this repository. Review the upstream license and model terms when preparing a deployment or redistribution.
