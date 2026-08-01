<div align="center">

# Role-Aware Tactile Path Perception from UAV Videos

### Motion-Semantic Guidance for Traversability-Aware Segmentation

[![ICANN 2026](https://img.shields.io/badge/ICANN-2026-124E78.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-UTP--9K-FFD21E.svg)](https://huggingface.co/datasets/prettyyefan/UTP-9K)

**Yefan Wang · Yusen Wu · Lingling Qu**

University of Shanghai for Science and Technology · Fujian University of Technology · Yangzhou University

## Overview

Appearance alone cannot reliably distinguish a stationary blockage, a moving pedestrian, and a shadow-like artifact from UAV videos. We therefore formulate tactile-path monitoring as **role-aware perception**, where the model reasons about how an occupied region affects traversability.

The framework combines:

- SAM2 candidate masks;
- RAFT optical flow;
- region-level motion statistics;
- soft semantic role priors;
- tactile-path spatial and altitude cues.

## Demonstration

https://github.com/user-attachments/assets/6ae03295-22e4-48de-b8e3-9d7fd2b0d5fc

<p>
  <i>Role-aware tactile-path perception from UAV videos.</i>
</p>

## UTP-9K

UTP-9K contains approximately **9,000 UAV frames** from **300 video clips**, captured at **30 m and 60 m**. The benchmark covers stationary blockages, moving targets, shadow/artifact interference, weak tactile patterns, and occlusion.

| Property | Value |
|---|---:|
| UAV frames | ~9,000 |
| Video clips | 300 |
| Flight heights | 30 m / 60 m |
| Semantic roles | Stationary / Moving / Artifact-Shadow |
| Task A | Binary obstacle perception |
| Task B | Three-way role-aware perception |

<p align="center">
  <img src="./Figs/utp9k.jpg" width="92%" alt="UTP-9K dataset overview">
</p>

## Method

For each candidate region, optical flow is summarized by mean magnitude, magnitude variance, and directional entropy. A lightweight MLP converts these cues into a soft distribution over stationary, moving, and artifact roles. The role prompt is then fused with tactile-path spatial and altitude embeddings and injected into the SAM2 decoding pathway.

## Results

| Task | Main metric | Ours |
|---|---:|---:|
| Task A | Overall mIoU | **80.0** |
| Task A | Overall F1 | **81.4** |
| Task A | Moving FP ↓ | **4.8** |
| Task A | Artifact FP ↓ | **5.2** |
| Task B | Macro-F1 | **79.2** |

<table>
<tr>
<td width="50%" align="center">
  <img src="./Figs/task_a_results.png" width="100%" alt="Task A results"><br>
  <b>Task A</b>
</td>
<td width="50%" align="center">
  <img src="./Figs/task_b_results.png" width="100%" alt="Task B results"><br>
  <b>Task B</b>
</td>
</tr>
<tr>
<td colspan="2" align="center">
  <img src="./Figs/task_a_false_positive.png" width="72%" alt="False-positive comparison"><br>
  <b>Region-level false positives</b>
</td>
</tr>
</table>

## Qualitative Comparison

<p align="center">
  <img src="./Figs/comparison.jpg" width="96%" alt="Qualitative comparison on UAV tactile-path scenes">
</p>

## Installation

```bash
git clone https://github.com/prettyyefan/UTP-9K.git
cd UTP-9K

conda create -n utp9k python=3.10 -y
conda activate utp9k

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e .
```

Install the official SAM2 package and place the required checkpoint under `checkpoints/`.

## Quick Start

```bash
python scripts/smoke_test.py
pytest -q
```

Training:

```bash
python scripts/train_role_head.py \
  --manifest data/manifests/train.jsonl \
  --root data/UTP-9K \
  --epochs 40 \
  --batch-size 16 \
  --output outputs/role_head.pt
```

Video inference:

```bash
python scripts/infer_video.py \
  --video data/demo/input.mov \
  --roi-mask data/demo/tactile_roi.png \
  --checkpoint outputs/role_head.pt \
  --altitude 30 \
  --output outputs/demonstration.mp4
```

## Citation

```bibtex
@inproceedings{wang2026roleaware,
  title     = {Role-Aware Tactile Path Perception from UAV Videos via Motion-Semantic Guidance},
  author    = {Wang, Yefan and Wu, Yusen and Qu, Lingling},
  booktitle = {International Conference on Artificial Neural Networks (ICANN)},
  year      = {2026}
}
```

## Acknowledgements

This project builds on [SAM2](https://github.com/facebookresearch/sam2) and the RAFT implementation in [torchvision](https://pytorch.org/vision/stable/models/optical_flow.html).
