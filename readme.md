# MD-IoU Loss for Oriented Object Detection

## 📋 Introduction

This repository provides the official implementation of the paper **MD-IoU: A Decoupled Regression Framework for Rotated Object Detection via Mahalanobis Center Alignment and Discretized IoU Approximation**.

MD-IoU is a novel loss function that addresses three fundamental challenges in oriented object detection:
1. **Metric Misalignment**: Ensures optimization directly targets the evaluation metric (SkewIoU)
2. **Boundary Discontinuity**: Provides continuous, geometrically correct gradients at angular boundaries
3. **Square-like Problem**: Resolves parameter ambiguity for near-square objects through geometric optimization

The loss employs a **cooperative coarse-to-fine framework**:
- **M-Loss (Mahalanobis Center Alignment)**: Provides stable, long-range attraction for center regression
- **D-IoU (Discretized IoU)**: Offers high-fidelity geometric refinement via differentiable ray-casting approximation
- **Dynamic Activation Weight**: Seamlessly coordinates the two components based on optimization progress

**Note**: This repository only contains the configuration files and the loss implementation file. To use MD-IoU, you need to integrate these files into your existing MMRotate 0.3.4 installation.

**Note on HERO**: HERO's official implementation is built on MMRotate 1.0, therefore its corresponding configuration files are only applicable to MMRotate 1.0 and are not compatible with the MMRotate 0.3.4 setup described above. Note that the `md_iou_loss.py` file provided here is written for MMRotate 0.3.4, so you also need to modify the loss file to adapt it to MMRotate 1.0 before it can be used with HERO.

**Note on Strip R-CNN**: Strip R-CNN also requires its official code to be usable. Although it is based on MMRotate 0.3.4, the base installation of MMRotate does not include Strip R-CNN, so you need to additionally integrate the official Strip R-CNN implementation into your MMRotate 0.3.4 environment.

## 🔧 Installation & Integration

This repository only provides the MD-IoU loss implementation (`md_iou_loss.py`) and the configuration files. The exact setup depends on which model you want to train, because the configs target different codebases. Please follow the branch that matches your config:

### Option A — MMRotate 0.3.4 models (R3Det, RoI Transformer, Rotated RetinaNet, Strip R-CNN)

1. **Prerequisites**
   - Install [MMRotate 0.3.4](https://github.com/open-mmlab/mmrotate/tree/v0.3.4) following the official instructions
   - Ensure you have PyTorch, CUDA, and other dependencies as required by MMRotate

2. **Additional code for Strip R-CNN**
   - The Strip R-CNN configs (`strip_rcnn_s_fpn_*_mdiou.py`) rely on custom modules that are **not** part of base MMRotate 0.3.4. Install the official Strip R-CNN code, e.g. by cloning [https://github.com/HVision-NKU/Strip-R-CNN](https://github.com/HVision-NKU/Strip-R-CNN), into your MMRotate 0.3.4 environment before these configs can run.

3. **Integration Steps**
   - Clone this repository and copy `md_iou_loss.py` and the desired config files into your MMRotate installation. Make sure the loss is imported properly.

### Option B — HERO (MMRotate 1.0 / HERO-Det)

1. **Prerequisites**
   - HERO's official implementation is [HERO-Det](https://github.com/Qian-CV/HERO-Det), which is built on MMRotate 1.0. The HERO config (`HERO_le90_r50_fpn_1x_dota_mdiou.py`) is **only** compatible with MMRotate 1.0 and will **not** work with the MMRotate 0.3.4 setup described in Option A.

2. **Integration Steps**
   - Set up the HERO-Det / MMRotate 1.0 environment. Note that `md_iou_loss.py` here is written for MMRotate 0.3.4 and will **not** work as-is under MMRotate 1.0 — you must first modify the loss file to adapt it to MMRotate 1.0 (e.g. update the imports/registry usage to the mmengine-based API). Then copy the adapted loss file and the HERO config into the environment and make sure the loss is imported properly.

