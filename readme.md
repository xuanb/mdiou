# MD-IoU Loss for Oriented Object Detection

## 📋 Introduction

This repository provides the official implementation of the **MD-IoU (Mahalanobis center alignment and Discretized IoU approximation) loss** for rotated object detection.

MD-IoU is a novel loss function that addresses three fundamental challenges in oriented object detection:
1. **Metric Misalignment**: Ensures optimization directly targets the evaluation metric (SkewIoU)
2. **Boundary Discontinuity**: Provides continuous, geometrically correct gradients at angular boundaries
3. **Square-like Problem**: Resolves parameter ambiguity for near-square objects through geometric optimization

The loss employs a **cooperative coarse-to-fine framework**:
- **M-Loss (Mahalanobis Center Alignment)**: Provides stable, long-range attraction for center regression
- **D-IoU (Discretized IoU)**: Offers high-fidelity geometric refinement via differentiable ray-casting approximation
- **Dynamic Activation Weight**: Seamlessly coordinates the two components based on optimization progress

**Note**: This repository only contains the configuration files and the loss implementation file. To use MD-IoU, you need to integrate these files into your existing MMRotate 0.3.4 installation.

## 🔧 Installation & Integration

### 1. Prerequisites
- Install [MMRotate 0.3.4](https://github.com/open-mmlab/mmrotate/tree/v0.3.4) following the official instructions
- Ensure you have PyTorch, CUDA, and other dependencies as required by MMRotate

### 2. Integration Steps
Clone this repository and copy the files to your MMRotate installation. Make sure you import the loss properly.

