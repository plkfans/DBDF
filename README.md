# DBDF

This repository contains the PyTorch implementation of **DBDF** (Distillation-Based Deep Feature), a knowledge distillation method for remote sensing scene classification.  
The baseline is a standard CNN (e.g., ResNet-18) trained on the AID dataset, and DBDF improves its performance via distillation.

## Installation

```bash
conda create -n DBDF python=3.8
conda activate DBDF
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu113
pip install opencv-python
pip install pycm
pip install matplotlib
pip install scikit-learn
```

## Dataset Preparation
1. Download the AID dataset and place the images in ./datasets/AID/images/.
The folder structure should be:
datasets/AID/images/
├── Airport/
├── BareLand/
├── ...
