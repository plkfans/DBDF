# DBDF

This is the Pytorch implementation of DBDF for remote sensing scene image classification

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
```bash
datasets/AID/images/
├── Airport/
├── BareLand/
├── ...
```
2. Split the dataset into training/validation lists:
```bash
python build_list.py --data_dir ./datasets/AID/images --out_dir ./datasets/AID/splits-0.2 --train_ratio 0.2
```
This will create the split files under ./datasets/AID/splits-0.2.

⚠️ Important: To avoid path-related errors (e.g., files not found or permission issues), it is strongly recommended to use absolute paths for --data_dir and --out_dir. 

## Training
The training script (main.py) supports two modes:
va (Vanilla) – Standard training of the CNN backbone without DBDF (baseline).
kd (Knowledge Distillation) – Trains the CNN backbone with the proposed DBDF method.

Examples:
```bash
# Vanilla training
python -u main.py --device cuda:0 --arch resnet18 --mode va --epochs 50

# Knowledge distillation with DBDF
python -u main.py --device cuda:0 --arch resnet18 --mode kd --epochs 50 --temperature 3.0
```

## Evaluation
Generate the confusion matrix after training:
```bash
python test.py --device cuda:0 --arch resnet18 --mode va
```
Change --mode to kd if you want to evaluate the distilled model (make sure the corresponding checkpoint exists).

## Acknowledgements
The dataset splitting strategy and confusion matrix plotting are adapted from:
[SKAL](https://github.com/hw2hwei/SKAL).

The distillation strategy is inspired by:
[Self-distillation](https://github.com/ArchipLab-LinfengZhang/pytorch-self-distillation-final).

We gratefully acknowledge the authors of these repositories for their open-source contributions, which provided valuable references and insights for this research.
