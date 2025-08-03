# Wafer Defect Detection & Classification using CNNs

This project performs wafer defect detection and classification using a pre-trained CNN (ResNet18) on the MIR-WM811K dataset, with additional root cause analysis using feature embeddings and PCA-based similarity retrieval.

![sample](https://upload.wikimedia.org/wikipedia/commons/3/31/Wafer_defects_example.png)

---

## Objectives

- Detect wafer defects using binary and multi-class classifiers
- Fine-tune pre-trained CNNs (ResNet18) for classification
- Extract intermediate CNN features and analyze root causes
- Implement PCA-based dimensionality reduction and image retrieval

---

## Dataset

- Name: [MIR-WM811K](http://mirlab.org/dataset/public/)
- ~172,000 labeled wafer maps
- 9 classes: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`, `none`

---

## Model Architecture

- Base: Pre-trained ResNet18
- Input: 224x224 grayscale wafer maps
- Output: Binary or multi-class classifier (9 outputs)
- Framework: PyTorch

---

## Features

- Fine-tuned CNN classifier (~96% accuracy binary)
- Multi-class training (~91% on imbalanced data)
- Feature extractor using intermediate ResNet layer
- Root cause analysis using PCA + cosine similarity (in progress)

---

## Results (so far)

| Task | Accuracy |
|------|----------|
| Defect vs None | ~96% |
| Multi-class (9) | ~91% |

---

## Setup

```bash
pip install torch torchvision pandas numpy matplotlib tqdm scikit-learn

