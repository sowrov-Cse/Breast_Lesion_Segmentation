# Breast Lesion Segmentation & Classification — Full Ensemble Pipeline

A multi-task deep learning pipeline for **breast ultrasound lesion segmentation** and **3-class classification** (**benign**, **malignant**, **normal**) using an ensemble of CNN- and transformer-based architectures.

This project combines:

- **EfficientNet-B4 + U-Net**
- **Swin Transformer Tiny + U-Net**
- **SegFormer MiT-B2**
- **U-Net++ + DenseNet121**

with:

- **5-fold stratified cross-validation**
- **multi-task learning** (segmentation + classification in one forward pass)
- **strong augmentation**
- **soft voting / weighted ensembling / stacking**
- **test-time augmentation (TTA)**
- **explainable AI (Grad-CAM / attention-based inspection)**

---

## Overview

Breast ultrasound analysis often requires both:

1. **Lesion localization** through segmentation, and  
2. **Lesion categorization** through classification.

Instead of treating these as separate tasks, this project uses a **shared multi-task architecture** where each model predicts:

- a **binary segmentation mask** for lesion regions
- a **3-class label** for diagnosis:
  - `benign`
  - `malignant`
  - `normal`

The repository evaluates four complementary architectures and then combines them through ensemble learning to improve robustness and generalization.

---

## Key Features

- **Multi-task training** for segmentation and classification
- **4 complementary base architectures**
- **5-fold stratified CV** across all models
- **Strong augmentation pipeline** for ultrasound robustness
- **Class imbalance handling** with `WeightedRandomSampler`
- **Composite segmentation loss**:
  - BCE
  - Dice Loss
  - Focal Tversky Loss
- **Classification loss**:
  - CrossEntropy with label smoothing
- **Hyperparameter search** with proxy grid search
- **Ensemble strategies**:
  - Soft voting
  - Weighted ensemble
  - Stacking with MLP meta-learner
- **Test-time augmentation (TTA)**
- **XAI support**:
  - Grad-CAM for CNN-based models
  - transformer attention inspection / qualitative analysis

---

## Model Architectures

### 1) EfficientNet-B4 + U-Net
- **Encoder:** EfficientNet-B4 (ImageNet pretrained)
- **Decoder:** U-Net with SCSE attention
- **Strength:** local texture modeling and efficient convolutional features

### 2) Swin-T U-Net
- **Encoder:** Swin Transformer Tiny
- **Decoder:** U-Net-style decoder
- **Strength:** global context modeling via shifted-window attention

### 3) SegFormer MiT-B2
- **Encoder:** Mix Transformer B2
- **Decoder:** lightweight SegFormer MLP decode head
- **Strength:** hierarchical attention without positional encoding

### 4) U-Net++ + DenseNet121
- **Encoder:** DenseNet121 (ImageNet pretrained)
- **Decoder:** U-Net++ with nested skip connections and SCSE attention
- **Strength:** dense feature reuse and strong lesion boundary recovery

All models share a common interface:

```python
seg_logits, cls_logits = model(x)
```

---

## Training Strategy

### Multi-task Objective
Each model is trained to optimize:

- **Segmentation**
  - BCE Loss
  - Dice Loss
  - Focal Tversky Loss

- **Classification**
  - CrossEntropy Loss with label smoothing

### Default training setup
- **Input size:** 256×256
- **Swin input size:** 224×224
- **Epochs:** 60 max
- **Batch size:** 12
- **Gradient accumulation:** 2
- **Optimizer:** AdamW
- **Scheduler:** Linear warmup → cosine annealing
- **Early stopping patience:** 12
- **Cross-validation:** 5-fold stratified

### Class imbalance handling
The notebook explicitly addresses the lower malignant sample count through:
- `WeightedRandomSampler`
- Focal Tversky loss
- label smoothing

---

## Data Format

The notebook expects the dataset in the following structure:

```text
combined_breast_ultrasound_dataset/
├── benign/
│   ├── images/
│   └── masks/
├── malignant/
│   ├── images/
│   └── masks/
└── normal/
    ├── images/
    └── masks/
```

Supported image extensions include:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

If a sample has no mask available (for example in the `normal` class), the pipeline automatically uses a **zero mask**.

---

## Augmentation Pipeline

Two training phases are used:

### Phase 1 — No Augmentation
Minimal preprocessing only:
- resize
- normalize
- tensor conversion

### Phase 2 — Strong Augmentation
A richer augmentation pipeline with roughly 20 operations, including:
- random crop
- horizontal / vertical flip
- shift / scale / rotate
- elastic transform
- grid distortion
- optical distortion
- brightness / contrast changes
- CLAHE
- hue / saturation shifts
- Gaussian noise / blur
- sharpen
- coarse dropout
- random shadow / fog / sun flare

This phase is designed to improve robustness to ultrasound variation and acquisition artifacts.

---

## Evaluation Metrics

### Classification
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix

### Segmentation
- Dice coefficient
- IoU (Jaccard)
- Pixel accuracy
- Segmentation precision
- Segmentation recall

---

## Results

### Individual model performance

| Model | Phase | Classification Accuracy | F1 (Weighted) | Dice | IoU |
|---|---|---:|---:|---:|---:|
| EfficientNet-B4 | No Aug | 0.7330 | 0.7240 | 0.7293 | 0.6847 |
| EfficientNet-B4 | With Aug | 0.7738 | 0.7766 | 0.7003 | 0.6455 |
| Swin-T U-Net | No Aug | 0.8371 | 0.8370 | 0.8324 | 0.7813 |
| Swin-T U-Net | With Aug | **0.8462** | **0.8465** | 0.7965 | 0.7382 |
| SegFormer-B2 | No Aug | 0.8054 | 0.8059 | 0.7791 | 0.7326 |
| SegFormer-B2 | With Aug | 0.8281 | 0.8294 | **0.8008** | **0.7399** |
| U-Net++ DenseNet121 | No Aug | 0.7602 | 0.7513 | 0.7977 | 0.7532 |
| U-Net++ DenseNet121 | With Aug | 0.7964 | 0.7962 | 0.7676 | 0.7108 |

### Ensemble performance

| Method | Classification Accuracy | F1 (Weighted) | Dice | IoU |
|---|---:|---:|---:|---:|
| Best individual model (Phase 2) | 0.8462 | 0.8465 | 0.8008 | 0.7399 |
| Soft Voting | 0.8462 | 0.8462 | 0.8143 | 0.7582 |
| **Weighted Ensemble** | **0.8462** | **0.8462** | **0.8197** | **0.7637** |
| Stacking (MLP) | 0.8326 | 0.8322 | — | — |

### TTA impact
Test-time augmentation improved segmentation for several models, especially:

- **EfficientNet-B4:** Dice `0.7003 → 0.7263`
- **Swin-T U-Net:** Dice `0.7965 → 0.8061`
- **SegFormer-B2:** Dice `0.8008 → 0.8016`
- **U-Net++ DenseNet121:** Dice `0.7676 → 0.7974`

In this notebook, TTA generally improved segmentation more than classification.

---

## Main Takeaways

- **Swin-T U-Net** achieved the best standalone **classification accuracy**
- **SegFormer-B2** achieved the best standalone **segmentation Dice**
- **Weighted ensembling** produced the best overall segmentation performance
- Strong augmentation improved classification for all four models, but segmentation gains were model-dependent
- TTA provided additional segmentation improvements for several architectures

---

## Explainability

The pipeline includes explainability utilities for model interpretation:

- **Grad-CAM** for CNN-based architectures
- qualitative mask overlay visualization
- transformer-oriented feature inspection for attention-based models

This helps validate whether predictions are focused on clinically relevant lesion regions.

---

## Repository Structure

A typical repo layout for this project can be:

```text
.
├── README.md
├── breast-lesion-segmentation-ensemble-local.ipynb
├── checkpoints/
├── results/
│   ├── dataset_overview.png
│   ├── augmentation_preview.png
│   ├── phase1_summary.csv
│   ├── phase_comparison.csv
│   ├── ensemble_comparison.csv
│   ├── master_comparison_table.csv
│   └── *.png
└── data/
```

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, these are the main libraries used:

```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch timm transformers
pip install albumentations opencv-python pillow
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torchmetrics
pip install grad-cam
```

---

## How to Run

### 1. Update the dataset path
In the notebook configuration, set:

```python
CFG.DATA_ROOT = "path/to/combined_breast_ultrasound_dataset"
```

### 2. Open the notebook
Run the notebook step by step:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

### 3. Execute sections in order
Recommended order:

1. Install dependencies  
2. Imports and configuration  
3. Dataset discovery and EDA  
4. Augmentation setup  
5. Dataset and dataloaders  
6. Model definitions  
7. Losses and metrics  
8. Training utilities  
9. Grid search  
10. Phase 1 training and evaluation  
11. Phase 2 training and evaluation  
12. Ensemble evaluation  
13. TTA evaluation  
14. XAI / qualitative analysis  
15. Final comparison tables  

---

## Outputs

The notebook saves multiple artifacts to `./results`, including:

- dataset overview plots
- augmentation preview
- training curves
- confusion matrices
- phase comparison tables
- ensemble comparison tables
- radar chart comparison
- qualitative segmentation visualizations

Model checkpoints are saved to:

```text
./checkpoints
```

---

## Notes

- The current implementation is notebook-first and can be refactored into modular Python scripts for easier reuse.
- Some dataset- and path-related values are hardcoded in the notebook and should be adapted for your environment.
- The SegFormer section initializes task-specific segmentation heads from pretrained encoder weights, which is expected behavior for downstream fine-tuning.

---

## Future Improvements

- Convert the notebook into a clean package / script-based project
- Add reproducible `requirements.txt` and environment files
- Add CLI-based training and inference
- Add inference on single images
- Add external validation on a separate ultrasound dataset
- Log experiments with Weights & Biases or MLflow
- Export best ensemble checkpoints for deployment

---

## Acknowledgements

This project uses:
- **PyTorch**
- **segmentation-models-pytorch**
- **Hugging Face Transformers**
- **timm**
- **Albumentations**
- **scikit-learn**
- **Grad-CAM**

---

## Citation

If this repository helps your work, please consider citing it or linking back to the project.

```bibtex
@misc{breast_lesion_segmentation_ensemble,
  title  = {Breast Lesion Segmentation \& Classification: Full Ensemble Pipeline},
  author = {Your Name},
  year   = {2026},
  note   = {GitHub repository}
}
```
