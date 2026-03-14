# 🐟 Multiclass Fish Image Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project that classifies fish images into multiple species categories using both a custom-built CNN and five transfer learning models (VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0). Includes a Streamlit web application for real-time predictions.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Streamlit App](#streamlit-app)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [Workflow Diagram](#workflow-diagram)
- [Coding Standards](#coding-standards)

---

## 🎯 Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Image Classification |
| **Task** | Multiclass fish species identification |
| **Models** | 1 Custom CNN + 5 Transfer Learning models |
| **Framework** | TensorFlow / Keras |
| **Deployment** | Streamlit web application |

### Business Use Cases
1. **Enhanced Accuracy** – Determine the best model for fish image classification through systematic experimentation.
2. **Deployment Ready** – Serve predictions via a user-friendly web interface with confidence scores.
3. **Model Comparison** – Evaluate and compare accuracy, precision, recall, and F1-score across all models.

---

## 📂 Dataset Structure

The dataset is organised into three splits. Each split contains sub-folders named after the fish species:

```
dataset/
├── train/
│   ├── Black Sea Sprat/
│   ├── Gilt-Head Bream/
│   ├── Hourse Mackerel/
│   ├── Red Mullet/
│   ├── Red Sea Bream/
│   ├── Sea Bass/
│   ├── Shrimp/
│   ├── Striped Red Mullet/
│   └── Trout/
├── val/
│   └── <same structure>
└── test/
    └── <same structure>
```

**Dataset source:** [GitHub – Image_Classification_dataset](https://github.com/srijith0214/Image_Classification_dataset)

---

## 🏗 Architecture

### Custom CNN (from scratch)
```
Input (224×224×3)
  → Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout(0.25)
  → Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.25)
  → Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout(0.25)
  → Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool → Dropout(0.25)
  → GlobalAvgPool → Dense(512) → BN → Dropout(0.5)
  → Dense(num_classes, Softmax)
```

### Transfer Learning (5 models)
Each model follows a two-phase training strategy:
1. **Phase 1 (Feature Extraction):** Backbone frozen, only the custom head is trained.
2. **Phase 2 (Fine-tuning):** Top backbone layers unfrozen, trained at a lower learning rate (1e-5).

```
ImageNet Backbone (frozen) → GlobalAvgPool → Dense(512, ReLU) → BN → Dropout(0.4)
                           → Dense(256, ReLU) → Dropout(0.3) → Dense(N, Softmax)
```

---

## 📁 Project Structure

```
fish_classification/
├── app.py                    # Streamlit web application
├── train.py                  # Main training script (all 6 models)
├── predict.py                # Standalone inference script
├── requirements.txt
├── models/
│   ├── custom_cnn.py         # CNN architecture from scratch
│   └── transfer_models.py    # VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
├── utils/
│   ├── data_loader.py        # Data pipeline & augmentation
│   └── evaluation.py         # Metrics, plots, comparison report
└── outputs/                  # Generated after training
    ├── best_model.h5
    ├── class_names.json
    ├── results_summary.json
    ├── comparison_report.md
    ├── model_comparison.png
    ├── sample_images.png
    ├── class_distribution.png
    ├── <ModelName>_history.png         (per model)
    └── <ModelName>_confusion_matrix.png (per model)
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fish_classification.git
cd fish_classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```bash
python train.py \
    --data_dir /path/to/dataset \
    --output_dir outputs/ \
    --epochs 25 \
    --fine_tune_epochs 10 \
    --batch_size 32 \
    --img_size 224 \
    --lr 1e-3 \
    --fine_tune_lr 1e-5
```

**Key Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *(required)* | Root of dataset (must contain train/, val/, test/) |
| `--output_dir` | `outputs/` | Where to save models & reports |
| `--epochs` | `25` | Training epochs per model |
| `--fine_tune_epochs` | `10` | Extra epochs for fine-tuning |
| `--batch_size` | `32` | Mini-batch size |
| `--img_size` | `224` | Input image dimension |
| `--lr` | `1e-3` | Phase-1 learning rate |
| `--fine_tune_lr` | `1e-5` | Phase-2 fine-tuning learning rate |

After training, `outputs/` will contain all saved models, plots, and the comparison report.

---

### Prediction

```bash
# Single image
python predict.py --model outputs/best_model.h5 --image fish.jpg

# Batch prediction on a folder
python predict.py --model outputs/best_model.h5 --image_dir my_images/ --top_k 5
```

---

### Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. You can:
- Upload any fish image (JPG, PNG, BMP, WEBP)
- View the predicted species and confidence score
- See a ranked bar chart of top-K predictions
- Expand the full probability table for all species

Configure the model and class-names paths in the **sidebar**.

---

## 📊 Model Comparison

All 3 models were evaluated on the held-out **test set** (3,187 images, 11 classes). Training was run on Google Colab with a T4 GPU.

| Model | Accuracy | Precision | Recall | F1-Score | Train Time | Notes |
|-------|:--------:|:---------:|:------:|:--------:|:----------:|-------|
| CustomCNN | 0.9934 | 0.9894 | 0.9934 | 0.9914 | ~35 min | Trained from scratch, 25 epochs |
| **MobileNetV2** ⭐ | **0.9969** | **0.9968** | **0.9969** | **0.9968** | ~50 min | Transfer learning + 10 fine-tune epochs |
| EfficientNetB0 | 0.2099 | 0.1182 | 0.2099 | 0.0998 | ~26 min | ⚠️ Did not converge — see note below |

> ⭐ **Best Model: MobileNetV2** with **99.69% test accuracy**

### 📌 Key Observations

- **MobileNetV2** achieved the best results, reaching 99.45% validation accuracy after phase-1 training alone, and 99.69% test accuracy after fine-tuning from layer 100 onwards. It performed near-perfectly on all 11 species.
- **CustomCNN** performed remarkably well for a model trained from scratch, achieving 99.34% test accuracy. The class `animal fish bass` (only 13 test samples) scored 0.00 F1 due to severe class imbalance.
- **EfficientNetB0** failed to learn — accuracy plateaued at ~17–22% (near random for 11 classes). This is a known issue with EfficientNet on Colab when images are **not pre-scaled to [0, 255]** before being fed through its internal rescaling layer. The fix is shown below.

### ⚠️ EfficientNetB0 Fix

EfficientNet has its own built-in rescaling layer and **should NOT receive images pre-divided by 255**. To fix, use a separate generator without `rescale`:

```python
# For EfficientNetB0 only — no rescale in the generator
efficientnet_datagen = ImageDataGenerator(
    rotation_range=20, zoom_range=0.2,
    width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, fill_mode="nearest"
    # ← NO rescale=1./255 here
)
```

### 📋 Per-Class Results — MobileNetV2 (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| animal fish | 1.00 | 0.99 | 0.99 | 520 |
| animal fish bass | 0.91 | 0.77 | 0.83 | 13 |
| black sea sprat | 0.99 | 1.00 | 0.99 | 298 |
| gilt head bream | 1.00 | 1.00 | 1.00 | 305 |
| hourse mackerel | 1.00 | 1.00 | 1.00 | 286 |
| red mullet | 1.00 | 1.00 | 1.00 | 291 |
| red sea bream | 1.00 | 1.00 | 1.00 | 273 |
| sea bass | 0.99 | 1.00 | 1.00 | 327 |
| shrimp | 1.00 | 1.00 | 1.00 | 289 |
| striped red mullet | 0.99 | 1.00 | 1.00 | 293 |
| trout | 1.00 | 1.00 | 1.00 | 292 |
| **weighted avg** | **1.00** | **1.00** | **1.00** | **3187** |

---

## 🔁 Workflow Diagram

```
Raw Dataset (train/val/test)
        │
        ▼
  Data Preprocessing
  ┌─────────────────────┐
  │ Rescale → [0, 1]    │
  │ Augmentation:        │
  │  • Rotation ±20°    │
  │  • Zoom 20%         │
  │  • Horizontal flip  │
  │  • Shift, Shear     │
  └─────────────────────┘
        │
        ▼
  Model Training (×3)
  ┌──────────────────────────────────────┐
  │ CustomCNN      (from scratch)        │
  │ MobileNetV2    (transfer + finetune) │ ⭐ Best
  │ EfficientNetB0 (transfer + finetune) │
  └──────────────────────────────────────┘
        │
        ▼
  Evaluation (Test Set)
  ┌─────────────────────┐
  │ Accuracy            │
  │ Precision / Recall  │
  │ F1-Score            │
  │ Confusion Matrix    │
  │ Training Curves     │
  └─────────────────────┘
        │
        ▼
  Best Model → best_model.h5
        │
        ▼
  Streamlit App → Real-time Prediction
```

---

## 📐 Coding Standards

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) conventions:
- Snake_case for variables and functions
- PascalCase for classes
- Docstrings on all public functions
- Type hints throughout
- Modular, single-responsibility modules

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
