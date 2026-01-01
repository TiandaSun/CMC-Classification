# Chinese Minority Costume Classification (CMC-Classification)

This repository contains the official implementation of the paper **"Ensemble Learning for Fine-Grained Chinese Minority Costume Classification: A Comprehensive Empirical Study"**.

The project provides an automated framework to train, evaluate, and compare various deep learning architectures (CNNs, Vision Transformers, and Hybrid models) on the detailed classification of costumes from five Chinese ethnic groups.

## 📖 Overview

- **Goal:** Fine-grained classification of minority costumes.
- **Classes:** 5 (Hui, Zhuang, Man, Yao, and Bai).
- **Dataset:** 3,000 high-resolution images (600 per class), split 80/20 for training and validation.
- **Performance:** The proposed Ensemble method achieves **94.50%** accuracy, outperforming individual baselines.

## 🚀 Key Features

The core script `auto_baseline_runner.py` is a complete pipeline that handles:

1. **Automated Training:** Runs multiple models (ResNet, EfficientNet, ViT, etc.) in sequence.
2. **Ablation Studies:** Can test training strategies like "Training from Scratch" vs. "Transfer Learning".
3. **Analytics:** Automatically generates confusion matrices, accuracy plots, and comparison tables (CSV/LaTeX).
4. **Ensembling:** Supports voting-based fusion of top-performing models.

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TiandaSun/CMC-Classification.git
cd CMC-Classification
```

### 2. Install Dependencies

This project requires PyTorch and timm (PyTorch Image Models).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm numpy pandas matplotlib seaborn scikit-learn tqdm
```

(Note: Adjust the PyTorch installation command based on your CUDA version).

## 💻 Usage

### Data Preparation

Organize your dataset in standard ImageFolder format:

```
dataset/
├── Hui/
│   ├── image_01.jpg
│   └── ...
├── Zhuang/
├── Man/
├── Yao/
└── Bai/
```

### Running the Baseline Suite

The `auto_baseline_runner.py` script is the main entry point.

#### 1. Basic Run (All Models)

Train and evaluate all supported models (ResNet, VGG, EfficientNet, ViT, Swin, etc.):

```bash
python auto_baseline_runner.py --data_dir ./path/to/dataset --output_dir ./results
```

#### 2. Quick Test (Essential Models Only)

Run only the 5 core models to save time:

```bash
python auto_baseline_runner.py --data_dir ./dataset --essential_only
```

#### 3. Run Ablation Studies

Include experiments like training from scratch (no pre-trained weights):

```bash
python auto_baseline_runner.py --data_dir ./dataset --ablations
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./dataset` | Path to the dataset root directory. |
| `--output_dir` | `./baseline_results` | Folder to save logs, plots, and checkpoints. |
| `--batch_size` | `32` | Batch size for training. |
| `--max_epochs` | `100` | Maximum epochs per model. |
| `--patience` | `15` | Early stopping patience (epochs). |
| `--essential_only` | `False` | Run only the core 5 baseline models. |
| `--ablations` | `False` | Run additional ablation study experiments. |

## 📊 Models & Results

The code supports the following architectures via the `timm` library:

| Category | Models |
|----------|--------|
| Traditional CNNs | ResNet50, VGG16 |
| Efficient CNNs | EfficientNet-B0, MobileNetV2 |
| Transformers | ViT-Small, DeiT-Tiny, DeiT-Small |
| Hybrid / Modern | ConvNeXt-Tiny, Swin-Tiny |

### Performance Snapshot

(Based on paper results on the validation set)

| Model | Accuracy |
|-------|----------|
| Ensemble (Voting) | 94.50% |
| ConvNeXt-Tiny | 93.17% |
| VGG16 | 92.00% |
| ResNet50 | 90.17% |
| DeiT-Small | 83.67% |

## 📝 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{peng2024ensemble,
  title={Ensemble Learning for Fine-Grained Chinese Minority Costume Classification: A Comprehensive Empirical Study},
  author={Peng, Tianxiao and Sun, Tianda and Wang, Ping},
  journal={Springer Nature},
  year={2024}
}
```

## 👥 Authors

- Tianxiao Peng (Guangdong University of Science and Technology)
- Tianda Sun (University of York)
- Ping Wang (Guangdong University of Science and Technology)
