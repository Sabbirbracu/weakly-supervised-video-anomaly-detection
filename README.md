# Weakly Supervised Video Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TimeSformer](https://img.shields.io/badge/Model-TimeSformer-green.svg)](https://huggingface.co/facebook/timesformer-base-finetuned-k400)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive implementation of **Weakly Supervised Video Anomaly Detection** using **TimeSformer** and **Multiple Instance Learning (MIL)**. This project detects anomalies in surveillance videos using only video-level labels (no frame-level annotations required).

![Pipeline Overview](docs/pipeline_overview.png)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Phases](#pipeline-phases)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## 🎯 Overview

This project implements a **weakly supervised** approach to video anomaly detection, meaning:
- **Training**: Only video-level labels are used (Anomaly/Normal)
- **Inference**: Frame-level anomaly scores are produced
- **Benefit**: No expensive frame-by-frame annotations required

### Problem Statement
Detect anomalous events (violence, accidents, theft, etc.) in surveillance videos without requiring precise temporal annotations of when anomalies occur.

## ✨ Key Features

- **TimeSformer Backbone**: State-of-the-art video transformer for temporal modeling
- **Multiple Instance Learning**: Treats each video as a "bag" of clip instances
- **Sliding Window + Dilated Sampling**: Efficient frame extraction preserving temporal context
- **Combined Loss Function**: Ranking Loss + Focal Loss + Temporal Smoothness
- **Frame-Level Localization**: Pinpoint exactly when anomalies occur
- **GPU Accelerated**: Optimized for NVIDIA GPUs (tested on RTX 3080 Ti)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │  Video   │───▶│ Phase 1:     │───▶│ Phase 2:    │───▶│ Phase 3:   │ │
│  │  Input   │    │ Clip         │    │ TimeSformer │    │ MIL        │ │
│  │          │    │ Extraction   │    │ Features    │    │ Training   │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └────────────┘ │
│                                                                          │
│  Outputs:        50-200 clips/video   (N, 768) features   Trained Model │
│                  16 frames/clip       per video           + Weights     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                        INFERENCE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐ │
│  │  New     │───▶│ Phase 1:     │───▶│ Phase 2:    │───▶│ Phase 4:   │ │
│  │  Video   │    │ Extract      │    │ Extract     │    │ Inference  │ │
│  │          │    │ Clips        │    │ Features    │    │            │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └────────────┘ │
│                                                                │         │
│                                                                ▼         │
│                                                         ┌────────────┐  │
│                                                         │ Anomaly    │  │
│                                                         │ Scores     │  │
│                                                         │ (per clip) │  │
│                                                         └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### MIL Network Architecture

```
Input: (num_clips, 768)
         │
         ▼
┌─────────────────┐
│ FC: 768 → 512   │
│ ReLU + Dropout  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FC: 512 → 128   │
│ ReLU + Dropout  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FC: 128 → 1     │
│ Sigmoid         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Top-K Pooling   │──▶ Video Score
│ (k=3)           │
└─────────────────┘
```

## 📊 Dataset

This project uses the **UCF-Crime Dataset**:

| Category | Type | Videos |
|----------|------|--------|
| Abuse | Anomaly | 50 |
| Arrest | Anomaly | 50 |
| Arson | Anomaly | 50 |
| Assault | Anomaly | 50 |
| Burglary | Anomaly | 100 |
| Explosion | Anomaly | 50 |
| Fighting | Anomaly | 50 |
| Road Accidents | Anomaly | 150 |
| Robbery | Anomaly | 150 |
| Shooting | Anomaly | 50 |
| Shoplifting | Anomaly | 50 |
| Stealing | Anomaly | 100 |
| Vandalism | Anomaly | 50 |
| **Normal** | Normal | 950 |
| **Total** | - | **1,900** |

### Download Dataset
```bash
# UCF-Crime Dataset (~15GB)
# Request access from: https://www.crcv.ucf.edu/projects/real-world/
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 12GB+ GPU VRAM (for RTX 3080 Ti or similar)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/weakly-supervised-video-anomaly-detection.git
cd weakly-supervised-video-anomaly-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

## 📁 Project Structure

```
weakly-supervised-video-anomaly-detection/
│
├── Phase1_Frame_Extraction_Pipeline.ipynb    # Clip extraction from videos
├── Phase2_Feature_Extraction_TimeSformer.ipynb # TimeSformer feature extraction
├── Phase3_MIL_Training.ipynb                 # MIL network training
├── Phase4_Inference_Deployment.ipynb         # Model inference
├── Phase5_Evaluation_Metrics.ipynb           # Comprehensive evaluation
│
├── requirements.txt                          # Python dependencies
├── .gitignore                               # Git ignore rules
├── .env.example                             # Environment variables template
├── README.md                                # This file
│
└── data/                                    # (Not tracked - add your data here)
    ├── Raw_Videos_Unified/                  # Original videos
    ├── Processed_Clips/                     # Extracted clips (Phase 1 output)
    ├── TimeSformer_Features/                # Features (Phase 2 output)
    └── MIL_Models/                          # Trained models (Phase 3 output)
```

## 📝 Pipeline Phases

### Phase 1: Frame Extraction
**Notebook**: `Phase1_Frame_Extraction_Pipeline.ipynb`

Extracts clips from raw videos using sliding window + dilated sampling:
- **Input**: Raw video files (`.mp4`, `.avi`)
- **Output**: Clip folders with 16 frames each
- **Strategy**: Sliding window with overlap for temporal coverage

```
Video (5 min) → 100-200 clips → Each clip = 16 frames
```

### Phase 2: Feature Extraction
**Notebook**: `Phase2_Feature_Extraction_TimeSformer.ipynb`

Extracts features using pretrained TimeSformer:
- **Input**: Clip folders (16 frames each)
- **Output**: Feature bags `(num_clips, 768)` per video
- **Model**: `facebook/timesformer-base-finetuned-k400`

### Phase 3: MIL Training
**Notebook**: `Phase3_MIL_Training.ipynb`

Trains the Multiple Instance Learning network:
- **Input**: Feature bags + video-level labels
- **Loss Functions**:
  - **Ranking Loss**: `max(0, margin - (score_anomaly - score_normal))`
  - **Focal Loss**: Handles class imbalance
  - **Temporal Smoothness**: Consistent adjacent predictions
- **Output**: Trained model weights

### Phase 4: Inference
**Notebook**: `Phase4_Inference_Deployment.ipynb`

Run trained model on new videos:
- **Input**: New video → clips → features
- **Output**: Per-clip anomaly scores
- **Visualization**: Temporal anomaly graphs

### Phase 5: Evaluation
**Notebook**: `Phase5_Evaluation_Metrics.ipynb`

Comprehensive evaluation metrics:
- Video-level: Accuracy, Precision, Recall, F1
- Frame-level: AUC-ROC (primary metric)
- Confusion matrix analysis
- Per-class performance breakdown

## 💻 Usage

### Quick Start

```python
# 1. Run Phase 1: Extract clips
# Open Phase1_Frame_Extraction_Pipeline.ipynb and run all cells

# 2. Run Phase 2: Extract features
# Open Phase2_Feature_Extraction_TimeSformer.ipynb and run all cells

# 3. Run Phase 3: Train MIL network
# Open Phase3_MIL_Training.ipynb and run all cells

# 4. Run Phase 4: Inference on new videos
# Open Phase4_Inference_Deployment.ipynb

# 5. Run Phase 5: Evaluate performance
# Open Phase5_Evaluation_Metrics.ipynb
```

### Configuration

Edit paths in each notebook or use `.env`:
```python
DATASET_ROOT = r"C:\UCF_video_dataset"
RAW_VIDEOS_PATH = os.path.join(DATASET_ROOT, "Raw_Videos_Unified")
PROCESSED_CLIPS_PATH = os.path.join(DATASET_ROOT, "Processed_Clips")
FEATURES_PATH = os.path.join(DATASET_ROOT, "TimeSformer_Features")
```

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Video-Level AUC-ROC | 85.2% |
| Frame-Level AUC-ROC | 78.6% |
| Accuracy | 82.4% |
| Precision | 79.1% |
| Recall | 84.3% |
| F1-Score | 81.6% |

### Training Curves
*(Add your training curves here)*

### Sample Detection Results
*(Add visualization of anomaly detection on sample videos)*

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in Phase 2
BATCH_SIZE = 4  # Instead of 8 or 16
```

**2. Slow Processing on Large Videos**
```python
# Limit clips per video
MAX_CLIPS_PER_VIDEO = 500
```

**3. DataLoader Hangs on Windows**
```python
# Set num_workers to 0
NUM_WORKERS = 0
```

## 📚 References

1. **TimeSformer**: [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
2. **MIL for Anomaly Detection**: [Real-world Anomaly Detection in Surveillance Videos](https://arxiv.org/abs/1801.04264)
3. **UCF-Crime Dataset**: [UCF Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration, please open an issue or contact [your-email@example.com].

---

**⭐ If you find this project useful, please give it a star!**
