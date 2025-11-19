# GTSRB Traffic Sign Classifier 🚦

A deep learning traffic sign recognition system using PyTorch with ResNet50 and EfficientNet-B3 architectures, trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a state-of-the-art traffic sign classifier designed for self-driving car applications. It achieves 95-99% accuracy on the GTSRB dataset using advanced deep learning techniques including transfer learning, class balancing, and comprehensive data augmentation.

### Key Features

- **Dual Model Architecture Support**: ResNet50 (25.6M params) and EfficientNet-B3 (12M params)
- **Advanced Preprocessing**: ROI (Region of Interest) cropping to focus on sign content
- **Class Imbalance Handling**: Weighted sampling and weighted loss for 43-class imbalanced dataset
- **Comprehensive Overfitting Prevention**: Dropout, weight decay, early stopping, gradient clipping
- **Multi-Metric Tracking**: Top-1, Top-5 accuracy, loss curves, learning rate scheduling
- **Extensive Post-Processing**: Confusion matrices, per-class analysis, error analysis, 10+ visualizations
- **Production-Ready**: Clear model naming, checkpoint saving, reproducible training

## 📊 Dataset

**GTSRB (German Traffic Sign Recognition Benchmark)**
- **Training Samples**: 39,209 images (split into 90% train, 10% validation)
- **Test Samples**: 12,630 images
- **Classes**: 43 different traffic sign types
- **Image Resolution**: Variable (resized to 224×224 for training)
- **Class Imbalance**: 10-40× ratio between most/least populated classes
- **Data Format**: CSV-based with metadata (ClassId, Width, Height, ROI coordinates)

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
# NVIDIA GPU with CUDA support (recommended)
# 16GB RAM minimum, 32GB recommended
# 6GB GPU VRAM minimum
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/K1taru/Traffic-Sign-Classifier-Model.git
cd Traffic-Sign-Classifier-Model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download GTSRB dataset**
```bash
python utils/kaggle_dataset_installer.py
```
*Note: Requires Kaggle API credentials in `~/.kaggle/kaggle.json`*

### Training

1. **Open the Jupyter notebook**
```bash
jupyter notebook src/traffic_sign_classifier_model.ipynb
```

2. **Configure model architecture** (Cell 6)
```python
MODEL_ARCH = 'efficientnet_b3'  # or 'resnet50'
```

3. **Run all cells** to train the model

4. **Find trained model** in `models/` directory:
```
GTSRB_efficientnet_b3_E22_VAL99.12.pth
```

See [`docs/QUICK_START.md`](docs/QUICK_START.md) for detailed training guide.

## 📁 Project Structure

```
Traffic-Sign-Classifier-Model/
├── src/
│   ├── traffic_sign_classifier_model.ipynb  # Main training notebook
│   └── utils/
│       ├── gtsrb_dataset.py                  # Custom GTSRB Dataset class
│       ├── dataset_counter.py                # Dataset statistics utilities
│       └── gpu_utils.py                      # GPU detection and info
├── dataset/
│   ├── Train.csv                             # Training data metadata
│   ├── Test.csv                              # Test data metadata
│   ├── Meta.csv                              # Class name mappings
│   ├── Train/                                # Training images (43 folders)
│   └── Test/                                 # Test images
├── models/                                    # Trained model checkpoints
│   └── .gitkeep                              # Keep directory in git
├── docs/
│   ├── IMPLEMENTATION_VERIFICATION.md        # Complete implementation checklist
│   ├── QUICK_START.md                        # Quick training guide
│   └── documentation.txt                     # Additional notes
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## 🧠 Model Architectures

### ResNet50
- **Parameters**: 25.6M
- **Pretrained**: ImageNet (IMAGENET1K_V1)
- **Classifier**: Dropout(0.4) → Linear(2048 → 43)
- **Expected Accuracy**: 95-98%

### EfficientNet-B3 ⭐ **RECOMMENDED**
- **Parameters**: 12M (more efficient)
- **Pretrained**: ImageNet (IMAGENET1K_V1)
- **Classifier**: Dropout(0.4) → Linear(1536 → 43)
- **Expected Accuracy**: 96-99%

## 🎓 Training Configuration

### Hyperparameters
```python
LEARNING_RATE = 0.0001
MAX_EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.10
DROPOUT_RATE = 0.4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
```

### Optimization Techniques
1. **Transfer Learning**: ImageNet pretrained weights
2. **ROI Cropping**: Focus on sign content, remove background
3. **Data Augmentation**: Rotation (±15°), affine, color jitter, perspective, random erasing
4. **Class Balancing**: Weighted random sampling + weighted CrossEntropyLoss
5. **Regularization**: Dropout (0.4) + L2 weight decay (1e-4)
6. **Early Stopping**: Stop training after 7 epochs without validation improvement
7. **Gradient Clipping**: Prevent exploding gradients (max_norm=1.0)
8. **Adaptive Learning Rate**: ReduceLROnPlateau scheduler

### Training Time
- **Per Epoch**: 15-25 minutes (with GPU)
- **Total Training**: 2-4 hours (with early stopping)

## 📈 Performance Metrics

### Tracked Metrics (6 total)
- Top-1 Accuracy (Train & Validation)
- Top-5 Accuracy (Train & Validation)
- Loss (Train & Validation)
- Learning Rate per epoch

### Post-Training Evaluation
- Overall test accuracy
- Balanced accuracy (for imbalanced classes)
- Per-class precision, recall, F1-score
- Confusion matrices (absolute & normalized)
- Top-10 most confused class pairs
- Confidence distribution analysis
- Error analysis by class

## 📊 Visualizations

The notebook generates 10+ comprehensive visualizations:

1. **Training Curves** (4 panels): Top-1 acc, loss, top-5 acc, learning rate
2. **Confusion Matrices** (2 panels): Absolute counts & normalized
3. **Performance Analysis** (6 panels):
   - Per-class accuracy bar chart
   - Precision/Recall/F1 comparison
   - Test set class distribution
   - Confidence distribution (correct vs incorrect)
   - Top-10 confused class pairs
   - Accuracy vs support scatter plot

## 💾 Model Checkpoints

Models are saved with descriptive filenames:
```
models/GTSRB_{architecture}_E{epoch}_VAL{accuracy}.pth
```

**Example**:
```
GTSRB_efficientnet_b3_E22_VAL99.12.pth
```
- Architecture: EfficientNet-B3
- Best epoch: 22
- Validation accuracy: 99.12%

### Checkpoint Contents
- Model state dict (weights)
- Optimizer state
- Training history (all 6 metrics)
- Best epoch and validation accuracy
- Architecture name and configuration

## 🔧 Usage Examples

### Load Trained Model
```python
import torch
from torchvision import models
import torch.nn as nn

# Load checkpoint
checkpoint = torch.load('models/GTSRB_efficientnet_b3_E22_VAL99.12.pth')

# Recreate model architecture
model = models.efficientnet_b3(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4, inplace=True),
    nn.Linear(in_features, 43)
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
```

### Make Predictions
```python
from PIL import Image
from torchvision import transforms

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('path/to/traffic_sign.jpg')
input_tensor = preprocess(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
```

## 📚 Documentation

- **[QUICK_START.md](docs/QUICK_START.md)**: Step-by-step training guide, configuration reference, troubleshooting
- **[IMPLEMENTATION_VERIFICATION.md](docs/IMPLEMENTATION_VERIFICATION.md)**: Complete technical documentation, verification checklist, performance expectations

## 🛠️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.65.0
jupyter>=1.0.0
```

See [`requirements.txt`](requirements.txt) for complete list.

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce `BATCH_SIZE` from 64 to 32 or 16 in Global Configuration cell.

### Low Accuracy
- Verify ROI cropping is enabled: `USE_ROI_CROP = True`
- Check that weighted sampler is enabled: `USE_WEIGHTED_SAMPLER = True`
- Ensure data augmentation is on: `USE_AUGMENTATION = True`

### Dataset Not Found
Run `python utils/kaggle_dataset_installer.py` and ensure Kaggle API is configured.

See [`docs/QUICK_START.md`](docs/QUICK_START.md) for more troubleshooting tips.

## 🎯 Results

### Expected Performance
- **ResNet50**: 95-98% validation accuracy
- **EfficientNet-B3**: 96-99% validation accuracy

### Key Metrics
- **Balanced Accuracy**: ~97% (accounts for class imbalance)
- **Top-5 Accuracy**: ~99.5% (correct class in top 5 predictions)
- **Average Confidence**: ~98% on correct predictions

## 🚗 Use Case: Self-Driving Cars

This classifier is designed for safety-critical applications in autonomous vehicles:

- **High Accuracy**: 96-99% ensures reliable sign recognition
- **Balanced Performance**: Handles rare sign types effectively
- **Confidence Scores**: Provides uncertainty estimates for decision-making
- **Real-time Capable**: Optimized architectures for fast inference
- **Robust**: Heavy augmentation ensures generalization to various conditions

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Mixed precision training (FP16) for faster training
- Test-time augmentation (TTA) for higher accuracy
- Model ensemble techniques
- CutMix/MixUp augmentation strategies
- Deployment optimizations (ONNX, TensorRT)

## 📄 License

This project is licensed under the **MIT License** - see below for details.

### Why MIT License?
- ✅ **Permissive**: Allows commercial use
- ✅ **Simple**: Short and easy to understand
- ✅ **Popular**: Widely used in machine learning projects
- ✅ **Flexible**: Compatible with proprietary software
- ✅ **Attribution Only**: Only requires credit to original authors

**MIT License Summary**:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Must include license and copyright notice
- ❌ No warranty provided
- ❌ Authors not liable

To add the MIT License, create a `LICENSE` file with:
```
MIT License

Copyright (c) 2025 K1taru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Alternative License Options**:
- **Apache 2.0**: Similar to MIT but includes explicit patent grant
- **GPL v3**: Copyleft license, derivatives must be open source
- **BSD 3-Clause**: Similar to MIT with additional non-endorsement clause

## 🙏 Acknowledgments

- **Dataset**: [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html) by Institut für Neuroinformatik
- **Pretrained Models**: ImageNet weights from [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html)
- **Framework**: [PyTorch](https://pytorch.org/) by Meta AI

## 📧 Contact

- **GitHub**: [@K1taru](https://github.com/K1taru)
- **Project**: [Traffic-Sign-Classifier-Model](https://github.com/K1taru/Traffic-Sign-Classifier-Model)

## 📝 Citation

If you use this project in your research or application, please cite:

```bibtex
@software{gtsrb_classifier_2025,
  author = {K1taru},
  title = {GTSRB Traffic Sign Classifier},
  year = {2025},
  url = {https://github.com/K1taru/Traffic-Sign-Classifier-Model}
}
```

---

**⭐ Star this repository if you find it helpful!**

**🚀 Ready to train your traffic sign classifier? See [docs/QUICK_START.md](docs/QUICK_START.md) to get started!**