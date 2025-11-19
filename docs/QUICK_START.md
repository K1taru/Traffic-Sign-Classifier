# GTSRB Traffic Sign Classifier - Quick Start Guide

## 🚀 Quick Training Steps

### Step 1: Choose Your Model
In Cell 6 (Global Configuration), set:
```python
MODEL_ARCH = 'resnet50'  # or 'efficientnet_b3'
```

**Recommendations:**
- **ResNet50**: Baseline, well-tested, 25.6M parameters
- **EfficientNet-B3**: Better accuracy, more efficient, 12M parameters ⭐ **RECOMMENDED**

---

### Step 2: Run Training
Execute all cells sequentially from top to bottom.

**Key Cells:**
- **Cells 1-6**: Setup and configuration
- **Cells 7-16**: Dataset loading and preprocessing
- **Cells 17-26**: Model setup and optimizer
- **Cell 27**: Training function definition
- **Cell 28**: **TRAIN THE MODEL** (takes 2-4 hours)
- **Cells 29-30**: Training curves
- **Cells 31-46**: Post-processing and evaluation

---

### Step 3: Train Second Model (Optional)
To train the other architecture:

1. **Change** `MODEL_ARCH` in Cell 6
2. **Re-run** from Cell 24 (Load Pretrained Model) onwards
3. Model will save with different name automatically

---

## 📊 What to Expect

### During Training:
```
🎯 STARTING TRAINING
================================================================================

📆 Epoch 1/30
================================================================================
Learning Rate: 1.00e-04
🔥 TRAIN | Loss: 1.2345 | Acc:  78.45% | Top-5:  95.23%
✅ VAL   | Loss: 0.9876 | Acc:  82.31% | Top-5:  96.78%
✨ New best validation accuracy: 82.31%
```

### Training Progress:
- **Early epochs (1-10)**: Rapid accuracy improvement (60% → 90%)
- **Mid epochs (11-20)**: Slower gains (90% → 95%)
- **Late epochs (21-30)**: Fine-tuning (95% → 98%)
- **Early stopping**: May stop before epoch 30 if no improvement

### Expected Final Accuracy:
- **ResNet50**: 95-98% validation accuracy
- **EfficientNet-B3**: 96-99% validation accuracy

---

## 💾 Model Files

### File Naming:
```
models/GTSRB_resnet50_E18_VAL98.45.pth
models/GTSRB_efficientnet_b3_E22_VAL99.12.pth
```

Format: `GTSRB_{architecture}_E{best_epoch}_VAL{val_accuracy}.pth`

### What's Saved:
- Model weights (state_dict)
- Optimizer state
- Training history (all metrics)
- Best epoch and accuracy
- Architecture name and config

---

## 📈 Post-Training Analysis

After training completes, you'll see:

### 1. Training Curves (4 panels)
- Top-1 Accuracy (train vs val)
- Loss (train vs val)
- Top-5 Accuracy (train vs val)
- Learning Rate schedule

### 2. Confusion Matrices (2 panels)
- Absolute counts
- Normalized percentages

### 3. Performance Analysis (6 panels)
- Per-class accuracy bar chart
- Precision/Recall/F1 comparison
- Test set distribution
- Confidence distribution
- Top-10 confused pairs
- Accuracy vs support scatter

### 4. Detailed Tables
- Per-class metrics (top-10 and bottom-10)
- Error analysis
- Misclassification patterns
- Final summary report

---

## 🔧 Configuration Reference

### Key Variables (Cell 6):
```python
# Model Selection
MODEL_ARCH = 'efficientnet_b3'  # or 'resnet50'

# Dataset Paths
DATASET_DIR = "dataset/"
TRAIN_CSV = "dataset/Train.csv"
TEST_CSV = "dataset/Test.csv"
META_CSV = "dataset/Meta.csv"

# Data Configuration
NUM_CLASSES = 43
VALIDATION_SPLIT = 0.10  # 10% validation
BATCH_SIZE = 64
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Augmentation & Preprocessing
USE_AUGMENTATION = True
USE_ROI_CROP = True  # Crop to sign region
USE_WEIGHTED_SAMPLER = True  # Handle class imbalance

# Model Saving
MODEL_SAVE_DIR = "models/"
```

### Training Hyperparameters (Cell 22):
```python
LEARNING_RATE = 0.0001          # Initial LR
MAX_EPOCHS = 30                 # Maximum epochs
WEIGHT_DECAY = 1e-4             # L2 regularization
DROPOUT_RATE = 0.4              # Dropout rate
EARLY_STOPPING_PATIENCE = 7     # Stop after 7 epochs no improvement
MAX_GRAD_NORM = 1.0             # Gradient clipping
```

---

## 🎯 Training Strategy

### Optimization Techniques Applied:
1. ✅ **Transfer Learning**: ImageNet pretrained weights
2. ✅ **ROI Cropping**: Focus on sign content, remove background
3. ✅ **Class Balancing**: Weighted loss + weighted sampler
4. ✅ **Heavy Augmentation**: Rotation, affine, color jitter, etc.
5. ✅ **Dropout**: 0.4 in classifier head
6. ✅ **Weight Decay**: L2 regularization (1e-4)
7. ✅ **Early Stopping**: Patience=7, saves best model
8. ✅ **Gradient Clipping**: Prevents exploding gradients
9. ✅ **Adaptive LR**: ReduceLROnPlateau scheduler
10. ✅ **Multi-Metric Tracking**: Top-1, Top-5, Loss, LR

---

## ⚙️ Hardware Requirements

### Minimum:
- **GPU**: NVIDIA GPU with 6GB VRAM (GTX 1060 or better)
- **RAM**: 16GB
- **Storage**: 3GB (dataset + models)
- **CUDA**: Compatible version with PyTorch

### Recommended:
- **GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 10GB (for multiple model versions)

---

## 🐛 Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution:** Reduce `BATCH_SIZE` from 64 to 32 or 16 in Cell 6

### Issue: "FileNotFoundError: Train.csv"
**Solution:** Ensure GTSRB dataset is downloaded to `dataset/` folder

### Issue: Training accuracy > 99%, validation < 90%
**Solution:** Overfitting detected. Model already has:
- Early stopping
- Dropout
- Weight decay
- Augmentation
This is severe overfitting - check if augmentation is enabled.

### Issue: Training is very slow
**Solution:** 
- Verify GPU is being used (check output of Cell 4)
- Ensure CUDA is properly installed
- Close other GPU-intensive applications

### Issue: All predictions are same class
**Solution:** 
- Check if weighted sampler is enabled
- Verify class weights are applied to loss function
- May need to retrain from scratch

---

## 📝 Training Checklist

Before starting training, verify:

- [ ] GTSRB dataset downloaded and extracted to `dataset/`
- [ ] CSV files exist: `Train.csv`, `Test.csv`, `Meta.csv`
- [ ] GPU is available and CUDA is working (Cell 4)
- [ ] `MODEL_ARCH` is set to desired architecture (Cell 6)
- [ ] `models/` directory exists (created automatically)
- [ ] All required packages installed (Cell 2)
- [ ] `utils/gtsrb_dataset.py` exists and is accessible

---

## 📊 Comparing Both Models

### After Training Both:

**Metrics to Compare:**
1. **Validation Accuracy**: Higher is better
2. **Test Accuracy**: Real-world performance
3. **Top-5 Accuracy**: Robustness measure
4. **Balanced Accuracy**: Performance on minority classes
5. **Training Time**: Efficiency consideration
6. **Model Size**: Deployment consideration

**Expected Results:**
- **ResNet50**: 
  - Pros: Well-established, stable training
  - Cons: Larger model, more parameters
  
- **EfficientNet-B3**: 
  - Pros: Better accuracy, smaller, efficient
  - Cons: Slightly more training time per epoch

**Recommendation for Self-Driving Car:**
Use **EfficientNet-B3** for better accuracy-efficiency trade-off. Safety-critical applications benefit from the higher accuracy.

---

## 🎓 Training Tips

### For Maximum Accuracy:
1. ✅ Use **EfficientNet-B3** architecture
2. ✅ Keep **ROI cropping enabled**
3. ✅ Ensure **weighted sampler is on**
4. ✅ Don't reduce augmentation
5. ✅ Let training complete (don't interrupt)
6. ✅ Monitor for overfitting warnings

### For Faster Training:
1. ⚡ Increase batch size if GPU allows (64 → 128)
2. ⚡ Reduce early stopping patience (7 → 5)
3. ⚡ Use fewer epochs if time-limited
4. ⚡ Use ResNet50 (faster per epoch)

### For Better Generalization:
1. 🎯 Increase validation split (10% → 15%)
2. 🎯 Increase dropout rate (0.4 → 0.5)
3. 🎯 Increase weight decay (1e-4 → 5e-4)
4. 🎯 Add more augmentation

---

## 📧 Next Steps After Training

### 1. Evaluate Test Performance
Run all post-processing cells (31-46) to get comprehensive evaluation.

### 2. Analyze Errors
Check error analysis to understand which classes are confused.

### 3. Compare Models
If you trained both, compare their performance metrics.

### 4. Deploy Best Model
Use the model with highest test accuracy for your self-driving car application.

### 5. Consider Ensemble
Average predictions from both models for even better accuracy.

---

## 🎉 Ready to Train!

You now have everything needed to train your GTSRB traffic sign classifier. Simply:

1. Set `MODEL_ARCH` in Cell 6
2. Run all cells sequentially
3. Wait 2-4 hours for training
4. Analyze results in post-processing
5. (Optional) Train second architecture

**Good luck with your self-driving car project!** 🚗🚦

---

**Document Version:** 1.0  
**For Questions:** See `IMPLEMENTATION_VERIFICATION.md` for detailed technical documentation
