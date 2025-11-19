# GTSRB Preprocessing & Dataset Changes

## Overview
This document explains the modifications made to adapt the fruit classifier code for the GTSRB traffic sign dataset.

## Key Differences: Fruit Dataset vs GTSRB Dataset

### Original Fruit Classifier Structure
- **Structure**: All images in class folders (e.g., `dataset/apple/`, `dataset/banana/`)
- **Split**: Code dynamically splits into train/val/test (70/15/15)
- **Loading**: Uses `torchvision.datasets.ImageFolder`
- **Classes**: 5 fruit classes
- **Balance**: Relatively balanced dataset

### GTSRB Dataset Structure
- **Structure**: CSV-based with metadata (Train.csv, Test.csv, Meta.csv)
- **Split**: Pre-split training and test sets
- **Loading**: Custom `GTSRBDataset` class reading from CSV
- **Classes**: 43 traffic sign classes
- **Balance**: Highly imbalanced (ratio: ~40x between most/least populated)
- **Additional Info**: ROI coordinates, sign shapes, colors, official IDs

## Major Code Changes

### 1. New Utility File: `gtsrb_dataset.py`

Created a comprehensive utility module with:

#### `GTSRBDataset` Class
- Custom PyTorch Dataset that reads from CSV files
- Loads images using paths from CSV
- **ROI Cropping**: Optionally crops to sign bounding box before resizing
- Applies transforms
- Returns (image, label) pairs

#### `load_gtsrb_info()` Function
- Parses Train.csv, Test.csv, and Meta.csv
- Computes comprehensive statistics:
  - Class distribution (train/test)
  - Image size statistics
  - Shape and color distribution
  - Class imbalance metrics

#### `print_gtsrb_summary()` Function
- Formatted display of dataset statistics
- Shows overall metrics, image sizes, shape/color distribution
- Highlights class imbalance issues

#### `print_class_distribution_table()` Function
- Detailed per-class breakdown
- Shows top/bottom N classes by sample count
- Includes shape, color, and official sign IDs

### 2. Global Configuration Changes

**Updated Parameters**:
```python
NUM_CLASSES = 43              # Was 5 for fruits
BATCH_SIZE = 64               # Increased from 32
EPOCHS = 25                   # Increased from 20
VALIDATION_SPLIT = 0.15       # New: split from training set
```

**New Configuration Options**:
- `USE_ROI_CROP`: Enable/disable ROI bounding box cropping
- `USE_IMAGENET_NORM`: Choose between ImageNet or GTSRB-specific normalization
- `TRAIN_CSV`, `TEST_CSV`, `META_CSV`: CSV file paths

### 3. Data Augmentation Updates

**Traffic Sign Specific Augmentations**:
```python
- RandomRotation(15°)         # Signs can be at angles
- RandomAffine(translate/scale) # Viewing distance variation
- ColorJitter                  # Lighting/weather conditions
- RandomPerspective            # Viewing angle simulation
- RandomErasing                # Occlusion/dirt simulation
```

**Rationale**: Traffic signs are photographed in real-world conditions with:
- Varying lighting (day/night, shadows)
- Different viewing angles
- Weather effects (rain, snow, fog)
- Partial occlusions (trees, poles, dirt)

### 4. Dataset Loading Changes

**Before (Fruit Classifier)**:
```python
full_dataset = datasets.ImageFolder(root=DATASET_DIR)
train, val, test = random_split(full_dataset, [0.7, 0.15, 0.15])
```

**After (GTSRB)**:
```python
# Load from CSV files
full_train = GTSRBDataset(TRAIN_CSV, DATASET_DIR, use_roi=True)
test_dataset = GTSRBDataset(TEST_CSV, DATASET_DIR, use_roi=True)

# Split training into train/val
train_subset, val_subset = random_split(full_train, [train_size, val_size])
```

**Key Difference**: GTSRB test set is pre-defined and should NOT be used for validation during training.

### 5. Class Weight Computation

**Enhanced for Severe Imbalance**:
- Reads class distribution from Train.csv
- Computes inverse frequency weights
- Normalizes weights for stability
- Creates sample weights for WeightedRandomSampler

**Imbalance Statistics**:
- Most populated class: ~2,000 samples
- Least populated class: ~200 samples
- Ratio: ~10-40x depending on class pair

### 6. Visualization Additions

**New Cell: Sample Image Visualization**
- Displays 12 random traffic signs from different classes
- Shows both original and ROI-cropped versions
- Includes image dimensions and class IDs
- Helps understand data diversity and quality

### 7. Summary Output Enhancements

**Comprehensive Preprocessing Summary**:
- Dataset name and source
- Sample counts (train/val/test)
- Image processing details (size, ROI, normalization)
- Augmentation techniques applied
- Class balancing strategy
- Batch configuration
- Device information

## Important Considerations for High Accuracy

### 1. ROI Cropping (ENABLED by default)
- **Why**: Focuses on the sign itself, removing background noise
- **Impact**: Significantly improves accuracy (5-10% improvement expected)
- **Trade-off**: Loses context, but traffic signs are recognizable independently

### 2. Class Imbalance Handling
- **Weighted Sampling**: Ensures minority classes are seen equally often
- **Class Weights in Loss**: Penalizes misclassification of rare classes more
- **Both Enabled**: Recommended for maximum performance on imbalanced data

### 3. Normalization Strategy
- **ImageNet (Recommended)**: Better for transfer learning with pretrained ResNet50
- **GTSRB-Specific**: May provide marginal improvement but requires retraining from scratch

### 4. Data Augmentation
- **Moderate Augmentation**: Traffic signs must remain recognizable
- **Geometric**: Small rotations, translations, scales (real-world variation)
- **Photometric**: Color jitter for lighting/weather conditions
- **Avoid**: Heavy distortions that change sign appearance

### 5. Batch Size Optimization
- **64 recommended**: Traffic sign images are smaller (after crop: typically 30-100px original)
- **GPU Memory**: Adjust based on available VRAM
- **Trade-off**: Larger batches = more stable gradients but less frequent updates

## Expected Performance

### Baseline (Transfer Learning, Basic Config)
- **Expected Accuracy**: 95-98%
- **Training Time**: ~10-15 minutes on GPU (25 epochs)

### Optimized (All Techniques)
- **Expected Accuracy**: 98-99.5%
- **Best Published Results**: 99.5-99.7% (ensemble methods)
- **Critical for Self-Driving**: High accuracy crucial for safety

## Next Steps (After Preprocessing)

1. **Model Architecture**: Keep ResNet50 or try EfficientNet-B0
2. **Training Strategy**: 
   - Start with frozen backbone (fine-tuning)
   - Gradually unfreeze layers
   - Use learning rate scheduling
3. **Evaluation Metrics**: 
   - Per-class accuracy (identify weak classes)
   - Confusion matrix analysis
   - Error analysis on misclassified signs
4. **Hyperparameter Tuning**:
   - Learning rate (try 1e-3, 1e-4, 1e-5)
   - Dropout (add to FC layers if overfitting)
   - Weight decay (L2 regularization)
5. **Ensemble Methods** (for maximum accuracy):
   - Multiple models with different architectures
   - Test-time augmentation
   - Model averaging

## Files Modified

1. **Created**: `src/utils/gtsrb_dataset.py` - GTSRB dataset utilities
2. **Modified**: `src/traffic_sign_classifier_model.ipynb` - Main notebook
   - Cells 1-10: Preprocessing and data loading
   - Global configuration
   - Dataset information display
   - Augmentation definitions
   - DataLoader creation
   - Summary statistics

## Testing Checklist

Before training, verify:
- [ ] All CSV files are in `dataset/` directory
- [ ] Image directories exist (`Train/`, `Test/`, `Meta/`)
- [ ] GPU is detected and CUDA is available
- [ ] Sample visualization displays correctly
- [ ] Dataset info shows 43 classes, correct sample counts
- [ ] Class weights computed without errors
- [ ] DataLoaders created successfully
- [ ] First batch can be loaded (test with `next(iter(train_loader))`)

## Performance Tips for Self-Driving Application

1. **Real-time Inference**: After training, optimize model with TorchScript or ONNX
2. **Robustness**: Test on images with various lighting, weather, occlusions
3. **Uncertainty Estimation**: Consider using dropout at inference for confidence scores
4. **Continuous Learning**: Collect misclassified examples for retraining
5. **Safety Critical**: Implement fallback mechanisms for low-confidence predictions
