# Dataset Directory

This directory contains the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## 📥 Dataset Download

The dataset is automatically downloaded using:
```bash
python utils/kaggle_dataset_installer.py
```

**Requirements**: Kaggle API credentials in `~/.kaggle/kaggle.json`

## 📁 Expected Structure

After download, this directory should contain:

```
dataset/
├── .gitkeep                    # Keep directory in git
├── Train.csv                   # Training data metadata (39,209 samples)
├── Test.csv                    # Test data metadata (12,630 samples)
├── Meta.csv                    # Class ID to sign name mapping
├── Train/                      # Training images organized by class
│   ├── 0/                      # Class 0 images
│   ├── 1/                      # Class 1 images
│   ├── ...
│   └── 42/                     # Class 42 images
└── Test/                       # Test images (not organized by class)
    ├── 00000.png
    ├── 00001.png
    └── ...
```

## 📊 Dataset Information

### Statistics
- **Total Training Images**: 39,209
- **Total Test Images**: 12,630
- **Number of Classes**: 43
- **Image Format**: PNG
- **Image Sizes**: Variable (resized to 224×224 for training)
- **Class Imbalance**: 10-40× ratio between classes

### CSV Files

#### Train.csv
Columns: `Width`, `Height`, `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`, `ClassId`, `Path`

Example:
```csv
Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
48,48,5,7,42,41,0,Train/0/00000_00000.png
```

#### Test.csv
Columns: `Width`, `Height`, `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`, `Path`

Example:
```csv
Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,Path
59,60,4,5,53,55,Test/00000.png
```

#### Meta.csv
Columns: `ClassId`, `SignName`

Example:
```csv
ClassId,SignName
0,Speed limit (20km/h)
1,Speed limit (30km/h)
2,Speed limit (50km/h)
...
```

## 🔍 ROI (Region of Interest)

Each image has ROI coordinates that define the bounding box of the traffic sign:
- **Roi.X1, Roi.Y1**: Top-left corner
- **Roi.X2, Roi.Y2**: Bottom-right corner

The training pipeline uses these coordinates to crop and focus on the sign content, removing background noise.

## 🚫 Git Configuration

**Important**: Dataset files are too large for Git and should be excluded.

### Add to `.gitignore`:
```gitignore
# Dataset files (large, downloaded separately)
dataset/*.csv
dataset/Train/
dataset/Test/
dataset/*.zip
dataset/*.tar.gz

# Keep directory structure
!dataset/.gitkeep
!dataset/dataset_info.txt
```

## 📈 Class Distribution

The GTSRB dataset is highly imbalanced:

| Characteristic | Value |
|---------------|-------|
| Most populated class | ~2,000 samples |
| Least populated class | ~200 samples |
| Imbalance ratio | ~10-40× |

**Solution**: The training pipeline handles this using:
- Weighted random sampling
- Class-weighted loss function
- Per-class evaluation metrics

## 🔄 Dataset Verification

After download, verify the dataset:

```python
import os
import pandas as pd

# Check CSV files
assert os.path.exists('dataset/Train.csv'), "Train.csv not found"
assert os.path.exists('dataset/Test.csv'), "Test.csv not found"
assert os.path.exists('dataset/Meta.csv'), "Meta.csv not found"

# Check counts
train_df = pd.read_csv('dataset/Train.csv')
test_df = pd.read_csv('dataset/Test.csv')
meta_df = pd.read_csv('dataset/Meta.csv')

print(f"✅ Training samples: {len(train_df):,}")
print(f"✅ Test samples: {len(test_df):,}")
print(f"✅ Number of classes: {len(meta_df)}")

# Expected: 39,209 training, 12,630 test, 43 classes
```

## 📚 Dataset Citation

If you use the GTSRB dataset, please cite:

```bibtex
@inproceedings{Stallkamp2012,
  author = {Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  title = {Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition},
  booktitle = {Neural Networks},
  year = {2012},
  pages = {323-332}
}
```

## 🔗 Dataset Source

- **Official Website**: https://benchmark.ini.rub.de/gtsrb_dataset.html
- **Kaggle**: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- **Paper**: [Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition](https://benchmark.ini.rub.de/gtsrb_news.html)

## 💾 Disk Space

The complete dataset requires approximately:
- **Training images**: ~300 MB
- **Test images**: ~90 MB
- **CSV files**: ~5 MB
- **Total**: ~400 MB

---

**💡 Note**: This directory is kept in Git using `.gitkeep`, but the actual dataset files should be downloaded locally and not committed to version control.
