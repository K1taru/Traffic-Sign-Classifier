# Models Directory

This directory stores trained model checkpoints from the GTSRB traffic sign classifier.

## 📁 Model File Naming Convention

Models are automatically saved with descriptive filenames during training:

```
GTSRB_{architecture}_E{epoch}_VAL{accuracy}.pth
```

### Filename Components:
- **GTSRB**: Dataset identifier
- **{architecture}**: Model architecture name
  - `resnet50`: ResNet50 (25.6M parameters)
  - `efficientnet_b3`: EfficientNet-B3 (12M parameters)
- **E{epoch}**: Best epoch number during training
- **VAL{accuracy}**: Validation accuracy (percentage)

### Example Filenames:
```
GTSRB_resnet50_E18_VAL98.45.pth
GTSRB_efficientnet_b3_E22_VAL99.12.pth
GTSRB_resnet50_E15_VAL97.89.pth
```

## 💾 Checkpoint Contents

Each `.pth` file contains:

```python
{
    'epoch': int,                    # Best epoch number
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'val_acc': float,                # Best validation accuracy
    'model_arch': str,               # Architecture name
    'num_classes': int,              # Number of classes (43)
    'history': dict                  # Training history (all metrics)
}
```

## 🔄 Loading a Trained Model

```python
import torch
from torchvision import models
import torch.nn as nn

# Load checkpoint
checkpoint = torch.load('models/GTSRB_efficientnet_b3_E22_VAL99.12.pth')

# Get architecture info
arch = checkpoint['model_arch']
num_classes = checkpoint['num_classes']

# Recreate model structure
if arch == 'resnet50':
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
elif arch == 'efficientnet_b3':
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Print info
print(f"Loaded {arch} model")
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Validation accuracy: {checkpoint['val_acc']*100:.2f}%")
```

## 📊 Accessing Training History

```python
# Load checkpoint
checkpoint = torch.load('models/GTSRB_efficientnet_b3_E22_VAL99.12.pth')

# Get training history
history = checkpoint['history']

# Available metrics:
print("Available metrics:", history.keys())
# dict_keys(['train_loss', 'val_loss', 'train_acc', 'val_acc', 
#            'train_top5_acc', 'val_top5_acc', 'learning_rates'])

# Example: Plot validation accuracy
import matplotlib.pyplot as plt

plt.plot(history['val_acc'])
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```

## 📈 Model Comparison

To compare multiple trained models:

```python
import os
import torch

models_dir = 'models/'
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

for model_file in sorted(model_files):
    checkpoint = torch.load(os.path.join(models_dir, model_file))
    print(f"\n{model_file}")
    print(f"  Architecture: {checkpoint['model_arch']}")
    print(f"  Best Epoch: {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    # Get final training accuracy
    if 'history' in checkpoint:
        final_train_acc = checkpoint['history']['train_acc'][-1]
        print(f"  Train Accuracy: {final_train_acc*100:.2f}%")
```

## 🗑️ Disk Space Management

Model files can be large (~100-500MB each). To save space:

### Keep Only Best Models
```bash
# Keep only models with >98% accuracy
# Delete others manually or use script
```

### Compress Old Models
```bash
# On Windows (PowerShell)
Compress-Archive -Path "models/GTSRB_*.pth" -DestinationPath "models_archive.zip"
```

```bash
# On Linux/Mac
tar -czf models_archive.tar.gz models/GTSRB_*.pth
```

## 🚫 Git Configuration

**Important**: Model files are typically too large for Git and should be excluded.

### Add to `.gitignore`:
```gitignore
# Trained models (large files)
models/*.pth
models/*.pt

# Keep the directory structure
!models/.gitkeep
!models/README.md
```

### For Large Model Sharing:
Consider using:
- **Git LFS** (Large File Storage) for version control
- **Google Drive / Dropbox** for team sharing
- **Hugging Face Hub** for public model hosting
- **AWS S3 / Azure Blob** for cloud storage

## 📝 Model Metadata

Create a `model_registry.json` to track your models:

```json
{
  "models": [
    {
      "filename": "GTSRB_efficientnet_b3_E22_VAL99.12.pth",
      "architecture": "efficientnet_b3",
      "val_accuracy": 99.12,
      "test_accuracy": 98.95,
      "training_date": "2025-11-19",
      "training_time_hours": 3.5,
      "notes": "Best model - production ready"
    },
    {
      "filename": "GTSRB_resnet50_E18_VAL98.45.pth",
      "architecture": "resnet50",
      "val_accuracy": 98.45,
      "test_accuracy": 98.23,
      "training_date": "2025-11-19",
      "training_time_hours": 2.8,
      "notes": "Baseline model for comparison"
    }
  ]
}
```

## 🎯 Expected Model Sizes

| Architecture | Approximate Size |
|-------------|-----------------|
| ResNet50 | ~100 MB |
| EfficientNet-B3 | ~50 MB |

*Sizes include model weights, optimizer state, and training history*

## ✅ Verification

To verify a model checkpoint is valid:

```python
import torch

def verify_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        required_keys = ['epoch', 'model_state_dict', 'val_acc', 'model_arch']
        for key in required_keys:
            assert key in checkpoint, f"Missing key: {key}"
        
        print(f"✅ Valid checkpoint: {filepath}")
        print(f"   Architecture: {checkpoint['model_arch']}")
        print(f"   Validation Acc: {checkpoint['val_acc']*100:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Invalid checkpoint: {filepath}")
        print(f"   Error: {str(e)}")
        return False

# Verify all models
import os
models_dir = 'models/'
for f in os.listdir(models_dir):
    if f.endswith('.pth'):
        verify_checkpoint(os.path.join(models_dir, f))
```

## 🔒 Security Note

**Warning**: PyTorch models use pickle for serialization, which can execute arbitrary code. Only load models from trusted sources.

```python
# Safer loading (prevents code execution)
checkpoint = torch.load(filepath, weights_only=True)  # PyTorch 2.0+
```

---

**💡 Tip**: After training completes, always verify your model loads correctly before closing the notebook!
