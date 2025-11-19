"""
GTSRB Dataset Utilities
Handles loading and preprocessing of the German Traffic Sign Recognition Benchmark dataset
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """
    Custom PyTorch Dataset for GTSRB that reads from CSV files.
    Supports both Train and Test datasets with optional ROI cropping.
    """
    
    def __init__(self, csv_path, root_dir, transform=None, use_roi=True):
        """
        Args:
            csv_path (str): Path to the CSV file (Train.csv or Test.csv)
            root_dir (str): Root directory containing the dataset
            transform (callable, optional): Optional transform to be applied on images
            use_roi (bool): Whether to crop images using ROI coordinates
        """
        self.annotations = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.use_roi = use_roi
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['Path'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply ROI cropping if enabled
        if self.use_roi:
            roi_x1 = self.annotations.iloc[idx]['Roi.X1']
            roi_y1 = self.annotations.iloc[idx]['Roi.Y1']
            roi_x2 = self.annotations.iloc[idx]['Roi.X2']
            roi_y2 = self.annotations.iloc[idx]['Roi.Y2']
            image = image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
        
        label = self.annotations.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_gtsrb_info(dataset_dir):
    """
    Load and analyze GTSRB dataset information from CSV files.
    
    Args:
        dataset_dir (str): Path to dataset directory containing Train.csv, Test.csv, Meta.csv
        
    Returns:
        dict: Comprehensive dataset statistics
    """
    train_csv = os.path.join(dataset_dir, 'Train.csv')
    test_csv = os.path.join(dataset_dir, 'Test.csv')
    meta_csv = os.path.join(dataset_dir, 'Meta.csv')
    
    # Check if files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train.csv not found at {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test.csv not found at {test_csv}")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Meta.csv not found at {meta_csv}")
    
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    meta_df = pd.read_csv(meta_csv)
    
    # Basic statistics
    num_train = len(train_df)
    num_test = len(test_df)
    num_classes = len(meta_df)
    
    # Class distribution in training set
    train_class_counts = train_df['ClassId'].value_counts().sort_index()
    
    # Class distribution in test set
    test_class_counts = test_df['ClassId'].value_counts().sort_index()
    
    # Image size statistics
    train_sizes = train_df[['Width', 'Height']]
    avg_width = train_sizes['Width'].mean()
    avg_height = train_sizes['Height'].mean()
    min_width = train_sizes['Width'].min()
    max_width = train_sizes['Width'].max()
    min_height = train_sizes['Height'].min()
    max_height = train_sizes['Height'].max()
    
    # Shape and color distribution from meta
    shape_counts = meta_df['ShapeId'].value_counts().sort_index()
    color_counts = meta_df['ColorId'].value_counts().sort_index()
    
    # Create comprehensive info dictionary
    info = {
        'num_train': num_train,
        'num_test': num_test,
        'num_classes': num_classes,
        'train_class_counts': train_class_counts.to_dict(),
        'test_class_counts': test_class_counts.to_dict(),
        'avg_width': avg_width,
        'avg_height': avg_height,
        'min_width': min_width,
        'max_width': max_width,
        'min_height': min_height,
        'max_height': max_height,
        'shape_distribution': shape_counts.to_dict(),
        'color_distribution': color_counts.to_dict(),
        'meta_df': meta_df,
        'train_df': train_df,
        'test_df': test_df
    }
    
    return info


def print_gtsrb_summary(info):
    """
    Print a formatted summary of GTSRB dataset information.
    
    Args:
        info (dict): Dataset information from load_gtsrb_info()
    """
    print("\n" + "="*80)
    print("🚦 GTSRB DATASET SUMMARY - German Traffic Sign Recognition Benchmark")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'Total Classes:':<30} {info['num_classes']}")
    print(f"{'Training Samples:':<30} {info['num_train']:,}")
    print(f"{'Test Samples:':<30} {info['num_test']:,}")
    print(f"{'Total Samples:':<30} {info['num_train'] + info['num_test']:,}")
    
    print(f"\n🖼️  IMAGE SIZE STATISTICS (Original)")
    print(f"{'Average Size:':<30} {info['avg_width']:.1f} x {info['avg_height']:.1f} pixels")
    print(f"{'Size Range (Width):':<30} {info['min_width']} - {info['max_width']} pixels")
    print(f"{'Size Range (Height):':<30} {info['min_height']} - {info['max_height']} pixels")
    
    # Shape distribution
    shape_names = {0: 'Circular', 1: 'Triangular', 2: 'Octagonal', 3: 'Inverted Triangle', 4: 'Diamond'}
    print(f"\n🔷 SIGN SHAPE DISTRIBUTION")
    for shape_id, count in sorted(info['shape_distribution'].items()):
        shape_name = shape_names.get(shape_id, f'Unknown ({shape_id})')
        percentage = (count / info['num_classes']) * 100
        print(f"  {shape_name:<20} {count:>3} classes ({percentage:>5.1f}%)")
    
    # Color distribution
    color_names = {0: 'Red Border', 1: 'Blue', 2: 'Other', 3: 'White/Blue'}
    print(f"\n🎨 SIGN COLOR DISTRIBUTION")
    for color_id, count in sorted(info['color_distribution'].items()):
        color_name = color_names.get(color_id, f'Unknown ({color_id})')
        percentage = (count / info['num_classes']) * 100
        print(f"  {color_name:<20} {count:>3} classes ({percentage:>5.1f}%)")
    
    # Class balance analysis
    train_counts = list(info['train_class_counts'].values())
    max_samples = max(train_counts)
    min_samples = min(train_counts)
    avg_samples = np.mean(train_counts)
    std_samples = np.std(train_counts)
    imbalance_ratio = max_samples / min_samples
    
    print(f"\n⚖️  CLASS BALANCE (Training Set)")
    print(f"{'Most populated class:':<30} {max_samples:,} samples")
    print(f"{'Least populated class:':<30} {min_samples:,} samples")
    print(f"{'Average per class:':<30} {avg_samples:.1f} samples")
    print(f"{'Standard deviation:':<30} {std_samples:.1f}")
    print(f"{'Imbalance ratio:':<30} {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print(f"  ⚠️  High class imbalance detected! Consider using weighted sampling.")
    elif imbalance_ratio > 2:
        print(f"  ⚡ Moderate class imbalance. Weighted sampling recommended.")
    else:
        print(f"  ✅ Classes are reasonably balanced.")
    
    print("\n" + "="*80)


def print_class_distribution_table(info, top_n=10):
    """
    Print detailed per-class distribution table.
    
    Args:
        info (dict): Dataset information from load_gtsrb_info()
        top_n (int): Number of top/bottom classes to show in detail
    """
    print(f"\n📋 PER-CLASS DISTRIBUTION (Top {top_n} and Bottom {top_n})")
    print("="*95)
    print(f"{'ClassID':<10} {'SignID':<12} {'Train':<12} {'Test':<12} {'Total':<12} {'Shape':<15} {'Color':<15}")
    print("="*95)
    
    # Combine data
    train_counts = info['train_class_counts']
    test_counts = info['test_class_counts']
    meta_df = info['meta_df']
    
    # Create combined list
    class_data = []
    for class_id in range(info['num_classes']):
        meta_row = meta_df[meta_df['ClassId'] == class_id].iloc[0]
        train_count = train_counts.get(class_id, 0)
        test_count = test_counts.get(class_id, 0)
        total_count = train_count + test_count
        
        class_data.append({
            'ClassId': class_id,
            'SignId': meta_row['SignId'],
            'ShapeId': meta_row['ShapeId'],
            'ColorId': meta_row['ColorId'],
            'Train': train_count,
            'Test': test_count,
            'Total': total_count
        })
    
    # Sort by total count
    class_data_sorted = sorted(class_data, key=lambda x: x['Total'], reverse=True)
    
    shape_names = {0: 'Circular', 1: 'Triangular', 2: 'Octagonal', 3: 'Inv.Triangle', 4: 'Diamond'}
    color_names = {0: 'Red Border', 1: 'Blue', 2: 'Other', 3: 'White/Blue'}
    
    # Print top N
    print(f"\n🔝 TOP {top_n} MOST POPULATED CLASSES:")
    for i, data in enumerate(class_data_sorted[:top_n], 1):
        shape_name = shape_names.get(data['ShapeId'], 'Unknown')
        color_name = color_names.get(data['ColorId'], 'Unknown')
        print(f"{data['ClassId']:<10} {str(data['SignId']):<12} {data['Train']:<12} {data['Test']:<12} "
              f"{data['Total']:<12} {shape_name:<15} {color_name:<15}")
    
    # Print bottom N
    print(f"\n🔻 BOTTOM {top_n} LEAST POPULATED CLASSES:")
    for i, data in enumerate(class_data_sorted[-top_n:], 1):
        shape_name = shape_names.get(data['ShapeId'], 'Unknown')
        color_name = color_names.get(data['ColorId'], 'Unknown')
        print(f"{data['ClassId']:<10} {str(data['SignId']):<12} {data['Train']:<12} {data['Test']:<12} "
              f"{data['Total']:<12} {shape_name:<15} {color_name:<15}")
    
    print("="*95)


def compute_dataset_statistics(csv_path, root_dir):
    """
    Compute mean and standard deviation for normalization.
    
    Args:
        csv_path (str): Path to Train.csv
        root_dir (str): Root directory of dataset
        
    Returns:
        tuple: (mean, std) for RGB channels
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    print("🔄 Computing dataset mean and std for normalization...")
    print("   This may take a few minutes...")
    
    # Create dataset without normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = GTSRBDataset(csv_path, root_dir, transform=transform, use_roi=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    print(f"✅ Dataset statistics computed:")
    print(f"   Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"   Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    
    return mean.tolist(), std.tolist()
