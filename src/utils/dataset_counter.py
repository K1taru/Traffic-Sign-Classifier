import os
from collections import defaultdict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def get_folder_size(path: str) -> int:
    """Calculate total size of all files in a folder."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    return total_size


def CountDataset(dataset_path: str) -> dict:
    """
    Count images and compute statistics for each class in the dataset.
    
    Args:
        dataset_path: Path to the dataset directory containing class folders
        
    Returns:
        Dictionary containing dataset statistics and per-class information
    """
    class_counts = defaultdict(int)
    folder_sizes = {}

    # Get all valid subfolders (class directories)
    try:
        class_folders = sorted([
            f for f in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('.')
        ])
    except FileNotFoundError:
        print(f"❌ Folder not found: {dataset_path}")
        return {}
    except PermissionError:
        print(f"⚠️  Permission denied: {dataset_path}")
        return {}

    # Count images & size per class
    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        count = 0
        for root, _, files in os.walk(class_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    count += 1
        class_counts[class_name] = count
        folder_sizes[class_name] = get_folder_size(class_path)

    if not class_counts:
        print("⚠️  No images found in dataset.")
        return {}

    # Compute summary values
    max_samples_per_class = max(class_counts.values())
    total_samples = sum(class_counts.values())
    total_size_bytes = sum(folder_sizes.values())

    # Build dataset_info dictionary
    dataset_info = {}
    for class_name, count in class_counts.items():
        class_ratio = count / max_samples_per_class if count > 0 else 0
        deficit = max_samples_per_class - count
        size_mb = folder_sizes[class_name] / (1024 ** 2)
        dataset_info[class_name] = {
            "samples": count,
            "class_ratio": class_ratio,
            "deficit": deficit,
            "size_mb": size_mb
        }

    dataset_info["total_samples"] = total_samples
    dataset_info["max_samples_per_class"] = max_samples_per_class
    dataset_info["total_size_mb"] = total_size_bytes / (1024 ** 2)
    dataset_info["num_classes"] = len(class_folders)

    # Dynamic column widths for pretty printing
    class_w = max(10, max((len(f) for f in class_folders), default=10) + 2)
    samples_w = max(9, max((len(str(dataset_info[f]["samples"])) for f in class_folders), default=9) + 2)
    ratio_w = 14
    pad_between_ratio_deficit = 3
    deficit_w = max(10, max((len(str(dataset_info[f]["deficit"])) for f in class_folders), default=10) + 3)
    size_w = 12
    mb_suffix = " MB"

    # Compute separator length for clean borders
    sep_len = class_w + samples_w + ratio_w + pad_between_ratio_deficit + deficit_w + size_w + len(mb_suffix) + 3

    # Print Header
    print("\n📊 DATASET SUMMARY")
    print("=" * sep_len)
    header = (
        f"{'Class':<{class_w}}"
        f"{'samples':>{samples_w}}"
        f"{'class_ratio':>{ratio_w}}"
        f"{' ' * pad_between_ratio_deficit}"
        f"{'deficit':>{deficit_w}}"
        f"{'size_mb':>{size_w + len(mb_suffix)}}"
    )
    print(header)
    print("=" * sep_len)

    # Print each class
    for class_name in class_folders:
        info = dataset_info[class_name]
        print(
            f"{class_name:<{class_w}}"
            f"{info['samples']:>{samples_w}d}"
            f"{info['class_ratio']:>{ratio_w}.4f}"
            f"{' ' * pad_between_ratio_deficit}"
            f"{info['deficit']:>{deficit_w}d}"
            f"{info['size_mb']:>{size_w}.2f}{mb_suffix}"
        )

    # Print summary values
    print("=" * sep_len)
    print(f"{'num_classes':<{class_w}}{dataset_info['num_classes']:>{samples_w}}")
    print(f"{'total_samples':<{class_w}}{dataset_info['total_samples']:>{samples_w}}")
    print(f"{'max_samples_per_class':<{class_w}}{dataset_info['max_samples_per_class']:>{samples_w}}")
    print(
        f"{'total_size_mb':<{class_w}}"
        f"{dataset_info['total_size_mb']:>{samples_w + ratio_w + pad_between_ratio_deficit + deficit_w + size_w}.2f}{mb_suffix}"
    )
    print("=" * sep_len)

    return dataset_info