"""
Organize the Cows and Buffalo dataset into the required directory structure.
"""
import os
import shutil
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "data" / "ssl"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

def create_directories():
    """Create the required directory structure."""
    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in ["cattle", "buffalo"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def organize_dataset():
    """Organize the dataset into train/val/test splits."""
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset")
        print(f"And extract it to: {DATASET_DIR}")
        return False
    
    # Map original class names to our class names
    class_mapping = {
        'cow': 'cattle',
        'buffalo': 'buffalo'
    }
    
    # Get all image files
    image_files = []
    for cls in class_mapping.keys():
        class_dir = DATASET_DIR / cls
        if class_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend([(str(f), class_mapping[cls]) for f in class_dir.glob(ext)])
    
    if not image_files:
        print("No images found in the dataset directory.")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Split into train, val, test
    random.seed(SEED)
    random.shuffle(image_files)
    
    # Calculate split indices
    train_end = int(len(image_files) * TRAIN_RATIO)
    val_end = train_end + int(len(image_files) * VAL_RATIO)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Copy files to their respective directories
    for split_name, files in splits.items():
        print(f"Processing {split_name} set: {len(files)} images")
        for src_path, cls in files:
            src = Path(src_path)
            dst = OUTPUT_DIR / split_name / cls / src.name
            shutil.copy2(src, dst)
    
    return True

def check_data_distribution():
    """Check the distribution of the organized dataset."""
    print("\n=== Data Distribution ===")
    total = 0
    for split in ["train", "val", "test"]:
        print(f"\n{split.capitalize()}:")
        split_total = 0
        for cls in ["cattle", "buffalo"]:
            count = len(list((OUTPUT_DIR / split / cls).glob("*")))
            print(f"- {cls}: {count} images")
            split_total += count
        print(f"Total: {split_total} images")
        total += split_total
    print(f"\nTotal images: {total}")

if __name__ == "__main__":
    print("=== Dataset Organization ===")
    create_directories()
    if organize_dataset():
        check_data_distribution()
        print("\nDataset organization complete!")
        print(f"Organized dataset saved to: {OUTPUT_DIR}")
    else:
        print("Failed to organize dataset. Please check the error messages above.")
