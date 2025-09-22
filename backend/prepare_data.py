"""
Prepare training data for species classification.
"""
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
SSL_DIR = DATA_DIR / "ssl"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

def organize_images():
    """Organize images into train/val/test splits."""
    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in ["cattle", "buffalo"]:
            (SSL_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {IMAGES_DIR}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # For now, we'll need to manually label the images
    # In a real scenario, you would have labeled data or a way to get labels
    print("\n=== Manual Labeling Required ===")
    print("Please move the images to the appropriate directories:")
    print(f"- Cattle images: {SSL_DIR}/[train/val/test]/cattle/")
    print(f"- Buffalo images: {SSL_DIR}/[train/val/test]/buffalo/")
    print("\nSuggested split ratios:")
    print(f"- Training: {TRAIN_RATIO*100:.0f}%")
    print(f"- Validation: {VAL_RATIO*100:.0f}%")
    print(f"- Test: {TEST_RATIO*100:.0f}%")
    print("\nAfter organizing the images, you can train the model.")

def check_data_balance():
    """Check the balance of the dataset."""
    print("\n=== Data Distribution ===")
    for split in ["train", "val", "test"]:
        print(f"\n{split.capitalize()}:")
        for cls in ["cattle", "buffalo"]:
            count = len(list((SSL_DIR / split / cls).glob("*")))
            print(f"- {cls}: {count} images")

if __name__ == "__main__":
    organize_images()
    check_data_balance()
