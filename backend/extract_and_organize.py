"""
Extract and organize the dataset from ZIP file.
"""
import os
import zipfile
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
ZIP_PATH = DATASET_DIR / "archive (1).zip"
EXTRACT_DIR = DATASET_DIR / "extracted"
OUTPUT_DIR = BASE_DIR / "data" / "ssl"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

def extract_zip():
    """Extract the dataset ZIP file."""
    print(f"Extracting {ZIP_PATH}...")
    
    # Create extraction directory
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Extracted to: {EXTRACT_DIR}")
        return True
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return False

def organize_dataset():
    """Organize the dataset into train/val/test splits."""
    print("\nOrganizing dataset...")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in ["cattle", "buffalo"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Check for YOLO format (images and labels folders)
    train_dir = EXTRACT_DIR / "train"
    if not (train_dir / "images").exists() or not (train_dir / "labels").exists():
        print("Expected YOLO format with 'images' and 'labels' folders not found.")
        return False
    
    print(f"Found YOLO format dataset at: {train_dir}")
    
    # Get all image files
    image_files = list((train_dir / "images").glob("*.*"))
    
    if not image_files:
        print("No images found in the dataset directory.")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Process image files
    image_data = []
    for img_path in image_files:
        # Get corresponding label file
        label_path = (train_dir / "labels" / img_path.name).with_suffix(".txt")
        
        if not label_path.exists():
            print(f"Warning: No label file found for {img_path}")
            continue
            
        # Read the first line of the label file to get the class
        try:
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    print(f"Warning: Empty label file {label_path}")
                    continue
                    
                # YOLO format: class_id x_center y_center width height
                class_id = int(first_line.split()[0])
                
                # Map class ID to class name
                class_name = "cattle" if class_id == 0 else "buffalo"
                
                image_data.append((str(img_path), class_name))
                
        except Exception as e:
            print(f"Error processing {label_path}: {e}")
    
    if not image_data:
        print("No valid labeled images found.")
        return False
        
    print(f"Successfully processed {len(image_data)} labeled images")
    
    if not image_files:
        print("No images found in the dataset directory.")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Split into train (70%), val (15%), test (15%)
    random.seed(SEED)
    random.shuffle(image_data)
    
    train_data, test_val_data = train_test_split(
        image_data, 
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )
    
    val_data, test_data = train_test_split(
        test_val_data,
        test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # Copy files to their respective directories
    for split_name, files in splits.items():
        print(f"Processing {split_name} set: {len(files)} images")
        for src_path, cls in files:
            src = Path(src_path)
            dst = OUTPUT_DIR / split_name / cls / src.name
            
            # Create destination directory if it doesn't exist
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copy2(src, dst)
                # Copy corresponding label file if it exists
                label_src = (src.parent.parent / "labels" / src.name).with_suffix(".txt")
                if label_src.exists():
                    label_dst = (OUTPUT_DIR / split_name / "labels" / src.name).with_suffix(".txt")
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_src, label_dst)
            except Exception as e:
                print(f"Error copying {src}: {e}")
                continue
    
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
    print("=== Dataset Extraction and Organization ===")
    
    # Extract the ZIP file
    if not extract_zip():
        print("Failed to extract the dataset. Please check the ZIP file.")
        exit(1)
    
    # Organize the dataset
    if organize_dataset():
        check_data_distribution()
        print("\nDataset organization complete!")
        print(f"Organized dataset saved to: {OUTPUT_DIR}")
    else:
        print("Failed to organize the dataset.")
