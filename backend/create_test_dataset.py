"""
Create a small test dataset for cattle vs buffalo classification.
"""
import os
import shutil
from pathlib import Path
import urllib.request
import tempfile

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "data" / "ssl"

# Sample image URLs (public domain/CC0 images)
SAMPLE_IMAGES = {
    'cattle': [
        'https://upload.wikimedia.org/wikipedia/commons/0/0c/Cow_female_black_white.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/3/30/Cow_%28Fleckvieh_breed%29_Oeschinensee_SLA16.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/8/8c/Cow_%28Fleckvieh_breed%29_Oeschinensee_SLA16.jpg',
    ],
    'buffalo': [
        'https://upload.wikimedia.org/wikipedia/commons/1/13/Water_Buffalo_%28Bubalus_bubalis%29_by_N_A_Nazeer.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/7/73/Mindanao_dwarf_buffalo.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/8/80/Water_buffalo_in_Tirunelveli.jpg',
    ]
}

def download_images():
    """Download sample images for testing."""
    print("Downloading sample images...")
    
    # Create directories
    for cls in SAMPLE_IMAGES.keys():
        (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    # Download images
    for cls, urls in SAMPLE_IMAGES.items():
        print(f"Downloading {cls} images...")
        for i, url in enumerate(urls):
            try:
                filename = f"{cls}{i+1}.jpg"
                filepath = DATASET_DIR / cls / filename
                
                if not filepath.exists():
                    with urllib.request.urlopen(url) as response, open(filepath, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                    print(f"  - Downloaded {filename}")
                else:
                    print(f"  - {filename} already exists")
                    
            except Exception as e:
                print(f"  - Error downloading {url}: {str(e)}")
    
    print("\nSample dataset created at:", DATASET_DIR)

def organize_dataset():
    """Organize the dataset into train/val/test splits."""
    from sklearn.model_selection import train_test_split
    import random
    
    print("\nOrganizing dataset...")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in SAMPLE_IMAGES.keys():
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for cls in SAMPLE_IMAGES.keys():
        class_dir = DATASET_DIR / cls
        if class_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend([(str(f), cls) for f in class_dir.glob(ext)])
    
    if not image_files:
        print("No images found in the dataset directory.")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Split into train (60%), val (20%), test (20%)
    random.seed(42)
    random.shuffle(image_files)
    
    train_files, test_files = train_test_split(image_files, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
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
    for split in ["train", "val", "test"]:
        print(f"\n{split.capitalize()}:")
        for cls in SAMPLE_IMAGES.keys():
            count = len(list((OUTPUT_DIR / split / cls).glob("*")))
            print(f"- {cls}: {count} images")

if __name__ == "__main__":
    print("=== Creating Test Dataset ===")
    download_images()
    if organize_dataset():
        check_data_distribution()
        print("\nTest dataset created successfully!")
        print(f"Dataset location: {DATASET_DIR}")
        print(f"Organized data: {OUTPUT_DIR}")
        print("\nYou can now train the model using:")
        print("python -m backend.train_ssl --data-dir data/ssl --output-dir output/ssl --epochs 10")
    else:
        print("Failed to create test dataset.")
