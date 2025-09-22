"""
Data preparation utilities for semi-supervised learning.
"""
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import json

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Utility class for preparing and splitting the dataset."""
    
    def __init__(self, data_dir: Union[str, Path], 
                 output_dir: Union[str, Path] = None,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Root directory containing the dataset
            output_dir: Directory to save processed data (default: data_dir/processed)
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'processed'
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Create output directories
        self.labeled_dir = self.output_dir / 'labeled'
        self.unlabeled_dir = self.output_dir / 'unlabeled'
        self.test_dir = self.output_dir / 'test'
        
        # Class labels
        self.classes = ['cattle', 'buffalo']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
    def prepare_dataset(self, 
                       labeled_ratio: float = 0.1,
                       min_samples_per_class: int = 5,
                       copy_files: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        Prepare the dataset for semi-supervised learning.
        
        Args:
            labeled_ratio: Fraction of data to use as labeled (rest is unlabeled)
            min_samples_per_class: Minimum number of labeled samples per class
            copy_files: Whether to copy files to the output directory
            
        Returns:
            Dictionary containing the dataset split
        """
        # Create output directories
        for cls_name in self.classes:
            (self.labeled_dir / cls_name).mkdir(parents=True, exist_ok=True)
            (self.test_dir / cls_name).mkdir(parents=True, exist_ok=True)
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all images
        dataset = {split: {cls_name: [] for cls_name in self.classes} 
                  for split in ['train', 'val', 'test']}
        
        for cls_name in self.classes:
            # Find all images for this class
            class_dir = self.data_dir / 'raw' / cls_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            image_paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if not image_paths:
                print(f"Warning: No images found in {class_dir}")
                continue
            
            # Split into train/val/test
            train_val_paths, test_paths = train_test_split(
                image_paths, 
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            train_paths, val_paths = train_test_split(
                train_val_paths,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state
            )
            
            # Ensure minimum number of labeled samples
            n_labeled = max(min_samples_per_class, int(len(train_paths) * labeled_ratio))
            labeled_paths = train_paths[:n_labeled]
            unlabeled_paths = train_paths[n_labeled:]
            
            # Update dataset dictionary
            dataset['train'][cls_name] = labeled_paths
            dataset['val'][cls_name] = val_paths
            dataset['test'][cls_name] = test_paths
            
            # Copy files to output directory
            if copy_files:
                # Copy labeled data
                for src_path in labeled_paths:
                    dst_path = self.labeled_dir / cls_name / src_path.name
                    shutil.copy2(src_path, dst_path)
                
                # Copy unlabeled data
                for src_path in unlabeled_paths:
                    dst_path = self.unlabeled_dir / f"{cls_name}_{src_path.name}"
                    shutil.copy2(src_path, dst_path)
                
                # Copy test data
                for src_path in test_paths:
                    dst_path = self.test_dir / cls_name / src_path.name
                    shutil.copy2(src_path, dst_path)
        
        # Save dataset split information
        split_info = {
            'classes': self.classes,
            'splits': {
                split: {cls_name: [str(p) for p in paths] 
                       for cls_name, paths in split_data.items()}
                for split, split_data in dataset.items()
            },
            'stats': {
                split: {cls_name: len(paths) 
                       for cls_name, paths in split_data.items()}
                for split, split_data in dataset.items()
            }
        }
        
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print("Dataset prepared successfully!")
        print("Statistics:")
        for split, stats in split_info['stats'].items():
            print(f"  {split.capitalize()}:")
            for cls_name, count in stats.items():
                print(f"    {cls_name}: {count} samples")
        
        return dataset
    
    def create_label_map(self, output_file: str = 'label_map.json') -> Dict[str, int]:
        """
        Create a label map file.
        
        Args:
            output_file: Path to save the label map
            
        Returns:
            Dictionary mapping class names to indices
        """
        label_map = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        with open(self.output_dir / output_file, 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"Label map saved to {self.output_dir/output_file}")
        return label_map
    
    @staticmethod
    def create_data_list(data_dir: Union[str, Path], 
                        output_file: str = 'data_list.txt',
                        class_to_idx: Dict[str, int] = None) -> List[Tuple[str, int]]:
        """
        Create a list of (image_path, label) pairs.
        
        Args:
            data_dir: Directory containing class subdirectories
            output_file: Path to save the data list
            class_to_idx: Optional mapping from class names to indices
            
        Returns:
            List of (image_path, label) pairs
        """
        data_dir = Path(data_dir)
        data_list = []
        
        if class_to_idx is None:
            # Infer classes from subdirectories
            classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = data_dir / cls_name
            if not cls_dir.exists():
                continue
                
            for img_path in cls_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    data_list.append((str(img_path), cls_idx))
        
        # Save to file
        with open(data_dir.parent / output_file, 'w') as f:
            for img_path, label in data_list:
                f.write(f"{img_path} {label}\n")
        
        print(f"Data list saved to {data_dir.parent/output_file}")
        return data_list


def prepare_ssl_data(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path] = None,
    labeled_ratio: float = 0.1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    min_samples_per_class: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    """
    Prepare data for semi-supervised learning.
    
    Args:
        data_dir: Root directory containing the dataset
        output_dir: Directory to save processed data
        labeled_ratio: Fraction of training data to use as labeled
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        min_samples_per_class: Minimum number of labeled samples per class
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing the dataset split
    """
    preprocessor = DataPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Prepare the dataset
    dataset = preprocessor.prepare_dataset(
        labeled_ratio=labeled_ratio,
        min_samples_per_class=min_samples_per_class
    )
    
    # Create label map
    preprocessor.create_label_map()
    
    # Create data lists
    for split in ['train', 'val', 'test']:
        preprocessor.create_data_list(
            data_dir=preprocessor.output_dir / ('labeled' if split == 'train' else split),
            output_file=f'{split}_list.txt',
            class_to_idx=preprocessor.class_to_idx
        )
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for semi-supervised learning')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root directory containing the dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save processed data')
    parser.add_argument('--labeled-ratio', type=float, default=0.1,
                        help='Fraction of training data to use as labeled')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Minimum number of labeled samples per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    prepare_ssl_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        labeled_ratio=args.labeled_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
        min_samples_per_class=args.min_samples,
        random_state=args.seed
    )
