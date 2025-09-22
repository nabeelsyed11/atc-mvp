#!/usr/bin/env python3
"""
Script to prepare data for semi-supervised learning.

This script organizes your raw data into the required directory structure
for the semi-supervised learning pipeline.
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ssl.data_utils import prepare_ssl_data

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for semi-supervised learning')
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root directory containing the raw dataset')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save processed data (default: data-dir/processed)')
    parser.add_argument('--labeled-ratio', type=float, default=0.1,
                        help='Fraction of training data to use as labeled (default: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Fraction of training data to use for validation (default: 0.1)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Minimum number of labeled samples per class (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--copy-files', action='store_true',
                        help='Copy files instead of creating symlinks')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Prepare the dataset
    dataset = prepare_ssl_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        labeled_ratio=args.labeled_ratio,
        test_size=args.test_size,
        val_size=args.val_size,
        min_samples_per_class=args.min_samples,
        random_state=args.seed
    )
    
    print("\nData preparation complete!")
    print(f"Output directory: {args.output_dir or os.path.join(args.data_dir, 'processed')}")
    print("\nNext steps:")
    print("1. Train the model:")
    print(f"   python -m backend.ssl.train --data-dir {args.output_dir or os.path.join(args.data_dir, 'processed')}")
    print("\n2. Export the trained model to ONNX:")
    print("   python -m backend.ssl.export --checkpoint output/ssl/model_best.pth")

if __name__ == "__main__":
    main()
