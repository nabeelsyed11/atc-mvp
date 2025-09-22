#!/usr/bin/env python3
"""
Script to train a semi-supervised learning model for species classification.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ssl import __main__ as ssl_main

def parse_args():
    parser = argparse.ArgumentParser(description='Train a semi-supervised learning model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing the processed dataset')
    parser.add_argument('--labeled-dir', type=str, default=None,
                        help='Directory containing labeled data (overrides data-dir/labeled)')
    parser.add_argument('--unlabeled-dir', type=str, default=None,
                        help='Directory containing unlabeled data (overrides data-dir/unlabeled)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'],
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability (default: 0.2)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (default: 10)')
    parser.add_argument('--pseudo-threshold', type=float, default=0.9,
                        help='Confidence threshold for pseudo-labels (default: 0.9)')
    parser.add_argument('--consistency-weight', type=float, default=0.5,
                        help='Weight for consistency loss (default: 0.5)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output/ssl',
                        help='Directory to save model checkpoints and logs (default: output/ssl)')
    
    # Misc
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Prepare command line arguments for the SSL module
    sys.argv = [sys.argv[0]]  # Clear existing arguments
    
    # Add required arguments
    sys.argv.extend([
        '--data-dir', args.data_dir,
        '--output-dir', args.output_dir,
        '--backbone', args.backbone,
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--patience', str(args.patience),
        '--pseudo-threshold', str(args.pseudo_threshold),
        '--consistency-weight', str(args.consistency_weight),
        '--num-workers', str(args.num_workers),
        '--seed', str(args.seed)
    ])
    
    # Add optional arguments if provided
    if args.labeled_dir:
        sys.argv.extend(['--labeled-dir', args.labeled_dir])
    if args.unlabeled_dir:
        sys.argv.extend(['--unlabeled-dir', args.unlabeled_dir])
    if args.pretrained:
        sys.argv.append('--pretrained')
    if args.dropout != 0.2:
        sys.extend(['--dropout', str(args.dropout)])
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    
    # Run the training script
    ssl_main.main()
    
    print("\nTraining complete!")
    print(f"Model checkpoints and logs saved to: {args.output_dir}")
    print("\nTo export the trained model to ONNX, run:")
    print(f"python -m backend.ssl.export --checkpoint {os.path.join(args.output_dir, 'model_best.pth')}")

if __name__ == "__main__":
    main()
