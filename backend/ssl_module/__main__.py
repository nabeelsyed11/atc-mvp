#!/usr/bin/env python3
"""
Semi-Supervised Learning for Species Classification

This script trains a semi-supervised learning model for cattle vs. buffalo classification.
It uses a teacher-student framework with consistency regularization.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from .models import create_model
from .trainer import SSLTrainer
from .utils import (
    SSLDataset, 
    prepare_data_directories,
    get_device
)

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-Supervised Learning for Species Classification')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/ssl',
                        help='Base directory for training data')
    parser.add_argument('--labeled-dir', type=str, default=None,
                        help='Directory containing labeled data (overrides data-dir/labeled)')
    parser.add_argument('--unlabeled-dir', type=str, default=None,
                        help='Directory containing unlabeled data (overrides data-dir/unlabeled)')
    parser.add_argument('--output-dir', type=str, default='output/ssl',
                        help='Directory to save model checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--pseudo-threshold', type=float, default=0.9,
                        help='Confidence threshold for pseudo-labels')
    parser.add_argument('--consistency-weight', type=float, default=0.5,
                        help='Weight for consistency loss')
    
    # Misc
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Prepare directories
    data_dirs = prepare_data_directories(args.data_dir)
    
    # Override data directories if specified
    if args.labeled_dir:
        data_dirs['labeled'] = args.labeled_dir
    if args.unlabeled_dir:
        data_dirs['unlabeled'] = args.unlabeled_dir
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    # TODO: Implement dataset loading
    # For now, using dummy data
    train_dataset = SSLDataset(labeled_data=[], unlabeled_data=[])
    val_dataset = SSLDataset(labeled_data=[], unlabeled_data=[])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        backbone=args.backbone,
        num_classes=2,  # cattle vs. buffalo
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    # Create trainer
    trainer = SSLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=2,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        pseudo_label_threshold=args.pseudo_threshold,
        consistency_weight=args.consistency_weight
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train()
    
    # Save final model
    final_model_path = output_dir / 'model_final.pth'
    trainer.save_model(str(final_model_path))
    print(f"Saved final model to {final_model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    trainer.save_history(str(history_path))
    print(f"Saved training history to {history_path}")

if __name__ == '__main__':
    main()
