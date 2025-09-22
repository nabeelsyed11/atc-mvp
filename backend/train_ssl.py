#!/usr/bin/env python3
"""
Train a semi-supervised learning model for species classification.

This script sets up and runs the training pipeline with the latest enhancements,
including mixed precision training, gradient accumulation, and advanced monitoring.
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ssl_module.augmentation import get_weak_strong_augment
from ssl_module.models import create_ssl_model
from ssl_module.trainer import SSLTrainer
from ssl_module.utils import (
    SSLDataset,
    get_device,
    setup_logging, 
    save_checkpoint, 
    load_checkpoint,
    prepare_data_directories
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSL Model for Species Classification')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/ssl',
                        help='Base directory containing train/val/test splits')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # SSL specific arguments
    parser.add_argument('--pseudo-threshold', type=float, default=0.9,
                        help='Confidence threshold for pseudo-labeling')
    parser.add_argument('--consistency-weight', type=float, default=1.0,
                        help='Weight for consistency loss')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/ssl_model',
                        help='Directory to save model checkpoints and logs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def create_datasets(args):
    """Create labeled and unlabeled datasets with appropriate transforms."""
    try:
        # Define transforms
        weak_augment = get_weak_strong_augment(input_size=224, is_weak=True)
        strong_augment = get_weak_strong_augment(input_size=224, is_weak=False)
        
        # Create transform for labeled data (weak augmentation only)
        labeled_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create transform for unlabeled data (weak + strong augmentation)
        unlabeled_transform = SSLTransform(
            weak_transform=weak_augment,
            strong_transform=strong_augment
        )
        
        # Create datasets
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
        
        print(f"Loading data from: {os.path.abspath(args.data_dir)}")
        print(f"Train directory: {os.path.abspath(train_dir)}")
        print(f"Validation directory: {os.path.abspath(val_dir)}")
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        # Load training data
        print("Loading training data...")
        train_dataset = ImageFolder(train_dir, transform=labeled_transform)
        print(f"Found {len(train_dataset)} training samples in {len(train_dataset.classes)} classes")
        
        # Load validation data
        print("Loading validation data...")
        val_dataset = ImageFolder(val_dir, transform=labeled_transform)
        print(f"Found {len(val_dataset)} validation samples in {len(val_dataset.classes)} classes")
        
        # For unlabeled data, we'll use training data with unlabeled transform
        print("Preparing unlabeled data...")
        unlabeled_dataset = ImageFolder(train_dir, transform=unlabeled_transform)
        
        # Create SSL datasets
        print("Creating SSL datasets...")
        train_ssl_dataset = SSLDataset(
            labeled_data=train_dataset,
            unlabeled_data=unlabeled_dataset,
            transform=labeled_transform
        )
        
        val_ssl_dataset = SSLDataset(
            labeled_data=val_dataset,
            unlabeled_data=None,  # No unlabeled data in validation
            transform=labeled_transform
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Class names: {train_dataset.classes}")
        
        return train_ssl_dataset, val_ssl_dataset
        
    except Exception as e:
        print(f"Error in create_datasets: {str(e)}")
        raise
    
    return train_dataset, val_dataset, labeled_dataset.classes

def main():
    try:
        args = parse_args()
        
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
        
        # Setup logging
        logger = setup_logging(os.path.join(args.output_dir, 'training.log'))
        logger.info(f"Starting training with arguments: {args}")
        
        # Create datasets and data loaders
        print("Creating datasets...")
        train_dataset, val_dataset = create_datasets(args)
        
        # Get class names from the training dataset
        classes = train_dataset.classes
        print(f"Classes: {classes}")
        
        # Save class names
        class_names_path = os.path.join(args.output_dir, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(classes, f)
        print(f"Saved class names to {class_names_path}")
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() // 2),  # Use half of available CPUs
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(2, os.cpu_count() // 4),  # Use quarter of available CPUs
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        # Create model
        print(f"Creating {args.backbone} model...")
        model = create_ssl_model(
            arch=args.backbone,
            num_classes=len(classes),
            pretrained=args.pretrained
        ).to(device)
        
        print(f"Model architecture:")
        print(model)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
            eta_min=1e-6
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        
        # Create trainer
        print("Creating trainer...")
        trainer = SSLTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=len(classes),
            device=device,
            output_dir=args.output_dir,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=max(5, args.epochs // 10),  # Early stopping after 10% of epochs
            pseudo_label_threshold=args.pseudo_threshold,
            consistency_weight=args.consistency_weight,
            use_ema=True,
            ema_decay=0.999,
        )
        
        # Start training
        print("Starting training...")
        trainer.train(start_epoch=start_epoch)
        
        print("Training completed successfully!")
        print(f"Model and logs saved to: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
