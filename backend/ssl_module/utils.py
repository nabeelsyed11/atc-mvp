import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SSLDataset(Dataset):
    """Dataset for semi-supervised learning with labeled and unlabeled data."""
    
    def __init__(self, 
                 labeled_data: List[Tuple[str, int]], 
                 unlabeled_data: List[str],
                 transform=None,
                 pseudo_label_threshold: float = 0.9):
        """
        Args:
            labeled_data: List of (image_path, label) tuples
            unlabeled_data: List of image paths for unlabeled data
            transform: Optional transform to be applied on images
            pseudo_label_threshold: Confidence threshold for pseudo-labels
        """
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.transform = transform or self._default_transform()
        self.pseudo_label_threshold = pseudo_label_threshold
        
    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return max(len(self.labeled_data), len(self.unlabeled_data))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get labeled data
        labeled_idx = idx % len(self.labeled_data)
        img_path, label = self.labeled_data[labeled_idx]
        img = self._load_image(img_path)
        
        # Get unlabeled data (will be used with pseudo-labels during training)
        unlabeled_idx = idx % len(self.unlabeled_data)
        unlabeled_img_path = self.unlabeled_data[unlabeled_idx]
        unlabeled_img = self._load_image(unlabeled_img_path)
        
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'unlabeled_image': unlabeled_img,
            'unlabeled_path': unlabeled_img_path
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        img = Image.open(path).convert('RGB')
        return self.transform(img)


def prepare_data_directories(base_dir: str) -> Dict[str, str]:
    """Create necessary directories for SSL data."""
    base_path = Path(base_dir)
    dirs = {
        'labeled': base_path / 'labeled',
        'unlabeled': base_path / 'unlabeled',
        'pseudo_labeled': base_path / 'pseudo_labeled',
        'models': base_path / 'models',
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Create class subdirectories for labeled data
    for cls_name in ['cattle', 'buffalo']:
        (dirs['labeled'] / cls_name).mkdir(exist_ok=True)
        (dirs['pseudo_labeled'] / cls_name).mkdir(exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}

def save_checkpoint(model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer, 
                  epoch: int, 
                  metrics: Dict,
                  path: str,
                  is_best: bool = False):
    """Save training checkpoint."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save checkpoint
    torch.save(state, path)
    
    # Save best model separately
    if is_best:
        best_path = str(Path(path).parent / 'model_best.pth')
        shutil.copyfile(path, best_path)

def load_checkpoint(path: str, model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer = None):
    """Load a training checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'model': model,
        'optimizer': optimizer
    }

def get_device() -> torch.device:
    """Get the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
