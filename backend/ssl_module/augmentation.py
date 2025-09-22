"""
Data augmentation utilities for semi-supervised learning.
"""
import random
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Callable, Optional, Tuple
import torch
from torchvision import transforms as T

class RandomAugment:
    """Random augmentation policy for semi-supervised learning."""
    
    def __init__(self, n: int = 2, m: int = 10):
        """
        Initialize RandomAugment.
        
        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude (0-10) of augmentation strength
        """
        self.n = n
        self.m = m
        self.augment_list = [
            (self._identity, 0, 1),
            (self._autocontrast, 0, 1),
            (self._equalize, 0, 1),
            (self._rotate, -30, 30),
            (self._solarize, 0, 256),
            (self._color, 0.1, 1.9),
            (self._posterize, 4, 8),
            (self._contrast, 0.1, 1.9),
            (self._sharpness, 0.1, 1.9),
            (self._brightness, 0.1, 1.9),
            (self._shear_x, -0.3, 0.3),
            (self._shear_y, -0.3, 0.3),
            (self._translate_x, -0.3, 0.3),
            (self._translate_y, -0.3, 0.3),
        ]
    
    def __call__(self, img):
        """Apply random augmentations to the input image."""
        ops = random.choices(self.augment_list, k=self.n)
        
        for op, min_val, max_val in ops:
            if random.random() < 0.5:  # 50% chance to apply each augmentation
                val = min_val + (max_val - min_val) * (self.m / 10)
                img = op(img, val)
        
        return img
    
    def _identity(self, img, _):
        return img
    
    def _autocontrast(self, img, _):
        return ImageOps.autocontrast(img)
    
    def _equalize(self, img, _):
        return ImageOps.equalize(img)
    
    def _rotate(self, img, degrees):
        return img.rotate(degrees, resample=Image.BILINEAR, expand=False)
    
    def _solarize(self, img, threshold):
        return ImageOps.solarize(img, threshold)
    
    def _color(self, img, factor):
        return ImageEnhance.Color(img).enhance(factor)
    
    def _posterize(self, img, bits):
        return ImageOps.posterize(img, int(bits))
    
    def _contrast(self, img, factor):
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def _sharpness(self, img, factor):
        return ImageEnhance.Sharpness(img).enhance(factor)
    
    def _brightness(self, img, factor):
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def _shear_x(self, img, factor):
        return img.transform(
            img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0),
            resample=Image.BILINEAR
        )
    
    def _shear_y(self, img, factor):
        return img.transform(
            img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0),
            resample=Image.BILINEAR
        )
    
    def _translate_x(self, img, factor):
        return img.transform(
            img.size, Image.AFFINE, (1, 0, factor * img.size[0], 0, 1, 0),
            resample=Image.BILINEAR
        )
    
    def _translate_y(self, img, factor):
        return img.transform(
            img.size, Image.AFFINE, (1, 0, 0, 0, 1, factor * img.size[1]),
            resample=Image.BILINEAR
        )


def get_transforms(input_size=224, augment=True, mean=None, std=None):
    """
    Get data transformations for training and validation.
    
    Args:
        input_size: Size of input images
        augment: Whether to apply data augmentation
        mean: Dataset mean for normalization
        std: Dataset standard deviation for normalization
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet std
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(n=2, m=9),  # Strong augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


class SSLTransform:
    """
    Transform that applies weak and strong augmentations to the same image.
    Returns a tuple of (weak_augmented, strong_augmented) images.
    """
    def __init__(self, weak_transform: Callable, strong_transform: Callable):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __call__(self, img):
        weak_aug = self.weak_transform(img)
        strong_aug = self.strong_transform(img)
        return weak_aug, strong_aug
    
    def __repr__(self):
        return f"{self.__class__.__name__}(weak_transform={self.weak_transform}, strong_transform={self.strong_transform})"


def get_weak_strong_augment(input_size=224, mean=None, std=None):
    """
    Get weak and strong augmentations for consistency training.
    
    Args:
        input_size: Size of input images
        mean: Dataset mean for normalization
        std: Dataset standard deviation for normalization
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Weak augmentation (flip + crop)
    weak_augment = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Strong augmentation
    strong_augment = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandomAugment(n=3, m=9),  # Strong augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    # Test/validation transform
    val_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return {
        'weak': weak_augment,
        'strong': strong_augment,
        'val': val_transform
    }
