import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Union, List, Dict, Any

class SpeciesClassifier(nn.Module):
    """Base classifier model for species classification."""
    
    def __init__(self, 
                 num_classes: int = 2, 
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 dropout: float = 0.2):
        """
        Initialize the species classifier.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture (resnet18, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load backbone
        if hasattr(models, backbone):
            backbone_model = getattr(models, backbone)(pretrained=pretrained)
            
            # Replace the final fully connected layer
            in_features = backbone_model.fc.in_features
            features = list(backbone_model.children())[:-1]  # Remove the original FC layer
            
            self.features = nn.Sequential(*features)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(in_features, num_classes)
            
            # Initialize the new FC layer
            nn.init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)
        else:
            raise ValueError(f"Backbone '{backbone}' is not supported. "
                             "Choose from torchvision.models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the feature embedding (before the final FC layer)."""
        with torch.no_grad():
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x


def create_model(backbone: str = 'resnet18', 
                num_classes: int = 2,
                pretrained: bool = True,
                dropout: float = 0.2) -> nn.Module:
    """Create a species classifier model."""
    return SpeciesClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )

def load_pretrained(model_path: str, 
                  device: str = None,
                  **kwargs) -> nn.Module:
    """Load a pretrained model from a checkpoint."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model with default settings
    model = create_model(**kwargs)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict with strict=False to ignore missing keys
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model
