# Semi-Supervised Learning for Species Classification

This module implements semi-supervised learning (SSL) for cattle vs. buffalo classification. It uses a teacher-student framework with consistency regularization to leverage both labeled and unlabeled data.

## Features

- **Semi-Supervised Learning**: Combines labeled and unlabeled data for training
- **Consistency Regularization**: Enforces consistent predictions for different augmentations of the same image
- **Pseudo-Labeling**: Generates pseudo-labels for unlabeled data with confidence thresholding
- **Flexible Backbones**: Supports various CNN architectures (ResNet, EfficientNet, etc.)
- **Training Utilities**: Includes learning rate scheduling, early stopping, and model checkpointing

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) For GPU acceleration, install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
   ```

## Directory Structure

```
ssl/
├── __init__.py           # Package initialization
├── __main__.py           # Training script
├── models.py             # Model definitions
├── trainer.py            # Training loop and utilities
├── utils.py              # Data loading and helper functions
└── requirements.txt      # Python dependencies
```

## Usage

### Training

1. Prepare your dataset in the following structure:
   ```
   data/ssl/
   ├── labeled/
   │   ├── cattle/
   │   │   ├── image1.jpg
   │   │   └── ...
   │   └── buffalo/
   │       ├── image1.jpg
   │       └── ...
   └── unlabeled/
       ├── image1.jpg
       └── ...
   ```

2. Start training:
   ```bash
   python -m backend.ssl.train \
     --data-dir data/ssl \
     --output-dir output/ssl \
     --backbone resnet18 \
     --batch-size 32 \
     --epochs 50 \
     --lr 1e-4 \
     --pseudo-threshold 0.9
   ```

### Evaluation

To evaluate a trained model:

```python
from backend.ssl.models import load_pretrained
from backend.ssl.utils import evaluate

# Load the trained model
model = load_pretrained('path/to/checkpoint.pth', num_classes=2)

# Evaluate on test set
metrics = evaluate(model, test_loader, device='cuda')
print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
```

### Inference

```python
import torch
from torchvision import transforms
from PIL import Image
from backend.ssl.models import load_pretrained

# Load model
model = load_pretrained('path/to/checkpoint.pth', num_classes=2)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open('path/to/image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Predicted class: {predicted_class} (confidence: {confidence:.2f})")
```

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Base directory for training data | `data/ssl` |
| `--labeled-dir` | Directory containing labeled data | `None` (uses `data-dir`/labeled) |
| `--unlabeled-dir` | Directory containing unlabeled data | `None` (uses `data-dir`/unlabeled) |
| `--output-dir` | Directory to save outputs | `output/ssl` |
| `--backbone` | Backbone architecture | `resnet18` |
| `--pretrained` | Use pretrained weights | `False` |
| `--batch-size` | Batch size | `32` |
| `--epochs` | Number of training epochs | `50` |
| `--lr` | Learning rate | `1e-4` |
| `--weight-decay` | Weight decay | `1e-4` |
| `--pseudo-threshold` | Confidence threshold for pseudo-labels | `0.9` |
| `--consistency-weight` | Weight for consistency loss | `0.5` |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `0` |
| `TORCH_HOME` | Directory for pretrained models | `~/.cache/torch` |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
