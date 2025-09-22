# Semi-Supervised Learning for Species Classification

This directory contains the implementation of a semi-supervised learning (SSL) pipeline for cattle vs. buffalo classification. The implementation uses a teacher-student framework with consistency regularization.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-ssl.txt
   ```

## Data Preparation

Organize your data in the following structure:

```
data/
  ssl/
    labeled/
      cattle/
        image1.jpg
        image2.jpg
        ...
      buffalo/
        image1.jpg
        image2.jpg
        ...
    unlabeled/
      image1.jpg
      image2.jpg
      ...
```

- `labeled/`: Contains labeled images organized by class (cattle/buffalo)
- `unlabeled/`: Contains unlabeled images (optional)

## Training

### Basic Training

```bash
python -m backend.train_ssl \
  --data-dir data/ssl \
  --output-dir output/ssl \
  --backbone resnet18 \
  --batch-size 32 \
  --epochs 100 \
  --lr 3e-4 \
  --pseudo-threshold 0.9 \
  --consistency-weight 1.0
```

### Advanced Options

- `--pretrained`: Use pretrained weights
- `--grad-accum-steps`: Number of gradient accumulation steps (default: 4)
- `--warmup-epochs`: Number of warmup epochs (default: 5)
- `--label-smoothing`: Label smoothing epsilon (default: 0.1)
- `--use-amp`: Enable mixed precision training (default: True)
- `--num-workers`: Number of data loading workers (default: 4)

### Resuming Training

To resume training from a checkpoint:

```bash
python -m backend.train_ssl \
  --resume output/ssl/checkpoint.pth \
  --output-dir output/ssl_resume
```

## Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=output/ssl/logs
```

## Model Export

After training, you can export the model to ONNX format:

```python
import torch
from backend.ssl.export import export_onnx

model = create_model(backbone='resnet18', num_classes=2)
checkpoint = torch.load('output/ssl/model_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

export_onnx(
    model,
    input_size=(1, 3, 224, 224),
    output_path='output/ssl/model.onnx',
    opset_version=12
)
```

## Evaluation

To evaluate the trained model:

```python
from backend.ssl.eval import evaluate_model

metrics = evaluate_model(
    model_path='output/ssl/model_best.pth',
    data_dir='data/ssl/val',
    batch_size=32,
    num_workers=4
)

print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## Hyperparameter Tuning

Key hyperparameters to tune:

1. `--consistency-weight`: Controls the weight of the consistency loss (default: 1.0)
2. `--pseudo-threshold`: Confidence threshold for pseudo-labels (default: 0.9)
3. `--lr`: Learning rate (default: 3e-4)
4. `--batch-size`: Batch size (default: 32)
5. `--warmup-epochs`: Number of warmup epochs (default: 5)

## Troubleshooting

1. **Out of Memory (OOM) Errors**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training (--use-amp)

2. **Slow Training**:
   - Increase number of workers (--num-workers)
   - Use a smaller model (e.g., resnet18 instead of resnet50)
   - Disable mixed precision if not using a GPU

3. **Poor Performance**:
   - Increase the amount of labeled data
   - Adjust the consistency weight
   - Lower the pseudo-label threshold
   - Try a different backbone architecture

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
