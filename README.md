# Animal Type Classification (ATC) using Semi-Supervised Learning

A comprehensive solution for automated classification of cattle and buffaloes using advanced computer vision and deep learning techniques. This project implements a Semi-Supervised Learning (SSL) approach to leverage both labeled and unlabeled data for improved model performance.

It lets a field user:
- Upload a side-profile image of an animal.
- Optionally click a few guided landmarks on the image (withers, brisket, front hoof, shoulder, hook, pin).
- Compute features (body length ratio, chest depth ratio, rump angle) and objective per-trait scores (1–9) with an overall score.
- Auto-save records locally (JSONL + CSV) and export CSV.
- Optionally try a basic auto mode using silhouette bbox when landmarks are not provided.
- Send a stub payload toward BPA (configurable URL) to illustrate integration.

Important: This is an MVP for demonstration. It does not replace expert scoring. Targets and mappings are adjustable in `backend/scoring.py`.

## 🚀 Features

- **Semi-Supervised Learning**: Leverages both labeled and unlabeled data for improved model performance
- **High Accuracy**: Distinguishes between cattle and buffalo with state-of-the-art accuracy
- **Advanced Augmentation**: Implements sophisticated data augmentation techniques
- **Model Checkpointing**: Automatic saving of model weights during training
- **Training Visualization**: Real-time monitoring with TensorBoard
- **Easy Deployment**: Simple API for model inference
- **Scalable**: Designed to work with large datasets
- **Reproducible**: Complete environment specification

## SSL Model Training

The SSL model is trained using a combination of labeled and unlabeled data. The labeled data is used to train a supervised model, while the unlabeled data is used to fine-tune the model using self-supervised learning techniques.

### Training Data

The training data consists of a mix of labeled and unlabeled images. The labeled images are annotated with the corresponding animal type (cattle or buffalo), while the unlabeled images are used to fine-tune the model.

### Training Process

The training process involves the following steps:

1. Preprocessing: The images are preprocessed to normalize the pixel values and resize the images to a fixed size.
2. Supervised Training: The labeled images are used to train a supervised model using a convolutional neural network (CNN) architecture.
3. Self-Supervised Fine-Tuning: The unlabeled images are used to fine-tune the model using self-supervised learning techniques, such as contrastive learning.

### Prerequisites
- Python 3.9+ (3.10 recommended)
- CUDA-compatible GPU (recommended) or CPU
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/atc-mvp.git
   cd atc-mvp
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-ssl.txt
   ```

Steps:
1) Create a virtual environment
```
py -m venv .venv
```
2) Install dependencies
```
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```
3) Run the server
```
.\.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
```
4) Open the UI in your browser at:
```
http://localhost:8000/
```

### Quick Start with Docker (no local Python needed)

Prerequisites:
- Docker Desktop installed and running

Steps:
1) Build the image
```
docker build -t atc-mvp .
```
2) Run the container
```
docker run --rm -p 8000:8000 --name atc-mvp atc-mvp
```
3) Open the UI in your browser at:
```
http://localhost:8000/
```

## 🚀 Training the Model

Train the SSL model with the following command:

```bash
python backend/train_ssl.py \
    --data-dir data/ssl \
    --backbone resnet18 \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --pretrained \
    --output-dir models/ssl_model
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Path to dataset directory | `data/ssl` |
| `--backbone` | Model architecture | `resnet18` |
| `--batch-size` | Batch size | `32` |
| `--epochs` | Number of training epochs | `100` |
| `--lr` | Learning rate | `0.001` |
| `--weight-decay` | Weight decay | `1e-4` |
| `--pretrained` | Use pretrained weights | `False` |
| `--pseudo-threshold` | Confidence threshold for pseudo-labeling | `0.9` |
| `--consistency-weight` | Weight for consistency loss | `1.0` |
| `--output-dir` | Output directory for checkpoints | `models/ssl_model` |
| `--resume` | Path to checkpoint to resume from | `None` |

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=models/ssl_model/logs
```
Then open `http://localhost:6006` in your browser.

## 📁 Project Structure

```
atc-mvp/
├── backend/               # Backend code
│   ├── ssl/              # SSL model implementation
│   ├── train_ssl.py      # Training script
│   ├── inference.py      # Inference script
│   └── ...
├── data/                 # Dataset directory
│   └── ssl/              # SSL dataset
│       ├── train/        # Training data
│       │   ├── cattle/   # Cattle images
│       │   └── buffalo/  # Buffalo images
│       └── val/          # Validation data
│           ├── cattle/
│           └── buffalo/
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── .gitignore            # Git ignore file
├── README.md             # This file
└── requirements-ssl.txt  # Python dependencies
```

## 🤖 Model Architecture

The model uses a ResNet-18 backbone with a custom classification head. The semi-supervised learning approach combines:

1. **Supervised Loss**: Cross-entropy loss on labeled data
2. **Consistency Loss**: KL divergence between predictions on weakly and strongly augmented unlabeled data
3. **Confidence-based Pseudo-labeling**: For unlabeled data with high confidence predictions

## How to Use
- Select species (cattle/buffalo).
- Or check "Auto-detect species (ML)" to let the system infer cattle vs buffalo.
- Upload a side-profile image.
- Either:
  - Click "Start Landmarking" and follow the guided sequence of points: withers, brisket, front_hoof, shoulder, hook, pin.
    - Click on the canvas to place each point.
    - Use "Undo" or "Clear" as needed.
  - Or check "Auto silhouette (no landmarks)" to attempt a simple bbox-based analysis.
- Click "Compute & Save" to generate scores. Results show per-trait and overall score. Records are saved locally.
- Use "Records" to view recent entries and "Export CSV" to download a summary file.
- Use "Send to BPA (stub)" to simulate sending a record to a configurable endpoint.

## Landmarks Guide
- withers: highest point over the shoulders.
- brisket: bottom of the chest between forelimbs.
- front_hoof: ground contact point of the fore hoof.
- shoulder: point of shoulder (anterior end of scapula).
- hook: hip bone (tuber coxae).
- pin: pin bone (tuber ischii / tailhead area).

These yield scale-free ratios:
- body_length_ratio = shoulder→pin length / height_at_withers
- chest_depth_ratio = withers→brisket / height_at_withers
- rump_angle_deg = angle of hook→pin line (deg) vs horizontal

## Configuration (.env)
Copy `.env.example` to `.env` and fill values if desired:
```
BPA_URL=
BPA_API_KEY=
BPA_AUTH_TYPE=Bearer
```
If `BPA_URL` is unset, `Send to BPA` remains a no-op stub.

## Data Storage
- JSONL: `data/records.jsonl`
- CSV summary: `data/records.csv`
- Uploaded images (copies): `data/images/{record_id}.jpg`

## Project Structure
```
atc-mvp/
  backend/
    __init__.py
    main.py
    image_processing.py
    scoring.py
    storage.py
    models.py
  frontend/
    index.html
    app.js
    styles.css
  data/               # created at runtime
  requirements.txt
  .env.example
  README.md
```

## Notes & Limitations
- Auto silhouette uses a simple largest-contour bbox and is not robust.
- Scoring targets are placeholders; tune them per breed policies in `scoring.py`.
- This MVP does not perform full conformation trait coverage; it demonstrates the pipeline.
- Ensure side-profile, unobstructed images for best landmarking.

## Species Auto-Detection (ML)
- Enabled via the checkbox "Auto-detect species (ML)" on the UI.
- By default, the app uses a simple heuristic if no ML model/runtime is available.
- To enable ML inference with an ONNX model:
  1) Install ONNX Runtime:
     - `python -m pip install onnxruntime`
  2) Place your model at `models/species_classifier.onnx` and optional labels at `models/species_labels.txt` (two lines: `cattle` and `buffalo`).
     - Or set env variables in `.env`:
       - `SPECIES_ONNX_PATH=full\path\to\your_model.onnx`
       - `SPECIES_LABELS_PATH=full\path\to\labels.txt`
       - `SPECIES_INPUT_SIZE=224` (optional; must match your model)
       - `SPECIES_USE_CROP=true` (optional; crop to main subject before classification)
       - `SPECIES_TTA=true` (optional; average prediction with horizontal flip)
  3) Restart the server. The response will include `detected_species`, `species_confidence`, and `species_method` (onnx/heuristic).

## License
For evaluation and internal prototyping only.
