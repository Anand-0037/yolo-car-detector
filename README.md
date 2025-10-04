# YOLO Car Detector

A computer vision project that uses a pre-trained YOLOv8 model to detect cars in images, fine-tuned on a [car dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data)

## Features

- **Pre-trained YOLO Model**: Uses YOLOv8 nano for fast object detection
- **Custom Fine-tuning**: Fine-tuned on car-specific dataset
- **Image Detection**: Detect cars in static images
- **GPU Acceleration**: CUDA support for faster training and inference
- **High Accuracy**: Achieves good mAP on validation set

## Dataset

- **Source**: [Kaggle Car Object Detection Dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data)
- **Classes**: 1 (car)
- **Split**: 80% train, 10% validation, 10% test
- **Format**: YOLO format with normalized bounding boxes

## Quick Start

### Prerequisites

**Install uv** (Python package manager): [Astral uv Documentation](https://docs.astral.sh/uv/getting-started/)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize and sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

#### 1. Prepare Dataset

```bash
# Download dataset from Kaggle
python fetch_dataset.py

# Split dataset and convert to YOLO format
python split_dataset.py
```

#### 2. Train Model

```bash
# Train YOLOv8 on car detection
python train_and_detect.py
```

#### 3. Run Detection

```bash
# Run inference on a test image
python inference.py --image dataset/images/test/vid_5_25200.jpg --conf 0.5
```

Or programmatically:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('fine_tuned_car_yolo.pt')

# Detect cars in image
results = model('path/to/image.jpg', conf=0.5)
```

## Project Structure

```
yolo-car-detector/
├── dataset/
│   ├── images/
│   │   ├── train/             # Training images
│   │   ├── val/               # Validation images
│   │   └── test/              # Test images
│   ├── labels/
│   │   ├── train/             # Training labels (YOLO format)
│   │   ├── val/               # Validation labels
│   │   └── test/              # Test labels
│   └── data/                  # Original dataset
├── runs/
│   └── detect/
│       ├── car_detector_runs2/   # Training results
│       │   ├── weights/
│       │   │   ├── best.pt       # Best model checkpoint
│       │   │   └── last.pt       # Last epoch checkpoint
│       │   ├── results.png       # Training metrics plot
│       │   ├── confusion_matrix.png
│       │   └── results.csv       # Training metrics data
│       └── predict/              # Detection results
├── fetch_dataset.py           # Dataset download script
├── split_dataset.py           # Dataset preparation script
├── train_and_detect.py        # Training and inference script
├── inference.py               # Standalone inference script
├── dataset.yaml              # YOLO dataset configuration
├── pyproject.toml            # Project dependencies (uv)
├── requirements.txt          # Alternative dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Configuration

### Training Parameters (Used)

The following parameters were used for the trained model:

```yaml
Model: YOLOv8n (nano)
Epochs: 50
Image Size: 640x640
Batch Size: 16
Device: CUDA GPU (device 0)
Optimizer: Auto (AdamW)
Learning Rate: 0.01 (initial)
LR Schedule: Cosine annealing
Weight Decay: 0.0005
Warmup Epochs: 3
Patience: 10 (early stopping)
```

### Augmentation Settings

```yaml
HSV-Hue: 0.015
HSV-Saturation: 0.7
HSV-Value: 0.4
Flip Left-Right: 0.5
Mosaic: 1.0 (disabled last 10 epochs)
Auto Augment: RandAugment
Random Erasing: 0.4
```

### Dataset Configuration (dataset.yaml)

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: ['car']
```

## Results & Performance

### Training Metrics

The model was trained for 50 epochs with the following best results:

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.979 |
| **mAP@0.5:0.95** | 0.634 |
| **Precision** | 0.978 |
| **Recall** | 0.925 |
| **Training Time** | ~6 minutes (epoch 20) |

### Training Progress

The model shows excellent convergence with:
- **Box Loss**: Decreased from 1.52 → 1.12
- **Classification Loss**: Decreased from 2.62 → 0.62
- **DFL Loss**: Decreased from 1.14 → 1.02

Key observations:
- Peak performance achieved around epoch 18-20
- Early stopping patience of 10 prevents overfitting
- Cosine LR scheduling helps smooth convergence

### Visualizations

Training outputs are saved in `runs/detect/car_detector_runs2/`:

- **results.png** - Training/validation metrics over time
- **confusion_matrix.png** - Model confusion matrix
- **BoxPR_curve.png** - Precision-Recall curve
- **BoxF1_curve.png** - F1-score curve
- **train_batch*.jpg** - Training batch samples with augmentations
- **val_batch*_pred.jpg** - Validation predictions

### Detection Examples

Example inference results from `runs/detect/predict/`:

```bash
# Example detection output
Detecting cars in: dataset/images/test/vid_5_25200.jpg
Detected 3 cars
```

Detection results include:
- Bounding boxes around detected cars
- Confidence scores for each detection
- Saved annotated images in `runs/detect/predict/`

### Model Files

Trained model checkpoints in `runs/detect/car_detector_runs2/weights/`:
- **best.pt** - Best model based on validation mAP (use this for inference)
- **last.pt** - Final epoch checkpoint
- **fine_tuned_car_yolo.pt** - Exported model for easy deployment

## Usage Examples

### Quick Inference

```bash
# Using the trained model
python inference.py --image dataset/images/test/vid_4_980.jpg --conf 0.5

# Using a different confidence threshold
python inference.py --image path/to/image.jpg --conf 0.7
```

### Use Different Model

```python
# For better accuracy (slower)
model = YOLO('yolov8s.pt')

# For maximum speed
model = YOLO('yolov8n.pt')
```

## Requirements

- Python 3.11+
- PyTorch with CUDA support (recommended)
- OpenCV
- Ultralytics YOLO
- scikit-learn
- pandas
