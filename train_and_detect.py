import os
os.environ['YOLO_OFFLINE'] = '1'

from ultralytics import YOLO
import torch
import argparse

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        print("GPU not available, using CPU")
        return False

def train_model(device_to_use, model_name='yolov8n.pt'):
    try:
        print(f"Loading pre-trained YOLO model: {model_name}")
        model = YOLO(model_name)

        print("Starting training...")
        results = model.train(
            data='dataset.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            device=device_to_use,
            name='car_detector_runs',
            plots=True,
            save=True,
            patience=10,
            lr0=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            close_mosaic=10,
        )

        print("Training completed!")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def validate_model(model):
    print("Validating model...")
    metrics = model.val()

    print(f"Validation Results:")
    print(f"mAP@0.5: {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")

    return metrics

def save_model(model):
    model_path = 'fine_tuned_car_yolo.pt'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    return model_path

def detect_cars(model, test_image_path=None):
    if test_image_path is None:
        test_dir = 'dataset/images/val'
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if test_images:
            test_image_path = os.path.join(test_dir, test_images[0])
        else:
            print("No test images found!")
            return

    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    print(f"Detecting cars in: {test_image_path}")

    model(test_image_path, save=True, conf=0.5, project='runs/detect', name='predict')
    return 'runs/detect/predict'

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for car detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO model to use (e.g., yolov8n.pt, yolov8s.pt)')
    args = parser.parse_args()
    
    print("=== YOLO Car Detection Training ===")

    device_to_use = 'cuda' if check_gpu() else 'cpu'

    train_images_dir = 'dataset/images/train'
    has_images = os.path.isdir(train_images_dir) and any(
        p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))
        for p in os.listdir(train_images_dir)
    )
    if not has_images:
        print("Error: Training data not found or empty in 'dataset/images/train'.")
        print("Please ensure you have run the following steps:")
        print("1. Download the dataset: 'uv run python fetch_dataset.py'")
        print("2. Split the dataset:   'uv run python split_dataset.py'")
        return

    model = train_model(device_to_use, args.model)

    metrics = validate_model(model)

    model_path = save_model(model)

    print("\n=== Testing Detection ===")
    detect_cars(model)

    print("\n=== Training Complete ===")
    print("Check the following for results:")
    print("- runs/detect/car_detector_runs/ for training plots and metrics")
    print("- runs/detect/predict/ for detection results")
    print(f"- {model_path} for the trained model")

if __name__ == "__main__":
    main()
