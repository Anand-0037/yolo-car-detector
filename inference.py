import os
os.environ['YOLO_OFFLINE'] = '1'

from ultralytics import YOLO
import argparse

def detect_cars(model_path, image_path, conf_threshold=0.5, save_result=True):
    model = YOLO(model_path)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Detecting cars in: {image_path}")

    results = model(image_path, conf=conf_threshold, save=save_result, project='runs/detect', name='predict')
    detection_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"Detected {detection_count} cars")
    return detection_count

def main():
    parser = argparse.ArgumentParser(description='Car Detection Inference')
    parser.add_argument('--model', type=str, default='fine_tuned_car_yolo.pt',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using train_and_detect.py")
        return

    detect_cars(args.model, args.image, args.conf)

if __name__ == "__main__":
    main()
