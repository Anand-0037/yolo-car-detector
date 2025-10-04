import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import argparse
import logging

def _detect_images_dir() -> str:
    candidates = [
        'dataset/data/training_images',
        'dataset/data/testing_images',
    ]
    for c in candidates:
        if os.path.isdir(c) and any(fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')) for fn in os.listdir(c)):
            return c
    raise FileNotFoundError("No images found in 'dataset/data/training_images' or 'dataset/data/testing_images'. Ensure images are present.")

def split_dataset(csv_path=None, img_dir=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting dataset splitting process...")
    if csv_path is None:
        csv_candidates = [
            'dataset/data/train_solution_bounding_boxes.csv',
            'dataset/data/train_solution_bounding_boxes (1).csv',
        ]
        csv_path = None
        for path in csv_candidates:
            if os.path.exists(path):
                csv_path = path
                break
        if csv_path is None:
            raise FileNotFoundError("Could not find the bounding box CSV file in 'dataset/data/'. Expected one of: " + ", ".join(csv_candidates))
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Specified CSV file not found: {csv_path}")

    if img_dir is None:
        img_dir = _detect_images_dir()
    else:
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Specified image directory not found: {img_dir}")

    train_img_dir = 'dataset/images/train'
    train_lbl_dir = 'dataset/labels/train'
    val_img_dir = 'dataset/images/val'
    val_lbl_dir = 'dataset/labels/val'
    test_img_dir = 'dataset/images/test'
    test_lbl_dir = 'dataset/labels/test'

    df = pd.read_csv(csv_path)
    csv_images = df['image'].astype(str).unique().tolist()
    existing_images = {fn for fn in os.listdir(img_dir) if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))}
    images = sorted([im for im in csv_images if im in existing_images])

    logger.info(f"Images in CSV: {len(csv_images)} | Existing in '{img_dir}': {len(existing_images)} | Usable: {len(images)}")
    skipped_images = [im for im in csv_images if im not in existing_images]
    if skipped_images:
        print(f"Skipped images (not found in '{img_dir}'): {len(skipped_images)} examples, e.g. {skipped_images[:10]}")
        with open("skipped_images.txt", "w") as f:
            for img in skipped_images:
                f.write(f"{img}\n")
    if len(images) == 0:
        raise FileNotFoundError(f"No overlapping images between CSV and '{img_dir}'. Please verify dataset paths.")
    
    min_images = 10
    if len(images) < min_images:
        raise ValueError(f"Dataset too small: {len(images)} images found, minimum {min_images} required for meaningful splits.")

    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)

    for img_list, img_out, lbl_out in [
        (train_imgs, train_img_dir, train_lbl_dir),
        (val_imgs, val_img_dir, val_lbl_dir),
        (test_imgs, test_img_dir, test_lbl_dir)
    ]:

        for img in img_list:
            src_img = os.path.join(img_dir, img)
            dst_img = os.path.join(img_out, img)
            if not os.path.exists(src_img):
                print(f"Warning: missing image {src_img}")
                continue
            try:
                shutil.copy2(src_img, dst_img)
            except Exception as e:
                print(f"Copy failed for {img}: {e}")
                continue

        split_df = df[df['image'].isin(img_list)]
        convert_bbox_to_yolo_split(split_df, img_out, lbl_out)

    print("Dataset split complete!")
    print(f"Train: {len(train_imgs)} images")
    print(f"Val: {len(val_imgs)} images")
    print(f"Test: {len(test_imgs)} images")

def convert_bbox_to_yolo_split(df, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in df['image'].unique():
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping...")
            continue

        img_height, img_width = img.shape[:2]
        
        if img_width == 0 or img_height == 0:
            print(f"Warning: Invalid image dimensions for {img_path} ({img_width}x{img_height}), skipping...")
            continue

        img_boxes = df[df['image'] == img_name]

        label_file = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.txt')

        with open(label_file, 'w') as f:
            for _, box in img_boxes.iterrows():
                x_center = (box['xmin'] + box['xmax']) / 2.0 / img_width
                y_center = (box['ymin'] + box['ymax']) / 2.0 / img_height
                width = (box['xmax'] - box['xmin']) / img_width
                height = (box['ymax'] - box['ymin']) / img_height

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset and convert to YOLO format')
    parser.add_argument('--csv', type=str, help='Path to bounding box CSV file (auto-detect if not specified)')
    parser.add_argument('--images', type=str, help='Path to images directory (auto-detect if not specified)')
    
    args = parser.parse_args()
    split_dataset(csv_path=args.csv, img_dir=args.images)
