import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def prepare_dataset(source_dir, train_ratio=0.8):
    """
    Prepare dataset by splitting images and labels into train/val sets.
    
    Args:
        source_dir: Directory containing images and labels
        train_ratio: Ratio of training data (default: 0.8)
    """
    # Get package directory
    pkg_dir = Path(__file__).parent.parent
    data_dir = pkg_dir / 'training' / 'data'
    
    # Create train/val directories
    train_img_dir = data_dir / 'images' / 'train'
    train_label_dir = data_dir / 'labels' / 'train'
    val_img_dir = data_dir / 'images' / 'val'
    val_label_dir = data_dir / 'labels' / 'val'
    
    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    source_dir = Path(source_dir)
    image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
    random.shuffle(image_files)
    
    # Split into train/val
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copy files to train/val directories
    for files, img_dir, label_dir in [
        (train_files, train_img_dir, train_label_dir),
        (val_files, val_img_dir, val_label_dir)
    ]:
        for img_path in tqdm(files, desc=f"Copying to {img_dir.parent.name}"):
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            
            # Copy corresponding label file
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy2(label_path, label_dir / label_path.name)
    
    print(f"Dataset prepared:")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare YOLOv8 training dataset')
    parser.add_argument('source_dir', help='Directory containing images and labels')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of training data (default: 0.8)')
    args = parser.parse_args()
    
    prepare_dataset(args.source_dir, args.train_ratio) 