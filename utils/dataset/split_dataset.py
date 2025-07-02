import os
import random
import shutil
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

# Dataset related configurations
DATASET_VERSION = os.getenv('DATASET_VERSION')  # Default to 'v1' if not specified
DATASET_PATH = os.getenv('DATASET_PATH')  # Default to 'data' directory
LABEL_STUDIO_DATA_PATH = os.getenv('LABEL_STUDIO_DATA_PATH')

# Training related configurations
TRAIN_SPLIT = 0.8  # Default to 80% for training
VAL_SPLIT = 0.2    # Default to 10% for validation
TEST_SPLIT = 0

# Ensure splits sum to 1
assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, "Split ratios must sum to 1"

def split_yolo_dataset(base_dir, output_dir=None, val_size=None, test_size=None, random_state=42, additional_files=None):
    """
    Split YOLO dataset into train, validation and test sets.
    
    Args:
        base_dir (str): Path to source dataset directory
        output_dir (str): Path to destination directory (if None, creates 'dataset_split' in base_dir parent)
        val_size (float): Proportion for validation set (if None, uses config VAL_SPLIT)
        test_size (float): Proportion for test set (if None, uses config TEST_SPLIT)
        random_state (int): Random seed for reproducibility
        additional_files (list): Additional files to copy to all splits
    """
    
    # Convert to Path object for better path handling
    base_path = Path(base_dir)
    images_dir = base_path / 'images'
    labels_dir = base_path / 'labels'
    
    # Create output directory
    if output_dir is None:
        output_path = Path('data/splitted_datasets') / base_path.name
    else:
        output_path = Path(output_dir)
    
    print(f"Creating split dataset in: {output_path}")
    
    # Validate input directories exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Use configuration values if not provided
    val_size = val_size if val_size is not None else VAL_SPLIT
    test_size = test_size if test_size is not None else TEST_SPLIT
    
    # Output directories in the separate destination folder
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    # Create directories if they don't exist
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Only create test directory if test_size > 0
    if test_size > 0:
        (test_dir / 'images').mkdir(parents=True, exist_ok=True)
        (test_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get list of image files (common image extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(images_dir) 
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError("No image files found in the images directory")
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle for randomness
    random.seed(random_state)
    random.shuffle(image_files)
    
    # Handle different split scenarios
    if test_size == 0:
        # Only split into train and val
        if val_size > 0:
            train_files, val_files = train_test_split(
                image_files,
                test_size=val_size,
                random_state=random_state
            )
            test_files = []
        else:
            # No validation or test split
            train_files = image_files
            val_files = []
            test_files = []
    else:
        # First split into train and temp (val + test)
        train_files, temp_files = train_test_split(
            image_files,
            test_size=(val_size + test_size),
            random_state=random_state
        )
        
        # Then split temp into val and test
        if val_size > 0:
            val_files, test_files = train_test_split(
                temp_files,
                test_size=(test_size / (val_size + test_size)),
                random_state=random_state
            )
        else:
            val_files = []
            test_files = temp_files
    
    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")
    
    def copy_files_with_validation(files, dest_dir, split_name):
        """Copy files and validate that corresponding labels exist."""
        if not files:  # Skip if no files to copy
            print(f"No files to copy for {split_name} set")
            return 0
            
        missing_labels = []
        copied_count = 0
        
        for file in files:
            # Copy image
            src_img = images_dir / file
            dest_img = dest_dir / 'images' / file
            shutil.copy2(src_img, dest_img)
            
            # Copy corresponding label file
            label_file = Path(file).stem + '.txt'
            src_label = labels_dir / label_file
            dest_label = dest_dir / 'labels' / label_file
            
            if src_label.exists():
                shutil.copy2(src_label, dest_label)
                copied_count += 1
            else:
                missing_labels.append(label_file)
        
        if missing_labels:
            print(f"Warning: {len(missing_labels)} label files missing for {split_name} set:")
            for label in missing_labels[:5]:  # Show first 5
                print(f"  - {label}")
            if len(missing_labels) > 5:
                print(f"  ... and {len(missing_labels) - 5} more")
        
        print(f"Successfully copied {copied_count} image-label pairs to {split_name} set")
        return copied_count
    
    # Copy training and validation files
    train_copied = copy_files_with_validation(train_files, train_dir, "training")
    val_copied = copy_files_with_validation(val_files, val_dir, "validation")
    test_copied = copy_files_with_validation(test_files, test_dir, "test") if test_size > 0 else 0
    
    # Create a summary report
    summary = {
        "source_dataset": str(base_path),
        "output_directory": str(output_path),
        "total_images": len(image_files),
        "train_images": len(train_files),
        "val_images": len(val_files),
        "test_images": len(test_files),
        "train_copied": train_copied,
        "val_copied": val_copied,
        "test_copied": test_copied,
        "train_split": 1 - (val_size + test_size),
        "val_split": val_size,
        "test_split": test_size,
        "random_state": random_state
    }
    
    # Copy additional files to the root of output directory (not to train/val subfolders)
    if additional_files is None:
        additional_files = ['classes.txt', 'notes.json', 'data.yaml', 'dataset.yaml']
    
    for file in additional_files:
        src_file = base_path / file
        if src_file.exists():
            shutil.copy2(src_file, output_path / file)
            print(f"Copied {file} to dataset root")
        else:
            print(f"Info: {file} not found, skipping...")
    
    # Create data.yaml if it doesn't exist (standard YOLO config file)
    data_yaml_path = output_path / 'data.yaml'
    if not data_yaml_path.exists():
        # Try to read classes from classes.txt
        classes_file = base_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            
            # Prepare paths for data.yaml
            data_yaml_content = ["# YOLO dataset configuration",
                               f"train: {output_path}/train/images",
                               f"val: {output_path}/val/images"]
            
            # Only add test path if test split exists
            if test_size > 0:
                data_yaml_content.append(f"test: {output_path}/test/images")
            
            data_yaml_content.extend([
                "",
                f"# Number of classes",
                f"nc: {len(classes)}",
                "",
                f"# Class names",
                f"names: {classes}"
            ])
            
            with open(data_yaml_path, 'w') as f:
                f.write('\n'.join(data_yaml_content))
            print("Created data.yaml file")
        else:
            print("Warning: No classes.txt found, couldn't create data.yaml")
    
    # Save summary to output directory
    with open(output_path / 'split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset split complete!")
    print(f"Split dataset saved to: {output_path}")
    print(f"Summary saved to: {output_path / 'split_summary.json'}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Split a YOLO dataset into train/val/test folders.")
    parser.add_argument('dataset_path', type=str, help='Path to the non-splitted dataset directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for the split dataset (default: data/splitted_datasets/<dataset_name>)')
    parser.add_argument('--val_size', type=float, default=None, help='Validation split size (default: 0.2)')
    parser.add_argument('--test_size', type=float, default=None, help='Test split size (default: 0)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    try:
        summary = split_yolo_dataset(
            base_dir=args.dataset_path,
            output_dir=args.output_dir,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
            additional_files=['classes.txt', 'notes.json']
        )
        print("\nSplit Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()