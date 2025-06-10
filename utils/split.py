import os
import random
import shutil
import json
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_yolo_dataset(base_dir, output_dir=None, test_size=0.2, random_state=42, additional_files=None):
    """
    Split YOLO dataset into train and validation sets.
    
    Args:
        base_dir (str): Path to source dataset directory
        output_dir (str): Path to destination directory (if None, creates 'dataset_split' in base_dir parent)
        test_size (float): Proportion for validation set (default: 0.2)
        random_state (int): Random seed for reproducibility
        additional_files (list): Additional files to copy to both splits
    """
    
    # Convert to Path object for better path handling
    base_path = Path(base_dir)
    images_dir = base_path / 'images'
    labels_dir = base_path / 'labels'
    
    # Create output directory
    if output_dir is None:
        output_path = base_path.parent / f"{base_path.name}_split"
    else:
        output_path = Path(output_dir)
    
    print(f"Creating split dataset in: {output_path}")
    
    # Validate input directories exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Output directories in the separate destination folder
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    # Create directories if they don't exist
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
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
    
    # Split the dataset
    train_files, val_files = train_test_split(
        image_files, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    
    def copy_files_with_validation(files, dest_dir, split_name):
        """Copy files and validate that corresponding labels exist."""
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
    
    # Create a summary report
    summary = {
        "source_dataset": str(base_path),
        "output_directory": str(output_path),
        "total_images": len(image_files),
        "train_images": len(train_files),
        "val_images": len(val_files),
        "train_copied": train_copied,
        "val_copied": val_copied,
        "test_size": test_size,
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
            
            data_yaml_content = f"""# YOLO dataset configuration
train: train/images
val: val/images

# Number of classes
nc: {len(classes)}

# Class names
names: {classes}
"""
            with open(data_yaml_path, 'w') as f:
                f.write(data_yaml_content)
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

# Example usage
if __name__ == "__main__":
    folder_name = 'pennon-label-yolo-00'
    # Define paths
    base_dir = f'labeled_data/{folder_name}'
    output_dir = f'datasets/{folder_name}'
    
    try:
        summary = split_yolo_dataset(
            base_dir=base_dir,
            output_dir=output_dir,  # Specify output directory
            test_size=0.2,
            random_state=42,
            additional_files=['classes.txt', 'notes.json']
        )
        print("\nSplit Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")