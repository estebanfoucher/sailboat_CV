import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from utils.augmentation.augmentation_utils import parse_dataset, count_class_occurrences_and_images
except ModuleNotFoundError:
    from augmentation_utils import parse_dataset, count_class_occurrences_and_images

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print class distribution and statistics for a YOLO dataset.")
    parser.add_argument('parent_dir', type=str, help="Parent directory containing 'images' and 'labels' subfolders.")
    parser.add_argument('--n_classes', type=int, default=3, help="Number of classes in your dataset (default: 3)")
    args = parser.parse_args()

    PARENT_DIR = args.parent_dir
    N_CLASSES = args.n_classes

    class_occurrences, class_image_presence = count_class_occurrences_and_images(PARENT_DIR, N_CLASSES)
    total_occurrences = sum(class_occurrences)

    dist_vector = [count / total_occurrences if total_occurrences > 0 else 0.0 for count in class_occurrences]

    print("Class statistics:")
    for i in range(N_CLASSES):
        print(f"Class {i}: {class_occurrences[i]} occurrences, present in {len(class_image_presence[i])} images")

    print(f"\nDistribution vector: {dist_vector}")
    print(f"Total images: {len(set().union(*class_image_presence))}")
    print(f"Total class occurrences: {total_occurrences}") 