import os
import shutil
import random
from typing import Dict, List, Set, Tuple

def parse_dataset(parent_dir: str, n_classes: int) -> Dict[str, Set[int]]:
    """
    Map each image filename to the set of classes present in its label file.
    parent_dir: directory containing 'images' and 'labels' subfolders
    Returns: {image_filename: set(class_indices)}
    """
    images_dir = os.path.join(parent_dir, 'images')
    labels_dir = os.path.join(parent_dir, 'labels')
    image_to_classes = {}
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            classes = set()
            for line in f:
                if line.strip():
                    class_idx = int(line.split()[0])
                    if 0 <= class_idx < n_classes:
                        classes.add(class_idx)
            if classes:
                # Assume image has same base name as label
                base = os.path.splitext(label_file)[0]
                # Try both .jpg and .png
                for ext in ['.jpg', '.png']:
                    img_path = os.path.join(images_dir, base + ext)
                    if os.path.exists(img_path):
                        image_to_classes[img_path] = classes
                        break
    return image_to_classes

def compute_class_distribution(image_list: List[str], image_to_classes: Dict[str, Set[int]], n_classes: int) -> List[float]:
    """
    Compute the class distribution vector for a list of images.
    Returns: [proportion for each class]
    """
    class_counts = [0] * n_classes
    for img in image_list:
        for c in image_to_classes.get(img, []):
            class_counts[c] += 1
    total = sum(class_counts)
    if total == 0:
        return [0.0] * n_classes
    return [count / total for count in class_counts]

def copy_image_and_label(src_img: str, src_lbl: str, dst_img: str, dst_lbl: str):
    """
    Copy image and label files to new destination paths.
    """
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)

def get_label_path_from_image(image_path: str, parent_dir: str) -> str:
    """
    Given an image path, return the corresponding label file path in labels_dir.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    labels_dir = os.path.join(parent_dir, 'labels')
    return os.path.join(labels_dir, base + '.txt')

def count_class_occurrences_and_images(parent_dir: str, n_classes: int):
    """
    Returns (class_occurrences, class_image_presence)
    - class_occurrences[i]: total number of times class i appears in all label files
    - class_image_presence[i]: set of images where class i appears at least once
    """
    images_dir = os.path.join(parent_dir, 'images')
    labels_dir = os.path.join(parent_dir, 'labels')
    class_occurrences = [0] * n_classes
    class_image_presence = [set() for _ in range(n_classes)]
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        base = os.path.splitext(label_file)[0]
        img_path = None
        for ext in ['.jpg', '.png']:
            candidate = os.path.join(images_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        with open(label_path, 'r') as f:
            present_classes = set()
            for line in f:
                if line.strip():
                    class_idx = int(line.split()[0])
                    if 0 <= class_idx < n_classes:
                        class_occurrences[class_idx] += 1
                        present_classes.add(class_idx)
            for c in present_classes:
                class_image_presence[c].add(img_path)
    return class_occurrences, class_image_presence 