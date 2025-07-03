import sys
import os
import argparse
import shutil
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from utils.augmentation.augmentation_utils import (
        parse_dataset, compute_class_distribution, copy_image_and_label, get_label_path_from_image, count_class_occurrences_and_images
    )
except ModuleNotFoundError:
    from augmentation_utils import (
        parse_dataset, compute_class_distribution, copy_image_and_label, get_label_path_from_image, count_class_occurrences_and_images
    )

# --- CONFIGURATION ---
N_CLASSES = 3  # Number of classes in your dataset
TARGET_DIST = [1/3, 1/3, 1/3]  # Target class distribution (must sum to 1)
EPSILON = 0.1  # Allowed deviation per class
MAX_ITERATIONS = 3000  # Maximum number of additional samples to add

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oversample a YOLO dataset to balance class distribution.")
    parser.add_argument('parent_dir', type=str, help="Parent directory containing 'images' and 'labels' subfolders.")
    args = parser.parse_args()

    PARENT_DIR = args.parent_dir
    OUTPUT_PARENT_DIR = PARENT_DIR.rstrip('/') + '_oversampled'
    OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_PARENT_DIR, 'images')
    OUTPUT_LABELS_DIR = os.path.join(OUTPUT_PARENT_DIR, 'labels')

    # --- SETUP OUTPUT DIRECTORIES ---
    if os.path.exists(OUTPUT_PARENT_DIR):
        print(f"Output folder {OUTPUT_PARENT_DIR} already exists. Removing it.")
        shutil.rmtree(OUTPUT_PARENT_DIR)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # --- PARSE DATASET ---
    image_to_classes = parse_dataset(PARENT_DIR, N_CLASSES)
    all_images = list(image_to_classes.keys())

    # --- INITIALIZE OUTPUT WITH ORIGINAL DATASET ---
    output_images = []
    for img_path in all_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        label_path = get_label_path_from_image(img_path, PARENT_DIR)
        out_img_path = os.path.join(OUTPUT_IMAGES_DIR, base + ext)
        out_lbl_path = os.path.join(OUTPUT_LABELS_DIR, base + '.txt')
        copy_image_and_label(img_path, label_path, out_img_path, out_lbl_path)
        output_images.append(out_img_path)

    def get_output_parent_dir():
        return OUTPUT_PARENT_DIR

    # --- COMPUTE INITIAL CLASS OCCURRENCES AND DISTRIBUTION ---
    output_parent_dir = get_output_parent_dir()
    class_occurrences, class_image_presence = count_class_occurrences_and_images(output_parent_dir, N_CLASSES)
    total_occurrences = sum(class_occurrences)
    dist_vector = [count / total_occurrences if total_occurrences > 0 else 0.0 for count in class_occurrences]

    print("Initial class statistics:")
    for i in range(N_CLASSES):
        print(f"Class {i}: {class_occurrences[i]} occurrences, present in {len(class_image_presence[i])} images")
    print(f"Distribution vector: {dist_vector}")
    print(f"Total images: {len(set().union(*class_image_presence))}")
    print(f"Total class occurrences: {total_occurrences}\n")

    def l1_distance(vec1, vec2):
        return sum(abs(a - b) for a, b in zip(vec1, vec2))

    # --- SAMPLING LOOP ---
    for i in range(MAX_ITERATIONS):
        img_path = random.choice(all_images)
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        label_path = get_label_path_from_image(img_path, PARENT_DIR)
        # Simulate adding this image
        simulated_class_occurrences = class_occurrences.copy()
        simulated_image_presence = [s.copy() for s in class_image_presence]
        present_classes = set()
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    class_idx = int(line.split()[0])
                    if 0 <= class_idx < N_CLASSES:
                        simulated_class_occurrences[class_idx] += 1
                        present_classes.add(class_idx)
        for c in present_classes:
            simulated_image_presence[c].add(img_path)
        simulated_total = sum(simulated_class_occurrences)
        simulated_dist = [count / simulated_total if simulated_total > 0 else 0.0 for count in simulated_class_occurrences]
        current_dist = [count / total_occurrences if total_occurrences > 0 else 0.0 for count in class_occurrences]
        dist_now = l1_distance(current_dist, TARGET_DIST)
        dist_sim = l1_distance(simulated_dist, TARGET_DIST)
        if dist_sim < dist_now:
            # Accept: copy image/label, update stats
            dup_img_name = f"{base}_dup{i}{ext}"
            dup_lbl_name = f"{base}_dup{i}.txt"
            out_img_path = os.path.join(OUTPUT_IMAGES_DIR, dup_img_name)
            out_lbl_path = os.path.join(OUTPUT_LABELS_DIR, dup_lbl_name)
            copy_image_and_label(img_path, label_path, out_img_path, out_lbl_path)
            output_images.append(out_img_path)
            class_occurrences = simulated_class_occurrences
            class_image_presence = simulated_image_presence
            total_occurrences = simulated_total
            dist_vector = simulated_dist
            print(f"Iteration {i+1}:")
            for j in range(N_CLASSES):
                print(f"  Class {j}: {class_occurrences[j]} occurrences, present in {len(class_image_presence[j])} images")
            print(f"  Distribution vector: {dist_vector}")
            print(f"  Total images: {len(set().union(*class_image_presence))}")
            print(f"  Total class occurrences: {total_occurrences}\n")
            if all(abs(dist_vector[j] - TARGET_DIST[j]) <= EPSILON for j in range(N_CLASSES)):
                print(f"Target distribution reached at iteration {i+1}!")
                break
        # else: skip, try next

    print("Final class statistics:")
    for i in range(N_CLASSES):
        print(f"Class {i}: {class_occurrences[i]} occurrences, present in {len(class_image_presence[i])} images")
    print(f"Distribution vector: {dist_vector}")
    print(f"Total images: {len(set().union(*class_image_presence))}")
    print(f"Total class occurrences: {total_occurrences}") 