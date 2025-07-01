import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import contextlib
import sys

# Paths
MODEL_PATH = f'runs/detect/train-03-nano-augment/weights/best.pt'
INPUT_ROOT = Path('seb_perso_24_06_2025/video_frames_extracted')
OUTPUT_ROOT = Path('seb_perso_24_06_2025/video_frames_extracted_filtered')

# Supported image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def count_images_per_subfolder():
    subfolder_counts = {}
    for subdir, _, files in os.walk(INPUT_ROOT):
        rel_dir = os.path.relpath(subdir, INPUT_ROOT)
        if rel_dir == ".":
            rel_dir = "(root)"
        count = sum(1 for fname in files if Path(fname).suffix.lower() in IMAGE_EXTS)
        if count > 0:
            subfolder_counts[rel_dir] = count
    return subfolder_counts

# Context manager to suppress stdout and stderr
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def filter_and_copy_images(confidence: float = 0.5):
    # Load YOLO model with verbose=False
    with suppress_stdout_stderr():
        model = YOLO(MODEL_PATH, verbose=False)
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f'Input directory not found: {INPUT_ROOT}')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Count images per subfolder and log
    subfolder_counts = count_images_per_subfolder()
    print("Image count per subfolder:")
    for folder, count in sorted(subfolder_counts.items(), key=lambda x: x[0]):
        print(f"  {folder}: {count}")

    total = 0
    kept = 0
    subfolders = [k for k in subfolder_counts.keys()]
    with tqdm(subfolders, desc="Folders", unit="folder") as folder_bar:
        for rel_dir in folder_bar:
            in_dir = INPUT_ROOT / rel_dir if rel_dir != "(root)" else INPUT_ROOT
            out_dir = OUTPUT_ROOT / rel_dir if rel_dir != "(root)" else OUTPUT_ROOT
            out_dir.mkdir(parents=True, exist_ok=True)
            # List images in this folder
            images = [fname for fname in os.listdir(in_dir) if Path(fname).suffix.lower() in IMAGE_EXTS]
            with tqdm(images, desc=f"Images in {rel_dir}", unit="img", leave=False) as img_bar:
                for fname in img_bar:
                    in_path = in_dir / fname
                    # Run YOLO inference with output suppressed
                    with suppress_stdout_stderr():
                        results = model(in_path, verbose=False)
                    # If any objects detected above threshold, copy
                    has_detection = False
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        scores = results[0].boxes.conf.cpu().numpy()
                        if (scores >= confidence).any():
                            has_detection = True
                    if has_detection:
                        shutil.copy2(in_path, out_dir / fname)
                        kept += 1
                    total += 1
    print(f"Done. {kept}/{total} images kept (with detections >= {confidence}). Output: {OUTPUT_ROOT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter images by YOLO detections.")
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence score for detections (default: 0.5)')
    args = parser.parse_args()
    filter_and_copy_images(confidence=args.confidence) 