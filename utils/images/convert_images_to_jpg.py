#!/usr/bin/env python3
"""
Convert images in a folder (including .heic) to .jpg in another folder.

Usage:
    python convert_images_to_jpg.py /path/to/source_folder /path/to/dest_folder
"""
import argparse
import os
from pathlib import Path
from PIL import Image

# Register HEIC/HEIF support if pillow-heif is available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

def convert_image_to_jpg(src_path, dest_path):
    with Image.open(src_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(dest_path, 'JPEG')

def main():
    parser = argparse.ArgumentParser(description="Convert all images in a folder (including .heic) to .jpg in another folder.")
    parser.add_argument('src_folder', help='Source folder with images')
    parser.add_argument('dest_folder', help='Destination folder for .jpg images')
    args = parser.parse_args()

    src_folder = Path(args.src_folder)
    dest_folder = Path(args.dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.heic'}

    for img_file in src_folder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in supported_exts:
            dest_file = dest_folder / (img_file.stem + '.jpg')
            try:
                convert_image_to_jpg(img_file, dest_file)
                print(f"Converted {img_file} -> {dest_file}")
            except Exception as e:
                print(f"Failed to convert {img_file}: {e}")

if __name__ == '__main__':
    main() 