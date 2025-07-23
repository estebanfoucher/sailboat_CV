import os
import shutil
import argparse
import uuid


def process_name(name):
    # Remove underscores, lowercase
    return name.replace('_', '').lower()


def rename_in_place(input_dir):
    renamed_count = 0
    for sub in ['images', 'videos']:
        subfolder = os.path.join(input_dir, sub)
        if not os.path.isdir(subfolder):
            continue
        for fname in os.listdir(subfolder):
            fpath = os.path.join(subfolder, fname)
            if os.path.isdir(fpath):
                continue
            name, ext = os.path.splitext(fname)
            short_uuid = str(uuid.uuid4())[:6]
            new_name = f"{process_name(name)}{short_uuid}{ext.lower()}"
            new_path = os.path.join(subfolder, new_name)
            os.rename(fpath, new_path)
            print(f"Renamed: {fpath} -> {new_path}")
            renamed_count += 1
    
    print(f"All files renamed in: {input_dir}")
    print(f"Total files renamed: {renamed_count}")
    return renamed_count


def main():
    parser = argparse.ArgumentParser(description='Rename images/videos in-place with short UUID and folder prefix, keeping folder structure.')
    parser.add_argument('input_dir', help='Path to the folder containing images/videos subfolders')
    args = parser.parse_args()
    rename_in_place(args.input_dir)


if __name__ == '__main__':
    main() 