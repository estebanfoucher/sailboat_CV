import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_label_prefix.py <labels_folder>")
        sys.exit(1)
    labels_folder = sys.argv[1]
    if not os.path.isdir(labels_folder):
        print(f"Error: {labels_folder} is not a valid directory.")
        sys.exit(1)
    for filename in os.listdir(labels_folder):
        if filename.endswith('.txt'):
            idx = filename.find('frame')
            if idx != -1:
                new_filename = filename[idx:]
                old_path = os.path.join(labels_folder, filename)
                new_path = os.path.join(labels_folder, new_filename)
                if old_path != new_path:
                    if os.path.exists(new_path):
                        print(f"Warning: {new_filename} already exists. Skipping {filename}.")
                    else:
                        os.rename(old_path, new_path)
                        print(f"Renamed {filename} -> {new_filename}")
    print(f"Prefixes removed from all .txt filenames in {labels_folder}") 