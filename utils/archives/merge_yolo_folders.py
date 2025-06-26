import os
import shutil


def merge_yolo_folders(source, target):
    # Get parent directory and new folder name
    parent_dir = os.path.dirname(target)
    source_name = os.path.basename(source)
    target_name = os.path.basename(target)
    new_folder_name = f"{target_name}_extends_{source_name}"
    new_folder_path = os.path.join(parent_dir, new_folder_name)

    # Copy the target folder to the new location
    if os.path.exists(new_folder_path):
        raise FileExistsError(f"{new_folder_path} already exists.")
    shutil.copytree(target, new_folder_path)

    for subfolder in ["images", "labels"]:
        src_sub = os.path.join(source, subfolder)
        dst_sub = os.path.join(new_folder_path, subfolder)
        if not os.path.exists(src_sub):
            continue
        if not os.path.exists(dst_sub):
            os.makedirs(dst_sub)
        for fname in os.listdir(src_sub):
            src_file = os.path.join(src_sub, fname)
            dst_file = os.path.join(dst_sub, fname)
            # Only copy if it's a file and doesn't already exist
            if os.path.isfile(src_file):
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Already exists, not copied: {os.path.relpath(dst_file, new_folder_path)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python merge_yolo_folders.py <source_folder> <target_folder>")
        sys.exit(1)
    source_folder = sys.argv[1]
    target_folder = sys.argv[2]
    merge_yolo_folders(source_folder, target_folder)
    print(f"Merged '{source_folder}' into a copy of '{target_folder}'.") 