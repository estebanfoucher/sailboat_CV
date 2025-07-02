import os
import shutil
import argparse
import filetype


def organize_raw_data(input_dir):
    images_dir = os.path.join(input_dir, 'images')
    videos_dir = os.path.join(input_dir, 'videos')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    image_count = 0
    video_count = 0

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if os.path.isdir(fpath):
            continue  # skip directories
        kind = filetype.guess(fpath)
        if kind is None:
            continue  # unknown file type, leave as is
        if kind.mime.startswith('image'):
            shutil.move(fpath, os.path.join(images_dir, fname))
            image_count += 1
        elif kind.mime.startswith('video'):
            shutil.move(fpath, os.path.join(videos_dir, fname))
            video_count += 1
        # else: leave as is

    print(f"Images found and moved: {image_count}")
    print(f"Videos found and moved: {video_count}")


def main():
    parser = argparse.ArgumentParser(description='Organize raw data by file type.')
    parser.add_argument('input_dir', help='Path to the folder containing raw data')
    args = parser.parse_args()
    organize_raw_data(args.input_dir)


if __name__ == '__main__':
    main() 