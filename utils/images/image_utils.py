from PIL import Image
from pathlib import Path


def get_image_resolution(image_path):
    """Get the resolution (width, height) of an image file."""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception:
        return None


def is_image_file(filename):
    """Check if a file is a supported image format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions

# I have an image which name is in format frame_<id_string>_<time_sec>.suffix
# I want to get the id_string and time_sec from the image name and return a dict
# Example: frame_123456_123456.jpg
# Output: {'id_string': '123456', 'time_sec': 123456}

def get_id_string_and_time_sec(image_name):
    """Get the id_string and time_sec from the image name."""
    return {'id_string': image_name.split('_')[1], 'time_sec': image_name.split('_')[2].split('.')[0]}



