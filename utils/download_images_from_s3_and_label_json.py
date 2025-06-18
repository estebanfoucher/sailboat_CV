import os
import boto3
import json
import logging
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('s3_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_images_from_s3(json_file_path, images_folder_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize a session using Boto3
    s3 = boto3.client('s3')

    # Ensure the images folder exists
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path)

    total_entries = len(data)
    successful_downloads = 0
    failed_downloads = []
    skipped_downloads = []

    logger.info(f"Starting download of {total_entries} images")

    # Iterate over the JSON data to extract S3 paths
    for entry in data:
        try:
            s3_path = entry['data']['image']
            # Parse the S3 path
            path_parts = s3_path.replace('s3://', '').split('/')
            bucket_name = path_parts[0]
            s3_key = '/'.join(path_parts[1:])

            # Extract the video_id and frame_image
            video_id = path_parts[-2]
            frame_image = path_parts[-1]

            # Create the new file name
            new_file_name = video_id + '__' + frame_image
            target_path = os.path.join(images_folder_path, new_file_name)

            # Skip if file already exists
            if os.path.exists(target_path):
                skipped_downloads.append({
                    'file': new_file_name,
                    'reason': 'File already exists'
                })
                logger.info(f"Skipped {new_file_name} - already exists")
                continue

            # Download the file
            try:
                s3.download_file(bucket_name, s3_key, target_path)
                successful_downloads += 1
                logger.info(f"Downloaded ({successful_downloads}/{total_entries}): {s3_key} to {new_file_name}")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                failed_downloads.append({
                    'file': new_file_name,
                    's3_path': s3_path,
                    'error': f"{error_code}: {error_message}"
                })
                logger.error(f"Failed to download {s3_path}: {error_code} - {error_message}")

        except KeyError as e:
            failed_downloads.append({
                'entry': entry,
                'error': f"Missing key in JSON: {str(e)}"
            })
            logger.error(f"Invalid JSON entry structure: {str(e)}")
        except Exception as e:
            failed_downloads.append({
                'entry': entry,
                'error': str(e)
            })
            logger.error(f"Unexpected error: {str(e)}")

    # Print summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Successfully downloaded: {successful_downloads}")
    logger.info(f"Failed downloads: {len(failed_downloads)}")
    logger.info(f"Skipped (already exist): {len(skipped_downloads)}")

    if failed_downloads:
        logger.info("\nFailed Downloads:")
        for fail in failed_downloads:
            logger.info(f"- {fail}")

    if skipped_downloads:
        logger.info("\nSkipped Downloads:")
        for skip in skipped_downloads:
            logger.info(f"- {skip}")

    return successful_downloads, failed_downloads, skipped_downloads

# Example usage
json_file_path = './docker/label_studio/data/export/yolo/pennon-label-yolo-01/pennon-label-yolo-01.json'
images_folder_path = './docker/label_studio/data/export/yolo/pennon-label-yolo-01/images'
download_images_from_s3(json_file_path, images_folder_path)