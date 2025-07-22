#!/usr/bin/env python3
import os
import boto3
import json
import logging
from botocore.exceptions import ClientError
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('s3_cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def delete_unannotated_objects(s3_folder_path, annotated_json_path):
    """
    Delete all files from S3 folder except those listed in the annotated JSON file.
    
    Args:
        s3_folder_path (str): S3 folder path (e.g., s3://bucket-name/folder/path/)
        annotated_json_path (str): Path to JSON file containing annotated tasks
    """
    # Parse S3 folder path
    if not s3_folder_path.startswith('s3://'):
        logger.error(f"Invalid S3 path format: {s3_folder_path}")
        return
    
    folder_parts = s3_folder_path.replace('s3://', '').split('/')
    bucket_name = folder_parts[0]
    folder_prefix = '/'.join(folder_parts[1:])
    
    # Ensure folder prefix ends with / if not empty
    if folder_prefix and not folder_prefix.endswith('/'):
        folder_prefix += '/'
    
    logger.info(f"Bucket: {bucket_name}")
    logger.info(f"Folder prefix: {folder_prefix}")
    
    # Load annotated tasks JSON
    try:
        with open(annotated_json_path, 'r') as file:
            annotated_data = json.load(file)
    except FileNotFoundError:
        logger.error(f"Annotated JSON file not found: {annotated_json_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {annotated_json_path}: {str(e)}")
        return

    # Extract files to keep (annotated files)
    files_to_keep = set()
    invalid_entries = []
    
    logger.info(f"Processing {len(annotated_data)} annotated entries")
    
    for i, entry in enumerate(annotated_data):
        try:
            s3_path = entry['data']['image']
            # Parse the S3 path to get just the filename
            if s3_path.startswith('s3://'):
                path_parts = s3_path.replace('s3://', '').split('/')
                filename = path_parts[-1]
                # Store the full key path within the bucket
                entry_bucket = path_parts[0]
                entry_key = '/'.join(path_parts[1:])
                
                # Only consider files from the same bucket and folder
                if entry_bucket == bucket_name and entry_key.startswith(folder_prefix):
                    files_to_keep.add(entry_key)
                    logger.debug(f"Will keep: {entry_key}")
                
        except KeyError as e:
            invalid_entries.append({
                'entry_index': i,
                'error': f"Missing key in JSON: {str(e)}"
            })
        except Exception as e:
            invalid_entries.append({
                'entry_index': i,
                'error': f"Unexpected error: {str(e)}"
            })
    
    logger.info(f"Found {len(files_to_keep)} files to keep from annotations")
    
    if invalid_entries:
        logger.warning(f"Found {len(invalid_entries)} invalid entries in JSON")
        for invalid in invalid_entries[:3]:  # Show first 3
            logger.warning(f"  - Entry {invalid['entry_index']}: {invalid['error']}")
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # List all objects in the folder
    try:
        logger.info(f"Listing all objects in {s3_folder_path}...")
        
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=folder_prefix
        )
        
        all_objects = []
        for page in page_iterator:
            if 'Contents' in page:
                all_objects.extend(page['Contents'])
        
        logger.info(f"Found {len(all_objects)} total objects in folder")
        
    except ClientError as e:
        logger.error(f"Failed to list objects in bucket {bucket_name}: {e}")
        return
    
    # Find objects to delete (not in keep list)
    objects_to_delete = []
    
    for obj in all_objects:
        key = obj['Key']
        # Skip if it's a folder (ends with /)
        if key.endswith('/'):
            continue
            
        if key not in files_to_keep:
            objects_to_delete.append({
                'key': key,
                'size': obj['Size'],
                'last_modified': obj['LastModified']
            })
    
    # Display summary
    print("\n" + "="*70)
    print("S3 FOLDER CLEANUP SUMMARY")
    print("="*70)
    print(f"S3 Folder: {s3_folder_path}")
    print(f"Annotated JSON: {annotated_json_path}")
    print(f"Total objects in folder: {len(all_objects)}")
    print(f"Annotated files to KEEP: {len(files_to_keep)}")
    print(f"Objects to DELETE: {len(objects_to_delete)}")
    
    if len(objects_to_delete) == 0:
        print("\n✅ No objects to delete. All files in folder are annotated.")
        return
    
    # Show sample of files to be deleted
    print(f"\nSample files to be deleted (showing up to 10):")
    for i, obj in enumerate(objects_to_delete[:10]):
        size_mb = obj['size'] / (1024 * 1024)
        print(f"  {i+1}. {obj['key']} ({size_mb:.2f} MB)")
    
    if len(objects_to_delete) > 10:
        print(f"  ... and {len(objects_to_delete) - 10} more files")
    
    # Calculate total size
    total_size_bytes = sum(obj['size'] for obj in objects_to_delete)
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\nTotal size to delete: {total_size_mb:.2f} MB")
    
    # Ask for confirmation
    print("\n" + "="*70)
    print("⚠️  WARNING: This will PERMANENTLY DELETE the objects listed above!")
    print("Only annotated files will be preserved.")
    print("="*70)
    
    confirmation = input(f"\nDo you want to proceed with deleting {len(objects_to_delete)} objects? (yes/no): ").strip().lower()
    
    if confirmation not in ['yes', 'y']:
        print("Deletion cancelled by user.")
        return
    
    # Proceed with deletion
    successful_deletions = 0
    failed_deletions = []
    
    print(f"\nStarting deletion of {len(objects_to_delete)} objects...")
    
    # Delete objects in batches (S3 allows up to 1000 objects per batch)
    batch_size = 1000
    
    for i in range(0, len(objects_to_delete), batch_size):
        batch = objects_to_delete[i:i + batch_size]
        
        # Prepare batch delete request
        delete_request = {
            'Objects': [{'Key': obj['key']} for obj in batch]
        }
        
        try:
            response = s3.delete_objects(
                Bucket=bucket_name,
                Delete=delete_request
            )
            
            # Process successful deletions
            if 'Deleted' in response:
                batch_success = len(response['Deleted'])
                successful_deletions += batch_success
                logger.info(f"Deleted batch: {batch_success} objects (Total: {successful_deletions}/{len(objects_to_delete)})")
            
            # Process errors
            if 'Errors' in response:
                for error in response['Errors']:
                    failed_deletions.append({
                        'key': error['Key'],
                        'error': f"{error['Code']}: {error['Message']}"
                    })
                    logger.error(f"Failed to delete {error['Key']}: {error['Code']} - {error['Message']}")
            
        except ClientError as e:
            # If batch delete fails, try individual deletions
            logger.warning(f"Batch delete failed, trying individual deletions: {e}")
            
            for obj in batch:
                try:
                    s3.delete_object(Bucket=bucket_name, Key=obj['key'])
                    successful_deletions += 1
                    logger.info(f"Deleted: {obj['key']} ({successful_deletions}/{len(objects_to_delete)})")
                except ClientError as individual_error:
                    failed_deletions.append({
                        'key': obj['key'],
                        'error': str(individual_error)
                    })
                    logger.error(f"Failed to delete {obj['key']}: {individual_error}")
    
    # Print final summary
    print("\n" + "="*70)
    print("DELETION SUMMARY")
    print("="*70)
    print(f"Total objects processed: {len(objects_to_delete)}")
    print(f"Successfully deleted: {successful_deletions}")
    print(f"Failed deletions: {len(failed_deletions)}")
    print(f"Files preserved (annotated): {len(files_to_keep)}")
    
    if failed_deletions:
        print(f"\nFailed Deletions ({len(failed_deletions)}):")
        for fail in failed_deletions[:10]:  # Show first 10
            print(f"- {fail['key']}: {fail['error']}")
        if len(failed_deletions) > 10:
            print(f"... and {len(failed_deletions) - 10} more failures")
    
    print(f"\n✅ Cleanup completed. {successful_deletions} objects deleted, {len(files_to_keep)} annotated files preserved.")
    logger.info("Cleanup process completed")
    
    return successful_deletions, failed_deletions, len(files_to_keep)

def main():
    parser = argparse.ArgumentParser(
        description="Delete all unannotated files from an S3 folder, keeping only annotated ones",
        epilog="Example: python delete_unannotated_s3_objects.py s3://my-bucket/images/ labels/annotated.json"
    )
    parser.add_argument(
        's3_folder_path', 
        type=str, 
        help='S3 folder path (e.g., s3://bucket-name/folder/path/)'
    )
    parser.add_argument(
        'annotated_json_path', 
        type=str, 
        help='Path to JSON file containing annotated tasks (Label Studio format)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotated_json_path):
        print(f"Error: Annotated JSON file '{args.annotated_json_path}' not found")
        return 1
    
    delete_unannotated_objects(args.s3_folder_path, args.annotated_json_path)
    return 0

if __name__ == "__main__":
    exit(main())