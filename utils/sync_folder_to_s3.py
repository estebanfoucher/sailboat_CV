import os
import sys
import subprocess
from dotenv import load_dotenv
import config  # loads .env

AWS_BUCKET_VERSION = os.getenv('AWS_BUCKET_VERSION')
if not AWS_BUCKET_VERSION:
    print('Error: AWS_BUCKET_VERSION not set in .env')
    sys.exit(1)

BUCKET_NAME = f"bucket-sail-{AWS_BUCKET_VERSION}"
LOCAL_DIR = "./data/datasets/pennon-label-yolo-03/images/"
S3_PATH = f"s3://{BUCKET_NAME}/frames/"

cmd = [
    "aws", "s3", "sync",
    LOCAL_DIR,
    S3_PATH,
    "--delete"
]

print(f"Running: {' '.join(cmd)}")
try:
    subprocess.run(cmd, check=True)
    print("Sync complete.")
except subprocess.CalledProcessError as e:
    print(f"Sync failed: {e}")
    sys.exit(1) 