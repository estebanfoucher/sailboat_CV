import os
import subprocess
from dotenv import load_dotenv
from loguru import logger
# Load environment variables from .env file
load_dotenv(".env")

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'localhost:8080'
# API key is available at the Account & Settings > Access Tokens page in Label Studio UI
API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

project_id = input("Enter the project id to delete: ")
confirm = input(f"Are you sure you want to delete this project with id: {project_id}? (y/n): ")
if confirm.lower() == 'y':
    cmd = [
        "curl", "-X", "DELETE",
        f"http://{LABEL_STUDIO_URL}/api/projects/{project_id}/",
        "-H", f"Authorization: Token {API_KEY}"
    ]
    subprocess.run(cmd)
    print(f"Project {project_id} deleted.")
else:
    print("Deletion cancelled.")