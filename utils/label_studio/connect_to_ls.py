import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(".env")

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'
# API key is available at the Account & Settings > Access Tokens page in Label Studio UI
API_KEY = os.getenv('LABEL_STUDIO_API_KEY')


# Import the SDK and the client module
from label_studio_sdk.client import LabelStudio
# Connect to the Label Studio API and check the connection
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)