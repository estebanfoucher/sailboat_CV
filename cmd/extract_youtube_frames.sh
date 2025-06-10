#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Execute the frame extractor
python ./utils/youtube_utils/frame_extract.py --file ./utils/youtube_utils/urls.txt 