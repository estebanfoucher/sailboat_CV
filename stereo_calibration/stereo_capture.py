#!/usr/bin/env python3
"""
Stereo Image Capture Tool

Captures simultaneous images from 2 camera streams using ffmpeg.
Saves images with timestamps in JSON format.
"""

import cv2
import numpy as np
import json
import yaml
import argparse
import os
import subprocess
import time
import threading
from datetime import datetime

class StereoImageCapture:
    """Simple simultaneous stereo image capture using ffmpeg."""
    
    def __init__(self, output_dir: str, config_path: str = "config.yml"):
        """Initialize with output directory and config."""
        self.output_dir = output_dir
        self.config = self._load_config(config_path)
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _capture_single_stream(self, stream_config, capture_id: str):
        """Capture a single image from a stream."""
        stream_name = stream_config['name']
        stream_url = stream_config['url']
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{capture_id}_{stream_name}_{timestamp}.jpg"
        output_path = os.path.join(self.output_dir, filename)
        
        # Build ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', stream_url,
            '-vframes', '1', '-q:v', '2',
            '-loglevel', 'error', output_path
        ]
        
        # Execute capture
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                return None
        except:
            return None
    
    def capture_stereo_images(self) -> bool:
        """Capture simultaneous images from both camera streams."""
        streams = self.config['streams']
        
        # Generate capture ID
        capture_id = f"capture_{int(time.time())}"
        
        print(f"Capturing stereo images...")
        
        # Capture from both streams using threads
        results = [None, None]
        
        def capture_thread(stream_idx, stream_config):
            results[stream_idx] = self._capture_single_stream(stream_config, capture_id)
        
        # Start threads
        threads = []
        for i, stream_config in enumerate(streams):
            thread = threading.Thread(target=capture_thread, args=(i, stream_config))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=15)
        
        # Check results
        successful = 0
        for i, image_path in enumerate(results):
            if image_path and os.path.exists(image_path):
                print(f"✓ Captured {streams[i]['name']}: {os.path.basename(image_path)}")
                successful += 1
            else:
                print(f"✗ Failed to capture {streams[i]['name']}")
        
        print(f"Capture completed: {successful}/2 successful")
        return successful == 2
    
    def get_latest_captures(self, count: int = 1):
        """Get the latest capture files."""
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
        files.sort()
        return [os.path.join(self.output_dir, f) for f in files[-count*2:]]
    
    def cleanup(self):
        """Clean up any remaining ffmpeg processes."""
        try:
            subprocess.run(['pkill', '-f', 'ffmpeg'], capture_output=True, timeout=3)
        except:
            pass
