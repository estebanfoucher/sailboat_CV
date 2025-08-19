#!/usr/bin/env python3
"""
Wrapper script to run SparseDC inference from the root directory.
This script changes to the SparseDC directory and runs the inference there.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run SparseDC inference with SuperGlue + triangulation')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--matches', required=True, help='Path to SuperGlue matches .npz file')
    parser.add_argument('--calibration', required=True, help='Path to stereo calibration JSON file')
    parser.add_argument('--output', default='SparseDC/sparsedc_results', help='Output directory')
    parser.add_argument('--model', required=True, help='Path to SparseDC model (.ckpt or .pth)')
    parser.add_argument('--resolution', default='512,1024', help='Model input resolution (H,W)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0, help='Triangulation confidence threshold')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        resolution = tuple(map(int, args.resolution.split(',')))
        if len(resolution) != 2:
            raise ValueError("Resolution must be in format H,W")
    except ValueError as e:
        print(f"❌ Invalid resolution format: {e}")
        return 1
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Change to SparseDC directory
    original_dir = os.getcwd()
    sparsedc_dir = os.path.join(original_dir, 'SparseDC')
    
    if not os.path.exists(sparsedc_dir):
        print(f"❌ SparseDC directory not found: {sparsedc_dir}")
        return 1
    
    try:
        os.chdir(sparsedc_dir)
        print(f"📁 Changed to directory: {sparsedc_dir}")
        
        # Build the command
        cmd = [
            sys.executable, 'sparsedc_direct_inference.py',
            '--image', os.path.join(original_dir, args.image),
            '--matches', os.path.join(original_dir, args.matches),
            '--calibration', os.path.join(original_dir, args.calibration),
            '--output', os.path.join(original_dir, args.output),
            '--model', os.path.join(original_dir, args.model),
            '--resolution', args.resolution,
            '--confidence-threshold', str(args.confidence_threshold)
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        # Set environment variable for depth_estimator path
        env = os.environ.copy()
        env['DEPTH_ESTIMATOR_PATH'] = os.path.join(original_dir, 'depth_estimator')
        
        # Run the inference
        result = subprocess.run(cmd, check=True, env=env)
        
        print("✅ Inference completed successfully!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Change back to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())
