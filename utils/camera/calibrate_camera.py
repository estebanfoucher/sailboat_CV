#!/usr/bin/env python3
"""
Camera calibration script using OpenCV's calibrateCamera method.
Takes a folder of checkerboard images and checkerboard specs to compute camera intrinsics.
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import argparse
from loguru import logger


def load_checkerboard_specs(specs_file: str) -> Dict:
    """Load checkerboard specifications from JSON or YAML file."""
    specs_path = Path(specs_file)
    if specs_path.suffix.lower() in ['.json']:
        with open(specs_path, 'r') as f:
            return json.load(f)
    elif specs_path.suffix.lower() in ['.yml', '.yaml']:
        with open(specs_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {specs_path.suffix}")


def find_corners_in_images(images_folder: str, pattern_size: Tuple[int, int], 
                          square_size: float) -> Tuple[List, List, List]:
    """Find checkerboard corners in all images in the folder."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    successful_images = []
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    images_path = Path(images_folder)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)
            image_points.append(corners2)
            successful_images.append(img_file.name)
            logger.info(f"Found corners in {img_file.name}")
        else:
            logger.warning(f"No corners found in {img_file.name}")
    
    logger.info(f"Successfully processed {len(successful_images)}/{len(image_files)} images")
    return object_points, image_points, successful_images


def calibrate_camera(object_points: List, image_points: List, 
                    image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run camera calibration using OpenCV's calibrateCamera."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], 
                                        camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    logger.info(f"Calibration error: {mean_error/len(object_points):.4f} pixels")
    return camera_matrix, dist_coeffs, mean_error/len(object_points)


def save_calibration_results(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                           error: float, output_file: str):
    """Save calibration results to JSON file."""
    results = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "calibration_error": float(error),
        "image_width": int(camera_matrix[0, 2] * 2),
        "image_height": int(camera_matrix[1, 2] * 2)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Calibration results saved to {output_file}")


def test_calibration(image_path: str, camera_matrix: np.ndarray, 
                    dist_coeffs: np.ndarray, output_path: str):
    """Test calibration by undistorting a sample image."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop the image
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    
    cv2.imwrite(output_path, undistorted)
    logger.info(f"Undistorted test image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Camera calibration using checkerboard")
    parser.add_argument("images_folder", help="Folder containing checkerboard images")
    parser.add_argument("specs_file", help="JSON/YAML file with checkerboard specifications")
    parser.add_argument("--output", default="calibration_results.json", 
                       help="Output file for calibration results")
    parser.add_argument("--test-image", help="Path to test image for undistortion")
    
    args = parser.parse_args()
    
    # Load specifications
    specs = load_checkerboard_specs(args.specs_file)
    pattern_size = (specs['inner_corners_x'], specs['inner_corners_y'])
    square_size = specs['square_size_mm']
    
    # Find corners
    object_points, image_points, successful_images = find_corners_in_images(
        args.images_folder, pattern_size, square_size
    )
    
    if len(object_points) < 5:
        logger.error("Need at least 5 successful images for calibration")
        return
    
    # Get image size from first successful image
    first_image = Path(args.images_folder) / successful_images[0]
    img = cv2.imread(str(first_image))
    image_size = (img.shape[1], img.shape[0])
    
    # Calibrate camera
    camera_matrix, dist_coeffs, error = calibrate_camera(
        object_points, image_points, image_size
    )
    
    # Save results
    save_calibration_results(camera_matrix, dist_coeffs, error, args.output)
    
    # Test calibration if test image provided
    if args.test_image:
        test_output = Path(args.output).with_suffix('_undistorted.jpg')
        test_calibration(args.test_image, camera_matrix, dist_coeffs, str(test_output))


if __name__ == "__main__":
    main() 