#!/usr/bin/env python3
"""
Script to load and read SuperGlue NPZ files containing feature matches.
"""

import numpy as np
import os
import argparse

def load_superglue_matches(npz_path):
    """
    Load SuperGlue matches from NPZ file.
    
    Args:
        npz_path: Path to the NPZ file
        
    Returns:
        Dictionary containing the match data
    """
    print(f"📂 Loading SuperGlue matches from: {npz_path}")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load the NPZ file
    data = np.load(npz_path)
    
    # Extract the data
    matches_data = {
        'keypoints0': data['keypoints0'],  # Keypoints from first image
        'keypoints1': data['keypoints1'],  # Keypoints from second image
        'matches': data['matches'],        # Match indices (into keypoints1)
        'match_confidence': data['match_confidence']  # Confidence scores
    }
    
    return matches_data

def print_basic_info(matches_data):
    """Print basic information about the matches."""
    keypoints0 = matches_data['keypoints0']
    keypoints1 = matches_data['keypoints1']
    matches = matches_data['matches']
    confidences = matches_data['match_confidence']
    
    # Find valid matches (where match index > -1)
    valid_matches = matches > -1
    num_valid_matches = np.sum(valid_matches)
    
    print(f"\n📊 Basic Match Information:")
    print(f"   Keypoints in image 0: {len(keypoints0)}")
    print(f"   Keypoints in image 1: {len(keypoints1)}")
    print(f"   Total match attempts: {len(matches)}")
    print(f"   Valid matches: {num_valid_matches}")
    print(f"   Match ratio: {num_valid_matches / len(matches):.3f} ({num_valid_matches / len(matches) * 100:.1f}%)")
    
    if num_valid_matches > 0:
        # Get matched keypoints
        matched_kpts0 = keypoints0[valid_matches]
        matched_kpts1 = keypoints1[matches[valid_matches]]
        matched_confidences = confidences[valid_matches]
        
        # Calculate match distances
        distances = np.linalg.norm(matched_kpts0 - matched_kpts1, axis=1)
        
        print(f"\n🎯 Match Statistics:")
        print(f"   Mean confidence: {np.mean(matched_confidences):.4f}")
        print(f"   Min confidence: {np.min(matched_confidences):.4f}")
        print(f"   Max confidence: {np.max(matched_confidences):.4f}")
        print(f"   Mean distance: {np.mean(distances):.2f} pixels")
        print(f"   Min distance: {np.min(distances):.2f} pixels")
        print(f"   Max distance: {np.max(distances):.2f} pixels")
        
        print(f"\n🔗 Sample Matches (first 10):")
        for i in range(min(10, len(matched_kpts0))):
            kpt0 = matched_kpts0[i]
            kpt1 = matched_kpts1[i]
            conf = matched_confidences[i]
            dist = distances[i]
            print(f"   Match {i+1}: ({kpt0[0]:.1f}, {kpt0[1]:.1f}) -> ({kpt1[0]:.1f}, {kpt1[1]:.1f}) "
                  f"[conf: {conf:.3f}, dist: {dist:.1f}px]")
    else:
        print(f"\n⚠️  No valid matches found!")

def main():
    """Main function to read and display NPZ file contents."""
    parser = argparse.ArgumentParser(description='Read SuperGlue NPZ match files')
    parser.add_argument('--npz_path', type=str, 
                       default='outputs/capture_1754654952_camera1_20250808_140912_350_capture_1754654952_camera2_20250808_140912_351_matches.npz',
                       help='Path to the NPZ file')
    
    args = parser.parse_args()
    
    try:
        # Load the matches
        matches_data = load_superglue_matches(args.npz_path)
        
        # Print basic information
        print_basic_info(matches_data)
        
        print(f"\n✅ NPZ file read successfully!")
        
    except Exception as e:
        print(f"❌ Error reading NPZ file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
