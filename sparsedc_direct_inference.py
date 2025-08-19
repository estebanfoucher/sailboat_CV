#!/usr/bin/env python3
"""
Minimal SparseDC inference script that bypasses PyTorch Lightning framework.
Directly loads the model and runs inference without the evaluation pipeline.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image

# Add SparseDC to path
sys.path.insert(0, 'SparseDC')

# Add depth_estimator to path for triangulation
sys.path.insert(0, 'depth_estimator')

from src.models.base.uncertainty import Uncertainty_
from src.models.backbones import PVTV2, ResNetU_
from src.models.decodes.uncertainty import UncertaintyFuse_
from src.criterion.loss import DepthLoss

# Import triangulation components
from depth_estimator.match_loader import MatchLoader
from depth_estimator.triangulation import StereoCalibration, Triangulator


class SparseDCDirectInference:
    def __init__(self, model_path, resolution=(720, 1280)):
        self.model_path = model_path
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing SparseDC Direct Inference")
        print(f"   Model: {model_path}")
        print(f"   Resolution: {resolution}")
        print(f"   Device: {self.device}")
        
        self.model = self._load_model()
    
    def _load_model(self):
        """Load SparseDC model directly from checkpoint."""
        print(" Loading model from checkpoint...")
        
        # Load checkpoint
        if self.model_path.endswith('.ckpt'):
            # PyTorch Lightning checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'net.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('net.'):
                        new_state_dict[k[4:]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            else:
                state_dict = checkpoint
        else:
            # Raw PyTorch state dict
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Filter out refiner-related weights to avoid DCN dependency
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not any(refiner_key in k for refiner_key in ['refiner', 'guide_layer']):
                filtered_state_dict[k] = v
        
        print(f"   Loaded {len(state_dict)} keys, filtered to {len(filtered_state_dict)} keys (removed refiner)")
        
        # Create model architecture
        model = self._create_model_architecture()
        
        # Load state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        print("✅ Model loaded successfully")
        return model
    
    def _create_model_architecture(self):
        """Create the SparseDC model architecture."""
        # Backbone components
        backbone_g = PVTV2(
            model_name='pvt_v2_b1',
            pretrained=None,
            is_fill=True
        )
        
        backbone_l = ResNetU_(
            model_name='resnet18',
            is_fill=True
        )
        
        # Decoder
        decode = UncertaintyFuse_(
            max_depth=10.0,
            bot_channel=64,
            g_in_channels=[512, 320, 128, 64],
            l_in_channels=[512, 512, 256, 128, 64],
            fuse_channel=128,
            is_gate_fuse=True
        )
        
        # Main model
        model = Uncertainty_(
            backbone_l=backbone_l,
            backbone_g=backbone_g,
            decode=decode,
            refiner=None,  # Disable refiner to avoid DCN dependency
            criterion=None,  # Not needed for inference
            is_padding=False,  # No padding for custom resolution
            padding_size=self.resolution,
            max_depth=10.0,
            channels=64,
            is_fill=True
        )
        
        return model
    
    def create_sparse_depth_from_matches(self, image_path, matches_path, calibration_path, confidence_threshold=0.0):
        """Create sparse depth map from SuperGlue matches and triangulation."""
        print("🔄 Creating sparse depth from SuperGlue matches...")
        
        # Load matches
        match_loader = MatchLoader(matches_path)
        matched_pairs = match_loader.get_matched_pairs(confidence_threshold)
        
        # Load calibration
        calibration = StereoCalibration(calibration_path)
        
        # Create triangulator
        triangulator = Triangulator(calibration)
        
        # Load image to get dimensions
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        # Create sparse depth map
        sparse_depth = np.zeros((image_height, image_width), dtype=np.float32)
        
        # Triangulate matches
        if matched_pairs is not None and len(matched_pairs) > 0:
            print(f"   Found {len(matched_pairs)} matches")
            
            # Triangulate all matches at once
            try:
                points_3d = triangulator.triangulate_from_matches(matched_pairs)
                
                if len(points_3d) > 0:
                    triangulated_points = 0
                    for i, (point1_2d, point2_2d) in enumerate(matched_pairs):
                        try:
                            # Get 3D point
                            point_3d_coords = points_3d[i]
                            
                            # Use original camera 1 coordinates (no need to reproject)
                            x, y = int(point1_2d[0]), int(point1_2d[1])
                            
                            # Check bounds
                            if 0 <= x < image_width and 0 <= y < image_height:
                                # Store depth value (Z coordinate)
                                depth_value = point_3d_coords[2]
                                if depth_value > 0:  # Valid depth
                                    sparse_depth[y, x] = depth_value
                                    triangulated_points += 1
                        
                        except Exception as e:
                            print(f"   Warning: Failed to process triangulated point {i}: {e}")
                            continue
                    
                    print(f"   Successfully triangulated {triangulated_points} 3D points")
                else:
                    print("   No valid 3D points from triangulation")
            
            except Exception as e:
                print(f"   Warning: Failed to triangulate matches: {e}")
        else:
            print("   No matches found")
        
        return sparse_depth
    
    def preprocess_input(self, image, sparse_depth):
        """Preprocess input for SparseDC model."""
        # Convert image to tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            # Convert PIL image to numpy then tensor
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).float()
        
        # Convert sparse depth to tensor
        if isinstance(sparse_depth, np.ndarray):
            depth_tensor = torch.from_numpy(sparse_depth).float()
        else:
            depth_tensor = torch.from_numpy(sparse_depth).float()
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        # Ensure correct shape [B, C, H, W]
        if image_tensor.shape[1] != 3:
            image_tensor = image_tensor.permute(0, 3, 1, 2)
        
        # Normalize image to [0, 1] if needed
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Resize to model's expected resolution
        import torch.nn.functional as F
        target_height, target_width = self.resolution
        
        # Resize image
        image_tensor = F.interpolate(image_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
        
        # Resize depth (use nearest neighbor to preserve sparse values)
        depth_tensor = F.interpolate(depth_tensor, size=(target_height, target_width), mode='nearest')
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        depth_tensor = depth_tensor.to(self.device)
        
        return image_tensor, depth_tensor
    
    def infer(self, image_path, matches_path, calibration_path, output_dir, confidence_threshold=0.0):
        """Run inference on image + matches + calibration."""
        print("🔄 Running complete inference pipeline...")
        
        # Create sparse depth from matches
        sparse_depth = self.create_sparse_depth_from_matches(
            image_path, matches_path, calibration_path, confidence_threshold
        )
        
        # Load image
        image = Image.open(image_path)
        
        # Preprocess inputs
        image_tensor, depth_tensor = self.preprocess_input(image, sparse_depth)
        
        # Create input dictionary
        sample = {
            "rgb": image_tensor,
            "dep": depth_tensor,
            "gt": depth_tensor  # Dummy ground truth (not used in eval mode)
        }
        
        # Save intermediate results for debugging
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save sparse depth
        np.save(os.path.join(debug_dir, "sparse_depth.npy"), sparse_depth)
        print(f"💾 Saved sparse depth to: {debug_dir}/sparse_depth.npy")
        
        # Save image tensor info
        print(f"   Image tensor shape: {image_tensor.shape}")
        print(f"   Depth tensor shape: {depth_tensor.shape}")
        
        # Run inference
        try:
            with torch.no_grad():
                prediction = self.model(sample)
            
            # Convert back to numpy
            if isinstance(prediction, tuple):
                prediction = prediction[0]  # Get first element if tuple
            
            prediction_np = prediction.cpu().numpy()
            
            print(f"✅ Inference completed. Output shape: {prediction_np.shape}")
            return prediction_np, sparse_depth
            
        except Exception as e:
            print(f"❌ Model inference failed: {e}")
            print("💾 Saving debug info...")
            
            # Save input tensors for debugging
            torch.save(image_tensor, os.path.join(debug_dir, "image_tensor.pt"))
            torch.save(depth_tensor, os.path.join(debug_dir, "depth_tensor.pt"))
            torch.save(sample, os.path.join(debug_dir, "sample.pt"))
            
            print(f"💾 Saved debug tensors to: {debug_dir}/")
            raise


def main():
    parser = argparse.ArgumentParser(description="Direct SparseDC inference with SuperGlue pipeline")
    parser.add_argument("--image", required=True, help="Path to RGB image")
    parser.add_argument("--matches", required=True, help="Path to SuperGlue matches .npz file")
    parser.add_argument("--calibration", required=True, help="Path to stereo calibration JSON")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model", required=True, help="Path to SparseDC model (.pth or .ckpt file)")
    parser.add_argument("--resolution", default="720,1280", help="Resolution as height,width")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Confidence threshold for triangulation")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        height, width = map(int, args.resolution.split(','))
        resolution = (height, width)
    except ValueError:
        print("❌ Invalid resolution format. Use 'height,width' (e.g., '720,1280')")
        return 1
    
    # Check input files
    for file_path in [args.image, args.matches, args.calibration, args.model]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize inference
    print("🚀 Initializing SparseDC direct inference...")
    try:
        inference = SparseDCDirectInference(args.model, resolution)
    except Exception as e:
        print(f"❌ Failed to initialize inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run inference
    try:
        prediction, sparse_depth = inference.infer(
            args.image, args.matches, args.calibration, args.output, args.confidence_threshold
        )
        
        # Save results
        output_base = os.path.splitext(os.path.basename(args.image))[0]
        
        # Save prediction
        prediction_path = os.path.join(args.output, f"{output_base}_depth_prediction.npy")
        np.save(prediction_path, prediction)
        print(f"💾 Saved depth prediction to: {prediction_path}")
        
        # Save sparse depth
        sparse_depth_path = os.path.join(args.output, f"{output_base}_sparse_depth.npy")
        np.save(sparse_depth_path, sparse_depth)
        print(f" Saved sparse depth to: {sparse_depth_path}")
        
        # Save visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        image = Image.open(args.image)
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Sparse depth
        sparse_vis = axes[1].imshow(sparse_depth, cmap='viridis')
        axes[1].set_title("Sparse Depth")
        axes[1].axis('off')
        plt.colorbar(sparse_vis, ax=axes[1])
        
        # Predicted depth
        pred_vis = axes[2].imshow(prediction[0, 0], cmap='viridis')
        axes[2].set_title("Predicted Depth")
        axes[2].axis('off')
        plt.colorbar(pred_vis, ax=axes[2])
        
        plt.tight_layout()
        vis_path = os.path.join(args.output, f"{output_base}_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved visualization to: {vis_path}")
        
        print("✅ Inference pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
