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

# Add current directory to path (SparseDC modules)
sys.path.insert(0, '.')

# Import triangulation components from local copies
print("✅ Using local depth_estimator modules")

from src.models.base.uncertainty import Uncertainty_
from src.models.backbones import PVTV2, ResNetU_
from src.models.decodes.uncertainty import UncertaintyFuse_
from src.criterion.loss import DepthLoss

# Import triangulation components from local copies
try:
    import match_loader
    import triangulation
    MatchLoader = match_loader.MatchLoader
    StereoCalibration = triangulation.StereoCalibration
    Triangulator = triangulation.Triangulator
    print("✅ Successfully imported local depth_estimator modules")
except ImportError as e:
    print(f"❌ Failed to import local depth_estimator modules: {e}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Available files: {os.listdir('.')}")
    sys.exit(1)


class SparseDCDirectInference:
    def __init__(self, model_path, resolution=(720, 1280)):
        self.model_path = model_path
        self.resolution = resolution
        # Force GPU if available, otherwise use CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"   Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("   Using CPU (CUDA not available)")
        
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
        """Preprocess input for SparseDC model with aspect ratio preservation."""
        # Convert image to numpy array if needed
        if isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = np.array(image)
        
        original_h, original_w = image_array.shape[:2]
        
        # Calculate scaling and padding to maintain aspect ratio
        target_h, target_w = self.resolution
        scale_h = target_h / original_h
        scale_w = target_w / original_w
        scale = min(scale_h, scale_w)  # Use smaller scale to fit within target
        
        # Calculate new dimensions after scaling
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)
        
        # Calculate padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        print(f"🔧 Resize transform: {original_h}x{original_w} -> {new_h}x{new_w} -> {target_h}x{target_w}")
        print(f"   Scale: {scale:.4f}, Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        
        # Convert image to tensor and normalize
        image_tensor = torch.from_numpy(image_array).float()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Resize image maintaining aspect ratio
        import torch.nn.functional as F
        image_tensor = F.interpolate(image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # Add padding
        image_tensor = F.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Transform sparse depth coordinates
        depth_tensor = self._transform_sparse_depth(sparse_depth, original_h, original_w, 
                                                   new_h, new_w, pad_top, pad_left, scale)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        depth_tensor = depth_tensor.to(self.device)
        
        return image_tensor, depth_tensor
    
    def _transform_sparse_depth(self, sparse_depth, orig_h, orig_w, new_h, new_w, pad_top, pad_left, scale):
        """Transform sparse depth coordinates to match resized image"""
        # Create new sparse depth map at target resolution
        target_h, target_w = self.resolution
        new_sparse = np.zeros((target_h, target_w), dtype=np.float32)
        
        # Find non-zero coordinates in original sparse depth
        if np.count_nonzero(sparse_depth) > 0:
            orig_coords = np.where(sparse_depth > 0)
            orig_y, orig_x = orig_coords
            orig_values = sparse_depth[orig_coords]
            
            # Transform coordinates
            # Scale coordinates
            new_y = (orig_y * scale).astype(int)
            new_x = (orig_x * scale).astype(int)
            
            # Add padding offset
            new_y = new_y + pad_top
            new_x = new_x + pad_left
            
            # Filter coordinates that are within bounds
            valid_mask = (new_y >= 0) & (new_y < target_h) & (new_x >= 0) & (new_x < target_w)
            
            if np.any(valid_mask):
                new_sparse[new_y[valid_mask], new_x[valid_mask]] = orig_values[valid_mask]
                print(f"   Transformed {np.sum(valid_mask)}/{len(orig_values)} sparse depth points")
            else:
                print("   ⚠️  No sparse depth points survived coordinate transformation")
        
        # Convert to tensor
        sparse_tensor = torch.from_numpy(new_sparse).unsqueeze(0).unsqueeze(0).float()
        return sparse_tensor
    
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
            
            # Also return the transformed sparse depth for visualization
            sparse_depth_transformed = depth_tensor[0, 0].cpu().numpy()
            return prediction_np, sparse_depth, sparse_depth_transformed
            
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
        prediction, sparse_depth, sparse_depth_transformed = inference.infer(
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
        print(f"💾 Saved sparse depth to: {sparse_depth_path}")
        
        # Debug: Check sparse depth data
        print(f"🔍 Debug - Original sparse depth shape: {sparse_depth.shape}")
        print(f"🔍 Debug - Original non-zero values: {np.count_nonzero(sparse_depth)}")
        if np.count_nonzero(sparse_depth) > 0:
            non_zero_coords = np.where(sparse_depth > 0)
            print(f"🔍 Debug - Original coordinates (first 5): {list(zip(non_zero_coords[0][:5], non_zero_coords[1][:5]))}")
            print(f"🔍 Debug - Original values (first 5): {sparse_depth[sparse_depth > 0][:5]}")
        else:
            print("🔍 Debug - WARNING: No non-zero values in original sparse depth!")
        
        # Save visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        image = Image.open(args.image)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Sparse depth (original resolution) - show as scatter plot for better visibility
        axes[0, 1].imshow(np.zeros_like(sparse_depth), cmap='gray', alpha=0.3)  # Background
        if np.count_nonzero(sparse_depth) > 0:
            non_zero_coords = np.where(sparse_depth > 0)
            scatter = axes[0, 1].scatter(non_zero_coords[1], non_zero_coords[0], 
                                       c=sparse_depth[sparse_depth > 0], 
                                       cmap='viridis', s=50, alpha=0.8)
            plt.colorbar(scatter, ax=axes[0, 1])
        axes[0, 1].set_title(f"Sparse Depth (Original: {sparse_depth.shape}) - {np.count_nonzero(sparse_depth)} points")
        axes[0, 1].axis('off')
        
        # Predicted depth (model resolution)
        pred_vis = axes[1, 0].imshow(prediction[0, 0], cmap='viridis', vmin=0, vmax=10)
        axes[1, 0].set_title(f"Predicted Depth (Model: {prediction[0, 0].shape})")
        axes[1, 0].axis('off')
        plt.colorbar(pred_vis, ax=axes[1, 0])
        
        # Use the properly transformed sparse depth from the inference process
        sparse_resized = sparse_depth_transformed
        
        # Debug: Check transformed sparse depth data
        print(f"🔍 Debug - Transformed sparse depth shape: {sparse_resized.shape}")
        print(f"🔍 Debug - Transformed non-zero values: {np.count_nonzero(sparse_resized)}")
        if np.count_nonzero(sparse_resized) > 0:
            non_zero_coords = np.where(sparse_resized > 0)
            print(f"🔍 Debug - Transformed coordinates (first 5): {list(zip(non_zero_coords[0][:5], non_zero_coords[1][:5]))}")
            print(f"🔍 Debug - Transformed values (first 5): {sparse_resized[sparse_resized > 0][:5]}")
        else:
            print("🔍 Debug - WARNING: No non-zero values in transformed sparse depth!")
        
        # Resized sparse depth - show as scatter plot for better visibility
        axes[1, 1].imshow(np.zeros_like(sparse_resized), cmap='gray', alpha=0.3)  # Background
        if np.count_nonzero(sparse_resized) > 0:
            non_zero_coords = np.where(sparse_resized > 0)
            scatter = axes[1, 1].scatter(non_zero_coords[1], non_zero_coords[0], 
                                       c=sparse_resized[sparse_resized > 0], 
                                       cmap='viridis', s=30, alpha=0.8)
            plt.colorbar(scatter, ax=axes[1, 1])
        axes[1, 1].set_title(f"Sparse Depth (Resized: {sparse_resized.shape}) - {np.count_nonzero(sparse_resized)} points")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(args.output, f"{output_base}_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved visualization to: {vis_path}")
        
        # Print statistics
        print(f"📊 Statistics:")
        print(f"   Original sparse depth: {np.count_nonzero(sparse_depth)} non-zero values, range [{sparse_depth.min():.3f}, {sparse_depth.max():.3f}]")
        print(f"   Transformed sparse depth: {np.count_nonzero(sparse_resized)} non-zero values, range [{sparse_resized.min():.3f}, {sparse_resized.max():.3f}]")
        print(f"   Prediction: range [{prediction[0, 0].min():.3f}, {prediction[0, 0].max():.3f}], mean {prediction[0, 0].mean():.3f}")
        print(f"   Point preservation: {np.count_nonzero(sparse_resized)}/{np.count_nonzero(sparse_depth)} points ({100*np.count_nonzero(sparse_resized)/max(1,np.count_nonzero(sparse_depth)):.1f}%)")
        
        print("✅ Inference pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
