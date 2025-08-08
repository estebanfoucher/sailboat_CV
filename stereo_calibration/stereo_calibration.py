import os
import yaml
import cv2
import shutil
import subprocess
from typing import List, Tuple, Dict, Any

try:
    from .stereo_capture import StereoImageCapture
    from .stereo_detection import StereoTagDetector
    from .utils import calibrate_stereo_many, print_summary, save_results
except ImportError:
    # When running as standalone script
    from stereo_capture import StereoImageCapture
    from stereo_detection import StereoTagDetector
    from utils import calibrate_stereo_many, print_summary, save_results


class StereoCalibration:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.output_dir = self.config['output']['directory']
        os.makedirs(self.output_dir, exist_ok=True)

        # Raw capture area where ffmpeg saves frames; we move them into per-pair dirs
        self.raw_dir = os.path.join(self.output_dir, 'raw_captures')
        os.makedirs(self.raw_dir, exist_ok=True)

        self.stream_names = [self.config['streams'][0]['name'], self.config['streams'][1]['name']]

        # Optional intrinsics paths (required for calibration)
        calib_cfg = self.config.get('calibration', {})
        self.intrinsics1_path = calib_cfg.get('intrinsics1')
        self.intrinsics2_path = calib_cfg.get('intrinsics2')

        self.detector = StereoTagDetector(config_path=config_path)
        # Don't create capture instance here - create fresh one each time

        # Start index from existing pairs
        self.collected_image_pairs_number = self._count_existing_pairs()

    def _count_existing_pairs(self) -> int:
        count = 0
        for name in os.listdir(self.output_dir):
            if name.startswith('pair_') and os.path.isdir(os.path.join(self.output_dir, name)):
                try:
                    idx = int(name.split('_')[1])
                    count = max(count, idx + 1)
                except Exception:
                    pass
        return count

    def _latest_captured_paths(self, capture_instance) -> List[str]:
        latest = capture_instance.get_latest_captures(1)
        if not latest:
            return []
        streams = latest[0].get('streams', [])
        paths = [s.get('image_path') for s in streams if s.get('success')]
        return [p for p in paths if p]

    def _store_pair(self, img_paths: List[str], pair_index: int) -> str:
        pair_dir = os.path.join(self.output_dir, f'pair_{pair_index}')
        os.makedirs(pair_dir, exist_ok=True)
        for p in img_paths:
            if os.path.exists(p):
                shutil.move(p, os.path.join(pair_dir, os.path.basename(p)))
                
        return pair_dir

    def _list_pair_dirs(self) -> List[str]:
        dirs = []
        for name in sorted(os.listdir(self.output_dir)):
            path = os.path.join(self.output_dir, name)
            if name.startswith('pair_') and os.path.isdir(path):
                dirs.append(path)
        return dirs

    def _load_pair_images(self, pair_dir: str) -> Tuple[Any, Any]:
        files = os.listdir(pair_dir)
        img1_path = None
        img2_path = None
        # Identify by stream names present in filenames
        for fn in files:
            if self.stream_names[0] in fn:
                img1_path = os.path.join(pair_dir, fn)
            elif self.stream_names[1] in fn:
                img2_path = os.path.join(pair_dir, fn)

        if not img1_path or not img2_path:
            # fallback: pick two images in alphabetical order
            jpgs = [os.path.join(pair_dir, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            jpgs.sort()
            if len(jpgs) >= 2:
                img1_path, img2_path = jpgs[:2]

        img1 = cv2.imread(img1_path) if img1_path else None
        img2 = cv2.imread(img2_path) if img2_path else None
        return img1, img2

    def _compute_calibration(self) -> Dict[str, Any]:
        if not self.intrinsics1_path or not self.intrinsics2_path:
            raise ValueError("Missing intrinsics paths. Please set 'calibration.intrinsics1' and 'calibration.intrinsics2' in config.yml")

        pair_dirs = self._list_pair_dirs()
        if not pair_dirs:
            raise ValueError("No pairs found to calibrate.")

        object_points_list = []
        image_points1_list = []
        image_points2_list = []
        image_size = None

        for pair_dir in pair_dirs:
            img1, img2 = self._load_pair_images(pair_dir)
            if img1 is None or img2 is None:
                print(f"Skipping {pair_dir}: could not load both images")
                continue

            if image_size is None:
                image_size = (img1.shape[1], img1.shape[0])

            try:
                p3d, p2d1, p2d2 = self.detector.get_correspondences(img1, img2)
                object_points_list.append(p3d)
                image_points1_list.append(p2d1)
                image_points2_list.append(p2d2)
            except Exception as e:
                print(f"Skipping {pair_dir}: {e}")

        if not object_points_list:
            raise ValueError("No valid correspondences found across pairs.")

        results = calibrate_stereo_many(
            object_points_list,
            image_points1_list,
            image_points2_list,
            self.intrinsics1_path,
            self.intrinsics2_path,
            image_size
        )
        return results

    def _detect_stream_features(self):
        """Detect stream resolution and FPS for both cameras."""
        print("\nDetecting stream features...")
        
        for i, stream in enumerate(self.config['streams']):
            stream_name = stream['name']
            stream_url = stream['url']
            
            print(f"\n{stream_name}:")
            print(f"  URL: {stream_url}")
            
            try:
                # Use ffprobe to get stream information
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', '-select_streams', 'v:0', stream_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    
                    if 'streams' in data and len(data['streams']) > 0:
                        stream_info = data['streams'][0]
                        
                        # Extract resolution
                        width = stream_info.get('width', 'Unknown')
                        height = stream_info.get('height', 'Unknown')
                        print(f"  Resolution: {width}x{height}")
                        
                        # Extract FPS
                        fps = stream_info.get('r_frame_rate', 'Unknown')
                        if fps != 'Unknown':
                            try:
                                # Parse fraction like "30/1"
                                if '/' in fps:
                                    num, den = fps.split('/')
                                    fps = float(num) / float(den)
                                else:
                                    fps = float(fps)
                                print(f"  FPS: {fps:.2f}")
                            except:
                                print(f"  FPS: {fps}")
                        else:
                            print(f"  FPS: {fps}")
                            
                        # Extract codec
                        codec = stream_info.get('codec_name', 'Unknown')
                        print(f"  Codec: {codec}")
                    else:
                        print("  Error: No video stream found")
                else:
                    print(f"  Error: ffprobe failed (return code: {result.returncode})")
                    if result.stderr:
                        print(f"  stderr: {result.stderr.strip()}")
                        
            except subprocess.TimeoutExpired:
                print("  Error: Timeout while detecting stream features")
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\nStream feature detection complete.")

    def run(self):
        print("StereoCalibration interactive session")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {self.collected_image_pairs_number} existing pairs")

        # Detect stream features at startup
        self._detect_stream_features()

        # Main options loop
        while True:
            print("\nChoose an option:")
            print("  1) Capture a new pair (then press Enter in the capture prompt)")
            print("  2) Compute the calibration upon all accumulated pairs and show result in terminal + store it to calibration.json in output_directory")
            print("  3) Compute calibration and exit")

            print("Waiting for user input...")
            try:
                choice = input("> ").strip()
                print(f"Received choice: '{choice}'")
            except KeyboardInterrupt:
                print("\nInterrupted while waiting for input")
                raise
            
            if choice == '1':
                # Create capture instance
                capture = StereoImageCapture(output_dir=self.raw_dir, config_path=self.config_path)
                
                # Wait for user input
                print("Press Enter to capture...")
                input()
                
                # Capture stereo images
                success = capture.capture_stereo_images()
                print(f"Capture success: {success}")
                if success:
                    # Get captured paths and store them
                    paths = capture.get_latest_captures(1)
                    if len(paths) == 2:
                        pair_dir = self._store_pair(paths, self.collected_image_pairs_number)
                        self.collected_image_pairs_number += 1
                        print(f"Stored pair to: {pair_dir}")
                        print(f"Collected pairs: {self.collected_image_pairs_number}")
                    else:
                        print("Capture did not produce two images")
                else:
                    print("Capture failed")
                
                del capture
                
            elif choice == '2':
                try:
                    results = self._compute_calibration()
                    print_summary(results)
                    save_results(results, os.path.join(self.output_dir, "calibration.json"))
                    print(f"Calibration saved to {os.path.join(self.output_dir, 'calibration.json')}")
                except Exception as e:
                    print(f"Calibration failed: {e}")
                    
            elif choice == '3':
                try:
                    results = self._compute_calibration()
                    print_summary(results)
                    save_results(results, os.path.join(self.output_dir, "calibration.json"))
                    print("Exiting.")
                except Exception as e:
                    print(f"Calibration failed: {e}")
                return
                
            else:
                print("Invalid choice. Enter 1, 2, or 3.")
            
            # Reset terminal settings
            os.system('stty sane')


def main():
    """Main function to run stereo calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration using April Tags')
    parser.add_argument('--config', '-c', default='config.yml', 
                       help='Path to configuration file (default: config.yml)')
    
    args = parser.parse_args()
    
    try:
        # Initialize stereo calibration
        calibrator = StereoCalibration(args.config)
        
        # Run interactive calibration session
        calibrator.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
        import traceback
        print("\nInterruption traceback:")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"Error during calibration: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
