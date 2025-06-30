import argparse
import os
import datetime
from video_utils import downsample_video, is_video_file, get_video_fps, get_video_resolution


def create_log_entry(input_path, target_fps=None, target_resolution=None, success=True, error_msg=None, input_fps=None, input_resolution=None):
    """Create a log entry for the processing operation."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_content = f"[{timestamp}] Processing: {input_path}\n"
    
    # Log input video information
    if input_fps is not None:
        log_content += f"  - Input FPS: {input_fps:.2f}\n"
    if input_resolution is not None:
        input_w, input_h = input_resolution
        log_content += f"  - Input Resolution: {input_w}x{input_h}\n"
    
    # Log target settings
    if target_fps:
        log_content += f"  - Target FPS: {target_fps}\n"
    if target_resolution:
        target_h, target_w = target_resolution
        log_content += f"  - Target Resolution: {target_w}x{target_h}\n"
    
    # Log final results
    if success:
        log_content += "  - Status: SUCCESS - Original file replaced\n"
        
        # Try to get final video info
        try:
            final_fps = get_video_fps(input_path)
            final_w, final_h = get_video_resolution(input_path)
            log_content += f"  - Final FPS: {final_fps:.2f}\n"
            log_content += f"  - Final Resolution: {final_w}x{final_h}\n"
        except Exception as e:
            log_content += f"  - Warning: Could not read final video properties: {e}\n"
    else:
        log_content += f"  - Status: FAILED - {error_msg}\n"
    
    log_content += "\n"
    
    # Create log file path
    base_dir = os.path.dirname(input_path)
    log_file = os.path.join(base_dir, "downsample_processing.log")
    
    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"Log entry added to: {log_file}")


def process_single_file(input_path, fps=None, resolution=None):
    """Process a single video file."""
    # Get input video properties before processing
    input_fps = None
    input_resolution = None
    
    try:
        input_fps = get_video_fps(input_path)
        input_resolution = get_video_resolution(input_path)
    except Exception as e:
        print(f"Warning: Could not read input video properties: {e}")
    
    try:
        result_path = downsample_video(input_path, None, fps, resolution)
        create_log_entry(input_path, fps, resolution, success=True,
                        input_fps=input_fps, input_resolution=input_resolution)
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {input_path}: {error_msg}")
        create_log_entry(input_path, fps, resolution, success=False, error_msg=error_msg,
                        input_fps=input_fps, input_resolution=input_resolution)
        return False


def parse_resolution(resolution_str):
    """Parse resolution string in format 'HxW' or 'H,W'."""
    if not resolution_str:
        return None
    
    # Try different separators
    for sep in ['x', 'X', ',', ':']:
        if sep in resolution_str:
            parts = resolution_str.split(sep)
            if len(parts) == 2:
                try:
                    height, width = int(parts[0]), int(parts[1])
                    return (height, width)
                except ValueError:
                    continue
    
    raise ValueError(f"Invalid resolution format: {resolution_str}. Use HxW (e.g., 720x1280)")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample video FPS and/or resolution using ffmpeg. Replaces original files and creates processing logs."
    )
    parser.add_argument("input", help="Input video file path or folder")
    parser.add_argument("--fps", type=int, help="Target FPS (e.g., 25)")
    parser.add_argument("--resolution", type=str, help="Target resolution as HxW (e.g., 720x1280)")
    parser.add_argument("--output", help="Output video file path (for single file input only, overrides replace behavior)")
    
    args = parser.parse_args()

    # Parse resolution if provided
    resolution = None
    if args.resolution:
        resolution = parse_resolution(args.resolution)
        print(f"Target resolution: {resolution[1]}x{resolution[0]} (WxH)")

    # Check if we have any processing to do
    if not args.fps and not args.resolution:
        print("Error: Must specify at least --fps or --resolution")
        return

    if os.path.isdir(args.input):
        # Process all videos in directory
        print(f"\n=== Scanning directory for video files ===")
        
        # First, collect all video files to get accurate count
        video_files = []
        for fname in os.listdir(args.input):
            fpath = os.path.join(args.input, fname)
            if os.path.isfile(fpath) and is_video_file(fpath):
                video_files.append((fname, fpath))
        
        total_count = len(video_files)
        processed_count = 0
        
        if total_count == 0:
            print("No video files found in directory.")
            return
        
        print(f"Found {total_count} video file(s) to process")
        print(f"=== Starting batch processing ===\n")
        
        for idx, (fname, fpath) in enumerate(video_files, 1):
            percentage = (idx / total_count) * 100
            print(f"[{idx}/{total_count}] ({percentage:.1f}%) Processing: {fname}")
            
            if process_single_file(fpath, args.fps, resolution):
                processed_count += 1
                print(f"  ✅ Success - {fname} processed")
            else:
                print(f"  ❌ Failed - {fname} processing failed")
        
        print(f"\n=== Processing Summary ===")
        print(f"Total videos found: {total_count}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {total_count - processed_count}")
        
        if processed_count == total_count:
            print("🎉 All videos processed successfully!")
        elif processed_count > 0:
            print(f"⚠️  {total_count - processed_count} video(s) failed to process")
        else:
            print("❌ No videos were processed successfully")
        
    else:
        # Process single file
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        
        if not is_video_file(args.input):
            print(f"Error: File is not a video: {args.input}")
            return
        
        # If output is specified, use different function call that doesn't replace
        if args.output:
            # Get input video properties before processing
            input_fps = None
            input_resolution = None
            
            try:
                input_fps = get_video_fps(args.input)
                input_resolution = get_video_resolution(args.input)
            except Exception as e:
                print(f"Warning: Could not read input video properties: {e}")
            
            try:
                downsample_video(args.input, args.output, args.fps, resolution)
                create_log_entry(args.input, args.fps, resolution, success=True,
                                input_fps=input_fps, input_resolution=input_resolution)
                print(f"Video saved to: {args.output}")
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing {args.input}: {error_msg}")
                create_log_entry(args.input, args.fps, resolution, success=False, error_msg=error_msg,
                                input_fps=input_fps, input_resolution=input_resolution)
        else:
            process_single_file(args.input, args.fps, resolution)


if __name__ == "__main__":
    main()