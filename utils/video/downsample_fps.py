import argparse
import subprocess
import os


def downsample_fps(input_path, output_path=None, fps=25):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fps{fps}{ext}"
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", str(fps),
        "-y",  # Overwrite output file without asking
        output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved downsampled video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Downsample a video's FPS using ffmpeg.")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("--fps", type=int, default=25, help="Target FPS (default: 25)")
    parser.add_argument("--output", help="Output video file path (default: <input>_fps<fps>.<ext>)")
    args = parser.parse_args()
    downsample_fps(args.input, args.output, args.fps)


if __name__ == "__main__":
    main() 