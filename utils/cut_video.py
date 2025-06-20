import subprocess
import os
import sys

def cut_video(input_path, start_sec, end_sec, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    output_path = os.path.join(output_dir, f"{name}_{start_sec}_{end_sec}{ext}")
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved cut video to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python cut_video.py <input.mkv> <start_sec> <end_sec> <output_dir>")
        sys.exit(1)
    cut_video(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]) 