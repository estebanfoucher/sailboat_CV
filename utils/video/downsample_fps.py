import argparse
from video_utils import downsample_fps


def main():
    parser = argparse.ArgumentParser(description="Downsample a video's FPS using ffmpeg.")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("--fps", type=int, default=25, help="Target FPS (default: 25)")
    parser.add_argument("--output", help="Output video file path (default: <input>_fps<fps>.<ext>)")
    args = parser.parse_args()
    downsample_fps(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()