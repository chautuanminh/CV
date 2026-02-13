"""
Frame extraction utility.
Extracts frames from video files at a specified FPS for labeling and training.
"""

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VIDEO, DATA_DIR


def extract_frames(
    video_path: str,
    output_dir: str | None = None,
    fps: int | None = None,
    max_frames: int | None = None,
    prefix: str = "frame",
) -> list[str]:
    """
    Extract frames from a video file.

    Args:
        video_path:  Path to the source video.
        output_dir:  Directory to save extracted frames (created if needed).
        fps:         Frames per second to extract (None = use config default).
        max_frames:  Maximum number of frames to extract (None = all).
        prefix:      Filename prefix for saved frames.

    Returns:
        List of saved frame file paths.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir) if output_dir else DATA_DIR / "frames" / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = fps or VIDEO["frame_extraction_fps"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(video_fps / fps))

    print(f"Video: {video_path.name}")
    print(f"  Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  Video FPS  : {video_fps:.1f}")
    print(f"  Extract FPS: {fps}  (every {frame_interval} frame(s))")
    print(f"  Total frames in video: {total_frames}")
    print(f"  Output dir : {output_dir}")

    saved_paths = []
    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=max_frames or (total_frames // frame_interval), desc="Extracting")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{prefix}_{frame_idx:06d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(str(filepath))
            saved_count += 1
            pbar.update(1)

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    pbar.close()
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a tennis match video.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("-o", "--output", help="Output directory for frames")
    parser.add_argument("-f", "--fps", type=int, default=None, help="Frames per second to extract")
    parser.add_argument("-n", "--max-frames", type=int, default=None, help="Maximum frames to extract")
    parser.add_argument("-p", "--prefix", default="frame", help="Filename prefix")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps, args.max_frames, args.prefix)
