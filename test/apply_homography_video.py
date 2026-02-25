import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.court_homography import CourtHomographyDetector

def process_video(
    video_path: str,
    court_model_path: str,
    output_path: str,
    court_conf: float = 0.25,
):
    print("=" * 60)
    print("  Applying Homography to Video")
    print("=" * 60)

    detector = CourtHomographyDetector(court_model_path, conf=court_conf)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"  Resolution: {w}x{h} | FPS: {fps:.0f} | Total Frames: {total_frames}")
    print(f"  Output: {output_path}")

    t_start = time.time()
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        corners, H, mask = detector.detect(frame)

        annotated = frame.copy()
        
        if H is not None:
            annotated = CourtHomographyDetector.project_court_lines(
                annotated, H,
                color=(0, 0, 255),
                thickness=2,
            )
            annotated = CourtHomographyDetector.draw_corners(annotated, corners)

        out.write(annotated)
        
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames", end='\r')

    print(f"\n  Finished processing in {time.time() - t_start:.1f}s")
    
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(Path(__file__).resolve().parent.parent / "res" / "src" / "Final.mp4"))
    parser.add_argument("--court-model", default=str(Path(__file__).resolve().parent.parent / "models" / "court_segment.pt"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "output" / "Final_homography.mp4"))
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    process_video(args.video, args.court_model, args.output)
