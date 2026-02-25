"""
Court Homography Pre-Check — Sample frames every 3 seconds
============================================================
Before running full inference, this script:
  1. Samples 1 frame every 3 seconds from the video (up to 30 frames).
  2. Runs YOLO court segmentation + homography on each sampled frame.
  3. Draws projected court lines on each frame.
  4. Saves every annotated frame as an individual image in a subfolder.

This lets you visually inspect whether the homography is stable and
correct across different moments in the video before committing to
a full pipeline run.

Usage:
    python test/homography_precheck.py --video res/src/Final.mp4
    python test/homography_precheck.py --video res/src/Final.mp4 --interval 5 --max-frames 20
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.court_homography import CourtHomographyDetector


def run_precheck(
    video_path: str,
    court_model_path: str,
    output_dir: str,
    interval_sec: float = 3.0,
    max_frames: int = 30,
    court_conf: float = 0.25,
):
    """
    Sample frames from the video at fixed intervals, run court detection
    + homography on each, and save annotated images.
    """

    # ── Load model ───────────────────────────────────────────────────
    print("=" * 60)
    print("  Court Homography Pre-Check")
    print("=" * 60)

    print(f"\n  Video    : {video_path}")
    print(f"  Model    : {court_model_path}")
    print(f"  Interval : every {interval_sec}s")
    print(f"  Max frames: {max_frames}")

    detector = CourtHomographyDetector(court_model_path, conf=court_conf)

    # ── Open video ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"  Resolution: {w}×{h} | FPS: {fps:.0f} | Duration: {duration_sec:.1f}s | Frames: {total_frames}")

    # ── Output subfolder ─────────────────────────────────────────────
    basename = Path(video_path).stem
    sub_dir = os.path.join(output_dir, f"{basename}_homography_precheck")
    os.makedirs(sub_dir, exist_ok=True)
    print(f"  Output   : {sub_dir}/")

    # ── Compute which frame indices to sample ────────────────────────
    frame_interval = int(fps * interval_sec)
    sample_indices = []
    idx = 0
    while idx < total_frames and len(sample_indices) < max_frames:
        sample_indices.append(idx)
        idx += frame_interval

    print(f"\n  Sampling {len(sample_indices)} frames: {sample_indices[:5]}{'…' if len(sample_indices) > 5 else ''}")
    print()

    # ── Process each sampled frame ───────────────────────────────────
    t_start = time.time()
    results_summary = []

    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  [{i+1}/{len(sample_indices)}] Frame {frame_idx} — could not read, skipping.")
            continue

        timestamp = frame_idx / fps

        # Run court detection
        corners, H, mask = detector.detect(frame)

        # Build annotated frame
        annotated = frame.copy()
        status = "FAIL"

        if H is not None:
            # Draw court lines
            annotated = CourtHomographyDetector.project_court_lines(
                annotated, H,
                color=(0, 0, 255),
                thickness=2,
            )
            # Draw corners
            annotated = CourtHomographyDetector.draw_corners(annotated, corners)
            status = "OK"

        # Add info text overlay
        info_text = f"Frame {frame_idx} | t={timestamp:.1f}s | Court: {status}"
        cv2.putText(annotated, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

        if corners is not None:
            corner_text = f"TL:({int(corners[0][0])},{int(corners[0][1])}) TR:({int(corners[1][0])},{int(corners[1][1])}) BR:({int(corners[2][0])},{int(corners[2][1])}) BL:({int(corners[3][0])},{int(corners[3][1])})"
            cv2.putText(annotated, corner_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, corner_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Save image
        img_name = f"frame_{frame_idx:06d}_t{timestamp:.1f}s.jpg"
        img_path = os.path.join(sub_dir, img_name)
        cv2.imwrite(img_path, annotated)

        print(f"  [{i+1}/{len(sample_indices)}] Frame {frame_idx:5d} | t={timestamp:6.1f}s | {status:4s} → {img_name}")

        results_summary.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "status": status,
            "corners": corners.tolist() if corners is not None else None,
        })

    cap.release()
    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────────
    ok_count = sum(1 for r in results_summary if r["status"] == "OK")
    fail_count = len(results_summary) - ok_count

    print(f"\n{'=' * 60}")
    print(f"  Pre-check complete in {elapsed:.1f}s")
    print(f"  Frames sampled : {len(results_summary)}")
    print(f"  Court detected : {ok_count} ✅")
    print(f"  Court missed   : {fail_count} ❌")
    print(f"  Output folder  : {sub_dir}")
    print(f"{'=' * 60}")

    if fail_count > 0:
        print(f"\n  ⚠️  Court detection failed on {fail_count} frames.")
        print(f"     Check the images in {sub_dir}/ to diagnose.")
    else:
        print(f"\n  ✅ All frames passed! Court is stable across the video.")

    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Court Homography Pre-Check — sample frames and verify court detection"
    )
    parser.add_argument(
        "--video",
        default=str(Path(__file__).resolve().parent.parent / "res" / "src" / "short_2023PPA.mp4"),
        help="Input video path",
    )
    parser.add_argument(
        "--court-model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "court_segment.pt"),
        help="Court segmentation model",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "output"))
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between samples (default: 3)")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to sample (default: 30)")
    parser.add_argument("--court-conf", type=float, default=0.25)
    args = parser.parse_args()

    run_precheck(
        video_path=args.video,
        court_model_path=args.court_model,
        output_dir=args.output_dir,
        interval_sec=args.interval,
        max_frames=args.max_frames,
        court_conf=args.court_conf,
    )


if __name__ == "__main__":
    main()
