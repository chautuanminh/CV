"""
Full Pickleball Pipeline — Court Homography + Player Tracking + Ball Tracking
==============================================================================
Runs all three YOLO models on a video and outputs an annotated video with:
  - Court lines projected via homography
  - Player bounding boxes with ByteTrack IDs
  - Ball detections with trailing path

Usage:
    python test/full_pipeline.py --video res/src/short_2023PPA.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.court_homography import CourtHomographyDetector


def run_full_pipeline(
    video_path: str,
    court_model_path: str,
    player_model_path: str,
    ball_model_path: str,
    output_dir: str,
    court_conf: float = 0.25,
    player_conf: float = 0.3,
    ball_conf: float = 0.15,
    player_iou: float = 0.5,
    show_preview: bool = False,
):
    """Run court homography + player tracking + ball tracking on a video."""

    # ── Load models ──────────────────────────────────────────────────
    print("=" * 60)
    print("  Pickleball Full Pipeline")
    print("=" * 60)

    print(f"\n[1/5] Loading models …")
    print(f"      Court  : {court_model_path}")
    court_detector = CourtHomographyDetector(court_model_path, conf=court_conf)

    print(f"      Player : {player_model_path}")
    player_model = YOLO(player_model_path)

    print(f"      Ball   : {ball_model_path}")
    ball_model = YOLO(ball_model_path)

    # ── Open video ───────────────────────────────────────────────────
    print(f"\n[2/5] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      Resolution: {w}×{h} | FPS: {fps} | Frames: {total_frames}")

    # ── Output writer ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{basename}_full_pipeline.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ── Colour palette for player track IDs ──────────────────────────
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    # ── Wait to compute court on each frame instead of first ─────────
    # We will detect the court dynamically frame-by-frame
    
    # Reset video to start (if we read it)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Ball trail buffer ────────────────────────────────────────────
    ball_trail = []  # list of (x, y) or None
    TRAIL_LENGTH = 30

    # ── Main processing loop ─────────────────────────────────────────
    print(f"\n[4/5] Processing {total_frames} frames …")
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # ── Court detect (every frame) ────────────────────────
        corners, court_H, court_mask = court_detector.detect(frame)
        if court_H is not None:
            annotated = CourtHomographyDetector.project_court_lines(
                annotated, court_H,
                color=(255, 50, 50),   # Blue-ish red
                thickness=2,
            )
            annotated = CourtHomographyDetector.draw_corners(annotated, corners)

        # ── Player tracking ──────────────────────────────────────────
        player_results = player_model.track(
            source=frame,
            tracker="bytetrack.yaml",
            conf=player_conf,
            iou=player_iou,
            persist=True,
            verbose=False,
        )

        p_result = player_results[0]
        if p_result.boxes is not None and len(p_result.boxes) > 0:
            boxes = p_result.boxes.xyxy.cpu().numpy().astype(int)
            confs = p_result.boxes.conf.cpu().numpy()
            track_ids = (
                p_result.boxes.id.cpu().numpy().astype(int)
                if p_result.boxes.id is not None
                else [None] * len(boxes)
            )
            class_names = p_result.names if hasattr(p_result, "names") else {}
            classes = p_result.boxes.cls.cpu().numpy().astype(int)

            for box, c, tid, cls_id in zip(boxes, confs, track_ids, classes):
                x1, y1, x2, y2 = box
                color = tuple(int(v) for v in palette[tid % len(palette)]) if tid is not None else (0, 255, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)
                label = f"ID {tid} {cls_name} {c:.2f}" if tid is not None else f"{cls_name} {c:.2f}"
                (tw, th_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th_txt - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Ball detection ───────────────────────────────────────────
        ball_results = ball_model(frame, conf=ball_conf, verbose=False)
        b_result = ball_results[0]

        ball_pos = None
        if b_result.boxes is not None and len(b_result.boxes) > 0:
            best_idx = b_result.boxes.conf.argmax()
            best = b_result.boxes[best_idx]
            bx = best.xyxy[0].cpu().numpy()
            cx = int((bx[0] + bx[2]) / 2)
            cy = int((bx[1] + bx[3]) / 2)
            ball_pos = (cx, cy)

        ball_trail.append(ball_pos)
        if len(ball_trail) > TRAIL_LENGTH:
            ball_trail.pop(0)

        # Draw ball trail
        valid_trail = [(i, pt) for i, pt in enumerate(ball_trail) if pt is not None]
        for idx in range(1, len(valid_trail)):
            i_prev, pt_prev = valid_trail[idx - 1]
            i_curr, pt_curr = valid_trail[idx]
            alpha = int(255 * ((i_curr + 1) / len(ball_trail)))
            trail_color = (0, alpha, 255)  # Yellow-orange gradient
            thickness = max(1, int(3 * (i_curr + 1) / len(ball_trail)))
            cv2.line(annotated, pt_prev, pt_curr, trail_color, thickness, cv2.LINE_AA)

        # Draw current ball position
        if ball_pos is not None:
            cv2.circle(annotated, ball_pos, 8, (0, 255, 255), -1)
            cv2.circle(annotated, ball_pos, 10, (0, 200, 200), 2)

        # ── Write frame ──────────────────────────────────────────────
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"      Frame {frame_idx}/{total_frames} ({fps_proc:.1f} fps)")

        if show_preview:
            cv2.imshow("Pickleball Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    fps_proc = frame_idx / elapsed if elapsed > 0 else 0

    print(f"\n[5/5] Done!")
    print("=" * 60)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Processing speed : {fps_proc:.1f} FPS ({elapsed:.1f}s)")
    print(f"  Output           : {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Full Pickleball Pipeline — Court + Players + Ball"
    )
    parser.add_argument(
        "--video",
        default=str(Path(__file__).resolve().parent.parent / "res" / "src" / "Final.mp4"),
        help="Input video path",
    )
    parser.add_argument(
        "--court-model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "court_segment.pt"),
        help="Court segmentation model",
    )
    parser.add_argument(
        "--player-model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "player.pt"),
        help="Player detection model",
    )
    parser.add_argument(
        "--ball-model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "ball_tracking" / "ball_tracking" / "train2" / "weights" / "best.pt"),
        help="Ball detection model",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "output"))
    parser.add_argument("--court-conf", type=float, default=0.25)
    parser.add_argument("--player-conf", type=float, default=0.3)
    parser.add_argument("--ball-conf", type=float, default=0.15)
    parser.add_argument("--player-iou", type=float, default=0.5)
    parser.add_argument("--show", action="store_true", help="Show live preview")
    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video,
        court_model_path=args.court_model,
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        output_dir=args.output_dir,
        court_conf=args.court_conf,
        player_conf=args.player_conf,
        ball_conf=args.ball_conf,
        player_iou=args.player_iou,
        show_preview=args.show,
    )


if __name__ == "__main__":
    main()
