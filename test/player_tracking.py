"""
Player Tracking — YOLO + ByteTrack
====================================
Runs YOLO object detection with ByteTrack tracking on a pickleball video.
Outputs an annotated video with tracked player bounding boxes and IDs.

Usage:
    python test/player_tracking.py [--model models/player.pt] [--video res/src/short_2023PPA.mp4]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def run_player_tracking(
    model_path: str,
    video_path: str,
    output_dir: str,
    conf: float = 0.3,
    iou: float = 0.5,
    show_preview: bool = False,
):
    """Run YOLO + ByteTrack inference on a video and save annotated output."""

    # ── Load model ───────────────────────────────────────────────────
    print(f"[1/3] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # ── Open video ───────────────────────────────────────────────────
    print(f"[2/3] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      Resolution: {w}×{h} | FPS: {fps} | Frames: {total_frames}")

    # ── Output writer ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{basename}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ── Colour palette for track IDs ─────────────────────────────────
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    # ── Tracking loop ────────────────────────────────────────────────
    print(f"[3/3] Running inference with ByteTrack (conf={conf}, iou={iou}) …")
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO + ByteTrack
        results = model.track(
            source=frame,
            tracker="bytetrack.yaml",
            conf=conf,
            iou=iou,
            persist=True,
            verbose=False,
        )

        result = results[0]
        annotated = frame.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else [None] * len(boxes)
            )
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names if hasattr(result, "names") else {}

            for box, c, tid, cls_id in zip(boxes, confs, track_ids, classes):
                x1, y1, x2, y2 = box
                color = tuple(int(v) for v in palette[tid % len(palette)]) if tid is not None else (0, 255, 0)

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label
                cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)
                label = f"ID {tid} {cls_name} {c:.2f}" if tid is not None else f"{cls_name} {c:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"      Frame {frame_idx}/{total_frames} ({fps_proc:.1f} fps)")

        if show_preview:
            cv2.imshow("Player Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n✅ Done! Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
    print(f"   Output → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Player tracking with YOLO + ByteTrack")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "player.pt"),
        help="Path to YOLO player model (default: models/player.pt)",
    )
    parser.add_argument(
        "--video",
        default=str(Path(__file__).resolve().parent.parent / "res" / "src" / "short_2023PPA.mp4"),
        help="Path to input video (default: res/src/short_2023PPA.mp4)",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent), help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    run_player_tracking(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        show_preview=args.show,
    )


if __name__ == "__main__":
    main()
