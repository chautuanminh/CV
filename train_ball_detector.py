"""
Train the ball detection model using YOLOv8.

Fine-tunes YOLOv8 at high resolution (1280px) for small-object (tennis ball)
detection. Includes post-processing for trajectory interpolation.

Usage:
    python train_ball_detector.py train
    python train_ball_detector.py train --epochs 100 --imgsz 1280
    python train_ball_detector.py detect video.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BALL_MODEL_YOLO, WEIGHTS_DIR


def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("  Ball Detection — YOLOv8 Training")
    print("=" * 60)

    model = YOLO(args.model or BALL_MODEL_YOLO["base_weights"])
    print(f"Base model : {args.model or BALL_MODEL_YOLO['base_weights']}")
    print(f"Dataset    : {args.data}")
    print(f"Epochs     : {args.epochs}")
    print(f"Image size : {args.imgsz}  (high-res for small objects)")
    print(f"Batch size : {args.batch}")
    print()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=BALL_MODEL_YOLO["optimizer"],
        lr0=BALL_MODEL_YOLO["lr"],
        project=str(WEIGHTS_DIR),
        name="ball_detector",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    val_results = model.val()
    print(f"  mAP@0.5     : {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")

    best_path = WEIGHTS_DIR / "ball_detector" / "weights" / "best.pt"
    print(f"\nBest weights: {best_path}")
    return str(best_path)


def interpolate_ball_positions(detections: list[dict], max_gap: int = 10) -> pd.DataFrame:
    """
    Fill gaps in ball detections using polynomial interpolation.

    Args:
        detections: List of {"frame": int, "x": float|None, "y": float|None}
        max_gap:    Maximum consecutive missing frames to interpolate.

    Returns:
        DataFrame with columns [frame, x, y, interpolated].
    """
    df = pd.DataFrame(detections)
    df = df.set_index("frame").sort_index()
    df = df.reindex(range(df.index.min(), df.index.max() + 1))

    # Mark which frames are interpolated
    df["interpolated"] = df["x"].isna()

    # Only interpolate gaps smaller than max_gap
    missing = df["x"].isna()
    gap_groups = (~missing).cumsum()
    gap_sizes = missing.groupby(gap_groups).transform("sum")
    fillable = missing & (gap_sizes <= max_gap)

    # Polynomial interpolation for fillable gaps
    order = BALL_MODEL_YOLO["interpolation_order"]
    df.loc[fillable, "x"] = df["x"].interpolate(method="polynomial", order=order)[fillable]
    df.loc[fillable, "y"] = df["y"].interpolate(method="polynomial", order=order)[fillable]

    return df.reset_index().rename(columns={"index": "frame"})


def detect_in_video(args):
    """Run ball detection on a video and save positions CSV."""
    from ultralytics import YOLO

    import cv2

    model_path = args.weights or str(WEIGHTS_DIR / "ball_detector" / "weights" / "best.pt")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total} frames at {fps:.1f} FPS")

    detections = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=BALL_MODEL_YOLO["confidence_threshold"], verbose=False)

        x, y = None, None
        if len(results[0].boxes) > 0:
            # Take the highest-confidence detection
            best = results[0].boxes[results[0].boxes.conf.argmax()]
            box = best.xyxy[0].cpu().numpy()
            x = float((box[0] + box[2]) / 2)
            y = float((box[1] + box[3]) / 2)

        detections.append({"frame": frame_idx, "x": x, "y": y})
        frame_idx += 1

    cap.release()

    # Interpolate missing positions
    df = interpolate_ball_positions(detections)

    output_csv = Path(args.video).stem + "_ball_positions.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} frame positions to {output_csv}")
    print(f"Detection rate: {(~df['interpolated']).mean():.1%}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / run YOLOv8 ball detector")
    sub = parser.add_subparsers(dest="command")

    # Train
    t = sub.add_parser("train", help="Train the ball detector")
    t.add_argument("--data", default=BALL_MODEL_YOLO["dataset_yaml"])
    t.add_argument("--model", default=None)
    t.add_argument("--epochs", type=int, default=BALL_MODEL_YOLO["epochs"])
    t.add_argument("--imgsz", type=int, default=BALL_MODEL_YOLO["imgsz"])
    t.add_argument("--batch", type=int, default=BALL_MODEL_YOLO["batch_size"])
    t.add_argument("--device", default="0")

    # Detect
    d = sub.add_parser("detect", help="Detect ball in video")
    d.add_argument("video", help="Path to video file")
    d.add_argument("--weights", default=None)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "detect":
        detect_in_video(args)
    else:
        parser.print_help()
